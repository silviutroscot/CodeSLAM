from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import CameraModel, LossConfig, OptimizationConfig, SLAMConfig, TrackingConfig
from .network import CodeSLAMDepthModel
from .optimization import (
    build_prior_factor,
    camera_pyramid,
    make_window_residual_fn,
    optimize_tracking_pose,
    optimize_window,
    prepare_frame_context,
)
from .pose import inverse, rotation_angle, translation_norm
from .types import FrameBatch, Keyframe, PriorFactor, TrackingResult


@dataclass
class ProcessedFrame:
    pose_w_c: torch.Tensor
    tracking: TrackingResult | None
    inserted_keyframe: bool


class CodeSLAMSystem:
    """Sliding-window monocular CodeSLAM system following the paper."""

    def __init__(
        self,
        model: CodeSLAMDepthModel,
        camera: CameraModel,
        *,
        loss_config: LossConfig | None = None,
        optimization_config: OptimizationConfig | None = None,
        tracking_config: TrackingConfig | None = None,
        slam_config: SLAMConfig | None = None,
    ) -> None:
        self.model = model
        self.camera = camera
        self.loss_config = loss_config or LossConfig()
        self.optimization_config = optimization_config or OptimizationConfig()
        self.tracking_config = tracking_config or TrackingConfig()
        self.slam_config = slam_config or SLAMConfig()

        self.keyframes: list[Keyframe] = []
        self.prior: PriorFactor | None = None
        self.latest_pose_w_c: torch.Tensor | None = None
        self.frame_counter = 0
        self.trajectory: dict[str, torch.Tensor] = {}

    def _new_frame_id(self, prefix: str) -> str:
        self.frame_counter += 1
        return f"{prefix}_{self.frame_counter:06d}"

    def _prediction_cameras(self) -> list[CameraModel]:
        return camera_pyramid(self.camera.scaled(0.5), self.model.config.pyramid_levels)

    def _mapping_edges(self, keyframes: list[Keyframe]) -> list[tuple[str, str]]:
        ids = [keyframe.keyframe_id for keyframe in keyframes]
        if len(ids) <= 1:
            return []
        if self.slam_config.mapping_edges == "consecutive":
            return [(ids[index], ids[index + 1]) for index in range(len(ids) - 1)]
        edges = []
        for left in range(len(ids)):
            for right in range(left + 1, len(ids)):
                edges.append((ids[left], ids[right]))
        return edges

    def _keyframe_contexts(self, keyframes: list[Keyframe]) -> dict[str, object]:
        return {
            keyframe.keyframe_id: prepare_frame_context(self.model, keyframe.keyframe_id, keyframe.intensity)
            for keyframe in keyframes
        }

    def _base_poses(self, keyframes: list[Keyframe]) -> dict[str, torch.Tensor]:
        return {keyframe.keyframe_id: keyframe.pose_w_c for keyframe in keyframes}

    def _base_codes(self, keyframes: list[Keyframe]) -> dict[str, torch.Tensor]:
        return {keyframe.keyframe_id: keyframe.code for keyframe in keyframes}

    def _update_keyframes(self, poses: dict[str, torch.Tensor], codes: dict[str, torch.Tensor]) -> None:
        for keyframe in self.keyframes:
            if keyframe.keyframe_id in poses:
                keyframe.pose_w_c = poses[keyframe.keyframe_id].detach().clone()
                keyframe.code = codes[keyframe.keyframe_id].detach().clone()

    def _should_create_keyframe(self, pose_w_c: torch.Tensor) -> bool:
        if not self.keyframes:
            return True
        reference_pose = self.keyframes[-1].pose_w_c
        relative = inverse(reference_pose) @ pose_w_c
        return (
            translation_norm(relative).item() > self.slam_config.keyframe_translation_threshold
            or rotation_angle(relative).item() > self.slam_config.keyframe_rotation_threshold_deg
        )

    def _insert_keyframe(self, frame: FrameBatch, pose_w_c: torch.Tensor) -> Keyframe:
        keyframe_id = frame.frame_id or self._new_frame_id("kf")
        code = self.model.zero_code(1, frame.intensity.device, frame.intensity.dtype).squeeze(0)
        keyframe = Keyframe(
            keyframe_id=keyframe_id,
            intensity=frame.intensity,
            pose_w_c=pose_w_c.detach().clone(),
            code=code,
            timestamp=frame.timestamp,
            depth=frame.depth,
            metadata=dict(frame.metadata),
        )
        self.keyframes.append(keyframe)
        self.trajectory[keyframe_id] = keyframe.pose_w_c
        return keyframe

    def bootstrap(self, first_frame: FrameBatch, second_frame: FrameBatch) -> tuple[Keyframe, Keyframe]:
        first_id = first_frame.frame_id or self._new_frame_id("boot")
        second_id = second_frame.frame_id or self._new_frame_id("boot")
        first_pose = torch.eye(4, device=first_frame.intensity.device, dtype=first_frame.intensity.dtype)
        second_pose = torch.eye(4, device=first_frame.intensity.device, dtype=first_frame.intensity.dtype)
        second_pose[2, 3] = self.slam_config.bootstrap_motion_prior

        contexts = {
            first_id: prepare_frame_context(self.model, first_id, first_frame.intensity),
            second_id: prepare_frame_context(self.model, second_id, second_frame.intensity),
        }
        base_poses = {first_id: first_pose, second_id: second_pose}
        zero_code = self.model.zero_code(1, first_frame.intensity.device, first_frame.intensity.dtype).squeeze(0)
        base_codes = {first_id: zero_code.clone(), second_id: zero_code.clone()}

        local_optimization = OptimizationConfig(
            iterations=self.optimization_config.pair_initialization_iterations,
            damping=self.optimization_config.damping,
            damping_multiplier=self.optimization_config.damping_multiplier,
            min_improvement=self.optimization_config.min_improvement,
            max_residuals_per_level=self.optimization_config.max_residuals_per_level,
            pair_initialization_iterations=self.optimization_config.pair_initialization_iterations,
            mapping_iterations=self.optimization_config.mapping_iterations,
            tracking_iterations_per_level=self.optimization_config.tracking_iterations_per_level,
        )

        poses, codes, _, _ = optimize_window(
            self.model,
            contexts,
            base_poses,
            base_codes,
            self._prediction_cameras(),
            [(first_id, second_id)],
            self.loss_config,
            local_optimization,
            prior=None,
            gauge_keyframe_id=first_id,
            gauge_reference_pose=first_pose,
            use_affine_brightness=self.tracking_config.use_affine_brightness,
            include_geometric_term=True,
        )

        first_keyframe = Keyframe(
            keyframe_id=first_id,
            intensity=first_frame.intensity,
            pose_w_c=poses[first_id].detach().clone(),
            code=codes[first_id].detach().clone(),
            timestamp=first_frame.timestamp,
            depth=first_frame.depth,
            metadata=dict(first_frame.metadata),
        )
        second_keyframe = Keyframe(
            keyframe_id=second_id,
            intensity=second_frame.intensity,
            pose_w_c=poses[second_id].detach().clone(),
            code=codes[second_id].detach().clone(),
            timestamp=second_frame.timestamp,
            depth=second_frame.depth,
            metadata=dict(second_frame.metadata),
        )
        self.keyframes = [first_keyframe, second_keyframe]
        self.latest_pose_w_c = second_keyframe.pose_w_c
        self.trajectory[first_id] = first_keyframe.pose_w_c
        self.trajectory[second_id] = second_keyframe.pose_w_c
        return first_keyframe, second_keyframe

    def map_window(self) -> None:
        if len(self.keyframes) < 2:
            return

        contexts = self._keyframe_contexts(self.keyframes)
        base_poses = self._base_poses(self.keyframes)
        base_codes = self._base_codes(self.keyframes)
        gauge_keyframe_id = self.keyframes[0].keyframe_id
        gauge_reference_pose = self.keyframes[0].pose_w_c
        edges = self._mapping_edges(self.keyframes)

        poses, codes, _, layout = optimize_window(
            self.model,
            contexts,
            base_poses,
            base_codes,
            self._prediction_cameras(),
            edges,
            self.loss_config,
            OptimizationConfig(
                iterations=self.optimization_config.mapping_iterations,
                damping=self.optimization_config.damping,
                damping_multiplier=self.optimization_config.damping_multiplier,
                min_improvement=self.optimization_config.min_improvement,
                max_residuals_per_level=self.optimization_config.max_residuals_per_level,
                pair_initialization_iterations=self.optimization_config.pair_initialization_iterations,
                mapping_iterations=self.optimization_config.mapping_iterations,
                tracking_iterations_per_level=self.optimization_config.tracking_iterations_per_level,
            ),
            prior=self.prior,
            gauge_keyframe_id=gauge_keyframe_id,
            gauge_reference_pose=gauge_reference_pose,
            use_affine_brightness=self.tracking_config.use_affine_brightness,
            include_geometric_term=True,
        )
        self._update_keyframes(poses, codes)

        if len(self.keyframes) > self.slam_config.max_keyframes:
            drop_id = self.keyframes[0].keyframe_id

            updated_contexts = self._keyframe_contexts(self.keyframes)
            updated_poses = self._base_poses(self.keyframes)
            updated_codes = self._base_codes(self.keyframes)
            residual_fn, updated_layout = make_window_residual_fn(
                self.model,
                updated_contexts,
                updated_poses,
                updated_codes,
                self._prediction_cameras(),
                self._mapping_edges(self.keyframes),
                self.loss_config,
                self.optimization_config,
                prior=self.prior,
                gauge_keyframe_id=gauge_keyframe_id,
                gauge_reference_pose=gauge_reference_pose,
                use_affine_brightness=self.tracking_config.use_affine_brightness,
                include_geometric_term=True,
            )
            self.prior = build_prior_factor(
                residual_fn,
                updated_poses,
                updated_codes,
                updated_layout,
                {drop_id},
                self.loss_config,
            )
            self.keyframes = [keyframe for keyframe in self.keyframes if keyframe.keyframe_id != drop_id]

    def process_frame(self, frame: FrameBatch) -> ProcessedFrame:
        if frame.frame_id is None:
            frame.frame_id = self._new_frame_id("frame")

        if not self.keyframes:
            keyframe = self._insert_keyframe(frame, torch.eye(4, device=frame.intensity.device, dtype=frame.intensity.dtype))
            self.latest_pose_w_c = keyframe.pose_w_c
            return ProcessedFrame(pose_w_c=keyframe.pose_w_c, tracking=None, inserted_keyframe=True)

        if len(self.keyframes) == 1:
            _, second_keyframe = self.bootstrap(
                FrameBatch(
                    intensity=self.keyframes[0].intensity,
                    depth=self.keyframes[0].depth,
                    timestamp=self.keyframes[0].timestamp,
                    frame_id=self.keyframes[0].keyframe_id,
                    metadata=dict(self.keyframes[0].metadata),
                ),
                frame,
            )
            self.latest_pose_w_c = second_keyframe.pose_w_c
            return ProcessedFrame(pose_w_c=second_keyframe.pose_w_c, tracking=None, inserted_keyframe=True)

        reference_keyframe = self.keyframes[-1]
        initial_pose = self.latest_pose_w_c if self.latest_pose_w_c is not None else reference_keyframe.pose_w_c
        tracking = optimize_tracking_pose(
            reference_keyframe,
            frame,
            initial_pose,
            self.model,
            self.camera,
            self.loss_config,
            self.optimization_config,
            self.tracking_config,
        )
        self.latest_pose_w_c = tracking.pose_w_c
        self.trajectory[frame.frame_id] = tracking.pose_w_c

        inserted = False
        if self._should_create_keyframe(tracking.pose_w_c):
            self._insert_keyframe(frame, tracking.pose_w_c)
            self.map_window()
            inserted = True

        return ProcessedFrame(pose_w_c=tracking.pose_w_c, tracking=tracking, inserted_keyframe=inserted)
