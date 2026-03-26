from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch

from .config import CameraModel, LossConfig, ModelConfig, OptimizationConfig, TrackingConfig
from .geometry import build_pyramid, fixed_grid_indices, surface_slant_weight, warp_from_reference
from .losses import weighted_residual
from .network import CodeSLAMDepthModel
from .pose import compose, inverse, relative_twist, se3_exp
from .types import FrameBatch, Keyframe, PriorFactor, TrackingResult


@dataclass
class FrameContext:
    frame_id: str
    intensity: torch.Tensor
    image_features: list[torch.Tensor]
    intensity_pyramid: list[torch.Tensor]


@dataclass
class LeastSquaresResult:
    solution: torch.Tensor
    cost: float
    converged: bool
    iterations: int
    residual: torch.Tensor


@dataclass(frozen=True)
class StateLayout:
    keyframe_ids: tuple[str, ...]
    code_dim: int

    @property
    def num_keyframes(self) -> int:
        return len(self.keyframe_ids)

    @property
    def total_dim(self) -> int:
        return self.num_keyframes * (6 + self.code_dim)

    def pose_slice(self, keyframe_id: str) -> slice:
        index = self.keyframe_ids.index(keyframe_id)
        return slice(index * 6, (index + 1) * 6)

    def code_slice(self, keyframe_id: str) -> slice:
        offset = self.num_keyframes * 6
        index = self.keyframe_ids.index(keyframe_id)
        start = offset + index * self.code_dim
        return slice(start, start + self.code_dim)


def camera_pyramid(camera: CameraModel, levels: int) -> list[CameraModel]:
    cameras = [camera]
    for _ in range(1, levels):
        cameras.append(cameras[-1].scaled(0.5))
    return list(reversed(cameras))


def prepare_frame_context(model: CodeSLAMDepthModel, frame_id: str, intensity: torch.Tensor) -> FrameContext:
    intensity = model._ensure_grayscale(intensity)
    half_resolution = torch.nn.functional.avg_pool2d(intensity, kernel_size=2, stride=2)
    intensity_pyramid = build_pyramid(half_resolution, model.config.pyramid_levels)
    image_features = model.encode_image(intensity)
    return FrameContext(frame_id=frame_id, intensity=intensity, image_features=image_features, intensity_pyramid=intensity_pyramid)


def _robust_solve(matrix: torch.Tensor, rhs: torch.Tensor, jitter: float = 1e-6) -> torch.Tensor:
    try:
        return torch.linalg.solve(matrix, rhs)
    except RuntimeError:
        stabilized = matrix + jitter * torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
        try:
            return torch.linalg.solve(stabilized, rhs)
        except RuntimeError:
            solution = torch.linalg.lstsq(stabilized, rhs.unsqueeze(-1) if rhs.ndim == 1 else rhs).solution
            return solution.squeeze(-1) if rhs.ndim == 1 else solution


def estimate_affine_brightness(
    reference: torch.Tensor,
    warped: torch.Tensor,
    valid: torch.Tensor,
    regularization: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    y = reference.reshape(-1)
    x = warped.reshape(-1)
    w = valid.reshape(-1).float()
    A = torch.stack([x, torch.ones_like(x)], dim=1)
    weighted_A = A * w.unsqueeze(1)
    normal_matrix = weighted_A.transpose(0, 1) @ A
    normal_matrix = normal_matrix + regularization * torch.eye(2, device=A.device, dtype=A.dtype)
    rhs = weighted_A.transpose(0, 1) @ y
    solution = _robust_solve(normal_matrix, rhs, regularization)
    return solution[0], solution[1]


def _vectorize_residual_map(residual: torch.Tensor, weight: torch.Tensor, limit: int, delta: float) -> torch.Tensor:
    residual = residual.squeeze(0).squeeze(0)
    weight = weight.squeeze(0).squeeze(0)
    indices = fixed_grid_indices(residual.shape[0], residual.shape[1], limit, residual.device)
    residual = residual.reshape(-1)[indices]
    weight = weight.reshape(-1)[indices]
    return weighted_residual(residual, weight, delta)


def _directional_residuals(
    reference_intensity: torch.Tensor,
    reference_depth: torch.Tensor,
    reference_scale: torch.Tensor,
    source_intensity: torch.Tensor,
    source_depth: torch.Tensor,
    source_scale: torch.Tensor,
    transform_source_from_reference: torch.Tensor,
    camera: CameraModel,
    loss_config: LossConfig,
    optimization_config: OptimizationConfig,
    use_affine_brightness: bool,
    include_geometric_term: bool,
) -> list[torch.Tensor]:
    warped_intensity, projected_depth, valid_photo, _ = warp_from_reference(
        reference_depth, source_intensity, transform_source_from_reference, camera
    )
    warped_depth, _, valid_depth, _ = warp_from_reference(reference_depth, source_depth, transform_source_from_reference, camera)
    warped_scale, _, valid_scale, _ = warp_from_reference(reference_depth, source_scale, transform_source_from_reference, camera)

    valid = valid_photo & valid_depth & valid_scale & (warped_depth > 0.0)

    if use_affine_brightness:
        alpha, beta = estimate_affine_brightness(reference_intensity, warped_intensity, valid, loss_config.affine_regularization)
        warped_intensity = alpha * warped_intensity + beta

    photo_residual = reference_intensity - warped_intensity
    projected_depth = projected_depth.unsqueeze(1)
    geometry_residual = warped_depth - projected_depth

    slanted_weight = surface_slant_weight(reference_depth, loss_config.slanted_surface_gamma)
    occlusion_weight = torch.sigmoid(
        loss_config.occlusion_sharpness * (warped_depth - projected_depth + loss_config.occlusion_margin)
    )
    valid_weight = valid.float()

    photo_weight = loss_config.photometric_weight * occlusion_weight * valid_weight
    geom_scale = 0.5 * (reference_scale + warped_scale)
    geom_weight = loss_config.geometric_weight * slanted_weight * valid_weight / geom_scale.clamp_min(1e-6)

    residuals = [
        _vectorize_residual_map(
            photo_residual,
            photo_weight,
            optimization_config.max_residuals_per_level,
            loss_config.huber_delta_photo,
        )
    ]
    if include_geometric_term:
        residuals.append(
            _vectorize_residual_map(
                geometry_residual,
                geom_weight,
                optimization_config.max_residuals_per_level,
                loss_config.huber_delta_geo,
            )
        )
    return residuals


def pairwise_residual_vector(
    prediction_a,
    intensity_pyramid_a: list[torch.Tensor],
    pose_w_a: torch.Tensor,
    prediction_b,
    intensity_pyramid_b: list[torch.Tensor],
    pose_w_b: torch.Tensor,
    cameras: list[CameraModel],
    loss_config: LossConfig,
    optimization_config: OptimizationConfig,
    *,
    use_affine_brightness: bool,
    include_geometric_term: bool,
) -> torch.Tensor:
    residuals: list[torch.Tensor] = []
    b_from_a = inverse(pose_w_b) @ pose_w_a
    a_from_b = inverse(pose_w_a) @ pose_w_b
    for level, camera in enumerate(cameras):
        residuals.extend(
            _directional_residuals(
                intensity_pyramid_a[level],
                prediction_a.depth_pyramid[level],
                prediction_a.scale_pyramid[level],
                intensity_pyramid_b[level],
                prediction_b.depth_pyramid[level],
                prediction_b.scale_pyramid[level],
                b_from_a,
                camera,
                loss_config,
                optimization_config,
                use_affine_brightness,
                include_geometric_term,
            )
        )
        residuals.extend(
            _directional_residuals(
                intensity_pyramid_b[level],
                prediction_b.depth_pyramid[level],
                prediction_b.scale_pyramid[level],
                intensity_pyramid_a[level],
                prediction_a.depth_pyramid[level],
                prediction_a.scale_pyramid[level],
                a_from_b,
                camera,
                loss_config,
                optimization_config,
                use_affine_brightness,
                include_geometric_term,
            )
        )
    return torch.cat([item.reshape(-1) for item in residuals], dim=0)


def tracking_residual_vector(
    reference_prediction,
    live_prediction,
    reference_pose_w_c: torch.Tensor,
    reference_intensity_pyramid: list[torch.Tensor],
    live_intensity_pyramid: list[torch.Tensor],
    live_pose_w_c: torch.Tensor,
    cameras: list[CameraModel],
    loss_config: LossConfig,
    optimization_config: OptimizationConfig,
    tracking_config: TrackingConfig,
) -> torch.Tensor:
    residuals: list[torch.Tensor] = []
    live_from_reference = inverse(live_pose_w_c) @ reference_pose_w_c
    for level, camera in enumerate(cameras):
        residuals.extend(
            _directional_residuals(
                reference_intensity_pyramid[level],
                reference_prediction.depth_pyramid[level],
                reference_prediction.scale_pyramid[level],
                live_intensity_pyramid[level],
                live_prediction.depth_pyramid[level],
                live_prediction.scale_pyramid[level],
                live_from_reference,
                camera,
                loss_config,
                optimization_config,
                tracking_config.use_affine_brightness,
                include_geometric_term=False,
            )
        )
    return torch.cat([item.reshape(-1) for item in residuals], dim=0)


def levenberg_marquardt(
    residual_fn: Callable[[torch.Tensor], torch.Tensor],
    initial_solution: torch.Tensor,
    optimization_config: OptimizationConfig,
) -> LeastSquaresResult:
    solution = initial_solution.clone()
    damping = optimization_config.damping
    last_cost = float("inf")
    converged = False
    last_residual = residual_fn(solution)

    for iteration in range(optimization_config.iterations):
        residual = residual_fn(solution)
        cost = 0.5 * torch.dot(residual, residual)
        jacobian = torch.autograd.functional.jacobian(residual_fn, solution, create_graph=False, vectorize=False)
        hessian = jacobian.transpose(0, 1) @ jacobian
        gradient = jacobian.transpose(0, 1) @ residual
        damped_hessian = hessian + damping * torch.eye(hessian.shape[0], device=hessian.device, dtype=hessian.dtype)
        step = _robust_solve(damped_hessian, -gradient, damping)
        candidate = solution + step
        candidate_residual = residual_fn(candidate)
        candidate_cost = 0.5 * torch.dot(candidate_residual, candidate_residual)
        improvement = cost - candidate_cost

        if candidate_cost < cost:
            solution = candidate
            last_residual = candidate_residual
            if improvement.abs() < optimization_config.min_improvement:
                converged = True
                last_cost = candidate_cost.item()
                break
            damping = damping / optimization_config.damping_multiplier
            last_cost = candidate_cost.item()
        else:
            damping = damping * optimization_config.damping_multiplier
            last_residual = residual
            last_cost = cost.item()

    return LeastSquaresResult(
        solution=solution,
        cost=last_cost,
        converged=converged,
        iterations=iteration + 1,
        residual=last_residual.detach(),
    )


def apply_state_delta(
    base_poses: dict[str, torch.Tensor],
    base_codes: dict[str, torch.Tensor],
    layout: StateLayout,
    delta: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    poses = {}
    codes = {}
    for keyframe_id in layout.keyframe_ids:
        pose_delta = delta[layout.pose_slice(keyframe_id)]
        code_delta = delta[layout.code_slice(keyframe_id)]
        poses[keyframe_id] = compose(se3_exp(pose_delta), base_poses[keyframe_id])
        codes[keyframe_id] = base_codes[keyframe_id] + code_delta
    return poses, codes


def prior_residual_vector(prior: PriorFactor, poses: dict[str, torch.Tensor], codes: dict[str, torch.Tensor]) -> torch.Tensor:
    blocks: list[torch.Tensor] = []
    for keyframe_id in prior.keyframe_ids:
        blocks.append(relative_twist(poses[keyframe_id], prior.pose_reference[keyframe_id]))
    for keyframe_id in prior.keyframe_ids:
        blocks.append(codes[keyframe_id] - prior.code_reference[keyframe_id])
    delta = torch.cat(blocks, dim=0)
    return prior.sqrt_hessian @ delta + prior.offset


def optimize_window(
    model: CodeSLAMDepthModel,
    contexts: dict[str, FrameContext],
    base_poses: dict[str, torch.Tensor],
    base_codes: dict[str, torch.Tensor],
    cameras: list[CameraModel],
    edges: Iterable[tuple[str, str]],
    loss_config: LossConfig,
    optimization_config: OptimizationConfig,
    *,
    prior: PriorFactor | None = None,
    gauge_keyframe_id: str | None = None,
    gauge_reference_pose: torch.Tensor | None = None,
    use_affine_brightness: bool = True,
    include_geometric_term: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], LeastSquaresResult, StateLayout]:
    residual_fn, layout = make_window_residual_fn(
        model,
        contexts,
        base_poses,
        base_codes,
        cameras,
        edges,
        loss_config,
        optimization_config,
        prior=prior,
        gauge_keyframe_id=gauge_keyframe_id,
        gauge_reference_pose=gauge_reference_pose,
        use_affine_brightness=use_affine_brightness,
        include_geometric_term=include_geometric_term,
    )
    initial_solution = next(iter(base_codes.values())).new_zeros(layout.total_dim)
    result = levenberg_marquardt(residual_fn, initial_solution, optimization_config)
    final_poses, final_codes = apply_state_delta(base_poses, base_codes, layout, result.solution)
    return final_poses, final_codes, result, layout


def make_window_residual_fn(
    model: CodeSLAMDepthModel,
    contexts: dict[str, FrameContext],
    base_poses: dict[str, torch.Tensor],
    base_codes: dict[str, torch.Tensor],
    cameras: list[CameraModel],
    edges: Iterable[tuple[str, str]],
    loss_config: LossConfig,
    optimization_config: OptimizationConfig,
    *,
    prior: PriorFactor | None = None,
    gauge_keyframe_id: str | None = None,
    gauge_reference_pose: torch.Tensor | None = None,
    use_affine_brightness: bool = True,
    include_geometric_term: bool = True,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], StateLayout]:
    layout = StateLayout(tuple(base_poses.keys()), model.config.code_dim)

    def residual_fn(delta: torch.Tensor) -> torch.Tensor:
        poses, codes = apply_state_delta(base_poses, base_codes, layout, delta)
        predictions = {
            keyframe_id: model.predict_from_image_features(contexts[keyframe_id].image_features, codes[keyframe_id].unsqueeze(0))
            for keyframe_id in layout.keyframe_ids
        }

        residuals = []
        for keyframe_a, keyframe_b in edges:
            residuals.append(
                pairwise_residual_vector(
                    predictions[keyframe_a],
                    contexts[keyframe_a].intensity_pyramid,
                    poses[keyframe_a],
                    predictions[keyframe_b],
                    contexts[keyframe_b].intensity_pyramid,
                    poses[keyframe_b],
                    cameras,
                    loss_config,
                    optimization_config,
                    use_affine_brightness=use_affine_brightness,
                    include_geometric_term=include_geometric_term,
                )
            )
        if prior is not None:
            residuals.append(prior_residual_vector(prior, poses, codes))
        if gauge_keyframe_id is not None and gauge_reference_pose is not None:
            residuals.append(loss_config.gauge_prior_weight * relative_twist(poses[gauge_keyframe_id], gauge_reference_pose))
        if not residuals:
            raise ValueError("Window optimization requires at least one residual term.")
        return torch.cat(residuals, dim=0)

    return residual_fn, layout


def build_prior_factor(
    residual_fn: Callable[[torch.Tensor], torch.Tensor],
    base_poses: dict[str, torch.Tensor],
    base_codes: dict[str, torch.Tensor],
    layout: StateLayout,
    marginalize_keyframe_ids: set[str],
    loss_config: LossConfig,
) -> PriorFactor:
    if not marginalize_keyframe_ids:
        raise ValueError("At least one keyframe must be marginalized to build a prior.")
    linearization_point = next(iter(base_codes.values())).new_zeros(layout.total_dim)
    residual = residual_fn(linearization_point)
    jacobian = torch.autograd.functional.jacobian(residual_fn, linearization_point, create_graph=False, vectorize=False)
    hessian = jacobian.transpose(0, 1) @ jacobian
    rhs = jacobian.transpose(0, 1) @ residual
    hessian = hessian + loss_config.prior_jitter * torch.eye(hessian.shape[0], device=hessian.device, dtype=hessian.dtype)

    marginalized_pose_indices = []
    marginalized_code_indices = []
    kept_pose_indices = []
    kept_code_indices = []
    kept_keyframes = []
    for keyframe_id in layout.keyframe_ids:
        pose_indices = list(range(layout.pose_slice(keyframe_id).start, layout.pose_slice(keyframe_id).stop))
        code_indices = list(range(layout.code_slice(keyframe_id).start, layout.code_slice(keyframe_id).stop))
        if keyframe_id in marginalize_keyframe_ids:
            marginalized_pose_indices.extend(pose_indices)
            marginalized_code_indices.extend(code_indices)
        else:
            kept_keyframes.append(keyframe_id)
            kept_pose_indices.extend(pose_indices)
            kept_code_indices.extend(code_indices)

    marginalized_indices = marginalized_pose_indices + marginalized_code_indices
    kept_indices = kept_pose_indices + kept_code_indices
    if not kept_indices:
        raise ValueError("Marginalization would remove the entire window; no prior can be formed.")

    hmm = hessian[marginalized_indices][:, marginalized_indices]
    hmk = hessian[marginalized_indices][:, kept_indices]
    hkm = hessian[kept_indices][:, marginalized_indices]
    hkk = hessian[kept_indices][:, kept_indices]
    bm = rhs[marginalized_indices]
    bk = rhs[kept_indices]

    solve_hmm_hmk = _robust_solve(hmm, hmk, loss_config.prior_jitter)
    solve_hmm_bm = _robust_solve(hmm, bm, loss_config.prior_jitter)
    schur_hessian = hkk - hkm @ solve_hmm_hmk
    schur_rhs = bk - hkm @ solve_hmm_bm
    schur_hessian = schur_hessian + loss_config.prior_jitter * torch.eye(
        schur_hessian.shape[0], device=schur_hessian.device, dtype=schur_hessian.dtype
    )

    sqrt_hessian = torch.linalg.cholesky(schur_hessian)
    offset = _robust_solve(sqrt_hessian.transpose(0, 1), schur_rhs, loss_config.prior_jitter)
    return PriorFactor(
        keyframe_ids=tuple(kept_keyframes),
        pose_reference={keyframe_id: base_poses[keyframe_id].detach().clone() for keyframe_id in kept_keyframes},
        code_reference={keyframe_id: base_codes[keyframe_id].detach().clone() for keyframe_id in kept_keyframes},
        sqrt_hessian=sqrt_hessian.detach().clone(),
        offset=offset.detach().clone(),
    )


def optimize_tracking_pose(
    reference_keyframe: Keyframe,
    live_frame: FrameBatch,
    initial_pose_w_c: torch.Tensor,
    model: CodeSLAMDepthModel,
    full_resolution_camera: CameraModel,
    loss_config: LossConfig,
    optimization_config: OptimizationConfig,
    tracking_config: TrackingConfig,
) -> TrackingResult:
    reference_context = prepare_frame_context(model, reference_keyframe.keyframe_id, reference_keyframe.intensity)
    reference_prediction = model.predict_from_image_features(reference_context.image_features, reference_keyframe.code.unsqueeze(0))

    live_context = prepare_frame_context(model, live_frame.frame_id or "live", live_frame.intensity)
    live_zero_code = model.zero_code(1, live_frame.intensity.device, live_frame.intensity.dtype)
    live_prediction = model.predict_from_image_features(live_context.image_features, live_zero_code)
    cameras = camera_pyramid(full_resolution_camera.scaled(0.5), tracking_config.pyramid_levels)

    pose = initial_pose_w_c
    for level, iterations in enumerate(optimization_config.tracking_iterations_per_level):
        level_camera = cameras[level:]
        reference_level_prediction = type(reference_prediction)(
            code=reference_prediction.code,
            proximity_pyramid=reference_prediction.proximity_pyramid[level:],
            depth_pyramid=reference_prediction.depth_pyramid[level:],
            scale_pyramid=reference_prediction.scale_pyramid[level:],
            posterior_mean=reference_prediction.posterior_mean,
            posterior_logvar=reference_prediction.posterior_logvar,
        )
        live_level_prediction = type(live_prediction)(
            code=live_prediction.code,
            proximity_pyramid=live_prediction.proximity_pyramid[level:],
            depth_pyramid=live_prediction.depth_pyramid[level:],
            scale_pyramid=live_prediction.scale_pyramid[level:],
            posterior_mean=live_prediction.posterior_mean,
            posterior_logvar=live_prediction.posterior_logvar,
        )
        reference_intensity = reference_context.intensity_pyramid[level:]
        live_intensity = live_context.intensity_pyramid[level:]
        local_optimization = OptimizationConfig(
            iterations=iterations,
            damping=optimization_config.damping,
            damping_multiplier=optimization_config.damping_multiplier,
            min_improvement=optimization_config.min_improvement,
            max_residuals_per_level=optimization_config.max_residuals_per_level,
            pair_initialization_iterations=optimization_config.pair_initialization_iterations,
            mapping_iterations=optimization_config.mapping_iterations,
            tracking_iterations_per_level=optimization_config.tracking_iterations_per_level,
        )

        def residual_fn(delta: torch.Tensor) -> torch.Tensor:
            candidate_pose = compose(se3_exp(delta), pose)
            return tracking_residual_vector(
                reference_level_prediction,
                live_level_prediction,
                reference_keyframe.pose_w_c,
                reference_intensity,
                live_intensity,
                candidate_pose,
                level_camera,
                loss_config,
                local_optimization,
                tracking_config,
            )

        result = levenberg_marquardt(residual_fn, pose.new_zeros(6), local_optimization)
        pose = compose(se3_exp(result.solution), pose)

    final_residual = tracking_residual_vector(
        reference_prediction,
        live_prediction,
        reference_keyframe.pose_w_c,
        reference_context.intensity_pyramid,
        live_context.intensity_pyramid,
        pose,
        cameras,
        loss_config,
        optimization_config,
        tracking_config,
    )
    return TrackingResult(
        pose_w_c=pose,
        cost=0.5 * torch.dot(final_residual, final_residual).item(),
        residual_count=int(final_residual.numel()),
        converged=True,
    )
