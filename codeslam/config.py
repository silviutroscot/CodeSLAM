from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CameraModel:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def scaled(self, scale_x: float, scale_y: float | None = None) -> "CameraModel":
        scale_y = scale_x if scale_y is None else scale_y
        return CameraModel(
            fx=self.fx * scale_x,
            fy=self.fy * scale_y,
            cx=self.cx * scale_x,
            cy=self.cy * scale_y,
            width=max(1, int(round(self.width * scale_x))),
            height=max(1, int(round(self.height * scale_y))),
        )


@dataclass(frozen=True)
class ModelConfig:
    input_height: int = 192
    input_width: int = 256
    code_dim: int = 128
    base_channels: int = 16
    pyramid_levels: int = 4
    latent_hidden_dim: int = 512
    linear_decoder: bool = True
    min_depth: float = 0.1
    max_depth: float = 40.0
    proximity_average_depth: float = 4.0
    min_uncertainty: float = 1e-3

    @property
    def prediction_height(self) -> int:
        return self.input_height // 2

    @property
    def prediction_width(self) -> int:
        return self.input_width // 2

    @property
    def proximity_transition(self) -> float:
        return self.proximity_average_depth


@dataclass(frozen=True)
class LossConfig:
    kl_weight: float = 1e-4
    geometric_weight: float = 1.0
    photometric_weight: float = 1.0
    huber_delta_photo: float = 0.05
    huber_delta_geo: float = 0.10
    occlusion_margin: float = 0.05
    occlusion_sharpness: float = 10.0
    slanted_surface_gamma: float = 8.0
    level_base_weight: float = 4.0
    affine_regularization: float = 1e-4
    gauge_prior_weight: float = 100.0
    prior_jitter: float = 1e-6


@dataclass(frozen=True)
class OptimizationConfig:
    iterations: int = 8
    damping: float = 1e-3
    damping_multiplier: float = 5.0
    min_improvement: float = 1e-7
    max_residuals_per_level: int = 4096
    pair_initialization_iterations: int = 10
    mapping_iterations: int = 5
    tracking_iterations_per_level: tuple[int, ...] = (12, 10, 8, 6)


@dataclass(frozen=True)
class TrackingConfig:
    pyramid_levels: int = 4
    use_affine_brightness: bool = True
    coarse_to_fine: bool = True


@dataclass(frozen=True)
class SLAMConfig:
    max_keyframes: int = 4
    keyframe_translation_threshold: float = 0.15
    keyframe_rotation_threshold_deg: float = 7.5
    mapping_edges: str = "consecutive"
    bootstrap_motion_prior: float = 0.0


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 8
    num_workers: int = 4
    learning_rate: float = 1e-4
    final_learning_rate: float = 1e-6
    weight_decay: float = 1e-6
    epochs: int = 6
    validation_interval: int = 1
    checkpoint_dir: str = "checkpoints"
    device: str = "cuda"
    extra_metrics: tuple[str, ...] = field(default_factory=lambda: ("abs_rel", "rmse", "delta1"))
