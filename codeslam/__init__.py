"""Core modules for the CodeSLAM paper implementation."""

from .config import CameraModel, LossConfig, ModelConfig, OptimizationConfig, SLAMConfig, TrackingConfig, TrainingConfig

__all__ = [
    "CameraModel",
    "LossConfig",
    "ModelConfig",
    "OptimizationConfig",
    "SLAMConfig",
    "TrackingConfig",
    "TrainingConfig",
]

try:  # pragma: no cover
    from .network import CodeSLAMDepthModel
    from .system import CodeSLAMSystem
    from .types import DepthPrediction, FrameBatch, Keyframe, PriorFactor, TrackingResult

    __all__.extend(
        [
            "CodeSLAMDepthModel",
            "CodeSLAMSystem",
            "DepthPrediction",
            "FrameBatch",
            "Keyframe",
            "PriorFactor",
            "TrackingResult",
        ]
    )
except ImportError:
    pass
