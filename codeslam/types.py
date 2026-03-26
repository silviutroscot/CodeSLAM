from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class FrameBatch:
    intensity: torch.Tensor
    depth: torch.Tensor | None = None
    timestamp: float | None = None
    frame_id: str | None = None
    pose_w_c: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DepthPrediction:
    code: torch.Tensor
    proximity_pyramid: list[torch.Tensor]
    depth_pyramid: list[torch.Tensor]
    scale_pyramid: list[torch.Tensor]
    posterior_mean: torch.Tensor | None = None
    posterior_logvar: torch.Tensor | None = None


@dataclass
class Keyframe:
    keyframe_id: str
    intensity: torch.Tensor
    pose_w_c: torch.Tensor
    code: torch.Tensor
    timestamp: float | None = None
    depth: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrackingResult:
    pose_w_c: torch.Tensor
    cost: float
    residual_count: int
    converged: bool


@dataclass
class PriorFactor:
    keyframe_ids: tuple[str, ...]
    pose_reference: dict[str, torch.Tensor]
    code_reference: dict[str, torch.Tensor]
    sqrt_hessian: torch.Tensor
    offset: torch.Tensor
