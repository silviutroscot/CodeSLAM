from __future__ import annotations

import torch


def depth_to_proximity(depth: torch.Tensor, transition: float) -> torch.Tensor:
    """Map depth to the paper's hybrid proximity parametrisation p = a / (d + a)."""
    transition_tensor = torch.as_tensor(transition, device=depth.device, dtype=depth.dtype)
    return transition_tensor / (depth + transition_tensor).clamp_min(1e-6)


def proximity_to_depth(proximity: torch.Tensor, transition: float) -> torch.Tensor:
    """Invert the paper's hybrid proximity parametrisation."""
    transition_tensor = torch.as_tensor(transition, device=proximity.device, dtype=proximity.dtype)
    return transition_tensor * (1.0 - proximity) / proximity.clamp_min(1e-6)
