from __future__ import annotations

import torch
import torch.nn.functional as F


def positive_scale(log_scale: torch.Tensor, minimum: float) -> torch.Tensor:
    return F.softplus(log_scale) + minimum


def laplace_nll(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scale: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    residual = torch.abs(prediction - target)
    loss = residual / scale.clamp_min(1e-6) + torch.log(scale.clamp_min(1e-6))
    if mask is not None:
        loss = loss * mask
        denom = mask.sum().clamp_min(1.0)
        return loss.sum() / denom
    return loss.mean()


def multiscale_laplace_nll(
    predictions: list[torch.Tensor],
    targets: list[torch.Tensor],
    scales: list[torch.Tensor],
    level_base_weight: float,
    masks: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    total = predictions[0].new_tensor(0.0)
    num_levels = len(predictions)
    for idx, (prediction, target, scale) in enumerate(zip(predictions, targets, scales)):
        weight = level_base_weight ** (num_levels - idx - 1)
        mask = None if masks is None else masks[idx]
        total = total + weight * laplace_nll(prediction, target, scale, mask)
    return total


def kl_divergence(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1.0 + logvar - mean.square() - logvar.exp())


def huberized_residual(residual: torch.Tensor, delta: float) -> torch.Tensor:
    abs_residual = residual.abs()
    weight = torch.where(abs_residual <= delta, torch.ones_like(abs_residual), torch.sqrt(delta / abs_residual.clamp_min(1e-12)))
    return residual * weight


def weighted_residual(residual: torch.Tensor, weight: torch.Tensor, delta: float) -> torch.Tensor:
    return huberized_residual(residual * torch.sqrt(weight.clamp_min(0.0)), delta)


def depth_metrics(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, float]:
    if mask is None:
        mask = torch.ones_like(target, dtype=torch.bool)
    prediction = prediction[mask]
    target = target[mask]
    ratio = torch.maximum(prediction / target.clamp_min(1e-6), target / prediction.clamp_min(1e-6))
    rmse = torch.sqrt(torch.mean((prediction - target) ** 2)).item()
    abs_rel = torch.mean(torch.abs(prediction - target) / target.clamp_min(1e-6)).item()
    delta1 = torch.mean((ratio < 1.25).float()).item()
    return {"rmse": rmse, "abs_rel": abs_rel, "delta1": delta1}
