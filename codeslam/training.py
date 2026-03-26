from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import LossConfig, ModelConfig, TrainingConfig
from .geometry import build_pyramid
from .losses import depth_metrics, kl_divergence, multiscale_laplace_nll
from .network import CodeSLAMDepthModel
from .proximity import depth_to_proximity


def build_depth_target_pyramid(depth: torch.Tensor, model_config: ModelConfig) -> list[torch.Tensor]:
    half_resolution = F.avg_pool2d(depth, kernel_size=2, stride=2)
    return build_pyramid(half_resolution, model_config.pyramid_levels)


def compute_training_loss(
    model: CodeSLAMDepthModel,
    batch: dict[str, torch.Tensor],
    loss_config: LossConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    intensity = batch["intensity"]
    depth = batch["depth"]
    prediction = model(intensity, depth, sample_posterior=True)

    target_depth_pyramid = build_depth_target_pyramid(depth, model.config)
    target_proximity_pyramid = [
        depth_to_proximity(target_depth, model.config.proximity_transition) for target_depth in target_depth_pyramid
    ]
    masks = [(target_depth > 0.0).float() for target_depth in target_depth_pyramid]

    reconstruction = multiscale_laplace_nll(
        prediction.proximity_pyramid,
        target_proximity_pyramid,
        prediction.scale_pyramid,
        loss_config.level_base_weight,
        masks,
    )
    kl = torch.zeros((), device=intensity.device, dtype=intensity.dtype)
    if prediction.posterior_mean is not None and prediction.posterior_logvar is not None:
        kl = kl_divergence(prediction.posterior_mean, prediction.posterior_logvar)
    total = reconstruction + loss_config.kl_weight * kl

    metrics = depth_metrics(prediction.depth_pyramid[-1].detach(), target_depth_pyramid[-1].detach(), masks[-1].bool())
    metrics["loss"] = total.item()
    metrics["reconstruction"] = reconstruction.item()
    metrics["kl"] = kl.item()
    return total, metrics


def train_one_epoch(
    model: CodeSLAMDepthModel,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_config: LossConfig,
) -> dict[str, float]:
    model.train()
    running = {"loss": 0.0, "reconstruction": 0.0, "kl": 0.0, "abs_rel": 0.0, "rmse": 0.0, "delta1": 0.0}
    batches = 0
    for batch in loader:
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = compute_training_loss(model, batch, loss_config)
        loss.backward()
        optimizer.step()

        for key in running:
            running[key] += metrics[key]
        batches += 1
    return {key: value / max(batches, 1) for key, value in running.items()}


@torch.no_grad()
def evaluate(
    model: CodeSLAMDepthModel,
    loader,
    device: torch.device,
    loss_config: LossConfig,
) -> dict[str, float]:
    model.eval()
    running = {"loss": 0.0, "reconstruction": 0.0, "kl": 0.0, "abs_rel": 0.0, "rmse": 0.0, "delta1": 0.0}
    batches = 0
    for batch in loader:
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        _, metrics = compute_training_loss(model, batch, loss_config)
        for key in running:
            running[key] += metrics[key]
        batches += 1
    return {key: value / max(batches, 1) for key, value in running.items()}


@torch.no_grad()
def evaluate_zero_code_prior(
    model: CodeSLAMDepthModel,
    loader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    running = {"abs_rel": 0.0, "rmse": 0.0, "delta1": 0.0}
    batches = 0
    for batch in loader:
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        intensity = batch["intensity"]
        depth = batch["depth"]
        zero_code = model.zero_code(intensity.shape[0], intensity.device, intensity.dtype)
        prediction = model(intensity, code=zero_code, sample_posterior=False)
        target_depth_pyramid = build_depth_target_pyramid(depth, model.config)
        mask = target_depth_pyramid[-1] > 0.0
        metrics = depth_metrics(prediction.depth_pyramid[-1], target_depth_pyramid[-1], mask)
        for key in running:
            running[key] += metrics[key]
        batches += 1
    return {key: value / max(batches, 1) for key, value in running.items()}


def fit(
    model: CodeSLAMDepthModel,
    train_loader,
    validation_loader,
    training_config: TrainingConfig,
    loss_config: LossConfig,
) -> None:
    device = torch.device(training_config.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    scheduler = None
    if training_config.epochs > 1 and training_config.final_learning_rate < training_config.learning_rate:
        gamma = (training_config.final_learning_rate / training_config.learning_rate) ** (1.0 / (training_config.epochs - 1))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    checkpoint_dir = Path(training_config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_validation = float("inf")
    for epoch in range(training_config.epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, loss_config)
        validation_metrics = evaluate(model, validation_loader, device, loss_config)
        print(f"epoch={epoch + 1} train={train_metrics} val={validation_metrics}")

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model.config),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
        }
        torch.save(checkpoint, checkpoint_dir / "latest.pt")
        if validation_metrics["loss"] < best_validation:
            best_validation = validation_metrics["loss"]
            torch.save(checkpoint, checkpoint_dir / "best.pt")
        if scheduler is not None:
            scheduler.step()
