from __future__ import annotations

from math import ceil, sqrt

import torch
import torch.nn.functional as F

from .config import CameraModel
from .pose import transform_points


def meshgrid(camera: CameraModel, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.arange(camera.height, device=device, dtype=dtype),
        torch.arange(camera.width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([xs, ys], dim=-1)


def backproject(depth: torch.Tensor, camera: CameraModel) -> torch.Tensor:
    if depth.ndim == 4:
        depth = depth[:, 0]
    grid = meshgrid(camera, depth.device, depth.dtype).unsqueeze(0)
    z = depth
    x = (grid[..., 0] - camera.cx) * z / camera.fx
    y = (grid[..., 1] - camera.cy) * z / camera.fy
    return torch.stack([x, y, z], dim=-1)


def project(points: torch.Tensor, camera: CameraModel) -> tuple[torch.Tensor, torch.Tensor]:
    z = points[..., 2].clamp_min(1e-6)
    u = points[..., 0] * camera.fx / z + camera.cx
    v = points[..., 1] * camera.fy / z + camera.cy
    return torch.stack([u, v], dim=-1), z


def normalize_grid(uv: torch.Tensor, camera: CameraModel) -> torch.Tensor:
    x = 2.0 * uv[..., 0] / max(camera.width - 1, 1) - 1.0
    y = 2.0 * uv[..., 1] / max(camera.height - 1, 1) - 1.0
    return torch.stack([x, y], dim=-1)


def sample_tensor(source: torch.Tensor, uv: torch.Tensor, camera: CameraModel) -> torch.Tensor:
    if source.ndim == 3:
        source = source.unsqueeze(0)
    grid = normalize_grid(uv, camera)
    return F.grid_sample(source, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def inside_image_mask(uv: torch.Tensor, camera: CameraModel) -> torch.Tensor:
    return (
        (uv[..., 0] >= 0.0)
        & (uv[..., 0] <= camera.width - 1)
        & (uv[..., 1] >= 0.0)
        & (uv[..., 1] <= camera.height - 1)
    )


def warp_from_reference(
    reference_depth: torch.Tensor,
    source_tensor: torch.Tensor,
    transform_source_from_reference: torch.Tensor,
    camera: CameraModel,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    reference_points = backproject(reference_depth, camera)
    source_points = transform_points(transform_source_from_reference, reference_points)
    uv_source, z_source = project(source_points, camera)
    sampled = sample_tensor(source_tensor, uv_source, camera)
    valid = inside_image_mask(uv_source, camera) & (z_source > 1e-6)
    return sampled, z_source, valid.unsqueeze(1), uv_source


def build_pyramid(tensor: torch.Tensor, levels: int) -> list[torch.Tensor]:
    pyramid = [tensor]
    for _ in range(1, levels):
        pyramid.append(F.avg_pool2d(pyramid[-1], kernel_size=2, stride=2))
    return list(reversed(pyramid))


def resample_like(tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return F.interpolate(tensor, size=reference.shape[-2:], mode="bilinear", align_corners=False)


def fixed_grid_indices(height: int, width: int, limit: int, device: torch.device) -> torch.Tensor:
    count = height * width
    if limit >= count:
        return torch.arange(count, device=device)
    stride = max(1, ceil(sqrt(count / limit)))
    ys = torch.arange(0, height, stride, device=device)
    xs = torch.arange(0, width, stride, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    flat = (grid_y * width + grid_x).reshape(-1)
    return flat[:limit]


def surface_slant_weight(depth: torch.Tensor, gamma: float) -> torch.Tensor:
    grad_x = depth[..., :, 1:] - depth[..., :, :-1]
    grad_y = depth[..., 1:, :] - depth[..., :-1, :]
    grad_x = F.pad(grad_x, (0, 1, 0, 0))
    grad_y = F.pad(grad_y, (0, 0, 0, 1))
    return torch.exp(-gamma * (grad_x.square() + grad_y.square()))
