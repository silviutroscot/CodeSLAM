from __future__ import annotations

import torch


def _ensure_batch(tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if tensor.ndim == 1:
        return tensor.unsqueeze(0), True
    if tensor.ndim == 2 and tensor.shape[-1] == 6:
        return tensor, False
    if tensor.ndim == 2 and tensor.shape == (3, 3):
        return tensor.unsqueeze(0), True
    if tensor.ndim == 2 and tensor.shape == (4, 4):
        return tensor.unsqueeze(0), True
    return tensor, False


def skew(vector: torch.Tensor) -> torch.Tensor:
    vector, squeezed = _ensure_batch(vector)
    zeros = torch.zeros_like(vector[:, 0])
    matrix = torch.stack(
        [
            zeros,
            -vector[:, 2],
            vector[:, 1],
            vector[:, 2],
            zeros,
            -vector[:, 0],
            -vector[:, 1],
            vector[:, 0],
            zeros,
        ],
        dim=-1,
    ).reshape(-1, 3, 3)
    return matrix[0] if squeezed else matrix


def _taylor_A(theta: torch.Tensor) -> torch.Tensor:
    return 1.0 - theta**2 / 6.0 + theta**4 / 120.0


def _taylor_B(theta: torch.Tensor) -> torch.Tensor:
    return 0.5 - theta**2 / 24.0 + theta**4 / 720.0


def _taylor_C(theta: torch.Tensor) -> torch.Tensor:
    return 1.0 / 6.0 - theta**2 / 120.0 + theta**4 / 5040.0


def so3_exp(omega: torch.Tensor) -> torch.Tensor:
    omega, squeezed = _ensure_batch(omega)
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True).clamp_min(1e-12)
    theta_sq = theta**2
    small = theta_sq < 1e-6
    A = torch.where(small, _taylor_A(theta), torch.sin(theta) / theta)
    B = torch.where(small, _taylor_B(theta), (1.0 - torch.cos(theta)) / theta_sq)
    Omega = skew(omega)
    eye = torch.eye(3, device=omega.device, dtype=omega.dtype).expand(omega.shape[0], 3, 3)
    rotation = eye + A[..., None] * Omega + B[..., None] * (Omega @ Omega)
    return rotation[0] if squeezed else rotation


def so3_log(rotation: torch.Tensor) -> torch.Tensor:
    rotation, squeezed = _ensure_batch(rotation)
    trace = rotation[:, 0, 0] + rotation[:, 1, 1] + rotation[:, 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    small = theta < 1e-5
    factor = torch.where(small, 0.5 + theta**2 / 12.0, theta / (2.0 * torch.sin(theta).clamp_min(1e-12)))
    omega_hat = factor[:, None, None] * (rotation - rotation.transpose(1, 2))
    omega = torch.stack(
        [omega_hat[:, 2, 1], omega_hat[:, 0, 2], omega_hat[:, 1, 0]],
        dim=-1,
    )
    return omega[0] if squeezed else omega


def se3_exp(xi: torch.Tensor) -> torch.Tensor:
    xi, squeezed = _ensure_batch(xi)
    rho = xi[:, :3]
    omega = xi[:, 3:]
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True).clamp_min(1e-12)
    theta_sq = theta**2
    small = theta_sq < 1e-6

    A = torch.where(small, _taylor_A(theta), torch.sin(theta) / theta)
    B = torch.where(small, _taylor_B(theta), (1.0 - torch.cos(theta)) / theta_sq)
    C = torch.where(small, _taylor_C(theta), (theta - torch.sin(theta)) / (theta * theta_sq))

    Omega = skew(omega)
    eye = torch.eye(3, device=xi.device, dtype=xi.dtype).expand(xi.shape[0], 3, 3)
    rotation = eye + A[..., None] * Omega + B[..., None] * (Omega @ Omega)
    V = eye + B[..., None] * Omega + C[..., None] * (Omega @ Omega)
    translation = (V @ rho.unsqueeze(-1)).squeeze(-1)

    transform = torch.eye(4, device=xi.device, dtype=xi.dtype).expand(xi.shape[0], 4, 4).clone()
    transform[:, :3, :3] = rotation
    transform[:, :3, 3] = translation
    return transform[0] if squeezed else transform


def se3_log(transform: torch.Tensor) -> torch.Tensor:
    transform, squeezed = _ensure_batch(transform)
    rotation = transform[:, :3, :3]
    translation = transform[:, :3, 3]
    omega = so3_log(rotation)
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True).clamp_min(1e-12)
    theta_sq = theta**2
    small = theta_sq < 1e-6
    Omega = skew(omega)
    eye = torch.eye(3, device=transform.device, dtype=transform.dtype).expand(transform.shape[0], 3, 3)
    coefficient = torch.where(
        small,
        torch.full_like(theta, 1.0 / 12.0),
        (1.0 / theta_sq) - (1.0 + torch.cos(theta)) / (2.0 * theta * torch.sin(theta)),
    )
    V_inv = eye - 0.5 * Omega + coefficient[..., None] * (Omega @ Omega)
    rho = (V_inv @ translation.unsqueeze(-1)).squeeze(-1)
    xi = torch.cat([rho, omega], dim=-1)
    return xi[0] if squeezed else xi


def compose(a_from_b: torch.Tensor, b_from_c: torch.Tensor) -> torch.Tensor:
    return a_from_b @ b_from_c


def inverse(transform: torch.Tensor) -> torch.Tensor:
    transform, squeezed = _ensure_batch(transform)
    rotation = transform[:, :3, :3]
    translation = transform[:, :3, 3]
    rotation_t = rotation.transpose(1, 2)
    inverse_transform = torch.eye(4, device=transform.device, dtype=transform.dtype).expand(transform.shape[0], 4, 4).clone()
    inverse_transform[:, :3, :3] = rotation_t
    inverse_transform[:, :3, 3] = -(rotation_t @ translation.unsqueeze(-1)).squeeze(-1)
    return inverse_transform[0] if squeezed else inverse_transform


def transform_points(transform: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    squeezed_transform = False
    if transform.ndim == 2:
        transform = transform.unsqueeze(0)
        squeezed_transform = True
    squeezed_points = False
    if points.ndim == 3:
        points = points.unsqueeze(0)
        squeezed_points = True
    rotation = transform[:, None, :3, :3]
    translation = transform[:, None, :3, 3]
    transformed = (rotation @ points.unsqueeze(-1)).squeeze(-1) + translation
    if squeezed_transform and squeezed_points:
        return transformed[0]
    return transformed


def relative_twist(current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return se3_log(current @ inverse(reference))


def translation_norm(transform: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(transform[..., :3, 3], dim=-1)


def rotation_angle(transform: torch.Tensor) -> torch.Tensor:
    rotation = transform[..., :3, :3]
    trace = rotation[..., 0, 0] + rotation[..., 1, 1] + rotation[..., 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos_theta))
