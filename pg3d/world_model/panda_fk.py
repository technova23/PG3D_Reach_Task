from __future__ import annotations

import math

import torch


def panda_end_effector_position(q: torch.Tensor) -> torch.Tensor:
    """Return Panda TCP positions for a batch of joint vectors.

    This is a small differentiable forward-kinematics wrapper for the Franka Panda
    arm. It is intentionally self-contained so ITPS-style steering can backprop
    through obstacle distance without relying on the simulator-backed ghost env.

    Args:
        q: Joint positions with shape ``[..., 7]``.

    Returns:
        TCP positions with shape ``[..., 3]``.
    """
    if q.ndim < 1 or q.shape[-1] != 7:
        raise ValueError(f"q must have shape [..., 7], got {tuple(q.shape)}")

    # Standard Panda kinematic chain parameters in meters.
    d = (0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.107)
    a = (0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088)
    alpha = (
        -math.pi / 2.0,
        math.pi / 2.0,
        math.pi / 2.0,
        -math.pi / 2.0,
        math.pi / 2.0,
        math.pi / 2.0,
        0.0,
    )
    theta_offset = (0.0, -math.pi / 2.0, 0.0, -math.pi / 2.0, 0.0, math.pi / 2.0, math.pi / 4.0)

    transform = torch.eye(4, device=q.device, dtype=q.dtype)
    batch_shape = q.shape[:-1]
    transform = transform.expand(*batch_shape, 4, 4).clone()
    for idx in range(7):
        transform = transform @ _dh_transform(
            q[..., idx] + theta_offset[idx],
            d[idx],
            a[idx],
            alpha[idx],
        )
    return transform[..., :3, 3]


def _dh_transform(theta: torch.Tensor, d: float, a: float, alpha: float) -> torch.Tensor:
    ct = torch.cos(theta)
    st = torch.sin(theta)
    ca = math.cos(alpha)
    sa = math.sin(alpha)
    zeros = torch.zeros_like(theta)
    ones = torch.ones_like(theta)
    row0 = torch.stack((ct, -st * ca, st * sa, a * ct), dim=-1)
    row1 = torch.stack((st, ct * ca, -ct * sa, a * st), dim=-1)
    row2 = torch.stack((zeros, torch.full_like(theta, sa), torch.full_like(theta, ca), torch.full_like(theta, d)), dim=-1)
    row3 = torch.stack((zeros, zeros, zeros, ones), dim=-1)
    return torch.stack((row0, row1, row2, row3), dim=-2)
