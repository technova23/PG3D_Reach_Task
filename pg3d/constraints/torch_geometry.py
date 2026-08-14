from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn.functional as F

from pg3d.constraints.geometry import (
    BoxRegion,
    CylinderRegion,
    RectRegion2D,
    SphereRegion,
)
from pg3d.constraints.programs import AvoidProjection, AvoidRegion

AvoidanceEnergyMode = Literal["smooth", "hinge"]
AvoidanceTarget = Literal["eef", "robot"]
AvoidanceConstraint = AvoidRegion | AvoidProjection


def avoidance_energy(
    points: torch.Tensor,
    constraints: Sequence[AvoidanceConstraint],
    *,
    target: AvoidanceTarget = "eef",
    mode: AvoidanceEnergyMode = "smooth",
    temperature: float = 0.01,
) -> torch.Tensor:
    """Return one differentiable EEF or whole-robot avoidance energy per batch item."""
    if target == "eef":
        expected_shape = "[B, T, 3]"
        valid_shape = points.ndim == 3 and points.shape[-1] == 3
    elif target == "robot":
        expected_shape = "[B, T, N, 3]"
        valid_shape = points.ndim == 4 and points.shape[-1] == 3 and points.shape[-2] > 0
    else:
        raise ValueError(f"unsupported avoidance target {target!r}")
    if not valid_shape:
        raise ValueError(
            f"{target} points must have shape {expected_shape}, got {tuple(points.shape)}"
        )
    if mode not in {"smooth", "hinge"}:
        raise ValueError(f"unsupported avoidance energy mode {mode!r}")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    reduction_dims = tuple(range(1, points.ndim - 1))
    total = points.sum(dim=tuple(range(1, points.ndim))) * 0.0
    for constraint in constraints:
        if constraint.target != target:
            raise ValueError(
                f"avoidance constraint target {constraint.target!r} does not match "
                f"guidance target {target!r}"
            )
        signed_distance = _signed_distance(points, constraint.region)
        violation = float(constraint.margin) - signed_distance
        if mode == "hinge":
            point_energy = torch.clamp(violation, min=0.0)
        else:
            point_energy = float(temperature) * F.softplus(violation / float(temperature))
        total = total + float(constraint.weight) * torch.amax(
            point_energy,
            dim=reduction_dims,
        )
    return total


def _signed_distance(
    points: torch.Tensor,
    region: SphereRegion | BoxRegion | CylinderRegion | RectRegion2D,
) -> torch.Tensor:
    if isinstance(region, SphereRegion):
        center = torch.as_tensor(region.center, device=points.device, dtype=points.dtype)
        return torch.linalg.norm(points - center.view(1, 1, 3), dim=-1) - float(region.radius)
    if isinstance(region, BoxRegion):
        center = torch.as_tensor(region.center, device=points.device, dtype=points.dtype)
        half_extents = torch.as_tensor(
            region.half_extents,
            device=points.device,
            dtype=points.dtype,
        )
        local = points - center.view(1, 1, 3)
        if region.yaw != 0.0:
            cos_yaw = torch.cos(points.new_tensor(region.yaw))
            sin_yaw = torch.sin(points.new_tensor(region.yaw))
            local_x = local[..., 0] * cos_yaw + local[..., 1] * sin_yaw
            local_y = -local[..., 0] * sin_yaw + local[..., 1] * cos_yaw
            local = torch.stack([local_x, local_y, local[..., 2]], dim=-1)
        q = torch.abs(local) - half_extents.view(1, 1, 3)
        outside = torch.linalg.norm(torch.clamp(q, min=0.0), dim=-1)
        inside = torch.minimum(torch.amax(q, dim=-1), torch.zeros_like(outside))
        return outside + inside
    if isinstance(region, CylinderRegion):
        center = torch.as_tensor(region.center, device=points.device, dtype=points.dtype)
        local = points - center.view(1, 1, 3)
        radial = torch.linalg.norm(local[..., :2], dim=-1) - float(region.radius)
        axial = torch.abs(local[..., 2]) - float(region.half_length)
        d = torch.stack([radial, axial], dim=-1)
        outside = torch.linalg.norm(torch.clamp(d, min=0.0), dim=-1)
        inside = torch.minimum(torch.amax(d, dim=-1), torch.zeros_like(outside))
        return outside + inside
    if isinstance(region, RectRegion2D):
        center = torch.as_tensor(region.center, device=points.device, dtype=points.dtype)
        half_extents = torch.as_tensor(
            region.half_extents,
            device=points.device,
            dtype=points.dtype,
        )
        q = torch.abs(points[..., :2] - center.view(1, 1, 2)) - half_extents.view(1, 1, 2)
        outside = torch.linalg.norm(torch.clamp(q, min=0.0), dim=-1)
        inside = torch.minimum(torch.amax(q, dim=-1), torch.zeros_like(outside))
        return outside + inside
    raise TypeError(f"unsupported avoidance region {type(region).__name__}")
