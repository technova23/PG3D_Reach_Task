from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from pg3d.constraints import AvoidProjection, AvoidRegion, BoxRegion, SphereRegion

DEFAULT_AVOID_COLOR = (255, 64, 16)
# Z range used to render the (height-agnostic) avoid_projection footprint as a
# visible extruded box in Rerun. Display-only; the constraint itself is infinite in Z.
PROJECTION_VISUAL_Z_RANGE = (0.0, 0.5)


@dataclass(frozen=True)
class ConstraintLineVisual:
    """Rerun-friendly line-strip representation of a constraint primitive."""

    name: str
    line_strips: list[np.ndarray]
    color: tuple[int, int, int] = DEFAULT_AVOID_COLOR


def avoid_region_line_visuals(
    constraints: Iterable[object],
    *,
    sphere_segments: int = 64,
) -> list[ConstraintLineVisual]:
    """Convert supported `AvoidRegion` objects into deterministic wireframe visuals."""
    if sphere_segments < 8:
        raise ValueError("sphere_segments must be at least 8")
    visuals: list[ConstraintLineVisual] = []
    for constraint_idx, constraint in enumerate(constraints):
        if isinstance(constraint, AvoidProjection):
            region = constraint.region
            z_lo, z_hi = PROJECTION_VISUAL_Z_RANGE
            center3 = np.array(
                [region.center[0], region.center[1], 0.5 * (z_lo + z_hi)], dtype=np.float32
            )
            half3 = np.array(
                [region.half_extents[0], region.half_extents[1], 0.5 * (z_hi - z_lo)],
                dtype=np.float32,
            )
            visuals.append(
                ConstraintLineVisual(
                    name=f"avoid_projection_{len(visuals)}",
                    line_strips=box_wireframe(center3, half3),
                )
            )
            continue
        if not isinstance(constraint, AvoidRegion):
            continue
        region = constraint.region
        name = f"avoid_region_{len(visuals)}"
        if isinstance(region, SphereRegion):
            visuals.append(
                ConstraintLineVisual(
                    name=name,
                    line_strips=sphere_wireframe(
                        region.center,
                        region.radius,
                        segments=sphere_segments,
                    ),
                )
            )
        elif isinstance(region, BoxRegion):
            visuals.append(
                ConstraintLineVisual(
                    name=name,
                    line_strips=box_wireframe(region.center, region.half_extents),
                )
            )
        else:
            raise TypeError(
                f"unsupported AvoidRegion primitive at index {constraint_idx}: "
                f"{type(region).__name__}"
            )
    return visuals


def sphere_wireframe(
    center: np.ndarray,
    radius: float,
    *,
    segments: int = 64,
) -> list[np.ndarray]:
    """Return three great-circle line strips for a sphere."""
    center_array = _vector3(center, name="center")
    radius = float(radius)
    if radius <= 0.0 or not np.isfinite(radius):
        raise ValueError("radius must be a positive finite value")
    if segments < 8:
        raise ValueError("segments must be at least 8")

    theta = np.linspace(0.0, 2.0 * np.pi, segments + 1, dtype=np.float32)
    cos = np.cos(theta) * radius
    sin = np.sin(theta) * radius
    zero = np.zeros_like(theta)
    circles = [
        np.stack([cos, sin, zero], axis=1),
        np.stack([cos, zero, sin], axis=1),
        np.stack([zero, cos, sin], axis=1),
    ]
    return [(circle + center_array.reshape(1, 3)).astype(np.float32) for circle in circles]


def box_wireframe(center: np.ndarray, half_extents: np.ndarray) -> list[np.ndarray]:
    """Return twelve edge line strips for an axis-aligned box."""
    center_array = _vector3(center, name="center")
    half_extents_array = _vector3(half_extents, name="half_extents")
    if np.any(half_extents_array <= 0.0) or not np.all(np.isfinite(half_extents_array)):
        raise ValueError("half_extents must be positive finite values")
    signs = np.asarray(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=np.float32,
    )
    corners = center_array.reshape(1, 3) + signs * half_extents_array.reshape(1, 3)
    edge_indices = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    return [corners[list(edge)].astype(np.float32) for edge in edge_indices]


def _vector3(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array
