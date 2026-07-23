from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoxObstacleComponent:
    """One box primitive in an obstacle family's local frame."""

    name: str
    half_extents: tuple[float, float, float]
    local_center: tuple[float, float, float]
    yaw_offset: float = 0.0


_DOOR_OPEN_YAW = float(np.deg2rad(70.0))
_DOOR_HINGE = np.asarray([-0.08, -0.085], dtype=np.float32)
_DOOR_CENTER_FROM_HINGE = np.asarray([0.08, 0.0], dtype=np.float32)
_DOOR_ROTATION = np.asarray(
    [
        [np.cos(_DOOR_OPEN_YAW), -np.sin(_DOOR_OPEN_YAW)],
        [np.sin(_DOOR_OPEN_YAW), np.cos(_DOOR_OPEN_YAW)],
    ],
    dtype=np.float32,
)
_DOOR_CENTER_XY = _DOOR_HINGE + _DOOR_ROTATION @ _DOOR_CENTER_FROM_HINGE

CABINET_COMPONENTS: tuple[BoxObstacleComponent, ...] = (
    BoxObstacleComponent("left_side", (0.005, 0.08, 0.20), (-0.075, 0.0, 0.0)),
    BoxObstacleComponent("right_side", (0.005, 0.08, 0.20), (0.075, 0.0, 0.0)),
    BoxObstacleComponent("top", (0.08, 0.08, 0.005), (0.0, 0.0, 0.195)),
    BoxObstacleComponent("bottom", (0.08, 0.08, 0.005), (0.0, 0.0, -0.195)),
    BoxObstacleComponent("back", (0.08, 0.005, 0.20), (0.0, 0.075, 0.0)),
    # The shelf is centered at the family origin and is the root-pose anchor.
    BoxObstacleComponent("shelf", (0.075, 0.075, 0.005), (0.0, 0.0, 0.0)),
    BoxObstacleComponent(
        "open_door",
        (0.08, 0.005, 0.19),
        (float(_DOOR_CENTER_XY[0]), float(_DOOR_CENTER_XY[1]), 0.0),
        _DOOR_OPEN_YAW,
    ),
)


def scaled_cabinet_components(half_height: float) -> tuple[BoxObstacleComponent, ...]:
    """Scale the cabinet vertically while preserving its tabletop-supported shape."""
    if not np.isfinite(half_height) or half_height <= 0.0:
        raise ValueError("cabinet half-height must be positive and finite")
    reference_half_height = 0.20
    scale = float(half_height) / reference_half_height
    return tuple(
        BoxObstacleComponent(
            name=component.name,
            half_extents=(
                component.half_extents[0],
                component.half_extents[1],
                component.half_extents[2] * scale,
            ),
            local_center=(
                component.local_center[0],
                component.local_center[1],
                component.local_center[2] * scale,
            ),
            yaw_offset=component.yaw_offset,
        )
        for component in CABINET_COMPONENTS
    )


def transform_box_component(
    component: BoxObstacleComponent,
    *,
    center: np.ndarray,
    yaw: float,
) -> tuple[np.ndarray, float]:
    """Transform a cabinet component from family-local to world coordinates."""
    center = np.asarray(center, dtype=np.float32).reshape(3)
    cos_yaw = float(np.cos(yaw))
    sin_yaw = float(np.sin(yaw))
    local = np.asarray(component.local_center, dtype=np.float32)
    world_offset = np.asarray(
        [
            cos_yaw * local[0] - sin_yaw * local[1],
            sin_yaw * local[0] + cos_yaw * local[1],
            local[2],
        ],
        dtype=np.float32,
    )
    return (center + world_offset).astype(np.float32), float(yaw + component.yaw_offset)
