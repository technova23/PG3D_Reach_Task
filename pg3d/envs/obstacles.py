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

U_SHAPE_REFERENCE_HALF_EXTENTS = (0.14, 0.15, 0.30)


def u_shape_components(
    half_extents: tuple[float, float, float] | np.ndarray,
) -> tuple[BoxObstacleComponent, ...]:
    """Build a tabletop-supported U from two side walls and one back wall.

    The family-local opening faces ``-Y`` and the closed back lies at ``+Y``.
    ``half_extents`` describes the complete U envelope. Wall thickness scales with
    the envelope so difficulty sweeps can resize the family without introducing
    separate hidden dimensions.
    """
    envelope = np.asarray(half_extents, dtype=np.float32)
    if envelope.shape != (3,) or not np.all(np.isfinite(envelope)) or np.any(envelope <= 0.0):
        raise ValueError("U-shape half-extents must contain three positive finite values")
    half_x, half_y, half_z = (float(value) for value in envelope)
    side_half_thickness = half_x / 7.0
    back_half_thickness = half_y * (2.0 / 15.0)
    side_center_x = half_x - side_half_thickness
    back_center_y = half_y - back_half_thickness
    return (
        BoxObstacleComponent(
            "left_side",
            (side_half_thickness, half_y, half_z),
            (-side_center_x, 0.0, 0.0),
        ),
        BoxObstacleComponent(
            "right_side",
            (side_half_thickness, half_y, half_z),
            (side_center_x, 0.0, 0.0),
        ),
        BoxObstacleComponent(
            "back",
            (half_x, back_half_thickness, half_z),
            (0.0, back_center_y, 0.0),
        ),
    )


GATE_ENVELOPE_HALF_EXTENTS = (0.14, 0.12, 0.20)
BRANCH_GATE_ENVELOPE_HALF_EXTENTS = (0.14, 0.12, 0.20)


def gate_gap_width(half_extents: tuple[float, float, float] | np.ndarray) -> float:
    """Open slot width (meters) that ``gate_components``/``branch_gate_components``
    would carve out of the given envelope -- exposed so a difficulty sweep can
    report the actual physical gap instead of the raw envelope half-extents."""
    envelope = np.asarray(half_extents, dtype=np.float32)
    return float(0.8 * envelope[0])


def gate_components(
    half_extents: tuple[float, float, float] | np.ndarray,
) -> tuple[BoxObstacleComponent, ...]:
    """Build a narrow slot from two parallel grounded panels (real, embodied --
    not a virtual/guidance-only region).

    The family-local gap opens along ``X``; both panels run the full ``Y``
    depth and ``Z`` height. ``half_extents`` describes the complete gate
    envelope, split 40/60 between the open gap and the two panels (see
    ``gate_gap_width``) so the gap scales down toward a gripper-width slot as
    the envelope narrows -- mirrors ``u_shape_components``' scale-with-envelope
    contract so a difficulty sweep can resize the family alone.
    """
    envelope = np.asarray(half_extents, dtype=np.float32)
    if envelope.shape != (3,) or not np.all(np.isfinite(envelope)) or np.any(envelope <= 0.0):
        raise ValueError("gate half-extents must contain three positive finite values")
    half_x, half_y, half_z = (float(value) for value in envelope)
    gap_half_width = half_x * 0.4
    panel_half_thickness = (half_x - gap_half_width) / 2.0
    panel_center_x = gap_half_width + panel_half_thickness
    return (
        BoxObstacleComponent(
            "left_panel",
            (panel_half_thickness, half_y, half_z),
            (-panel_center_x, 0.0, 0.0),
        ),
        BoxObstacleComponent(
            "right_panel",
            (panel_half_thickness, half_y, half_z),
            (panel_center_x, 0.0, 0.0),
        ),
    )


def branch_gate_components(
    half_extents: tuple[float, float, float] | np.ndarray,
) -> tuple[BoxObstacleComponent, ...]:
    """Gate slot plus one branch-blocking panel set back on the +X side.

    Composes ``gate_components`` (the narrow slot the arm must thread) with a
    third panel, offset along +Y beyond the right gate panel and aligned with
    it in X. A trajectory that clears the gate by favoring the +X-side branch
    is then forced into the blocker; only a route that favors -X through the
    slot (or re-crosses to -X after it) avoids both obstacles. This is the
    compound family: gate-threading and branch-avoidance must both be
    satisfied by *one* trajectory, not resolved independently -- the
    conjunction is the point, not either constraint alone.
    """
    gate = gate_components(half_extents)
    envelope = np.asarray(half_extents, dtype=np.float32)
    half_y, half_z = float(envelope[1]), float(envelope[2])
    right_panel = next(component for component in gate if component.name == "right_panel")
    blocker_half_thickness = half_y / 2.0
    blocker_center_x = right_panel.local_center[0]
    blocker_center_y = half_y + blocker_half_thickness
    blocker = BoxObstacleComponent(
        "branch_blocker",
        (right_panel.half_extents[0], blocker_half_thickness, half_z),
        (blocker_center_x, blocker_center_y, 0.0),
    )
    return (*gate, blocker)


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
    """Transform an obstacle component from family-local to world coordinates."""
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
