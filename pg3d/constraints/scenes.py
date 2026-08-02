"""Hand-authored and procedural non-convex obstacle scene builders.

Each scene is a small cluster of convex primitives (sphere/cylinder) arranged
so their *union* forms a non-convex keep-out shape between an episode's start
and goal. No new ``Region`` subclass is needed for this: the eval harness
already scores a list of ``AvoidRegion`` constraints together, so several
simultaneous convex regions already behave as one non-convex obstacle for
constraint-cost purposes.

Scenes are built from ``SphereRegion``/``CylinderRegion`` rather than
``BoxRegion`` because ``BoxRegion`` is axis-aligned only (see
``pg3d/constraints/geometry.py``) and these layouts need to orient relative to
the (arbitrary, per-episode) start-to-goal direction, not to world axes.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from pg3d.constraints import AvoidRegion, CylinderRegion, SphereRegion

Array = np.ndarray

ObstacleSceneName = Literal[
    "none",
    "wall_gap",
    "l_corner",
    "u_pocket",
    "pillar_cluster",
    "cluttered_field",
]

OBSTACLE_SCENE_NAMES: tuple[str, ...] = (
    "none",
    "wall_gap",
    "l_corner",
    "u_pocket",
    "pillar_cluster",
    "cluttered_field",
)


def _direct_frame(start: Array, goal: Array) -> tuple[Array, Array, Array, float]:
    """Return (midpoint, forward_unit, lateral_unit, path_length) for start->goal.

    ``lateral`` is horizontal (perpendicular to forward and to world-up), so
    scenes built from it stay in the same horizontal plane as the direct path
    rather than tilting arbitrarily with the path's incline.
    """
    start = np.asarray(start, dtype=np.float32).reshape(3)
    goal = np.asarray(goal, dtype=np.float32).reshape(3)
    delta = goal - start
    length = float(np.linalg.norm(delta))
    if length > 1e-6:
        forward = (delta / length).astype(np.float32)
    else:
        forward = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    lateral = np.cross(forward, world_up)
    lateral_norm = float(np.linalg.norm(lateral))
    if lateral_norm > 1e-6:
        lateral = (lateral / lateral_norm).astype(np.float32)
    else:
        lateral = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    midpoint = 0.5 * (start + goal)
    return midpoint, forward, lateral, length


def _sphere_row(
    origin: Array,
    direction: Array,
    count: int,
    radius: float,
    *,
    spacing_factor: float = 1.8,
) -> list[SphereRegion]:
    """Return ``count`` spheres of ``radius`` starting at ``origin``, stepping along ``direction``."""
    direction = np.asarray(direction, dtype=np.float32)
    norm = float(np.linalg.norm(direction))
    if norm > 1e-6:
        direction = direction / norm
    spacing = radius * spacing_factor
    return [
        SphereRegion(center=origin + direction * (i * spacing), radius=radius)
        for i in range(max(count, 0))
    ]


def _as_avoid_regions(
    regions: list[Any],
    *,
    weight: float,
    margin: float,
    name_prefix: str,
) -> list[AvoidRegion]:
    return [
        AvoidRegion(region=region, margin=margin, weight=weight, name=f"{name_prefix}_{i}")
        for i, region in enumerate(regions)
    ]


def wall_gap_scene(
    *,
    start: Array,
    goal: Array,
    object_radius: float = 0.05,
    arm_segments: int = 2,
    gap_width: float | None = None,
    weight: float = 1.0,
    margin: float = 0.0,
    name_prefix: str = "obstacle_scene_wall_gap",
) -> list[AvoidRegion]:
    """Wall of spheres across the direct path with a gap off to one side.

    Blocks the straight start-to-goal line; the only way through is the gap,
    which sits off-center (not on the direct line) so the shortest detour is
    a real lateral move, not just squeezing straight through.
    """
    midpoint, _forward, lateral, _length = _direct_frame(start, goal)
    gap = float(gap_width) if gap_width is not None else object_radius * 2.4
    right = _sphere_row(midpoint + lateral * (gap / 2.0 + object_radius), lateral, arm_segments, object_radius)
    left = _sphere_row(midpoint - lateral * (gap / 2.0 + object_radius), -lateral, arm_segments, object_radius)
    return _as_avoid_regions(right + left, weight=weight, margin=margin, name_prefix=name_prefix)


def l_corner_scene(
    *,
    start: Array,
    goal: Array,
    object_radius: float = 0.05,
    arm_segments: int = 3,
    weight: float = 1.0,
    margin: float = 0.0,
    name_prefix: str = "obstacle_scene_l_corner",
) -> list[AvoidRegion]:
    """Solid wall across the direct path plus a perpendicular arm off one end.

    The base wall blocks the straight line; the perpendicular arm closes off
    the nearer detour around one end, so escaping requires going around the
    open end instead.
    """
    midpoint, forward, lateral, _length = _direct_frame(start, goal)
    base = _sphere_row(
        midpoint - lateral * ((arm_segments - 1) * object_radius * 0.9),
        lateral,
        arm_segments * 2,
        object_radius,
    )
    corner = base[-1].center if base else midpoint
    arm = _sphere_row(corner, forward, arm_segments, object_radius)
    return _as_avoid_regions(base + arm, weight=weight, margin=margin, name_prefix=name_prefix)


def u_pocket_scene(
    *,
    start: Array,
    goal: Array,
    object_radius: float = 0.05,
    base_segments: int = 3,
    side_segments: int = 2,
    weight: float = 1.0,
    margin: float = 0.0,
    name_prefix: str = "obstacle_scene_u_pocket",
) -> list[AvoidRegion]:
    """Pocket (U-shape) with its mouth facing the start.

    A greedy straight-line policy drives into the mouth and dead-ends at the
    base wall; escaping requires backing out and going around one of the
    side walls -- the classic local-minimum trap.
    """
    midpoint, forward, lateral, _length = _direct_frame(start, goal)
    base_origin = midpoint - lateral * ((base_segments - 1) * object_radius * 0.9)
    base = _sphere_row(base_origin, lateral, base_segments * 2, object_radius)
    if not base:
        return []
    left_corner = base[0].center
    right_corner = base[-1].center
    left_arm = _sphere_row(left_corner, -forward, side_segments, object_radius)
    right_arm = _sphere_row(right_corner, -forward, side_segments, object_radius)
    return _as_avoid_regions(base + left_arm + right_arm, weight=weight, margin=margin, name_prefix=name_prefix)


def pillar_cluster_scene(
    *,
    start: Array,
    goal: Array,
    num_pillars: int = 5,
    pillar_radius: float = 0.03,
    pillar_height: float = 0.3,
    jitter: float = 0.08,
    weight: float = 1.0,
    margin: float = 0.0,
    name_prefix: str = "obstacle_scene_pillar",
    rng: np.random.Generator | None = None,
) -> list[AvoidRegion]:
    """Cluster of vertical cylindrical pillars scattered near the path midpoint."""
    midpoint, forward, lateral, length = _direct_frame(start, goal)
    rng = rng if rng is not None else np.random.default_rng()
    axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    regions: list[Any] = []
    for _ in range(max(num_pillars, 0)):
        along = float(rng.uniform(-0.5, 0.5)) * length * 0.4
        across = float(rng.uniform(-1.0, 1.0)) * jitter
        center = midpoint + forward * along + lateral * across
        regions.append(CylinderRegion(center=center, axis=axis, radius=pillar_radius, length=pillar_height))
    return _as_avoid_regions(regions, weight=weight, margin=margin, name_prefix=name_prefix)


def cluttered_field_scene(
    *,
    start: Array,
    goal: Array,
    num_objects: int = 6,
    object_radius: float = 0.04,
    field_span: float = 0.4,
    min_separation_from_endpoints: float = 0.1,
    weight: float = 1.0,
    margin: float = 0.0,
    name_prefix: str = "obstacle_scene_clutter",
    rng: np.random.Generator | None = None,
    max_attempts_per_object: int = 50,
) -> list[AvoidRegion]:
    """Procedurally scattered convex clutter (spheres + short vertical cylinders).

    Rejection-samples each object's placement to keep a minimum clearance
    from the start/goal and from previously placed objects, so the field is
    dense without guaranteeing an unsolvable episode. Meant for the
    "search across environment variants" exploration, not a fixed trap
    topology -- re-run with different ``rng``/``clutter_seed`` values to get
    different arrangements.
    """
    midpoint, forward, lateral, length = _direct_frame(start, goal)
    rng = rng if rng is not None else np.random.default_rng()
    start = np.asarray(start, dtype=np.float32).reshape(3)
    goal = np.asarray(goal, dtype=np.float32).reshape(3)
    placed_centers: list[Array] = []
    regions: list[Any] = []
    span = max(length, 1e-3) * field_span
    for obj_idx in range(max(num_objects, 0)):
        for _attempt in range(max_attempts_per_object):
            along = float(rng.uniform(-0.5, 0.5)) * span
            across = float(rng.uniform(-1.0, 1.0)) * span
            center = midpoint + forward * along + lateral * across
            if (
                np.linalg.norm(center - start) < min_separation_from_endpoints
                or np.linalg.norm(center - goal) < min_separation_from_endpoints
            ):
                continue
            if any(np.linalg.norm(center - other) < object_radius * 2.5 for other in placed_centers):
                continue
            placed_centers.append(center)
            if obj_idx % 2 == 0:
                regions.append(SphereRegion(center=center, radius=object_radius))
            else:
                regions.append(
                    CylinderRegion(
                        center=center,
                        axis=np.array([0.0, 0.0, 1.0], dtype=np.float32),
                        radius=object_radius,
                        length=object_radius * 6.0,
                    )
                )
            break
    return _as_avoid_regions(regions, weight=weight, margin=margin, name_prefix=name_prefix)


def build_obstacle_scene(
    name: str,
    *,
    start: Array,
    goal: Array,
    object_radius: float,
    arm_segments: int,
    weight: float,
    margin: float,
    num_clutter_objects: int,
    rng: np.random.Generator | None = None,
) -> list[AvoidRegion]:
    """Dispatch to the named scene builder using shared sizing parameters."""
    if name == "none":
        return []
    if name == "wall_gap":
        return wall_gap_scene(
            start=start,
            goal=goal,
            object_radius=object_radius,
            arm_segments=arm_segments,
            weight=weight,
            margin=margin,
        )
    if name == "l_corner":
        return l_corner_scene(
            start=start,
            goal=goal,
            object_radius=object_radius,
            arm_segments=arm_segments,
            weight=weight,
            margin=margin,
        )
    if name == "u_pocket":
        return u_pocket_scene(
            start=start,
            goal=goal,
            object_radius=object_radius,
            base_segments=arm_segments,
            side_segments=max(1, arm_segments - 1),
            weight=weight,
            margin=margin,
        )
    if name == "pillar_cluster":
        return pillar_cluster_scene(
            start=start,
            goal=goal,
            num_pillars=num_clutter_objects,
            pillar_radius=object_radius,
            weight=weight,
            margin=margin,
            rng=rng,
        )
    if name == "cluttered_field":
        return cluttered_field_scene(
            start=start,
            goal=goal,
            num_objects=num_clutter_objects,
            object_radius=object_radius,
            weight=weight,
            margin=margin,
            rng=rng,
        )
    raise ValueError(f"unknown obstacle scene {name!r}")
