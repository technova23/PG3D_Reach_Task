"""Hand-authored and procedural non-convex obstacle scene builders.

Each scene is a small cluster of convex primitives (sphere/box/cylinder)
arranged so their *union* forms a non-convex keep-out shape between an
episode's start and goal. No new ``Region`` subclass is needed for this: the
eval harness already scores a list of ``AvoidRegion`` constraints together,
so several simultaneous convex regions already behave as one non-convex
obstacle for constraint-cost purposes.

Wall-shaped scenes (``wall_gap``/``l_corner``/``u_pocket``/``shortcut_trap``)
default to solid ``BoxRegion`` walls (``primitive="box"``), oriented via
``BoxRegion.rotation`` to the (arbitrary, per-episode) start-to-goal frame
rather than world axes -- a single oriented box per straight segment is a
genuinely gap-free barrier, unlike a chain of spheres spaced along the same
line (``primitive="sphere"``, the original/legacy behavior, still available)
which can leave thin gaps a candidate path slips through cheaply. Wall height
also defaults far taller than a single ``object_radius`` so "hop over the
top" isn't a trivially cheap escape either -- see ``wall_height``.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from pg3d.constraints import AvoidRegion, BoxRegion, CylinderRegion, SphereRegion

Array = np.ndarray

ObstaclePrimitive = Literal["sphere", "box"]

# Default wall height (meters) for box-primitive walls -- deliberately much
# taller than a single object_radius so going over the top costs a real
# vertical detour instead of a cheap hop. Tune down for a shorter/easier wall.
DEFAULT_WALL_HEIGHT = 0.35

ObstacleSceneName = Literal[
    "none",
    "wall_gap",
    "l_corner",
    "u_pocket",
    "shortcut_trap",
    "pillar_cluster",
    "cluttered_field",
]

OBSTACLE_SCENE_NAMES: tuple[str, ...] = (
    "none",
    "wall_gap",
    "l_corner",
    "u_pocket",
    "shortcut_trap",
    "pillar_cluster",
    "cluttered_field",
)


def _direct_frame(
    start: Array,
    goal: Array,
    *,
    placement_fraction: float = 0.5,
) -> tuple[Array, Array, Array, float]:
    """Return (placement_point, forward_unit, lateral_unit, path_length) for start->goal.

    ``placement_point`` sits at ``placement_fraction`` of the way from start
    (0.0) to goal (1.0) -- 0.5 (default) is the midpoint used by every scene
    originally. Placing closer to start (e.g. 0.2-0.3) puts the obstacle
    where a greedy policy commits to it earliest, while most of the
    remaining path length is still ahead -- the "duck toward the obstacle
    because it looks cheap early on, discover it's a dead end" framing.

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
    fraction = float(np.clip(placement_fraction, 0.0, 1.0))
    placement_point = start + delta * fraction
    return placement_point, forward, lateral, length


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


def _box_segment(
    origin: Array,
    axis: Array,
    other_axis: Array,
    length: float,
    *,
    thickness: float,
    height: float,
) -> BoxRegion:
    """Return one solid ``BoxRegion`` spanning ``length`` along ``axis``.

    ``axis`` is the segment's long/spanning direction (e.g. lateral for a
    wall crossing the path, or forward for a perpendicular closing arm).
    ``other_axis`` need only be non-parallel to ``axis`` -- it's used to
    derive a right-handed local frame (``axis``, ``other_axis`` projected
    orthogonal to ``axis``, and their cross product as the remaining axis),
    so the caller doesn't have to pre-orthogonalize anything.
    """
    axis = np.asarray(axis, dtype=np.float32).reshape(3)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-6:
        raise ValueError("axis must be a non-zero vector")
    axis = axis / axis_norm
    other_axis = np.asarray(other_axis, dtype=np.float32).reshape(3)
    # Orthogonalize other_axis against axis (Gram-Schmidt) so an
    # approximately-but-not-exactly-perpendicular caller value still works.
    other_axis = other_axis - axis * float(np.dot(axis, other_axis))
    other_norm = float(np.linalg.norm(other_axis))
    if other_norm > 1e-6:
        other_axis = (other_axis / other_norm).astype(np.float32)
    else:
        fallback = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(float(np.dot(axis, fallback))) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        other_axis = fallback - axis * float(np.dot(axis, fallback))
        other_axis = (other_axis / float(np.linalg.norm(other_axis))).astype(np.float32)
    up = np.cross(axis, other_axis).astype(np.float32)
    rotation = np.column_stack([axis, other_axis, up]).astype(np.float32)
    return BoxRegion(
        center=np.asarray(origin, dtype=np.float32).reshape(3),
        half_extents=np.array(
            [max(length, 1e-3) / 2.0, max(thickness, 1e-3) / 2.0, max(height, 1e-3) / 2.0],
            dtype=np.float32,
        ),
        rotation=rotation,
    )


def _wall_segment(
    origin: Array,
    axis: Array,
    thickness_axis: Array,
    span_segments: int,
    object_radius: float,
    primitive: ObstaclePrimitive,
    *,
    wall_height: float = DEFAULT_WALL_HEIGHT,
) -> tuple[list[Any], Array]:
    """Build a wall segment starting at ``origin`` and extending along ``axis``.

    Returns ``(regions, far_end)``: ``regions`` is either a chain of spheres
    (``primitive="sphere"``, legacy behavior, spaced via ``_sphere_row``) or
    a single solid oriented box (``primitive="box"``, default -- gap-free
    and ``wall_height`` tall rather than ``2 * object_radius``). ``far_end``
    is the point at the segment's outer tip, for chaining a perpendicular
    segment onto its corner (see ``l_corner_scene``/``u_pocket_scene``).
    """
    origin = np.asarray(origin, dtype=np.float32).reshape(3)
    axis = np.asarray(axis, dtype=np.float32).reshape(3)
    axis_norm = float(np.linalg.norm(axis))
    axis_unit = (axis / axis_norm).astype(np.float32) if axis_norm > 1e-6 else axis
    span_segments = max(int(span_segments), 0)
    if primitive == "sphere":
        regions = _sphere_row(origin, axis_unit, span_segments, object_radius, spacing_factor=1.8)
        far_end = regions[-1].center if regions else origin
        return list(regions), far_end
    if primitive == "box":
        if span_segments == 0:
            return [], origin
        spacing = object_radius * 1.8
        length = span_segments * spacing + object_radius * 2.0
        center = origin + axis_unit * (length / 2.0)
        far_end = origin + axis_unit * length
        box = _box_segment(
            center,
            axis_unit,
            thickness_axis,
            length,
            thickness=object_radius * 2.0,
            height=wall_height,
        )
        return [box], far_end
    raise ValueError(f"unknown obstacle-scene primitive {primitive!r}; expected 'sphere' or 'box'")


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


def wall_gap_at(
    *,
    center: Array,
    forward: Array,
    lateral: Array,
    object_radius: float = 0.05,
    arm_segments: int = 2,
    gap_width: float | None = None,
    weight: float = 1.0,
    margin: float = 0.0,
    primitive: ObstaclePrimitive = "box",
    wall_height: float = DEFAULT_WALL_HEIGHT,
    name_prefix: str = "obstacle_scene_wall_gap",
) -> list[AvoidRegion]:
    """Build a wall-with-a-gap at an explicit center/orientation.

    Lower-level than ``wall_gap_scene``: takes ``forward``/``lateral`` frame
    vectors directly instead of deriving them from a straight start->goal
    line, so the wall can follow the local tangent of a *curved* reference
    path (e.g. the median of a rolled-out candidate-trajectory bundle) at
    one point along it, not just a straight chord. ``wall_gap_scene``
    computes its frame from ``_direct_frame`` and delegates here; call this
    directly for curved-path placement -- see
    ``scripts/eval_constrained_reach.py``'s ``--obstacle-spawn-first-shape
    wall_gap`` for the motivating use (placing a wall_gap on the candidate
    bundle's own median path instead of the raw start-goal chord).
    """
    center = np.asarray(center, dtype=np.float32).reshape(3)
    forward = np.asarray(forward, dtype=np.float32).reshape(3)
    lateral = np.asarray(lateral, dtype=np.float32).reshape(3)
    gap = float(gap_width) if gap_width is not None else object_radius * 2.4
    right, _ = _wall_segment(
        center + lateral * (gap / 2.0 + object_radius),
        lateral,
        forward,
        arm_segments,
        object_radius,
        primitive,
        wall_height=wall_height,
    )
    left, _ = _wall_segment(
        center - lateral * (gap / 2.0 + object_radius),
        -lateral,
        forward,
        arm_segments,
        object_radius,
        primitive,
        wall_height=wall_height,
    )
    return _as_avoid_regions(right + left, weight=weight, margin=margin, name_prefix=name_prefix)


def wall_gap_scene(
    *,
    start: Array,
    goal: Array,
    object_radius: float = 0.05,
    arm_segments: int = 2,
    gap_width: float | None = None,
    weight: float = 1.0,
    margin: float = 0.0,
    primitive: ObstaclePrimitive = "box",
    wall_height: float = DEFAULT_WALL_HEIGHT,
    placement_fraction: float = 0.5,
    name_prefix: str = "obstacle_scene_wall_gap",
) -> list[AvoidRegion]:
    """Wall across the direct path with a gap off to one side.

    Blocks the straight start-to-goal line; the only way through is the gap,
    which sits off-center (not on the direct line) so the shortest detour is
    a real lateral move, not just squeezing straight through.
    ``primitive="box"`` (default) makes each side a single solid oriented
    slab -- gap-free, unlike ``primitive="sphere"`` (legacy) which chains
    spheres and can leave thin cusps between them for a candidate path to
    clip through cheaply.
    """
    placement, forward, lateral, _length = _direct_frame(
        start, goal, placement_fraction=placement_fraction
    )
    return wall_gap_at(
        center=placement,
        forward=forward,
        lateral=lateral,
        object_radius=object_radius,
        arm_segments=arm_segments,
        gap_width=gap_width,
        weight=weight,
        margin=margin,
        primitive=primitive,
        wall_height=wall_height,
        name_prefix=name_prefix,
    )


def l_corner_scene(
    *,
    start: Array,
    goal: Array,
    object_radius: float = 0.05,
    arm_segments: int = 3,
    weight: float = 1.0,
    margin: float = 0.0,
    primitive: ObstaclePrimitive = "box",
    wall_height: float = DEFAULT_WALL_HEIGHT,
    placement_fraction: float = 0.5,
    name_prefix: str = "obstacle_scene_l_corner",
) -> list[AvoidRegion]:
    """Wall across the direct path plus a perpendicular arm off one end.

    The base wall blocks the straight line; the perpendicular arm closes off
    the nearer detour around one end, so escaping requires going around the
    open end instead. ``primitive="box"`` (default) builds both pieces as
    solid oriented slabs; ``primitive="sphere"`` keeps the original chained
    spheres.
    """
    placement, forward, lateral, _length = _direct_frame(
        start, goal, placement_fraction=placement_fraction
    )
    base_origin = placement - lateral * ((arm_segments - 1) * object_radius * 0.9)
    base, corner = _wall_segment(
        base_origin, lateral, forward, arm_segments * 2, object_radius, primitive, wall_height=wall_height
    )
    if not base:
        return []
    arm, _ = _wall_segment(
        corner, forward, lateral, arm_segments, object_radius, primitive, wall_height=wall_height
    )
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
    primitive: ObstaclePrimitive = "box",
    wall_height: float = DEFAULT_WALL_HEIGHT,
    placement_fraction: float = 0.5,
    name_prefix: str = "obstacle_scene_u_pocket",
) -> list[AvoidRegion]:
    """Pocket (U-shape) with its mouth facing the start.

    A greedy straight-line policy drives into the mouth and dead-ends at the
    base wall; escaping requires backing out and going around one of the
    side walls -- the classic local-minimum trap. ``primitive="box"``
    (default) builds all three walls as solid oriented slabs.
    """
    placement, forward, lateral, _length = _direct_frame(
        start, goal, placement_fraction=placement_fraction
    )
    base_origin = placement - lateral * ((base_segments - 1) * object_radius * 0.9)
    base, right_corner = _wall_segment(
        base_origin,
        lateral,
        forward,
        base_segments * 2,
        object_radius,
        primitive,
        wall_height=wall_height,
    )
    if not base:
        return []
    left_corner = np.asarray(base_origin, dtype=np.float32)
    left_arm, _ = _wall_segment(
        left_corner, -forward, lateral, side_segments, object_radius, primitive, wall_height=wall_height
    )
    right_arm, _ = _wall_segment(
        right_corner, -forward, lateral, side_segments, object_radius, primitive, wall_height=wall_height
    )
    return _as_avoid_regions(base + left_arm + right_arm, weight=weight, margin=margin, name_prefix=name_prefix)


def shortcut_trap_scene(
    *,
    start: Array,
    goal: Array,
    object_radius: float = 0.05,
    arm_segments: int = 3,
    weight: float = 1.0,
    margin: float = 0.0,
    primitive: ObstaclePrimitive = "box",
    wall_height: float = DEFAULT_WALL_HEIGHT,
    placement_fraction: float = 0.25,
    name_prefix: str = "obstacle_scene_shortcut_trap",
) -> list[AvoidRegion]:
    """L-shaped wall placed close to start, baiting a greedy policy early.

    Same base-wall-plus-closing-arm shape as ``l_corner_scene``, but placed
    near the start (``placement_fraction=0.25`` by default, vs. 0.5 for
    ``l_corner``) instead of the path midpoint. A greedy/local-cost policy
    commits to heading straight at it almost immediately -- most of the
    remaining path is still ahead when it discovers the dead end -- so the
    "locally cheapest direction is wrong" property bites earlier and harder
    than a midpoint-placed obstacle. Defaults to solid box walls (see
    ``wall_gap_scene`` for why) sized well beyond typical candidate-path
    lateral spread (``arm_segments``, ``wall_height``) so skirting around
    the end or hopping over the top both cost a real detour, not a cheap
    nudge.
    """
    return l_corner_scene(
        start=start,
        goal=goal,
        object_radius=object_radius,
        arm_segments=arm_segments,
        weight=weight,
        margin=margin,
        primitive=primitive,
        wall_height=wall_height,
        placement_fraction=placement_fraction,
        name_prefix=name_prefix,
    )


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
    primitive: ObstaclePrimitive = "box",
    wall_height: float = DEFAULT_WALL_HEIGHT,
    placement_fraction: float | None = None,
) -> list[AvoidRegion]:
    """Dispatch to the named scene builder using shared sizing parameters.

    ``primitive``/``wall_height`` only affect the wall-shaped scenes
    (``wall_gap``/``l_corner``/``u_pocket``/``shortcut_trap``); ignored by
    ``pillar_cluster``/``cluttered_field``, which are cylinder/sphere-only by
    design. ``placement_fraction`` likewise only applies to the wall-shaped
    scenes; ``None`` leaves each scene's own default (0.5 midpoint, except
    ``shortcut_trap``'s 0.25-near-start default).
    """
    if name == "none":
        return []
    if name == "wall_gap":
        kwargs: dict[str, Any] = dict(
            start=start,
            goal=goal,
            object_radius=object_radius,
            arm_segments=arm_segments,
            weight=weight,
            margin=margin,
            primitive=primitive,
            wall_height=wall_height,
        )
        if placement_fraction is not None:
            kwargs["placement_fraction"] = placement_fraction
        return wall_gap_scene(**kwargs)
    if name == "l_corner":
        kwargs = dict(
            start=start,
            goal=goal,
            object_radius=object_radius,
            arm_segments=arm_segments,
            weight=weight,
            margin=margin,
            primitive=primitive,
            wall_height=wall_height,
        )
        if placement_fraction is not None:
            kwargs["placement_fraction"] = placement_fraction
        return l_corner_scene(**kwargs)
    if name == "u_pocket":
        kwargs = dict(
            start=start,
            goal=goal,
            object_radius=object_radius,
            base_segments=arm_segments,
            side_segments=max(1, arm_segments - 1),
            weight=weight,
            margin=margin,
            primitive=primitive,
            wall_height=wall_height,
        )
        if placement_fraction is not None:
            kwargs["placement_fraction"] = placement_fraction
        return u_pocket_scene(**kwargs)
    if name == "shortcut_trap":
        kwargs = dict(
            start=start,
            goal=goal,
            object_radius=object_radius,
            arm_segments=arm_segments,
            weight=weight,
            margin=margin,
            primitive=primitive,
            wall_height=wall_height,
        )
        if placement_fraction is not None:
            kwargs["placement_fraction"] = placement_fraction
        return shortcut_trap_scene(**kwargs)
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
