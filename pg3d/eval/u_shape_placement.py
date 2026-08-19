from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from pg3d.constraints import AvoidRegion, BoxRegion
from pg3d.envs.obstacles import transform_box_component, u_shape_components


@dataclass(frozen=True)
class UShapeSearchConfig:
    """Deterministic, normalized search space for one U-shaped reach scene."""

    mouth_widths: tuple[float, ...] = (0.18, 0.20, 0.22, 0.24)
    fallback_mouth_widths: tuple[float, ...] = (0.26, 0.28)
    mouth_chunk_fractions: tuple[float, ...] = (0.5, 0.75, 1.0)
    back_chunk_ratios: tuple[float, ...] = (2.0, 2.5, 3.0)
    fallback_back_chunk_ratios: tuple[float, ...] = (
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
    )
    cavity_depths: tuple[float, ...] = (0.08, 0.10, 0.12, 0.14, 0.16)
    lateral_offsets: tuple[float, ...] = (-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06)
    full_height: float = 0.75
    target_back_chunks: float = 2.5
    min_clearance: float = 0.03
    min_mouth_clearance: float = 0.05
    min_goal_beyond_back: float = 0.05
    min_back_penetration: float = 0.005
    workspace_bounds: tuple[tuple[float, float], tuple[float, float]] = (
        (-0.90, 0.70),
        (-0.60, 0.60),
    )

    def __post_init__(self) -> None:
        positive_sequences = (
            self.mouth_widths,
            self.fallback_mouth_widths,
            self.mouth_chunk_fractions,
            self.back_chunk_ratios,
            self.fallback_back_chunk_ratios,
            self.cavity_depths,
        )
        if any(
            not values or any(value <= 0.0 for value in values) for values in positive_sequences
        ):
            raise ValueError("U-shape search sequences must contain positive values")
        if any(not np.isfinite(value) for values in positive_sequences for value in values):
            raise ValueError("U-shape search sequences must be finite")
        scalar_values = (
            self.full_height,
            self.target_back_chunks,
            self.min_clearance,
            self.min_mouth_clearance,
            self.min_goal_beyond_back,
            self.min_back_penetration,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in scalar_values):
            raise ValueError("U-shape search thresholds must be positive and finite")
        if len(self.workspace_bounds) != 2 or any(
            len(bounds) != 2 or bounds[0] >= bounds[1] for bounds in self.workspace_bounds
        ):
            raise ValueError("workspace_bounds must contain increasing XY bounds")


@dataclass(frozen=True)
class UShapeCandidate:
    """One root pose and envelope proposed by the normalized search."""

    root_center: np.ndarray
    yaw: float
    half_extents: np.ndarray
    mouth_width: float
    mouth_distance: float
    back_distance: float
    back_chunk_ratio: float
    mouth_chunk_fraction: float
    lateral_offset: float
    chunk_distance: float
    fallback: bool = False

    def __post_init__(self) -> None:
        center = np.asarray(self.root_center, dtype=np.float32).reshape(3)
        half_extents = np.asarray(self.half_extents, dtype=np.float32).reshape(3)
        if not np.isfinite(center).all() or not np.isfinite(half_extents).all():
            raise ValueError("candidate geometry must be finite")
        if np.any(half_extents <= 0.0):
            raise ValueError("candidate half-extents must be positive")
        object.__setattr__(self, "root_center", center.copy())
        object.__setattr__(self, "half_extents", half_extents.copy())

    @property
    def full_size(self) -> np.ndarray:
        return 2.0 * self.half_extents

    @property
    def footprint_area(self) -> float:
        return float(4.0 * self.half_extents[0] * self.half_extents[1])

    @property
    def back_target_error(self) -> float:
        return abs(float(self.back_chunk_ratio) - 2.5)


@dataclass(frozen=True)
class UShapeGeometryCheck:
    accepted: bool
    reasons: tuple[str, ...]
    initial_clearance: float
    goal_clearance: float
    mouth_clearance: float
    goal_beyond_back: float
    direct_back_clearance: float


@dataclass(frozen=True)
class ValidatedUShapeCandidate:
    candidate: UShapeCandidate
    geometry: UShapeGeometryCheck
    witness_clearance: float
    witness_path_length: float
    witness_steps: int
    witness_side: str
    obstacle_points: int


def path_aligned_yaw(start: np.ndarray, goal: np.ndarray) -> float:
    """Orient local +Y from start toward goal, leaving the opening at local -Y."""
    start_xy = np.asarray(start, dtype=np.float64).reshape(-1)[:2]
    goal_xy = np.asarray(goal, dtype=np.float64).reshape(-1)[:2]
    delta = goal_xy - start_xy
    if not np.isfinite(delta).all() or float(np.linalg.norm(delta)) <= 1e-8:
        raise ValueError("start and goal must have distinct finite XY positions")
    return float(math.atan2(float(delta[1]), float(delta[0])) - math.pi / 2.0)


def opening_toward_point_yaw(center: np.ndarray, point: np.ndarray) -> float:
    """Return a yaw whose local ``-Y`` opening points at ``point``.

    Unlike :func:`path_aligned_yaw`, this rule uses the obstacle center itself.
    It is therefore appropriate when an existing box pose is the placement
    anchor and only the replacement U orientation is free.
    """
    center_xy = np.asarray(center, dtype=np.float64).reshape(-1)[:2]
    point_xy = np.asarray(point, dtype=np.float64).reshape(-1)[:2]
    opening = point_xy - center_xy
    if not np.isfinite(opening).all() or float(np.linalg.norm(opening)) <= 1e-8:
        raise ValueError("center and opening target must have distinct finite XY positions")
    yaw = math.atan2(float(opening[1]), float(opening[0])) + math.pi / 2.0
    return float(math.atan2(math.sin(yaw), math.cos(yaw)))


def box_derived_u_shape_candidate(
    center: np.ndarray,
    half_extents: np.ndarray,
    *,
    start: np.ndarray,
    first_chunk_distance: float,
    yaw_delta: float = 0.0,
) -> UShapeCandidate:
    """Replace a box envelope with a same-envelope U opening toward the start."""
    root = np.asarray(center, dtype=np.float32).reshape(3)
    envelope = np.asarray(half_extents, dtype=np.float32).reshape(3)
    start_xyz = np.asarray(start, dtype=np.float64).reshape(3)
    if not np.isfinite(root).all() or not np.isfinite(envelope).all() or np.any(envelope <= 0.0):
        raise ValueError("box center and half-extents must be positive finite geometry")
    chunk = float(first_chunk_distance)
    if not np.isfinite(chunk) or chunk <= 0.0:
        raise ValueError("first_chunk_distance must be positive and finite")
    if not np.isfinite(yaw_delta):
        raise ValueError("yaw_delta must be finite")
    raw_yaw = opening_toward_point_yaw(root, start_xyz) + float(yaw_delta)
    yaw = float(math.atan2(math.sin(raw_yaw), math.cos(raw_yaw)))
    forward = np.asarray([-math.sin(yaw), math.cos(yaw)], dtype=np.float64)
    right = np.asarray([forward[1], -forward[0]], dtype=np.float64)
    root_from_start = root[:2].astype(np.float64) - start_xyz[:2]
    center_forward = float(np.dot(root_from_start, forward))
    mouth_distance = center_forward - float(envelope[1])
    back_distance = center_forward + float(envelope[1])
    lateral_offset = float(np.dot(root_from_start, right))
    mouth_width = float((10.0 / 7.0) * envelope[0])
    return UShapeCandidate(
        root_center=root,
        yaw=yaw,
        half_extents=envelope,
        mouth_width=mouth_width,
        mouth_distance=mouth_distance,
        back_distance=back_distance,
        back_chunk_ratio=back_distance / chunk,
        mouth_chunk_fraction=mouth_distance / chunk,
        lateral_offset=lateral_offset,
        chunk_distance=chunk,
        fallback=False,
    )


def half_extents_from_mouth_and_depth(
    mouth_width: float,
    usable_cavity_depth: float,
    *,
    full_height: float,
) -> np.ndarray:
    """Invert the existing scaled-U proportions into envelope half-extents.

    The current family has an inner mouth width ``10/7 * half_x`` and a usable
    mouth-to-back-front depth ``26/15 * half_y``.
    """
    values = (mouth_width, usable_cavity_depth, full_height)
    if any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("mouth width, cavity depth, and height must be positive and finite")
    return np.asarray(
        [0.7 * mouth_width, (15.0 / 26.0) * usable_cavity_depth, 0.5 * full_height],
        dtype=np.float32,
    )


def enumerate_u_shape_candidates(
    start: np.ndarray,
    goal: np.ndarray,
    first_chunk_distance: float,
    *,
    config: UShapeSearchConfig,
    fallback: bool = False,
) -> list[UShapeCandidate]:
    """Enumerate candidates in a stable order independent of method outcomes."""
    start_xyz = np.asarray(start, dtype=np.float64).reshape(3)
    goal_xyz = np.asarray(goal, dtype=np.float64).reshape(3)
    yaw = path_aligned_yaw(start_xyz, goal_xyz)
    forward = goal_xyz[:2] - start_xyz[:2]
    forward /= np.linalg.norm(forward)
    right = np.asarray([forward[1], -forward[0]], dtype=np.float64)
    chunk = float(first_chunk_distance)
    if not np.isfinite(chunk) or chunk <= 0.0:
        raise ValueError("first_chunk_distance must be positive and finite")
    widths = config.fallback_mouth_widths if fallback else config.mouth_widths
    back_ratios = config.fallback_back_chunk_ratios if fallback else config.back_chunk_ratios
    candidates: list[UShapeCandidate] = []
    for mouth_width in widths:
        for mouth_fraction in config.mouth_chunk_fractions:
            mouth_distance = float(mouth_fraction * chunk)
            for back_ratio in back_ratios:
                back_distance = float(back_ratio * chunk)
                usable_depth = back_distance - mouth_distance
                if usable_depth <= 0.0:
                    continue
                half_extents = half_extents_from_mouth_and_depth(
                    mouth_width,
                    usable_depth,
                    full_height=config.full_height,
                )
                for lateral_offset in config.lateral_offsets:
                    center_xy = (
                        start_xyz[:2]
                        + forward * (mouth_distance + float(half_extents[1]))
                        + right * float(lateral_offset)
                    )
                    candidates.append(
                        UShapeCandidate(
                            root_center=np.asarray(
                                [center_xy[0], center_xy[1], half_extents[2]],
                                dtype=np.float32,
                            ),
                            yaw=yaw,
                            half_extents=half_extents,
                            mouth_width=float(mouth_width),
                            mouth_distance=mouth_distance,
                            back_distance=back_distance,
                            back_chunk_ratio=float(back_ratio),
                            mouth_chunk_fraction=float(mouth_fraction),
                            lateral_offset=float(lateral_offset),
                            chunk_distance=chunk,
                            fallback=fallback,
                        )
                    )
    return candidates


def enumerate_u_shape_trajectory_candidates(
    start: np.ndarray,
    goal: np.ndarray,
    nominal_tcp_path: np.ndarray,
    *,
    config: UShapeSearchConfig,
    fallback: bool = False,
    action_steps_per_chunk: int = 8,
) -> list[UShapeCandidate]:
    """Place the closed back at actual nominal chunk-boundary progress.

    The primary phase evaluates 2.0, 2.5, and 3.0 chunks.  If those positions
    are too close to the articulated arm, the fallback phase advances one
    whole chunk at a time while preserving every clearance gate.
    """
    if action_steps_per_chunk <= 0:
        raise ValueError("action_steps_per_chunk must be positive")
    path = np.asarray(nominal_tcp_path, dtype=np.float64).reshape(-1, 3)
    if len(path) <= action_steps_per_chunk or not np.isfinite(path).all():
        raise ValueError("nominal_tcp_path must contain a finite complete first chunk")
    start_xyz = np.asarray(start, dtype=np.float64).reshape(3)
    goal_xyz = np.asarray(goal, dtype=np.float64).reshape(3)
    yaw = path_aligned_yaw(start_xyz, goal_xyz)
    forward = goal_xyz[:2] - start_xyz[:2]
    forward /= np.linalg.norm(forward)
    right = np.asarray([forward[1], -forward[0]], dtype=np.float64)
    first_chunk_distance = float(
        np.sum(np.linalg.norm(np.diff(path[: action_steps_per_chunk + 1], axis=0), axis=1))
    )
    if first_chunk_distance <= 0.0:
        raise ValueError("nominal first chunk has zero TCP displacement")
    if fallback:
        widths = config.mouth_widths + config.fallback_mouth_widths
        back_ratios = config.fallback_back_chunk_ratios
    else:
        widths = config.mouth_widths
        back_ratios = config.back_chunk_ratios
    candidates: list[UShapeCandidate] = []
    for mouth_width in widths:
        for back_ratio in back_ratios:
            back_step = int(round(back_ratio * action_steps_per_chunk))
            if back_step >= len(path):
                continue
            back_distance = float(np.dot(path[back_step, :2] - start_xyz[:2], forward))
            for usable_depth in config.cavity_depths:
                mouth_distance = back_distance - float(usable_depth)
                if mouth_distance <= 0.0:
                    continue
                half_extents = half_extents_from_mouth_and_depth(
                    float(mouth_width),
                    float(usable_depth),
                    full_height=config.full_height,
                )
                for lateral_offset in config.lateral_offsets:
                    center_xy = (
                        start_xyz[:2]
                        + forward * (mouth_distance + float(half_extents[1]))
                        + right * float(lateral_offset)
                    )
                    candidates.append(
                        UShapeCandidate(
                            root_center=np.asarray(
                                [center_xy[0], center_xy[1], half_extents[2]],
                                dtype=np.float32,
                            ),
                            yaw=yaw,
                            half_extents=half_extents,
                            mouth_width=float(mouth_width),
                            mouth_distance=mouth_distance,
                            back_distance=back_distance,
                            back_chunk_ratio=float(back_ratio),
                            mouth_chunk_fraction=mouth_distance / first_chunk_distance,
                            lateral_offset=float(lateral_offset),
                            chunk_distance=first_chunk_distance,
                            fallback=fallback,
                        )
                    )
    return candidates


def u_shape_constraints(
    candidate: UShapeCandidate,
    *,
    target: str,
    name_prefix: str = "u_shape_p1",
    clearance_scale: float = 0.05,
) -> list[AvoidRegion]:
    if target not in {"eef", "robot"}:
        raise ValueError("target must be 'eef' or 'robot'")
    constraints = []
    for component in u_shape_components(candidate.half_extents):
        center, yaw = transform_box_component(
            component,
            center=candidate.root_center,
            yaw=candidate.yaw,
        )
        constraints.append(
            AvoidRegion(
                BoxRegion(center=center, half_extents=component.half_extents, yaw=yaw),
                target=target,
                name=f"{name_prefix}/u_shape_{component.name}",
                clearance_scale=clearance_scale,
            )
        )
    return constraints


def evaluate_u_shape_geometry(
    candidate: UShapeCandidate,
    *,
    start_robot_points: np.ndarray,
    goal_robot_points: np.ndarray,
    nominal_tcp_path: np.ndarray,
    goal: np.ndarray,
    config: UShapeSearchConfig,
) -> UShapeGeometryCheck:
    constraints = u_shape_constraints(candidate, target="robot")
    initial_clearance = _min_clearance(start_robot_points, constraints)
    goal_clearance = _min_clearance(goal_robot_points, constraints)
    back = next(constraint for constraint in constraints if constraint.name.endswith("_back"))
    nominal = np.asarray(nominal_tcp_path, dtype=np.float32).reshape(-1, 3)
    if not len(nominal):
        raise ValueError("nominal_tcp_path must contain at least one point")
    # The hard gate is intentionally the direct start-to-goal consequence, not
    # whether an already-curved nominal rollout happens to touch the back.  A
    # dense deterministic segment also makes the result independent of rollout
    # logging frequency.
    direct_path = np.linspace(nominal[0], np.asarray(goal, dtype=np.float32), 2049)
    direct_back_clearance = float(np.min(back.region.signed_distance(direct_path)))
    mouth_clearance = 0.5 * candidate.mouth_width - abs(candidate.lateral_offset)
    forward = np.asarray(
        [-math.sin(candidate.yaw), math.cos(candidate.yaw)],
        dtype=np.float64,
    )
    goal_forward = float(
        np.dot(
            np.asarray(goal, dtype=np.float64).reshape(3)[:2] - candidate.root_center[:2],
            forward,
        )
    )
    goal_beyond_back = goal_forward - float(candidate.half_extents[1])
    reasons: list[str] = []
    if initial_clearance < config.min_clearance:
        reasons.append("initial_clearance")
    if goal_clearance < config.min_clearance:
        reasons.append("goal_clearance")
    if mouth_clearance < config.min_mouth_clearance:
        reasons.append("mouth_clearance")
    if goal_beyond_back < config.min_goal_beyond_back:
        reasons.append("goal_beyond_back")
    if direct_back_clearance > -config.min_back_penetration:
        reasons.append("direct_path_misses_back")
    if not _footprint_in_workspace(candidate, config.workspace_bounds):
        reasons.append("workspace")
    return UShapeGeometryCheck(
        accepted=not reasons,
        reasons=tuple(reasons),
        initial_clearance=initial_clearance,
        goal_clearance=goal_clearance,
        mouth_clearance=mouth_clearance,
        goal_beyond_back=goal_beyond_back,
        direct_back_clearance=direct_back_clearance,
    )


def validated_candidate_sort_key(item: ValidatedUShapeCandidate) -> tuple[float | str, ...]:
    candidate = item.candidate
    worst_clearance = min(
        item.geometry.initial_clearance,
        item.geometry.goal_clearance,
        item.witness_clearance,
    )
    return (
        abs(candidate.back_chunk_ratio - 2.5),
        -worst_clearance,
        item.witness_path_length,
        candidate.footprint_area,
        abs(candidate.lateral_offset),
        candidate.mouth_width,
        candidate.mouth_chunk_fraction,
        candidate.lateral_offset,
        item.witness_side,
    )


def component_geometry(candidate: UShapeCandidate) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for component in u_shape_components(candidate.half_extents):
        center, yaw = transform_box_component(
            component,
            center=candidate.root_center,
            yaw=candidate.yaw,
        )
        records.append(
            {
                "name": component.name,
                "center": center.astype(float).tolist(),
                "half_extents": [float(value) for value in component.half_extents],
                "yaw": float(yaw),
            }
        )
    return records


def _min_clearance(points: np.ndarray, constraints: Iterable[AvoidRegion]) -> float:
    cloud = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if not len(cloud):
        raise ValueError("whole-robot clearance requires a non-empty point cloud")
    values = [
        float(np.min(constraint.region.signed_distance(cloud) - float(constraint.margin)))
        for constraint in constraints
    ]
    return min(values)


def _footprint_in_workspace(
    candidate: UShapeCandidate,
    workspace_bounds: tuple[tuple[float, float], tuple[float, float]],
) -> bool:
    half_x, half_y = (float(value) for value in candidate.half_extents[:2])
    corners = np.asarray(
        [[x, y] for x in (-half_x, half_x) for y in (-half_y, half_y)],
        dtype=np.float64,
    )
    cos_yaw = math.cos(candidate.yaw)
    sin_yaw = math.sin(candidate.yaw)
    rotation = np.asarray([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float64)
    world = corners @ rotation.T + candidate.root_center[:2]
    return bool(
        np.all((world[:, 0] >= workspace_bounds[0][0]) & (world[:, 0] <= workspace_bounds[0][1]))
        and np.all(
            (world[:, 1] >= workspace_bounds[1][0]) & (world[:, 1] <= workspace_bounds[1][1])
        )
    )
