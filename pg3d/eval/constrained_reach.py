from __future__ import annotations

import hashlib
import json
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from pg3d.constraints import (
    AvoidProjection,
    AvoidRegion,
    BoxRegion,
    SceneContext,
    SphereRegion,
    constraints_from_json,
    constraints_to_json,
)
from pg3d.constraints.core import mean_squared_norm
from pg3d.utils.serialization import jsonable
from pg3d.world_model import ActionChunk, ImaginedRollout

EvalMethod = Literal["base", "rejection", "reranking"]
ArtifactSelection = Literal["periodic", "random", "all"]
SUCCESS_RATE_METRICS: tuple[tuple[str, str], ...] = (
    ("reach_success", "Reach"),
    ("constraint_satisfied", "Constraint"),
    ("combined_success", "Combined"),
    ("stable_combined_success", "Stable combined"),
)


@dataclass(frozen=True)
class AvoidOverlayConfig:
    """Configuration for the first constrained-reach avoid-region overlay."""

    radius: float = 0.08
    min_radius: float = 0.025
    margin: float = 0.0
    weight: float = 1.0
    tolerance: float = 1e-6
    name: str = "direct_path_avoid_region"
    path_fraction: float = 0.5
    shape: Literal["sphere", "box", "cuboid"] = "sphere"
    box_half_extents: tuple[float, float, float] | None = None
    target: Literal["eef", "robot"] = "eef"


@dataclass(frozen=True)
class NominalPathAvoidConfig:
    """Configuration for avoid regions placed on a nominal executed TCP path."""

    radius: float = 0.03
    path_fraction: float = 0.5
    margin: float = 0.0
    weight: float = 1.0
    tolerance: float = 1e-6
    name: str = "nominal_path_avoid_region"


@dataclass
class EpisodePath:
    """Executed simulator path used for constrained-reach metrics."""

    tcp_positions: list[np.ndarray] = field(default_factory=list)
    q: list[np.ndarray] = field(default_factory=list)
    target_distances: list[float] = field(default_factory=list)

    def append(self, *, tcp_position: Any, q: Any, target_distance: float) -> None:
        tcp = np.asarray(tcp_position, dtype=np.float32).reshape(3)
        qpos = np.asarray(q, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(tcp)):
            raise ValueError("tcp_position must be finite")
        if not np.all(np.isfinite(qpos)):
            raise ValueError("q must be finite")
        self.tcp_positions.append(tcp)
        self.q.append(qpos)
        self.target_distances.append(float(target_distance))

    @property
    def tcp_array(self) -> np.ndarray:
        if not self.tcp_positions:
            return np.zeros((0, 3), dtype=np.float32)
        return np.stack(self.tcp_positions, axis=0).astype(np.float32, copy=False)

    @property
    def q_array(self) -> np.ndarray:
        if not self.q:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack(self.q, axis=0).astype(np.float32, copy=False)


@dataclass
class TimingEvent:
    """One profiled timing event."""

    name: str
    seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seconds": float(self.seconds),
            "metadata": jsonable(self.metadata),
        }


class TimingRecorder:
    """Lightweight wall-clock timing recorder for eval profiling."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        sync_fn: Any | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.sync_fn = sync_fn
        self.events: list[TimingEvent] = []

    @contextmanager
    def time(self, name: str, **metadata: Any) -> Any:
        """Record elapsed wall-clock time for a named block when profiling is enabled."""
        if not self.enabled:
            yield
            return
        self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.events.append(
                TimingEvent(
                    name=name,
                    seconds=time.perf_counter() - start,
                    metadata=metadata,
                )
            )

    def summary(self) -> dict[str, dict[str, float]]:
        """Aggregate timing events by name."""
        by_name: dict[str, list[float]] = {}
        for event in self.events:
            by_name.setdefault(event.name, []).append(float(event.seconds))
        return {
            name: {
                "count": float(len(values)),
                "total": float(np.sum(values)),
                "mean": float(np.mean(values)),
                "max": float(np.max(values)),
            }
            for name, values in sorted(by_name.items())
        }

    def drain_events(self) -> list[TimingEvent]:
        """Return and clear pending timing events for incremental JSONL writing."""
        events = list(self.events)
        self.events.clear()
        return events

    def _sync(self) -> None:
        if self.sync_fn is not None:
            self.sync_fn()


def validate_planning_horizons(
    *,
    planning_horizon_chunks: int,
    execution_horizon_chunks: int,
) -> None:
    """Validate receding-horizon planning/execution chunk counts."""
    if planning_horizon_chunks <= 0:
        raise ValueError("planning_horizon_chunks must be positive")
    if execution_horizon_chunks <= 0:
        raise ValueError("execution_horizon_chunks must be positive")
    if execution_horizon_chunks > planning_horizon_chunks:
        raise ValueError("execution_horizon_chunks must be <= planning_horizon_chunks")


def direct_path_avoid_region(
    *,
    start_tcp: Any,
    target_position: Any,
    config: AvoidOverlayConfig | None = None,
) -> AvoidRegion:
    """Create an episode-specific sphere on the direct TCP-goal path at cfg.path_fraction."""
    cfg = config or AvoidOverlayConfig()
    start = _vector3(start_tcp, "start_tcp")
    goal = _vector3(target_position, "target_position")
    distance = float(np.linalg.norm(goal - start))
    if distance <= 1e-6:
        raise ValueError("start_tcp and target_position must be distinct")
    effective_radius = min(float(cfg.radius), max(float(cfg.min_radius), 0.45 * distance))
    fraction = float(cfg.path_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"path_fraction must be in [0, 1], got {fraction}")
    center = (start + fraction * (goal - start)).astype(np.float32)
    if cfg.shape in ("box", "cuboid"):
        if cfg.box_half_extents is not None:
            half_extents = np.asarray(cfg.box_half_extents, dtype=np.float32)
        else:
            half_extents = np.full(3, float(effective_radius), dtype=np.float32)
        region: SphereRegion | BoxRegion = BoxRegion(center=center, half_extents=half_extents)
    else:
        region = SphereRegion(center=center, radius=effective_radius)
    return AvoidRegion(
        region=region,
        target=cfg.target,
        margin=cfg.margin,
        weight=cfg.weight,
        tolerance=cfg.tolerance,
        name=cfg.name,
    )


def nominal_path_avoid_region(
    tcp_positions: Any,
    *,
    config: NominalPathAvoidConfig | None = None,
) -> AvoidRegion:
    """Create a sphere centered at a fixed arc-length fraction of an executed TCP path."""
    cfg = config or NominalPathAvoidConfig()
    if float(cfg.radius) <= 0.0:
        raise ValueError("nominal path avoid radius must be positive")
    if not 0.0 <= float(cfg.path_fraction) <= 1.0:
        raise ValueError("nominal path fraction must be in [0, 1]")
    if float(cfg.margin) < 0.0:
        raise ValueError("nominal path avoid margin must be non-negative")
    if float(cfg.tolerance) < 0.0:
        raise ValueError("nominal path avoid tolerance must be non-negative")
    center = _point_at_arc_fraction(tcp_positions, fraction=float(cfg.path_fraction))
    return AvoidRegion(
        region=SphereRegion(center=center, radius=float(cfg.radius)),
        margin=float(cfg.margin),
        weight=float(cfg.weight),
        tolerance=float(cfg.tolerance),
        name=cfg.name,
    )


def save_episode_constraints(path: Path, constraints: list[AvoidRegion]) -> None:
    """Persist one episode's constraint instances for repeatable evaluation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(constraints_to_json(constraints)), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def constraint_fingerprint(constraints: list[AvoidRegion]) -> str:
    """Return a stable SHA-256 identifier for serialized constraint geometry."""
    payload = json.dumps(
        jsonable(constraints_to_json(constraints)),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_episode_constraints(path: Path) -> list[AvoidRegion]:
    """Load one episode's avoid-region constraints from JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"constraints file must contain a list: {path}")
    constraints = constraints_from_json(payload)
    avoid_regions: list[AvoidRegion] = []
    for idx, constraint in enumerate(constraints):
        if not isinstance(constraint, (AvoidRegion, AvoidProjection)):
            raise ValueError(
                f"only AvoidRegion/AvoidProjection constraints are supported for "
                f"constrained reach; {path} item {idx} is {type(constraint).__name__}"
            )
        avoid_regions.append(constraint)
    return avoid_regions


def scene_context_for_constraints(
    *,
    target_position: Any,
    constraints: list[AvoidRegion],
    metadata: dict[str, Any] | None = None,
) -> SceneContext:
    """Build the scene context passed to composition controllers."""
    regions = {
        constraint.name: constraint.region
        for constraint in constraints
        if isinstance(constraint, (AvoidRegion, AvoidProjection))
    }
    return SceneContext(
        target_position=np.asarray(target_position, dtype=np.float32),
        regions=regions,
        metadata=metadata or {},
    )


def min_constraint_clearance(
    tcp_positions: Any,
    constraints: list[AvoidRegion],
) -> float | None:
    """Return minimum signed clearance to all avoid regions, after margins."""
    path = np.asarray(tcp_positions, dtype=np.float32)
    if path.size == 0 or not constraints:
        return None
    if path.ndim != 2 or path.shape[1] != 3:
        raise ValueError(f"tcp_positions must have shape [T, 3], got {path.shape}")
    clearances: list[float] = []
    for constraint in constraints:
        signed = constraint.region.signed_distance(path)
        clearances.append(float(np.min(signed - float(constraint.margin))))
    return min(clearances) if clearances else None


def constraint_clearance_series(
    tcp_positions: Any,
    constraints: list[AvoidRegion],
) -> np.ndarray:
    """Return minimum signed clearance at every path sample after constraint margins."""
    path = np.asarray(tcp_positions, dtype=np.float32)
    if path.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if path.ndim != 2 or path.shape[1] != 3:
        raise ValueError(f"tcp_positions must have shape [T, 3], got {path.shape}")
    if not np.all(np.isfinite(path)):
        raise ValueError("tcp_positions must be finite")
    if not constraints:
        return np.full((path.shape[0],), np.inf, dtype=np.float32)
    per_constraint = [
        np.asarray(constraint.region.signed_distance(path), dtype=np.float32)
        - np.float32(constraint.margin)
        for constraint in constraints
    ]
    return np.min(np.stack(per_constraint, axis=0), axis=0).astype(
        np.float32, copy=False
    )


def path_satisfies_constraints(
    tcp_positions: Any,
    constraints: list[AvoidRegion],
) -> bool:
    """Return whether an executed TCP path stays outside all avoid regions."""
    path = np.asarray(tcp_positions, dtype=np.float32)
    if path.size == 0 or not constraints:
        return True
    if path.ndim != 2 or path.shape[1] != 3:
        raise ValueError(f"tcp_positions must have shape [T, 3], got {path.shape}")
    return all(
        float(
            np.min(
                np.asarray(constraint.region.signed_distance(path), dtype=np.float32)
                - np.float32(constraint.margin)
            )
        )
        >= -float(constraint.tolerance)
        for constraint in constraints
    )


def constraint_violation_metrics(
    tcp_positions: Any,
    constraints: list[AvoidRegion],
    *,
    dt: float,
) -> dict[str, float | int]:
    """Summarize executed TCP constraint violations in physical units."""
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a positive finite value")
    clearances = constraint_clearance_series(tcp_positions, constraints)
    if clearances.size == 0 or not constraints:
        return {
            "max_violation_depth": 0.0,
            "violation_steps": 0,
            "violation_fraction": 0.0,
            "integrated_violation": 0.0,
            "violation_event_count": 0,
        }
    adjusted_clearances = np.min(
        np.stack(
            [
                np.asarray(constraint.region.signed_distance(tcp_positions), dtype=np.float32)
                - np.float32(constraint.margin)
                + np.float32(constraint.tolerance)
                for constraint in constraints
            ],
            axis=0,
        ),
        axis=0,
    )
    depths = np.maximum(0.0, -adjusted_clearances)
    violating = depths > 0.0
    starts = violating & ~np.concatenate(
        [np.zeros((1,), dtype=bool), violating[:-1]]
    )
    return {
        "max_violation_depth": float(np.max(depths)),
        "violation_steps": int(np.count_nonzero(violating)),
        "violation_fraction": float(np.mean(violating)),
        "integrated_violation": float(np.sum(depths) * float(dt)),
        "violation_event_count": int(np.count_nonzero(starts)),
    }


def trajectory_path_length(points: Any) -> float:
    """Return Euclidean path length for a [T, D] trajectory."""
    trajectory = np.asarray(points, dtype=np.float32)
    if trajectory.size == 0:
        return 0.0
    if trajectory.ndim != 2:
        raise ValueError(f"trajectory must have shape [T, D], got {trajectory.shape}")
    if not np.all(np.isfinite(trajectory)):
        raise ValueError("trajectory must be finite")
    if trajectory.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)))


def trajectory_derivative_mse(points: Any, *, order: int, dt: float) -> float:
    """Return mean squared derivative norm for a trajectory in physical time."""
    trajectory = np.asarray(points, dtype=np.float32)
    if trajectory.size == 0:
        return 0.0
    if trajectory.ndim != 2:
        raise ValueError(f"trajectory must have shape [T, D], got {trajectory.shape}")
    if not np.all(np.isfinite(trajectory)):
        raise ValueError("trajectory must be finite")
    if order <= 0:
        raise ValueError("order must be positive")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a positive finite value")
    if trajectory.shape[0] <= order:
        return 0.0
    derivative = np.diff(trajectory, n=order, axis=0) / np.float32(dt**order)
    return mean_squared_norm(derivative)


def max_joint_velocity(q: Any, *, dt: float) -> float:
    """Return maximum absolute finite-difference joint velocity."""
    trajectory = np.asarray(q, dtype=np.float32)
    if trajectory.size == 0:
        return 0.0
    if trajectory.ndim != 2:
        raise ValueError(f"q must have shape [T, D], got {trajectory.shape}")
    if not np.all(np.isfinite(trajectory)):
        raise ValueError("q must be finite")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a positive finite value")
    if trajectory.shape[0] < 2:
        return 0.0
    return float(np.max(np.abs(np.diff(trajectory, axis=0) / np.float32(dt))))


def stable_goal_reached(
    target_distances: Any,
    *,
    first_success_step: int | None,
    goal_threshold: float,
    hold_steps: int,
) -> bool:
    """Return whether the goal remains reached for the full post-success hold window."""
    if not np.isfinite(goal_threshold) or goal_threshold < 0.0:
        raise ValueError("goal_threshold must be a non-negative finite value")
    if hold_steps < 0:
        raise ValueError("hold_steps must be non-negative")
    if first_success_step is None:
        return False
    distances = np.asarray(target_distances, dtype=np.float32).reshape(-1)
    if first_success_step < 0 or first_success_step >= distances.size:
        raise ValueError("first_success_step must index target_distances")
    if hold_steps == 0:
        return bool(
            np.isfinite(distances[first_success_step])
            and distances[first_success_step] <= goal_threshold
        )
    start = first_success_step + 1
    hold = distances[start : start + hold_steps]
    return bool(
        hold.size == hold_steps
        and np.all(np.isfinite(hold))
        and np.all(hold <= goal_threshold)
    )


def q_trajectory_smoothness(q: Any, *, order: int = 2) -> float:
    """Return mean squared finite-difference norm for an executed q trajectory."""
    q_array = np.asarray(q, dtype=np.float32)
    if q_array.size == 0:
        return 0.0
    if q_array.ndim != 2:
        raise ValueError(f"q must have shape [T, D], got {q_array.shape}")
    if order not in {1, 2}:
        raise ValueError("order must be 1 or 2")
    if q_array.shape[0] <= order:
        return 0.0
    return mean_squared_norm(np.diff(q_array, n=order, axis=0))


def episode_metric_row(
    *,
    method: EvalMethod,
    episode: int,
    seed: int,
    path: EpisodePath,
    constraints: list[AvoidRegion],
    reach_success: bool,
    first_success_step: int | None,
    steps: int,
    replans: int,
    candidate_feasibility_fraction: float | None,
    fallback_count: int = 0,
    video: str | None = None,
    rerun: str | None = None,
    robot_clearance_points: Any | None = None,
    goal_threshold: float | None = None,
    hold_steps: int = 0,
    control_dt: float = 1.0,
    action_selection_times: list[float] | None = None,
) -> dict[str, Any]:
    """Build the stable episode-level constrained-reach metric row.

    When ``robot_clearance_points`` is provided (a [M, 3] cloud of the whole robot
    sampled across the executed trajectory), constraint satisfaction is evaluated
    against the entire robot rather than only the TCP. The TCP-only values are still
    reported under ``*_tcp`` keys for comparability.
    """
    distances = np.asarray(path.target_distances, dtype=np.float32)
    finite_distances = distances[np.isfinite(distances)]
    if not np.isfinite(control_dt) or control_dt <= 0.0:
        raise ValueError("control_dt must be a positive finite value")
    tcp_min_clearance = min_constraint_clearance(path.tcp_array, constraints)
    tcp_satisfied = path_satisfies_constraints(path.tcp_array, constraints)
    tcp_violation = constraint_violation_metrics(
        path.tcp_array,
        constraints,
        dt=control_dt,
    )
    if robot_clearance_points is not None:
        min_clearance = min_constraint_clearance(robot_clearance_points, constraints)
        constraint_satisfied = path_satisfies_constraints(robot_clearance_points, constraints)
        constraint_target = "robot"
    else:
        min_clearance = tcp_min_clearance
        constraint_satisfied = tcp_satisfied
        constraint_target = "eef"
    stable_reach = (
        stable_goal_reached(
            distances,
            first_success_step=first_success_step,
            goal_threshold=float(goal_threshold),
            hold_steps=hold_steps,
        )
        if goal_threshold is not None
        else bool(reach_success and hold_steps == 0)
    )
    max_violation_depth = (
        max(0.0, -float(min_clearance))
        if min_clearance is not None and np.isfinite(float(min_clearance))
        else 0.0
    )
    selection_times = np.asarray(action_selection_times or [], dtype=np.float64)
    if np.any(~np.isfinite(selection_times)) or np.any(selection_times < 0.0):
        raise ValueError("action_selection_times must be finite and non-negative")
    return {
        "method": method,
        "episode": int(episode),
        "seed": int(seed),
        "steps": int(steps),
        "replans": int(replans),
        "reach_success": bool(reach_success),
        "stable_goal_reached": stable_reach,
        "constraint_satisfied": bool(constraint_satisfied),
        "constraint_target": constraint_target,
        "constraint_satisfied_tcp": bool(tcp_satisfied),
        "combined_success": bool(reach_success and constraint_satisfied),
        "stable_combined_success": bool(stable_reach and constraint_satisfied),
        "first_success_step": first_success_step,
        "goal_threshold": float(goal_threshold) if goal_threshold is not None else None,
        "hold_steps": int(hold_steps),
        "control_dt": float(control_dt),
        "final_target_distance": (
            float(finite_distances[-1]) if finite_distances.size else None
        ),
        "min_target_distance": (
            float(np.min(finite_distances)) if finite_distances.size else None
        ),
        "min_clearance": min_clearance,
        "min_clearance_tcp": tcp_min_clearance,
        "max_violation_depth": max_violation_depth,
        "max_violation_depth_tcp": tcp_violation["max_violation_depth"],
        "violation_steps_tcp": tcp_violation["violation_steps"],
        "violation_fraction_tcp": tcp_violation["violation_fraction"],
        "integrated_violation_tcp": tcp_violation["integrated_violation"],
        "violation_event_count_tcp": tcp_violation["violation_event_count"],
        "tcp_path_length": trajectory_path_length(path.tcp_array),
        "joint_path_length": trajectory_path_length(path.q_array),
        "tcp_acceleration_mse": trajectory_derivative_mse(
            path.tcp_array, order=2, dt=control_dt
        ),
        "tcp_jerk_mse": trajectory_derivative_mse(
            path.tcp_array, order=3, dt=control_dt
        ),
        "joint_acceleration_mse": trajectory_derivative_mse(
            path.q_array, order=2, dt=control_dt
        ),
        "joint_jerk_mse": trajectory_derivative_mse(
            path.q_array, order=3, dt=control_dt
        ),
        "max_joint_velocity": max_joint_velocity(path.q_array, dt=control_dt),
        "action_selection_time_total": (
            float(np.sum(selection_times)) if selection_times.size else None
        ),
        "action_selection_time_median": (
            float(np.median(selection_times)) if selection_times.size else None
        ),
        "action_selection_time_p90": (
            float(np.percentile(selection_times, 90.0)) if selection_times.size else None
        ),
        "action_selection_time_p95": (
            float(np.percentile(selection_times, 95.0)) if selection_times.size else None
        ),
        "smoothness": q_trajectory_smoothness(path.q_array, order=2),
        "candidate_feasibility_fraction": candidate_feasibility_fraction,
        "fallback_count": int(fallback_count),
        "video": video,
        "rerun": rerun,
    }


def wilson_interval(successes: int, total: int, *, z: float = 1.96) -> tuple[float, float]:
    """Return a Wilson score interval for a binomial proportion."""
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("successes and total must satisfy 0 <= successes <= total")
    if total == 0:
        return (0.0, 0.0)
    n = float(total)
    p = float(successes) / n
    z2 = z * z
    center = (p + z2 / (2.0 * n)) / (1.0 + z2 / n)
    half_width = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / (1.0 + z2 / n)
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def summarize_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate episode rows by method with Wilson intervals for boolean rates."""
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)

    summary: dict[str, Any] = {}
    for method, method_rows in sorted(by_method.items()):
        total = len(method_rows)
        method_summary: dict[str, Any] = {"episodes": total}
        for key in [
            "reach_success",
            "constraint_satisfied",
            "constraint_satisfied_tcp",
            "combined_success",
            "stable_goal_reached",
            "stable_combined_success",
        ]:
            if not any(key in row for row in method_rows):
                continue
            successes = sum(1 for row in method_rows if bool(row.get(key)))
            low, high = wilson_interval(successes, total)
            method_summary[f"{key}_rate"] = successes / total if total else 0.0
            method_summary[f"{key}_wilson_low"] = low
            method_summary[f"{key}_wilson_high"] = high
        for key in [
            "final_target_distance",
            "min_target_distance",
            "min_clearance",
            "min_clearance_tcp",
            "max_violation_depth",
            "max_violation_depth_tcp",
            "violation_steps_tcp",
            "violation_fraction_tcp",
            "integrated_violation_tcp",
            "violation_event_count_tcp",
            "tcp_path_length",
            "joint_path_length",
            "tcp_acceleration_mse",
            "tcp_jerk_mse",
            "joint_acceleration_mse",
            "joint_jerk_mse",
            "max_joint_velocity",
            "action_selection_time_total",
            "action_selection_time_median",
            "action_selection_time_p90",
            "action_selection_time_p95",
            "smoothness",
            "candidate_feasibility_fraction",
        ]:
            method_summary.update(_mean_std(key, method_rows))
        summary[method] = method_summary
    return summary


def validate_paired_episode_rows(
    rows: list[dict[str, Any]],
    *,
    methods: list[str],
) -> None:
    """Validate complete method pairing and shared episode protocol identifiers."""
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("methods must be non-empty and unique")
    expected = set(methods)
    by_episode: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_episode.setdefault(int(row["episode"]), []).append(row)
    if not by_episode:
        raise ValueError("paired evaluation produced no episode rows")
    identity_keys = (
        "simulator_seed",
        "source",
        "dataset_episode_index",
        "policy_seed",
        "constraint_id",
        "checkpoint_id",
    )
    for episode, episode_rows in sorted(by_episode.items()):
        observed = [str(row["method"]) for row in episode_rows]
        if len(observed) != len(set(observed)):
            raise ValueError(f"episode {episode} has duplicate method rows: {observed}")
        if set(observed) != expected:
            raise ValueError(
                f"episode {episode} methods do not match protocol: "
                f"expected={sorted(expected)} observed={sorted(observed)}"
            )
        for key in identity_keys:
            values = {_identity_value(row.get(key)) for row in episode_rows}
            if len(values) != 1:
                raise ValueError(
                    f"episode {episode} has mismatched {key} across methods: {values}"
                )


def success_rate_ci_rows(
    summary: dict[str, Any],
    *,
    methods: list[str] | None = None,
    metrics: tuple[tuple[str, str], ...] = SUCCESS_RATE_METRICS,
) -> list[dict[str, Any]]:
    """Return bar-plot rows for success rates and Wilson intervals.

    Accepts either a full eval `summary.json` dict or the nested `by_method` value.
    """
    by_method = summary.get("by_method", summary)
    if not isinstance(by_method, dict):
        raise ValueError("summary must contain a by_method mapping")

    ordered_methods = methods or [
        method for method in ["base", "rejection", "reranking"] if method in by_method
    ]
    ordered_methods.extend(
        method for method in sorted(by_method) if method not in ordered_methods
    )

    rows: list[dict[str, Any]] = []
    for method in ordered_methods:
        method_summary = by_method.get(method)
        if method_summary is None:
            continue
        for key, label in metrics:
            rate_key = f"{key}_rate"
            low_key = f"{key}_wilson_low"
            high_key = f"{key}_wilson_high"
            if (
                rate_key not in method_summary
                or low_key not in method_summary
                or high_key not in method_summary
            ):
                raise ValueError(f"summary for method {method!r} is missing {key} CI fields")
            rate = float(method_summary[rate_key])
            low = float(method_summary[low_key])
            high = float(method_summary[high_key])
            rows.append(
                {
                    "method": method,
                    "metric": key,
                    "label": label,
                    "rate": rate,
                    "wilson_low": low,
                    "wilson_high": high,
                    "err_low": rate - low,
                    "err_high": high - rate,
                }
            )
    return rows


def concatenate_rollouts(
    rollouts: list[ImaginedRollout],
    *,
    metadata: dict[str, Any] | None = None,
) -> ImaginedRollout:
    """Concatenate imagined rollouts into one multi-chunk candidate rollout."""
    if not rollouts:
        raise ValueError("rollouts must not be empty")
    actions = np.concatenate([rollout.action_chunk.actions for rollout in rollouts], axis=0)
    action_mode = rollouts[0].action_chunk.action_mode
    dt = rollouts[0].action_chunk.dt
    if any(rollout.action_chunk.action_mode != action_mode for rollout in rollouts):
        raise ValueError("all rollouts must use the same action mode")
    if any(abs(float(rollout.action_chunk.dt) - float(dt)) > 1e-9 for rollout in rollouts):
        raise ValueError("all rollouts must use the same dt")

    chunk_metadata: dict[str, Any] = {
        "planning_horizon_chunks": len(rollouts),
    }
    if metadata:
        chunk_metadata.update(metadata)
    action_chunk = ActionChunk(
        actions=actions,
        action_mode=action_mode,
        dt=dt,
        metadata=chunk_metadata,
    )
    return ImaginedRollout(
        q=np.concatenate([rollout.q for rollout in rollouts], axis=0),
        eef_path=np.concatenate([rollout.eef_path for rollout in rollouts], axis=0),
        robot_point_clouds=[
            cloud for rollout in rollouts for cloud in rollout.robot_point_clouds
        ],
        scene_point_clouds=[
            cloud for rollout in rollouts for cloud in rollout.scene_point_clouds
        ],
        robot_masks=[mask for rollout in rollouts for mask in rollout.robot_masks],
        action_chunk=action_chunk,
        metadata=chunk_metadata,
    )


def candidate_feasibility_fraction(feasible: int, total: int) -> float | None:
    """Return a candidate feasibility fraction, or None when no candidates were scored."""
    if total < 0 or feasible < 0 or feasible > total:
        raise ValueError("feasible and total must satisfy 0 <= feasible <= total")
    if total == 0:
        return None
    return float(feasible) / float(total)


def should_emit_episode_artifact(episode_index: int, every_episodes: int) -> bool:
    """Return whether a periodic episode artifact should be emitted."""
    if episode_index < 0:
        raise ValueError("episode_index must be non-negative")
    if every_episodes <= 0:
        raise ValueError("every_episodes must be positive")
    return episode_index == 0 or (episode_index + 1) % every_episodes == 0


def select_artifact_episode_indices(
    episode_indices: list[int],
    *,
    selection: ArtifactSelection,
    count: int,
    seed: int,
    every_episodes: int,
) -> list[int]:
    """Select episode output indices for video/Rerun artifacts."""
    if count <= 0:
        raise ValueError("count must be positive")
    if every_episodes <= 0:
        raise ValueError("every_episodes must be positive")
    if any(index < 0 for index in episode_indices):
        raise ValueError("episode_indices must be non-negative")
    if selection == "all":
        return list(episode_indices)
    if selection == "periodic":
        return [
            index
            for index in episode_indices
            if should_emit_episode_artifact(index, every_episodes)
        ]
    if selection == "random":
        if not episode_indices:
            return []
        selected_count = min(count, len(episode_indices))
        rng = np.random.default_rng(seed)
        selected = rng.choice(
            np.asarray(episode_indices, dtype=np.int64),
            size=selected_count,
            replace=False,
        )
        return sorted(int(index) for index in selected)
    raise ValueError(f"unsupported artifact selection {selection!r}")


def progress_series(rows: list[dict[str, Any]]) -> dict[str, dict[str, list[float]]]:
    """Build cumulative per-method progress series for local/W&B plots."""
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)

    series: dict[str, dict[str, list[float]]] = {}
    for method, method_rows in sorted(by_method.items()):
        ordered = sorted(method_rows, key=lambda row: (int(row["episode"]), int(row["seed"])))
        method_series: dict[str, list[float]] = {
            "episode": [],
            "reach_success_rate": [],
            "constraint_satisfied_rate": [],
            "combined_success_rate": [],
            "stable_combined_success_rate": [],
            "final_target_distance": [],
            "min_clearance": [],
            "candidate_feasibility_fraction": [],
            "fallback_count": [],
        }
        reach = 0
        constraint = 0
        combined = 0
        stable_combined = 0
        for idx, row in enumerate(ordered, start=1):
            reach += int(bool(row["reach_success"]))
            constraint += int(bool(row["constraint_satisfied"]))
            combined += int(bool(row["combined_success"]))
            stable_combined += int(bool(row.get("stable_combined_success", False)))
            method_series["episode"].append(float(row["episode"]))
            method_series["reach_success_rate"].append(reach / idx)
            method_series["constraint_satisfied_rate"].append(constraint / idx)
            method_series["combined_success_rate"].append(combined / idx)
            method_series["stable_combined_success_rate"].append(stable_combined / idx)
            method_series["final_target_distance"].append(_optional_float(row))
            method_series["min_clearance"].append(_optional_float(row, key="min_clearance"))
            method_series["candidate_feasibility_fraction"].append(
                _optional_float(row, key="candidate_feasibility_fraction")
            )
            method_series["fallback_count"].append(float(row.get("fallback_count", 0)))
        series[method] = method_series
    return series


def _mean_std(key: str, rows: list[dict[str, Any]]) -> dict[str, float | None]:
    values = [
        float(row[key])
        for row in rows
        if row.get(key) is not None and np.isfinite(float(row[key]))
    ]
    if not values:
        return {
            f"{key}_mean": None,
            f"{key}_std": None,
            f"{key}_median": None,
            f"{key}_q1": None,
            f"{key}_q3": None,
        }
    array = np.asarray(values, dtype=np.float32)
    return {
        f"{key}_mean": float(np.mean(array)),
        f"{key}_std": float(np.std(array)),
        f"{key}_median": float(np.median(array)),
        f"{key}_q1": float(np.percentile(array, 25.0)),
        f"{key}_q3": float(np.percentile(array, 75.0)),
    }


def _point_at_arc_fraction(points: Any, *, fraction: float) -> np.ndarray:
    path = np.asarray(points, dtype=np.float32)
    if path.ndim != 2 or path.shape[1] != 3:
        raise ValueError(f"tcp_positions must have shape [T, 3], got {path.shape}")
    if path.shape[0] == 0:
        raise ValueError("tcp_positions must contain at least one point")
    if not np.all(np.isfinite(path)):
        raise ValueError("tcp_positions must be finite")
    if path.shape[0] == 1:
        return path[0].astype(np.float32, copy=True)
    segments = path[1:] - path[:-1]
    lengths = np.linalg.norm(segments, axis=1)
    total = float(np.sum(lengths))
    if total <= 1e-9:
        return path[0].astype(np.float32, copy=True)
    target_length = float(fraction) * total
    cumulative = np.cumsum(lengths)
    segment_idx = int(np.searchsorted(cumulative, target_length, side="left"))
    segment_idx = min(segment_idx, len(lengths) - 1)
    previous = 0.0 if segment_idx == 0 else float(cumulative[segment_idx - 1])
    length = float(lengths[segment_idx])
    alpha = 0.0 if length <= 1e-9 else (target_length - previous) / length
    point = path[segment_idx] + np.float32(alpha) * segments[segment_idx]
    return point.astype(np.float32, copy=False)


def _vector3(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _optional_float(row: dict[str, Any], *, key: str = "final_target_distance") -> float:
    value = row.get(key)
    if value is None:
        return float("nan")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return numeric if np.isfinite(numeric) else float("nan")


def _identity_value(value: Any) -> str:
    return json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"))
