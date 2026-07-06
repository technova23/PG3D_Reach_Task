from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from pg3d.constraints.core import SceneContext, mean_squared_norm, trajectory_points
from pg3d.constraints.geometry import (
    CylinderRegion,
    RectRegion2D,
    Region,
    SphereRegion,
    region_from_json,
)
from pg3d.world_model.types import Array, ImaginedRollout, as_float_array

ConstraintTarget = Literal["eef", "robot"]
SmoothnessTarget = Literal["q", "eef"]


@dataclass(frozen=True)
class CartesianPoseConstraint:
    """Soft Cartesian pose target over the whole rollout."""

    target_position: Array
    target_orientation: Array
    position_tolerance: float
    rotation_tolerance: float
    target: ConstraintTarget = "eef"
    weight: float = 1.0
    name: str = "cartesian_pose"
    constraint_type: str = "cartesian_pose"
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.target != "eef":
            raise ValueError("CartesianPoseConstraint currently supports only target='eef'")
        object.__setattr__(
            self,
            "target_position",
            _vector3(self.target_position, name="target_position"),
        )
        orientation = as_float_array(self.target_orientation, name="target_orientation", ndim=1)
        if orientation.shape not in {(4,), (9,)}:
            raise ValueError("target_orientation must have shape (4,) or (9,)")
        if orientation.shape == (4,):
            orientation = _normalize_quaternion(orientation)
        else:
            orientation = _normalize_rotation_matrix(orientation.reshape(3, 3)).reshape(9)
        object.__setattr__(self, "target_orientation", orientation)
        _validate_nonnegative(self.position_tolerance, "position_tolerance")
        _validate_nonnegative(self.rotation_tolerance, "rotation_tolerance")
        _validate_finite(self.weight, "weight")
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def cost(self, rollout: ImaginedRollout, scene: SceneContext | None = None) -> dict[str, float]:
        positions = np.asarray(rollout.eef_path, dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"eef_path must have shape [T, 3], got {positions.shape}")
        positions = positions.reshape(-1, 3)
        pos_errors = np.linalg.norm(positions - self.target_position.reshape(1, 3), axis=1)
        rot_errors = _rotation_distance_series(rollout.eef_orientations, self.target_orientation)
        combined = np.maximum(pos_errors - float(self.position_tolerance), 0.0) + np.maximum(
            rot_errors - float(self.rotation_tolerance), 0.0
        )
        if combined.size == 0:
            best_idx = -1
            best_total = float("inf")
            best_pos_error = float("inf")
            best_rot_error = float("inf")
        else:
            best_idx = int(np.argmin(combined))
            best_total = float(combined[best_idx])
            best_pos_error = float(pos_errors[best_idx])
            best_rot_error = float(rot_errors[best_idx])
        return {
            self.name: float(self.weight) * best_total,
            f"{self.name}/position_error": best_pos_error,
            f"{self.name}/rotation_error": best_rot_error,
            f"{self.name}/position_violation": max(
                best_pos_error - float(self.position_tolerance),
                0.0,
            ),
            f"{self.name}/rotation_violation": max(
                best_rot_error - float(self.rotation_tolerance),
                0.0,
            ),
            f"{self.name}/best_index": float(best_idx),
        }

    def satisfied(
        self,
        rollout: ImaginedRollout,
        scene: SceneContext | None = None,
    ) -> bool:
        positions = np.asarray(rollout.eef_path, dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] == 0:
            return False
        pos_errors = np.linalg.norm(positions - self.target_position.reshape(1, 3), axis=1)
        rot_errors = _rotation_distance_series(rollout.eef_orientations, self.target_orientation)
        return bool(
            np.any(
                (pos_errors <= float(self.position_tolerance))
                & (rot_errors <= float(self.rotation_tolerance))
            )
        )

    def to_json(self) -> dict[str, Any]:
        payload = {
            "type": self.constraint_type,
            "target": self.target,
            "target_position": self.target_position.tolist(),
            "target_orientation": self.target_orientation.tolist(),
            "position_tolerance": self.position_tolerance,
            "rotation_tolerance": self.rotation_tolerance,
            "weight": self.weight,
            "name": self.name,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class AvoidRegion:
    """Penalize an imagined trajectory for entering a keep-out region."""

    region: Region
    target: ConstraintTarget = "eef"
    margin: float = 0.0
    weight: float = 1.0
    tolerance: float = 1e-6
    name: str = "avoid_region"
    constraint_type: str = "avoid_region"

    def __post_init__(self) -> None:
        if self.target not in {"eef", "robot"}:
            raise ValueError("AvoidRegion supports only target='eef' or target='robot'")
        _validate_nonnegative(self.margin, "margin")
        _validate_nonnegative(self.tolerance, "tolerance")
        _validate_finite(self.weight, "weight")

    def cost(self, rollout: ImaginedRollout, scene: SceneContext | None = None) -> dict[str, float]:
        points = trajectory_points(rollout, target=self.target)
        signed_distance = self.region.signed_distance(points)
        violation = np.maximum(float(self.margin) - signed_distance, 0.0)
        # Penalize the WORST penetration (meters) rather than mean(violation**2):
        # a length-averaged, squared penalty is far too small to compete with the
        # goal_distance term during candidate selection. See AvoidProjection.cost.
        max_violation = float(np.max(violation)) if violation.size else 0.0
        return {
            self.name: float(self.weight) * max_violation,
            f"{self.name}/max_violation": max_violation,
            f"{self.name}/min_signed_distance": (
                float(np.min(signed_distance)) if signed_distance.size else float("inf")
            ),
            f"{self.name}/inside_count": float(np.count_nonzero(signed_distance < 0.0)),
        }

    def satisfied(
        self,
        rollout: ImaginedRollout,
        scene: SceneContext | None = None,
    ) -> bool:
        max_violation = self.cost(rollout, scene)[f"{self.name}/max_violation"]
        return max_violation <= float(self.tolerance)

    def to_json(self) -> dict[str, Any]:
        return {
            "type": self.constraint_type,
            "target": self.target,
            "region": self.region.to_json(),
            "margin": self.margin,
            "weight": self.weight,
            "tolerance": self.tolerance,
            "name": self.name,
        }


@dataclass(frozen=True)
class AvoidProjection:
    """Penalize the XY projection of a trajectory for passing over a footprint.

    Reach analog of no-overflight (P0.5): the ``(x, y)`` projection of the EEF
    (or whole robot) is penalized for entering a restricted tabletop rectangle,
    *regardless of height z*. Geometrically this is an axis-aligned 2-D keep-out
    rectangle extruded through all heights.

    The primary cost key (``name``, no ``/``) is summed by
    ``primary_constraint_penalty`` and so flows into the candidate ``total_score``
    alongside ``goal_distance``, ``trajectory_smoothness`` and
    ``consensus_deviations``. The slash-qualified keys are diagnostics only.
    """

    region: RectRegion2D
    target: ConstraintTarget = "eef"
    margin: float = 0.0
    weight: float = 1.0
    tolerance: float = 1e-6
    name: str = "avoid_projection"
    constraint_type: str = "avoid_projection"

    def __post_init__(self) -> None:
        if self.target not in {"eef", "robot"}:
            raise ValueError("AvoidProjection supports only target='eef' or target='robot'")
        _validate_nonnegative(self.margin, "margin")
        _validate_nonnegative(self.tolerance, "tolerance")
        _validate_finite(self.weight, "weight")

    def cost(self, rollout: ImaginedRollout, scene: SceneContext | None = None) -> dict[str, float]:
        points = trajectory_points(rollout, target=self.target)
        # RectRegion2D.signed_distance drops the Z column, so the [T, 3] EEF path
        # (or [M, 3] robot cloud) is scored purely on its XY footprint.
        signed_distance = self.region.signed_distance(points)
        violation = np.maximum(float(self.margin) - signed_distance, 0.0)
        # Penalize the WORST XY penetration (meters), not mean(violation**2). A mean
        # over every timestep dilutes the few violating steps by trajectory length
        # and the square crushes sub-meter depths, leaving the penalty too small to
        # influence candidate selection. The max keeps it length-invariant and in
        # the same units (and on the same scale) as goal_distance.
        max_violation = float(np.max(violation)) if violation.size else 0.0
        total = float(signed_distance.size)
        inside = float(np.count_nonzero(signed_distance < 0.0)) if signed_distance.size else 0.0
        return {
            self.name: float(self.weight) * max_violation,
            f"{self.name}/max_violation": max_violation,
            f"{self.name}/min_signed_distance": (
                float(np.min(signed_distance)) if signed_distance.size else float("inf")
            ),
            f"{self.name}/inside_count": inside,
            f"{self.name}/fraction_over": (inside / total) if total else 0.0,
        }

    def satisfied(
        self,
        rollout: ImaginedRollout,
        scene: SceneContext | None = None,
    ) -> bool:
        max_violation = self.cost(rollout, scene)[f"{self.name}/max_violation"]
        return max_violation <= float(self.tolerance)

    def to_json(self) -> dict[str, Any]:
        return {
            "type": self.constraint_type,
            "target": self.target,
            "region": self.region.to_json(),
            "margin": self.margin,
            "weight": self.weight,
            "tolerance": self.tolerance,
            "name": self.name,
        }


@dataclass(frozen=True)
class SmoothnessCost:
    """Penalize jagged joint or end-effector trajectories."""

    target: SmoothnessTarget = "q"
    order: int = 2
    weight: float = 1.0
    threshold: float | None = None
    name: str = "smoothness"
    constraint_type: str = "smoothness"

    def __post_init__(self) -> None:
        if self.target not in {"q", "eef"}:
            raise ValueError(f"unsupported smoothness target {self.target!r}")
        if self.order not in {1, 2}:
            raise ValueError("order must be 1 or 2")
        _validate_nonnegative(self.weight, "weight")
        if self.threshold is not None:
            _validate_nonnegative(self.threshold, "threshold")

    def cost(self, rollout: ImaginedRollout, scene: SceneContext | None = None) -> dict[str, float]:
        points = trajectory_points(rollout, target=self.target)
        diffs = _finite_difference(points, order=self.order)
        raw = mean_squared_norm(diffs)
        return {
            self.name: float(self.weight) * raw,
            f"{self.name}/raw": raw,
        }

    def satisfied(
        self,
        rollout: ImaginedRollout,
        scene: SceneContext | None = None,
    ) -> bool:
        if self.threshold is None:
            return True
        return self.cost(rollout, scene)[self.name] <= float(self.threshold)

    def to_json(self) -> dict[str, Any]:
        return {
            "type": self.constraint_type,
            "target": self.target,
            "order": self.order,
            "weight": self.weight,
            "threshold": self.threshold,
            "name": self.name,
        }


@dataclass(frozen=True)
class CylinderPassageConstraint:
    """Hold orientation while the EEF traverses a finite cylinder corridor."""

    region: CylinderRegion
    target_orientation: Array
    waypoint_start: int
    waypoint_end: int
    position_tolerance: float
    rotation_tolerance: float
    weight: float = 1.0
    name: str = "cylinder_passage"
    constraint_type: str = "cylinder_passage"

    def __post_init__(self) -> None:
        orientation = as_float_array(self.target_orientation, name="target_orientation", ndim=1)
        if orientation.shape not in {(4,), (9,)}:
            raise ValueError("target_orientation must have shape (4,) or (9,)")
        if orientation.shape == (4,):
            orientation = _normalize_quaternion(orientation)
        else:
            orientation = _normalize_rotation_matrix(orientation.reshape(3, 3)).reshape(9)
        object.__setattr__(self, "target_orientation", orientation)
        waypoint_start = int(self.waypoint_start)
        waypoint_end = int(self.waypoint_end)
        if waypoint_start < 0 or waypoint_end < 0:
            raise ValueError("waypoint bounds must be non-negative")
        if waypoint_end < waypoint_start:
            raise ValueError("waypoint_end must be >= waypoint_start")
        object.__setattr__(self, "waypoint_start", waypoint_start)
        object.__setattr__(self, "waypoint_end", waypoint_end)
        _validate_nonnegative(self.position_tolerance, "position_tolerance")
        _validate_nonnegative(self.rotation_tolerance, "rotation_tolerance")
        _validate_finite(self.weight, "weight")

    def cost(self, rollout: ImaginedRollout, scene: SceneContext | None = None) -> dict[str, float]:
        start, end = self._resolve_window(rollout)
        segment = rollout.eef_path[start : end + 1]
        signed_distance = self.region.signed_distance(segment)
        axial = self.region.project_axial(segment).reshape(-1)
        axial_span = float(np.max(axial) - np.min(axial)) if axial.size else 0.0
        max_radial_violation = (
            float(np.max(np.maximum(signed_distance, 0.0)))
            if signed_distance.size
            else 0.0
        )
        passage_violation = max(float(self.region.length) - axial_span, 0.0)
        pose_rotation_error = float(
            np.max(
                _rotation_distance_series(
                    rollout.eef_orientations[start : end + 1],
                    self.target_orientation,
                )
            )
        ) if segment.size else 0.0
        pose_rotation_violation = max(pose_rotation_error - float(self.rotation_tolerance), 0.0)
        total = float(self.weight) * (passage_violation + pose_rotation_violation)
        return {
            self.name: total,
            f"{self.name}/passage_violation": passage_violation,
            f"{self.name}/axial_span": axial_span,
            f"{self.name}/max_radial_violation": max_radial_violation,
            f"{self.name}/position_violation": max_radial_violation,
            f"{self.name}/rotation_violation": pose_rotation_violation,
            f"{self.name}/waypoint_start": float(start),
            f"{self.name}/waypoint_end": float(end),
        }

    def satisfied(
        self,
        rollout: ImaginedRollout,
        scene: SceneContext | None = None,
    ) -> bool:
        costs = self.cost(rollout, scene)
        return (
            costs[f"{self.name}/passage_violation"] <= 1e-6
            and costs[f"{self.name}/rotation_violation"] <= 1e-6
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "type": self.constraint_type,
            "region": self.region.to_json(),
            "target_orientation": self.target_orientation.tolist(),
            "waypoint_start": self.waypoint_start,
            "waypoint_end": self.waypoint_end,
            "position_tolerance": self.position_tolerance,
            "rotation_tolerance": self.rotation_tolerance,
            "weight": self.weight,
            "name": self.name,
        }

    def _resolve_window(self, rollout: ImaginedRollout) -> tuple[int, int]:
        horizon = rollout.eef_path.shape[0]
        if self.waypoint_end >= horizon:
            raise IndexError(
                f"waypoint_end {self.waypoint_end} is out of bounds for rollout horizon {horizon}"
            )
        return self.waypoint_start, self.waypoint_end


def make_obstructing_avoid_region(
    start: Array,
    goal: Array,
    *,
    radius: float = 0.07,
    margin: float = 0.0,
    weight: float = 1.0,
    name: str = "avoid_region",
) -> AvoidRegion:
    """Create a small sphere on the direct EEF path from start to goal."""
    start = _vector3(start, name="start")
    goal = _vector3(goal, name="goal")
    if np.linalg.norm(goal - start) <= 1e-6:
        raise ValueError("start and goal must be distinct to place an obstructing region")
    return AvoidRegion(
        region=SphereRegion(center=(start + goal) * 0.5, radius=radius),
        margin=margin,
        weight=weight,
        name=name,
    )


def constraint_from_json(
    config: dict[str, Any],
) -> (
    AvoidRegion
    | AvoidProjection
    | SmoothnessCost
    | CartesianPoseConstraint
    | CylinderPassageConstraint
):
    """Load a constraint from a JSON-safe config."""
    constraint_type = config.get("type")
    if constraint_type == "avoid_region":
        return AvoidRegion(
            region=region_from_json(config["region"]),
            target=config.get("target", "eef"),
            margin=float(config.get("margin", 0.0)),
            weight=float(config.get("weight", 1.0)),
            tolerance=float(config.get("tolerance", 1e-6)),
            name=str(config.get("name", "avoid_region")),
        )
    if constraint_type == "avoid_projection":
        region = region_from_json(config["region"])
        if not isinstance(region, RectRegion2D):
            raise ValueError(
                f"avoid_projection requires a rect2d region, got {region.region_type!r}"
            )
        return AvoidProjection(
            region=region,
            target=config.get("target", "eef"),
            margin=float(config.get("margin", 0.0)),
            weight=float(config.get("weight", 1.0)),
            tolerance=float(config.get("tolerance", 1e-6)),
            name=str(config.get("name", "avoid_projection")),
        )
    if constraint_type == "smoothness":
        threshold = config.get("threshold")
        return SmoothnessCost(
            target=config.get("target", "q"),
            order=int(config.get("order", 2)),
            weight=float(config.get("weight", 1.0)),
            threshold=None if threshold is None else float(threshold),
            name=str(config.get("name", "smoothness")),
        )
    if constraint_type == "cartesian_pose":
        return CartesianPoseConstraint(
            target_position=config["target_position"],
            target_orientation=config["target_orientation"],
            position_tolerance=float(config.get("position_tolerance", 0.0)),
            rotation_tolerance=float(config.get("rotation_tolerance", 0.0)),
            target=config.get("target", "eef"),
            weight=float(config.get("weight", 1.0)),
            name=str(config.get("name", "cartesian_pose")),
            metadata=dict(config.get("metadata", {})),
        )
    if constraint_type == "cylinder_passage":
        return CylinderPassageConstraint(
            region=region_from_json(config["region"]),
            target_orientation=config["target_orientation"],
            waypoint_start=int(config["waypoint_start"]),
            waypoint_end=int(config["waypoint_end"]),
            position_tolerance=float(config.get("position_tolerance", 0.0)),
            rotation_tolerance=float(config.get("rotation_tolerance", 0.0)),
            weight=float(config.get("weight", 1.0)),
            name=str(config.get("name", "cylinder_passage")),
        )
    raise ValueError(f"unknown constraint type {constraint_type!r}")


def constraints_to_json(
    constraints: list[
        AvoidRegion
        | AvoidProjection
        | SmoothnessCost
        | CartesianPoseConstraint
        | CylinderPassageConstraint
    ],
) -> list[dict[str, Any]]:
    """Serialize a list of constraints."""
    return [constraint.to_json() for constraint in constraints]


def constraints_from_json(
    configs: list[dict[str, Any]],
) -> list[
    AvoidRegion
    | AvoidProjection
    | SmoothnessCost
    | CartesianPoseConstraint
    | CylinderPassageConstraint
]:
    """Deserialize a list of constraints."""
    return [constraint_from_json(config) for config in configs]


def _finite_difference(points: Array, *, order: int) -> Array:
    if points.shape[0] <= order:
        return np.zeros((0, points.shape[1]), dtype=np.float32)
    return np.diff(points, n=order, axis=0).astype(np.float32, copy=False)


def _vector3(value: Any, *, name: str) -> Array:
    array = as_float_array(value, name=name, ndim=1)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}")
    return array


def _validate_finite(value: float, name: str) -> None:
    if not np.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


def _validate_nonnegative(value: float, name: str) -> None:
    _validate_finite(value, name)
    if float(value) < 0.0:
        raise ValueError(f"{name} must be non-negative")


def _normalize_quaternion(quaternion: Array) -> Array:
    quat = np.asarray(quaternion, dtype=np.float32)
    norm = float(np.linalg.norm(quat))
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("quaternion must be a non-zero finite vector")
    return quat / norm


def _normalize_rotation_matrix(matrix: Array) -> Array:
    rot = np.asarray(matrix, dtype=np.float32).reshape(3, 3)
    if not np.all(np.isfinite(rot)):
        raise ValueError("rotation matrix must contain only finite values")
    u, _, vh = np.linalg.svd(rot)
    rot = u @ vh
    if np.linalg.det(rot) < 0.0:
        u[:, -1] *= -1.0
        rot = u @ vh
    return rot.astype(np.float32, copy=False)


def _rotation_distance_series(orientations: Array | None, target_orientation: Array) -> Array:
    if orientations is None:
        raise ValueError(
            "imagined rollouts must provide eef_orientations for Cartesian pose constraints"
        )
    orientations = np.asarray(orientations, dtype=np.float32)
    if orientations.ndim != 2 or orientations.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if orientations.shape[1] == 4:
        target_q = _normalize_quaternion(target_orientation.reshape(4))
        return np.asarray(
            [
                float(
                    2.0
                    * np.arccos(
                        np.clip(
                            np.abs(np.dot(_normalize_quaternion(actual), target_q)),
                            -1.0,
                            1.0,
                        )
                    )
                )
                for actual in orientations
            ],
            dtype=np.float32,
        )
    if orientations.shape[1] == 9:
        target_r = _normalize_rotation_matrix(target_orientation.reshape(3, 3))
        errors = []
        for actual in orientations:
            actual_r = _normalize_rotation_matrix(actual.reshape(3, 3))
            delta = actual_r @ target_r.T
            trace = float(np.trace(delta))
            errors.append(float(np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))))
        return np.asarray(errors, dtype=np.float32)
    raise ValueError(f"unsupported eef orientation shape {orientations.shape}")


def _orientation_distance(actual: Array, target: Array) -> float:
    if actual.shape != target.shape:
        raise ValueError(f"orientation shapes must match, got {actual.shape} and {target.shape}")
    if actual.shape == (4,):
        actual_q = _normalize_quaternion(actual)
        target_q = _normalize_quaternion(target)
        dot = float(np.clip(np.abs(np.dot(actual_q, target_q)), -1.0, 1.0))
        return float(2.0 * np.arccos(dot))
    actual_r = _normalize_rotation_matrix(actual.reshape(3, 3))
    target_r = _normalize_rotation_matrix(target.reshape(3, 3))
    delta = actual_r @ target_r.T
    trace = float(np.trace(delta))
    return float(np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0)))
