from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from pg3d.constraints.core import SceneContext, mean_squared_norm, trajectory_points
from pg3d.constraints.geometry import Region, SphereRegion, region_from_json
from pg3d.world_model.types import Array, ImaginedRollout, as_float_array

ConstraintTarget = Literal["eef", "robot"]
SmoothnessTarget = Literal["q", "eef"]


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
        violation_sq = violation * violation
        return {
            self.name: float(self.weight) * float(np.mean(violation_sq)),
            f"{self.name}/max_violation": float(np.max(violation)) if violation.size else 0.0,
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


def constraint_from_json(config: dict[str, Any]) -> AvoidRegion | SmoothnessCost:
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
    if constraint_type == "smoothness":
        threshold = config.get("threshold")
        return SmoothnessCost(
            target=config.get("target", "q"),
            order=int(config.get("order", 2)),
            weight=float(config.get("weight", 1.0)),
            threshold=None if threshold is None else float(threshold),
            name=str(config.get("name", "smoothness")),
        )
    raise ValueError(f"unknown constraint type {constraint_type!r}")


def constraints_to_json(constraints: list[AvoidRegion | SmoothnessCost]) -> list[dict[str, Any]]:
    """Serialize a list of constraints."""
    return [constraint.to_json() for constraint in constraints]


def constraints_from_json(configs: list[dict[str, Any]]) -> list[AvoidRegion | SmoothnessCost]:
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
