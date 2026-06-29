from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from pg3d.constraints.geometry import Region
from pg3d.world_model.types import Array, ImaginedRollout, as_float_array


@dataclass
class SceneContext:
    """Eval/debug context available to constraint programs."""

    target_position: Array | None = None
    regions: dict[str, Region] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.target_position is not None:
            target = as_float_array(self.target_position, name="target_position", ndim=1)
            if target.shape != (3,):
                raise ValueError(f"target_position must have shape (3,), got {target.shape}")
            self.target_position = target
        self.regions = dict(self.regions)
        self.metadata = dict(self.metadata)


class Constraint(Protocol):
    """Python-first constraint interface for imagined rollouts."""

    constraint_type: str

    def cost(self, rollout: ImaginedRollout, scene: SceneContext | None = None) -> dict[str, float]:
        """Return scalar cost terms for one imagined rollout."""
        ...

    def satisfied(
        self,
        rollout: ImaginedRollout,
        scene: SceneContext | None = None,
    ) -> bool:
        """Return whether the rollout satisfies this constraint."""
        ...

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-safe constraint config."""
        ...


def trajectory_points(rollout: ImaginedRollout, *, target: str) -> Array:
    """Extract a trajectory from an imagined rollout for constraint scoring."""
    if target == "eef":
        return rollout.eef_path
    if target == "q":
        return rollout.q
    if target == "robot":
        clouds = [cloud for cloud in rollout.robot_point_clouds if cloud.size]
        if not clouds:
            return np.zeros((0, 3), dtype=np.float32)
        return np.concatenate(clouds, axis=0).astype(np.float32, copy=False)
    raise ValueError(f"unsupported trajectory target {target!r}")


def mean_squared_norm(values: Array) -> float:
    """Return the mean squared row norm for a 2D array."""
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return 0.0
    if values.ndim != 2:
        raise ValueError(f"values must have ndim=2, got shape {values.shape}")
    return float(np.mean(np.sum(values * values, axis=1)))
