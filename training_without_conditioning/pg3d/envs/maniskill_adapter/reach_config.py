from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReachGoalRegion:
    """One weighted Cartesian target-sampling region for pg3d reach tasks."""

    name: str
    weight: float
    center: tuple[float, float, float]
    half_extents: tuple[float, float, float]

    @property
    def bounds(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        return tuple(
            (round(center - half_extent, 10), round(center + half_extent, 10))
            for center, half_extent in zip(self.center, self.half_extents, strict=True)
        )

    def to_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "weight": float(self.weight),
            "center": list(self.center),
            "half_extents": list(self.half_extents),
            "bounds": [list(bounds) for bounds in self.bounds],
        }


@dataclass(frozen=True)
class ReachTaskSpec:
    """Simulator-free defaults for pg3d ManiSkill reach task variants."""

    env_id: str
    max_episode_steps: int
    goal_center: tuple[float, float, float]
    goal_half_extents: tuple[float, float, float]
    goal_regions: tuple[ReachGoalRegion, ...] = ()

    @property
    def goal_bounds(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        if self.goal_regions:
            return tuple(
                (
                    round(min(region.bounds[axis][0] for region in self.goal_regions), 10),
                    round(max(region.bounds[axis][1] for region in self.goal_regions), 10),
                )
                for axis in range(3)
            )
        return tuple(
            (round(center - half_extent, 10), round(center + half_extent, 10))
            for center, half_extent in zip(self.goal_center, self.goal_half_extents, strict=True)
        )


REACH_TASK_SPECS: dict[str, ReachTaskSpec] = {
    "PG3DReach-Narrow-v0": ReachTaskSpec(
        env_id="PG3DReach-Narrow-v0",
        max_episode_steps=50,
        goal_center=(0.0, 0.0, 0.35),
        goal_half_extents=(0.08, 0.08, 0.08),
    ),
    "PG3DReach-Medium-v0": ReachTaskSpec(
        env_id="PG3DReach-Medium-v0",
        max_episode_steps=60,
        goal_center=(0.02, 0.0, 0.38),
        goal_half_extents=(0.16, 0.16, 0.14),
    ),
    "PG3DReach-Workspace-v0": ReachTaskSpec(
        env_id="PG3DReach-Workspace-v0",
        max_episode_steps=100,
        goal_center=(0.05, 0.0, 0.45),
        goal_half_extents=(0.35, 0.35, 0.30),
    ),
    "PG3DReach-BalancedWorkspace-v0": ReachTaskSpec(
        env_id="PG3DReach-BalancedWorkspace-v0",
        max_episode_steps=100,
        goal_center=(0.04, 0.0, 0.44),
        goal_half_extents=(0.30, 0.30, 0.24),
        goal_regions=(
            ReachGoalRegion(
                name="core_practical",
                weight=0.70,
                center=(0.05, 0.0, 0.42),
                half_extents=(0.19, 0.20, 0.14),
            ),
            ReachGoalRegion(
                name="outer_practical",
                weight=0.30,
                center=(0.04, 0.0, 0.44),
                half_extents=(0.30, 0.30, 0.24),
            ),
        ),
    ),
}


def reach_task_metadata(env_id: str) -> dict[str, object]:
    """Return JSON-safe task-default metadata for a pg3d reach env id."""
    spec = REACH_TASK_SPECS.get(env_id)
    if spec is None:
        return {"env_id": env_id}
    metadata: dict[str, object] = {
        "env_id": spec.env_id,
        "max_episode_steps": spec.max_episode_steps,
        "goal_center": list(spec.goal_center),
        "goal_half_extents": list(spec.goal_half_extents),
        "goal_bounds": [list(bounds) for bounds in spec.goal_bounds],
    }
    if spec.goal_regions:
        metadata["goal_regions"] = [region.to_json() for region in spec.goal_regions]
    return metadata
