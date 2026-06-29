from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np

from pg3d.envs.maniskill_adapter.observation import adapt_observation
from pg3d.utils.arrays import to_numpy
from pg3d.world_model.types import Array, RobotGeometryProvider, as_float_array


@dataclass
class GhostGeometrySnapshot:
    """Rendered Panda geometry for one imagined joint state."""

    q: Array
    eef_position: Array
    robot_point_cloud: Array


class ManiSkillGhostPandaGeometryProvider(RobotGeometryProvider):
    """Render Panda robot clouds from a second ManiSkill env.

    This provider intentionally lives outside `pg3d.world_model`: it mutates a live ghost
    simulator env to a requested qpos, renders/adapts a point-cloud observation, and returns only
    the robot-segmented points to the pure world-model core.
    """

    def __init__(
        self,
        env: Any,
        *,
        task_name: str = "unknown",
        batch_index: int = 0,
        max_robot_points: int | None = None,
        crop_bounds: Array | None = None,
    ) -> None:
        self.env = env
        self.task_name = task_name
        self.batch_index = batch_index
        self.max_robot_points = max_robot_points
        self.crop_bounds = (
            None
            if crop_bounds is None
            else as_float_array(crop_bounds, name="crop_bounds", ndim=2)
        )
        if self.crop_bounds is not None and self.crop_bounds.shape != (3, 2):
            raise ValueError(f"crop_bounds must have shape (3, 2), got {self.crop_bounds.shape}")
        self._cache: GhostGeometrySnapshot | None = None

    def end_effector_position(self, q: Array) -> Array:
        """Return the ghost-env TCP position for one qpos."""
        return self._snapshot(q).eef_position

    def end_effector_position_only(self, q: Array) -> Array:
        """Return TCP position after setting qpos without rendering a point cloud."""
        qpos = as_float_array(q, name="q", ndim=1)
        self._set_robot_qpos(qpos)
        unwrapped = getattr(self.env, "unwrapped", self.env)
        agent = getattr(unwrapped, "agent", None)
        tcp_pose = getattr(agent, "tcp_pose", None)
        tcp_position = getattr(tcp_pose, "p", None)
        if tcp_position is None:
            raise RuntimeError("ghost ManiSkill env does not expose agent.tcp_pose.p")
        eef = to_numpy(tcp_position).astype(np.float32, copy=True)
        if eef.ndim == 2:
            if self.batch_index >= eef.shape[0]:
                raise IndexError(
                    f"batch_index={self.batch_index} is out of range for tcp shape {eef.shape}"
                )
            eef = eef[self.batch_index]
        eef = eef.reshape(-1)
        if eef.shape != (3,):
            raise RuntimeError(f"tcp position must have shape (3,), got {eef.shape}")
        self._cache = None
        return eef.astype(np.float32, copy=True)

    def robot_point_cloud(self, q: Array) -> Array:
        """Return robot-segmented point-cloud points for one qpos."""
        return self._snapshot(q).robot_point_cloud

    def reset(self, *, seed: int, options: dict[str, Any] | None = None) -> tuple[Any, Any]:
        """Reset the ghost env to match the comparison seed."""
        self._cache = None
        return self.env.reset(seed=seed, options=options or {"reconfigure": True})

    def set_robot_point_budget_from_mask(
        self,
        robot_mask: Array,
        *,
        point_valid_mask: Array | None = None,
        min_points: int = 1,
    ) -> None:
        """Match the rendered robot-point budget to a cropped policy observation."""
        mask = np.asarray(robot_mask, dtype=bool).reshape(-1)
        if point_valid_mask is not None:
            valid = np.asarray(point_valid_mask, dtype=bool).reshape(-1)
            if valid.shape != mask.shape:
                raise ValueError(
                    f"point_valid_mask shape {valid.shape} does not match robot_mask {mask.shape}"
                )
            mask = mask & valid
        count = int(np.count_nonzero(mask))
        next_max = max(min_points, count) if count > 0 else None
        if next_max != self.max_robot_points:
            self._cache = None
        self.max_robot_points = next_max

    def _snapshot(self, q: Array) -> GhostGeometrySnapshot:
        qpos = as_float_array(q, name="q", ndim=1)
        if self._cache is not None and np.array_equal(self._cache.q, qpos):
            return self._cache

        self._set_robot_qpos(qpos)
        unwrapped = getattr(self.env, "unwrapped", self.env)
        info = unwrapped.evaluate() if hasattr(unwrapped, "evaluate") else {}
        obs = unwrapped.get_obs(info) if hasattr(unwrapped, "get_obs") else self.env.get_obs(info)
        adapted = adapt_observation(
            obs,
            info=info,
            env=self.env,
            task_name=self.task_name,
            batch_index=self.batch_index,
        )
        if adapted.robot_mask is None:
            raise RuntimeError("ghost ManiSkill observation did not include a robot mask")
        robot_points = adapted.point_cloud[adapted.robot_mask]
        robot_points = _filter_bounds(robot_points, self.crop_bounds)
        robot_points = _deterministic_sample(robot_points, self.max_robot_points)
        if adapted.robot_state.tcp_pose is None or adapted.robot_state.tcp_pose.shape[0] < 3:
            raise RuntimeError("ghost ManiSkill observation did not include tcp_pose")
        snapshot = GhostGeometrySnapshot(
            q=qpos.astype(np.float32, copy=True),
            eef_position=adapted.robot_state.tcp_pose[:3].astype(np.float32, copy=True),
            robot_point_cloud=robot_points.astype(np.float32, copy=True),
        )
        self._cache = snapshot
        return snapshot

    def _set_robot_qpos(self, q: Array) -> None:
        unwrapped = getattr(self.env, "unwrapped", self.env)
        agent = getattr(unwrapped, "agent", None)
        robot = getattr(agent, "robot", None)
        if robot is None:
            raise RuntimeError("ghost ManiSkill env does not expose agent.robot")

        current_qpos = to_numpy(robot.get_qpos()).astype(np.float32, copy=True)
        if current_qpos.ndim == 1:
            current_qpos = current_qpos.reshape(1, -1)
        if current_qpos.shape[0] != 1:
            raise RuntimeError(
                "ManiSkill ghost geometry provider currently supports num_envs=1; "
                f"got qpos shape {current_qpos.shape}"
            )
        if q.shape[0] > current_qpos.shape[1]:
            raise ValueError(
                f"q has {q.shape[0]} joints but robot qpos has {current_qpos.shape[1]}"
            )
        next_qpos = current_qpos.copy()
        next_qpos[0, : q.shape[0]] = q
        next_qvel = np.zeros_like(next_qpos, dtype=np.float32)

        scene = getattr(unwrapped, "scene", None)
        with _all_envs_reset_mask(scene):
            robot.set_qpos(next_qpos)
            robot.set_qvel(next_qvel)


def _filter_bounds(points: Array, bounds: Array | None) -> Array:
    if bounds is None or points.size == 0:
        return points.astype(np.float32, copy=True)
    in_bounds = np.all((points >= bounds[:, 0]) & (points <= bounds[:, 1]), axis=1)
    return points[in_bounds].astype(np.float32, copy=True)


def _deterministic_sample(points: Array, max_points: int | None) -> Array:
    points = as_float_array(points, name="robot_point_cloud", ndim=2)
    if points.shape[1] != 3:
        raise ValueError(f"robot_point_cloud must have shape [N, 3], got {points.shape}")
    if max_points is None or points.shape[0] <= max_points:
        return points.astype(np.float32, copy=True)
    if max_points <= 0:
        raise ValueError("max_robot_points must be positive when provided")
    selected = np.linspace(0, points.shape[0] - 1, max_points)
    return points[np.rint(selected).astype(np.int64)].astype(np.float32, copy=True)


@contextmanager
def _all_envs_reset_mask(scene: Any) -> Iterator[None]:
    mask = getattr(scene, "_reset_mask", None)
    if mask is None:
        yield
        return

    previous = mask.clone() if hasattr(mask, "clone") else np.array(mask, copy=True)
    try:
        if hasattr(mask, "fill_"):
            mask.fill_(True)
        else:
            mask[...] = True
        yield
    finally:
        if hasattr(mask, "copy_"):
            mask.copy_(previous)
        else:
            mask[...] = previous
