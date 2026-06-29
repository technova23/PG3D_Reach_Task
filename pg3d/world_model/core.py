from __future__ import annotations

from typing import Any

import numpy as np

from pg3d.envs.maniskill_adapter.types import Observation
from pg3d.world_model.chunks import interpret_joint_chunk
from pg3d.world_model.compositor import compose_robot_cloud, static_scene_from_robot_mask
from pg3d.world_model.types import (
    ActionChunk,
    Array,
    ImaginedRollout,
    RobotGeometryProvider,
    as_float_array,
)


class GeometricWorldModel:
    """Robot-only kinematic point-cloud world model.

    The geometry provider is intentionally simulator-free here. A future ManiSkill/SAPIEN
    adapter should implement `RobotGeometryProvider` behind this boundary.
    """

    def __init__(
        self,
        geometry_provider: RobotGeometryProvider,
        *,
        controlled_dof: int | None = None,
    ) -> None:
        self.geometry_provider = geometry_provider
        self.controlled_dof = controlled_dof

    def imagine(
        self,
        observation: Observation,
        action_chunk: ActionChunk,
        *,
        start_q: Array | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ImaginedRollout:
        """Imagine future robot geometry for one action chunk."""
        q0 = (
            as_float_array(start_q, name="start_q", ndim=1)
            if start_q is not None
            else observation.robot_state.joint_positions
        )
        q = interpret_joint_chunk(
            action_chunk,
            q0,
            controlled_dof=self.controlled_dof,
        )
        static_scene = static_scene_from_robot_mask(
            observation.point_cloud,
            observation.robot_mask,
        )

        eef_positions: list[Array] = []
        robot_clouds: list[Array] = []
        scene_clouds: list[Array] = []
        robot_masks: list[Array] = []
        for q_step in q:
            eef_positions.append(_eef_position(self.geometry_provider, q_step))
            robot_cloud = _robot_point_cloud(self.geometry_provider, q_step)
            scene_cloud, robot_mask = compose_robot_cloud(static_scene, robot_cloud)
            robot_clouds.append(robot_cloud)
            scene_clouds.append(scene_cloud)
            robot_masks.append(robot_mask)

        rollout_metadata = {
            "static_scene_points": int(static_scene.shape[0]),
            "current_robot_points": (
                int(np.count_nonzero(observation.robot_mask))
                if observation.robot_mask is not None
                else 0
            ),
            "controlled_dof": int(action_chunk.action_dim),
            "world_model": "robot_only_kinematic_v0",
        }
        if metadata:
            rollout_metadata.update(metadata)

        return ImaginedRollout(
            q=q,
            eef_path=np.stack(eef_positions, axis=0).astype(np.float32, copy=False),
            robot_point_clouds=robot_clouds,
            scene_point_clouds=scene_clouds,
            robot_masks=robot_masks,
            action_chunk=action_chunk,
            metadata=rollout_metadata,
        )


def _eef_position(provider: RobotGeometryProvider, q: Array) -> Array:
    eef = as_float_array(provider.end_effector_position(q), name="end_effector_position", ndim=1)
    if eef.shape != (3,):
        raise ValueError(f"end_effector_position must have shape (3,), got {eef.shape}")
    return eef.astype(np.float32, copy=True)


def _robot_point_cloud(provider: RobotGeometryProvider, q: Array) -> Array:
    points = as_float_array(provider.robot_point_cloud(q), name="robot_point_cloud", ndim=2)
    if points.shape[1] != 3:
        raise ValueError(f"robot_point_cloud must have shape [N, 3], got {points.shape}")
    return points.astype(np.float32, copy=True)
