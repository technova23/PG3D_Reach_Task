from __future__ import annotations

from typing import Any

import numpy as np
import sapien
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

from pg3d.envs.maniskill_adapter.reach_config import REACH_TASK_SPECS, ReachGoalRegion


class PG3DReachEnv(BaseEnv):
    """Panda reach task base for pg3d data-generation variants."""

    SUPPORTED_ROBOTS = ["panda"]
    goal_thresh = 0.025

    def __init__(
        self,
        *args: Any,
        robot_uids: str = "panda",
        goal_center: tuple[float, float, float] = (0.0, 0.0, 0.35),
        goal_half_extents: tuple[float, float, float] = (0.08, 0.08, 0.08),
        goal_regions: tuple[ReachGoalRegion, ...] = (),
        goal_thresh: float = 0.025,
        require_static: bool = False,
        robot_init_qpos_noise: float = 0.0,
        **kwargs: Any,
    ) -> None:
        self.goal_center = goal_center
        self.goal_half_extents = goal_half_extents
        self.goal_regions = tuple(goal_regions)
        self.goal_thresh = goal_thresh
        self.require_static = require_static
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self) -> list[CameraConfig]:
        pose = sapien_utils.look_at(eye=[0.45, -0.55, 0.65], target=[0.0, 0.0, 0.25])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self) -> CameraConfig:
        pose = sapien_utils.look_at(eye=[0.75, -0.75, 0.7], target=[0.0, 0.0, 0.25])
        return CameraConfig("render_camera", pose, 512, 512, 1.0, 0.01, 100)

    def _load_agent(self, options: dict[str, Any]) -> None:
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0.0, 0.0]))

    def _load_scene(self, options: dict[str, Any]) -> None:
        self.table_scene = TableSceneBuilder(
            self,
            robot_init_qpos_noise=self.robot_init_qpos_noise,
        )
        self.table_scene.build()
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0.0, 1.0, 0.0, 1.0],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self.start_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[1.0, 0.0, 0.0, 1.0],
            name="start_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict[str, Any]) -> None:
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            batch_size = len(env_idx)
            if self.goal_regions:
                weights = torch.tensor(
                    [region.weight for region in self.goal_regions],
                    dtype=torch.float32,
                )
                region_indices = torch.multinomial(
                    weights / weights.sum(),
                    batch_size,
                    replacement=True,
                )
                centers = torch.tensor(
                    [region.center for region in self.goal_regions],
                    dtype=torch.float32,
                )
                half_extents = torch.tensor(
                    [region.half_extents for region in self.goal_regions],
                    dtype=torch.float32,
                )
                goal_xyz = (
                    centers[region_indices]
                    + (torch.rand((batch_size, 3)) * 2.0 - 1.0) * half_extents[region_indices]
                )
            else:
                center = torch.tensor(self.goal_center, dtype=torch.float32)
                half_extents = torch.tensor(self.goal_half_extents, dtype=torch.float32)
                goal_xyz = center + (torch.rand((batch_size, 3)) * 2.0 - 1.0) * half_extents
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))
            self.start_site.set_pose(Pose.create_from_pq(self.agent.tcp_pose.p))

    def _get_obs_extra(self, info: dict[str, Any]) -> dict[str, Any]:
        return {
            "tcp_pose": self.agent.tcp_pose.raw_pose,
            "goal_pos": self.goal_site.pose.p,
            "tcp_to_goal_pos": self.goal_site.pose.p - self.agent.tcp_pose.p,
        }

    def evaluate(self) -> dict[str, torch.Tensor]:
        tcp_to_goal_dist = torch.linalg.norm(self.goal_site.pose.p - self.agent.tcp_pose.p, axis=1)
        reached = tcp_to_goal_dist <= self.goal_thresh
        is_robot_static = self.agent.is_static(0.2)
        success = reached & is_robot_static if self.require_static else reached
        return {
            "success": success,
            "reached": reached,
            "is_robot_static": is_robot_static,
            "tcp_to_goal_dist": tcp_to_goal_dist,
        }

    def compute_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict[str, Any]
    ) -> torch.Tensor:
        reaching_reward = 1 - torch.tanh(5 * info["tcp_to_goal_dist"])
        static_bonus = 0.1 * self.agent.is_static(0.2)
        reward = reaching_reward + static_bonus
        reward[info["success"]] = 2.0
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict[str, Any]
    ) -> torch.Tensor:
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 2.0


@register_env("PG3DReach-Narrow-v0", max_episode_steps=50)
class PG3DReachNarrowEnv(PG3DReachEnv):
    """Small reset distribution for first DP3 reach dataset smoke."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        spec = REACH_TASK_SPECS["PG3DReach-Narrow-v0"]
        kwargs.setdefault("goal_center", spec.goal_center)
        kwargs.setdefault("goal_half_extents", spec.goal_half_extents)
        super().__init__(*args, **kwargs)


@register_env("PG3DReach-Medium-v0", max_episode_steps=60)
class PG3DReachMediumEnv(PG3DReachEnv):
    """Wider reset distribution for the next nominal reach dataset."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        spec = REACH_TASK_SPECS["PG3DReach-Medium-v0"]
        kwargs.setdefault("goal_center", spec.goal_center)
        kwargs.setdefault("goal_half_extents", spec.goal_half_extents)
        super().__init__(*args, **kwargs)


@register_env("PG3DReach-Workspace-v0", max_episode_steps=100)
class PG3DReachWorkspaceEnv(PG3DReachEnv):
    """Broad Cartesian goal distribution for constraint/reranking policy pretraining."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        spec = REACH_TASK_SPECS["PG3DReach-Workspace-v0"]
        kwargs.setdefault("goal_center", spec.goal_center)
        kwargs.setdefault("goal_half_extents", spec.goal_half_extents)
        super().__init__(*args, **kwargs)


@register_env("PG3DReach-BalancedWorkspace-v0", max_episode_steps=100)
class PG3DReachBalancedWorkspaceEnv(PG3DReachEnv):
    """Mixed practical/workspace distribution for P11 nominal reach reliability."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        spec = REACH_TASK_SPECS["PG3DReach-BalancedWorkspace-v0"]
        kwargs.setdefault("goal_center", spec.goal_center)
        kwargs.setdefault("goal_half_extents", spec.goal_half_extents)
        kwargs.setdefault("goal_regions", spec.goal_regions)
        super().__init__(*args, **kwargs)
