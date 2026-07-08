"""xArm7 no-gripper reach envs for pg3d data generation.

These reuse the Panda :class:`PG3DReachEnv` task logic wholesale (goal sampling,
success, reward, start/goal sites) and only swap the robot to ``xarm7_nogripper``.
Importing this module registers both the agent and the env ids.

NOTE: ``ROBOT_BASE_POSE`` and the goal workspace are inherited from the Panda
config for now; xArm7's reach envelope and base footprint differ, so re-tune them
when you move past the spawn smoke test (marked TODO).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import sapien
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from scipy.spatial.transform import Rotation as _Rotation

from pg3d.envs.maniskill_adapter.reach_config import REACH_TASK_SPECS
from pg3d.envs.maniskill_adapter.reach_env import PG3DReachEnv
from pg3d.envs.xarm_adapter.agents import (  # noqa: F401 - registers agents
    XArm7Gripper,
    XArm7NoGripper,
    XArm7Robotiq,
)
from pg3d.envs.xarm_adapter.reach_config import (
    XARM7_CAM_CALIB_ERROR_ROTATION_STD_DEG,
    XARM7_CAM_CALIB_ERROR_TRANSLATION_STD_M,
    XARM7_CAM_POSITION_JITTER_M,
    XARM7_CAM_Q_WXYZ,
    XARM7_CAM_ROTATION_JITTER_DEG,
    XARM7_CAM_T_BASE,
    XARM7_CAM_VFOV_RAD,
    XARM7_SIM_CAM_HEIGHT,
    XARM7_SIM_CAM_WIDTH,
)

ROBOT_BASE_POSE = sapien.Pose(p=[-0.615, 0.0, 0.0])




class PG3DReachXArm7Env(PG3DReachEnv):
    """xArm7 (7-DoF, no gripper) reach base."""

    SUPPORTED_ROBOTS = ["xarm7_nogripper"]

    def __init__(self, *args: Any, robot_uids: str = "xarm7_nogripper", **kwargs: Any) -> None:
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: dict[str, Any]) -> None:
        # Bypass PG3DReachEnv's Panda base pose; place the xArm7 base instead.
        BaseEnv._load_agent(self, options, ROBOT_BASE_POSE)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict[str, Any]) -> None:
        # TableSceneBuilder.initialize() only handles "panda" / "xarm6_robotiq" / etc. by name.
        # For all xArm7 variants (no-gripper, gripper, robotiq), apply the rest keyframe here
        # so the robot is NOT left at the URDF-default zeros pose. At zeros, xArm7 is near a
        # kinematic singularity and the PD controller causes immediate simulation explosion.
        rest_qpos = self.agent.keyframes["rest"].qpos
        b = len(env_idx)
        qpos = torch.tensor(rest_qpos, dtype=torch.float32, device=self.device).unsqueeze(0).expand(b, -1)
        self.agent.reset(qpos)
        super()._initialize_episode(env_idx, options)
        self._randomize_camera_pose()

    def _randomize_camera_pose(self) -> None:
        """Jitter base_camera's pose each episode -- position by up to +/-10cm and
        orientation by up to +/-2deg, uniform and independent per axis -- so
        training data covers a range of camera viewpoints instead of one fixed,
        perfectly-known pose. Called once per episode (from _initialize_episode);
        the resulting pose is held fixed (no per-step re-jitter) for the rest of
        the episode. Also (re-)samples this episode's calibration-error
        correction -- see _sample_camera_calibration_error. No-op if the current
        obs mode didn't build any cameras.

        set_local_pose only accepts one unbatched pose (applies to every parallel
        sub-scene camera equally) — fine since data-gen always runs num_envs=1.
        """
        camera = self._sensors.get("base_camera")
        if camera is None:
            self._camera_calib_correction = None
            return
        with torch.device(self.device):
            delta_p = (torch.rand(3) * 2.0 - 1.0) * XARM7_CAM_POSITION_JITTER_M
            delta_euler_deg = (torch.rand(3) * 2.0 - 1.0) * XARM7_CAM_ROTATION_JITTER_DEG
        nominal_p = np.array(ROBOT_BASE_POSE.p) + XARM7_CAM_T_BASE
        nominal_rot = _Rotation.from_quat(XARM7_CAM_Q_WXYZ, scalar_first=True)
        delta_rot = _Rotation.from_euler("xyz", delta_euler_deg.cpu().numpy(), degrees=True)
        self._camera_episode_p = nominal_p + delta_p.cpu().numpy()
        self._camera_episode_rot = nominal_rot * delta_rot
        camera.camera.set_local_pose(
            sapien.Pose(
                p=self._camera_episode_p.tolist(),
                q=self._camera_episode_rot.as_quat(scalar_first=True).tolist(),
            )
        )
        self._camera_calib_correction = self._sample_camera_calibration_error()

    def _sample_camera_calibration_error(self) -> tuple[np.ndarray, np.ndarray]:
        """Sample this episode's fixed calibration-error correction.

        Real eye-to-hand calibration is never perfectly accurate: even after
        calibrating, the camera pose used to convert depth into world/robot-frame
        points has residual error (measured RMS: see XARM7_CAM_CALIB_ERROR_* in
        reach_config.py). This does NOT move the physical/rendering camera --
        base_camera still renders from self._camera_episode_{p,rot} exactly, set
        above. Instead it models a second, independent camera pose (the
        mis-calibrated "belief") only used when interpreting depth into world
        coordinates -- see the get_obs override below, which re-projects
        already-world-frame points from the true pose's frame into the noisy
        pose's frame. Sampled once per episode (this method is only called from
        _randomize_camera_pose) and held fixed for the whole episode, since real
        calibration error doesn't change moment to moment within one recording.

        Returns (R, t) such that world_point_noisy = R @ world_point_true + t.
        """
        with torch.device(self.device):
            delta_p = torch.randn(3) * XARM7_CAM_CALIB_ERROR_TRANSLATION_STD_M
            delta_euler_deg = torch.randn(3) * XARM7_CAM_CALIB_ERROR_ROTATION_STD_DEG
        true_p = self._camera_episode_p
        true_rot = self._camera_episode_rot
        noisy_p = true_p + delta_p.cpu().numpy()
        noisy_rot = true_rot * _Rotation.from_euler("xyz", delta_euler_deg.cpu().numpy(), degrees=True)
        r_correction = (noisy_rot * true_rot.inv()).as_matrix().astype(np.float32)
        t_correction = (noisy_p - r_correction @ true_p).astype(np.float32)
        return r_correction, t_correction

    def get_obs(self, info: dict[str, Any] | None = None, unflattened: bool = False) -> Any:
        obs = super().get_obs(info, unflattened=True)
        correction = getattr(self, "_camera_calib_correction", None)
        if correction is not None and "pointcloud" in obs:
            r_correction, t_correction = correction
            xyzw = obs["pointcloud"]["xyzw"]
            xyz = xyzw[..., :3]
            r_t = torch.as_tensor(r_correction, dtype=xyz.dtype, device=xyz.device)
            t_t = torch.as_tensor(t_correction, dtype=xyz.dtype, device=xyz.device)
            xyzw[..., :3] = xyz @ r_t.transpose(0, 1) + t_t
        return obs if unflattened else self._flatten_raw_obs(obs)

    @property
    def _default_sensor_configs(self) -> list[CameraConfig]:
        # Camera pose from eye-to-hand calibration on real xArm7.
        # Position: robot base frame → world frame by adding ROBOT_BASE_POSE.p.
        # Orientation: SAPIEN [w,x,y,z] quaternion converted from the calibration
        # rotation matrix. The calibration script outputs an OpenCV/pinhole-optical
        # rotation (+z forward, +y down, +x right); SAPIEN's own convention is
        # (forward, right, up) = (+x, -y, +z) (see mani_skill.utils.sapien_utils.
        # look_at docstring) — a different axis assignment, not a relabeling. The
        # conversion lives in reach_config._opencv_camera_rotation_to_sapien;
        # passing the raw OpenCV matrix straight into SAPIEN silently pointed the
        # camera ~90° away from the workspace (confirmed empirically: 0/16384 raw
        # points ever landed inside the crop bounds before this was fixed).
        cam_p = (np.array(ROBOT_BASE_POSE.p) + XARM7_CAM_T_BASE).tolist()
        pose = sapien.Pose(p=cam_p, q=XARM7_CAM_Q_WXYZ.tolist())
        # near/far match RealSense D455 practical range (0.4 m – 6 m).
        return [CameraConfig("base_camera", pose, XARM7_SIM_CAM_WIDTH, XARM7_SIM_CAM_HEIGHT, XARM7_CAM_VFOV_RAD, 0.4, 6.0)]

    @property
    def _default_human_render_camera_configs(self) -> CameraConfig:
        # Front-facing third-person view: camera at world [0.7, 0, 0.45] looking toward
        # workspace center [-0.315, 0, 0.22].  Y=0 so no diagonal offset.
        pose = sapien_utils.look_at(eye=[0.7, 0.0, 0.45], target=[-0.315, 0.0, 0.22])
        return CameraConfig("render_camera", pose, 640, 480, float(np.deg2rad(60)), 0.1, 10.0)


class PG3DReachXArm7GripperEnv(PG3DReachXArm7Env):
    """xArm7 + xArm parallel-jaw gripper reach base.

    Gripper is passive (always open); TCP = ``link_tcp`` (172 mm past the arm
    flange). Action space stays 7-dim (arm joints only). Use for reach datasets
    where the gripper body is needed for sim-to-real point-cloud fidelity.
    """

    SUPPORTED_ROBOTS = ["xarm7_gripper"]

    def __init__(self, *args: Any, robot_uids: str = "xarm7_gripper", **kwargs: Any) -> None:
        # Skip PG3DReachXArm7Env.__init__ which hardcodes xarm7_nogripper,
        # go straight to PG3DReachEnv which accepts robot_uids generically.
        super(PG3DReachXArm7Env, self).__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: dict[str, Any]) -> None:
        BaseEnv._load_agent(self, options, ROBOT_BASE_POSE)


@register_env("PG3DReach-XArm7-Workspace-v0", max_episode_steps=100)
class PG3DReachXArm7WorkspaceEnv(PG3DReachXArm7Env):
    """Broad Cartesian goal distribution (xArm7), mirrors PG3DReach-Workspace-v0."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        spec = REACH_TASK_SPECS["PG3DReach-Workspace-v0"]
        kwargs.setdefault("goal_center", spec.goal_center)
        kwargs.setdefault("goal_half_extents", spec.goal_half_extents)
        super().__init__(*args, **kwargs)


@register_env("PG3DReach-XArm7-BalancedWorkspace-v0", max_episode_steps=100)
class PG3DReachXArm7BalancedWorkspaceEnv(PG3DReachXArm7Env):
    """Mixed practical/workspace distribution (xArm7), mirrors the balanced Panda env."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        spec = REACH_TASK_SPECS["PG3DReach-BalancedWorkspace-v0"]
        kwargs.setdefault("goal_center", spec.goal_center)
        kwargs.setdefault("goal_half_extents", spec.goal_half_extents)
        kwargs.setdefault("goal_regions", spec.goal_regions)
        super().__init__(*args, **kwargs)


class PG3DReachXArm7RobotiqEnv(PG3DReachXArm7Env):
    """xArm7 + Robotiq 2F-85 gripper reach base.

    Gripper is passive (always open); TCP = ``eef`` (150 mm past the arm flange).
    Action space stays 7-dim (arm joints only). Drop-in for the xArm-gripper env
    but with watertight Robotiq collision meshes, so mplib planning works.
    """

    SUPPORTED_ROBOTS = ["xarm7_robotiq"]

    def __init__(self, *args: Any, robot_uids: str = "xarm7_robotiq", **kwargs: Any) -> None:
        super(PG3DReachXArm7Env, self).__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: dict[str, Any]) -> None:
        BaseEnv._load_agent(self, options, ROBOT_BASE_POSE)


@register_env("PG3DReach-XArm7-Gripper-Workspace-v0", max_episode_steps=100)
class PG3DReachXArm7GripperWorkspaceEnv(PG3DReachXArm7GripperEnv):
    """Broad workspace goal distribution for xArm7 + gripper (TCP = link_tcp)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        spec = REACH_TASK_SPECS["PG3DReach-Workspace-v0"]
        kwargs.setdefault("goal_center", spec.goal_center)
        kwargs.setdefault("goal_half_extents", spec.goal_half_extents)
        super().__init__(*args, **kwargs)


@register_env("PG3DReach-XArm7-Gripper-BalancedWorkspace-v0", max_episode_steps=100)
class PG3DReachXArm7GripperBalancedWorkspaceEnv(PG3DReachXArm7GripperEnv):
    """Balanced goal distribution for xArm7 + gripper (TCP = link_tcp)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        spec = REACH_TASK_SPECS["PG3DReach-BalancedWorkspace-v0"]
        kwargs.setdefault("goal_center", spec.goal_center)
        kwargs.setdefault("goal_half_extents", spec.goal_half_extents)
        kwargs.setdefault("goal_regions", spec.goal_regions)
        super().__init__(*args, **kwargs)


@register_env("PG3DReach-XArm7-Robotiq-Workspace-v0", max_episode_steps=100)
class PG3DReachXArm7RobotiqWorkspaceEnv(PG3DReachXArm7RobotiqEnv):
    """Broad workspace goal distribution for xArm7 + Robotiq 2F-85 (TCP = eef)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        spec = REACH_TASK_SPECS["PG3DReach-Workspace-v0"]
        kwargs.setdefault("goal_center", spec.goal_center)
        kwargs.setdefault("goal_half_extents", spec.goal_half_extents)
        super().__init__(*args, **kwargs)


@register_env("PG3DReach-XArm7-Robotiq-BalancedWorkspace-v0", max_episode_steps=100)
class PG3DReachXArm7RobotiqBalancedWorkspaceEnv(PG3DReachXArm7RobotiqEnv):
    """Balanced goal distribution for xArm7 + Robotiq 2F-85 (TCP = eef)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        spec = REACH_TASK_SPECS["PG3DReach-BalancedWorkspace-v0"]
        kwargs.setdefault("goal_center", spec.goal_center)
        kwargs.setdefault("goal_half_extents", spec.goal_half_extents)
        kwargs.setdefault("goal_regions", spec.goal_regions)
        super().__init__(*args, **kwargs)

