"""xArm7 + gripper pick env: adds one dynamic, graspable cube to the reach scene.

Everything else (robot, table, origin, camera, workspace) is inherited
unchanged from :class:`PG3DReachXArm7GripperEnv` -- this module ONLY adds a
physical cube actor, so the reach-only policy, planner, and workspace tuning
done for the reach task all still apply here without modification.

Design note (see scripts/eval_pose_variety_pick_and_place.py, which is the
actual consumer of this env): the checkpoint being evaluated was trained as a
REACH policy -- it has no gripper output and never saw any object in its
point cloud if the dataset was generated with --robot-point-fraction 1.0 (see
XARM7_REACH_BOX_BASE / crop_point_cloud). This env does not change that. The
cube exists so a pick-and-place SCRIPT can (a) point the reach goal
conditioning at a pose located at the cube, (b) let the already-trained
reach/reranking machinery steer there exactly as it does for an abstract
goal point, then (c) hardcode gripper-close + lift once reach converges, and
(d) verify the grasp actually worked by reading the cube's own SAPIEN pose
before/after -- the env's only job is to provide that real, physically
simulated cube to grasp and to report its ground-truth pose.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import sapien
import torch
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from pg3d.envs.maniskill_adapter.reach_config import REACH_TASK_SPECS
from pg3d.envs.xarm_adapter.reach_config import XARM7_REACH_BOX_BASE
from pg3d.envs.xarm_adapter.reach_env import PG3DReachXArm7GripperEnv, ROBOT_BASE_POSE

# Default cube size. NOT verified against the real xArm gripper's actual jaw
# stroke/opening width -- that spec isn't recorded anywhere in this repo (see
# agents.py's XArm7Gripper docstring, which documents TCP offset and control
# gains but not finger travel in mm). 0.02m half-size (4cm cube) is a
# conservative small-object default; re-tune --cube-half-size against the
# real gripper's measured max opening (with margin) before trusting a grasp
# attempt sim-to-real.
DEFAULT_CUBE_HALF_SIZE = 0.02

# Radius of the "place" landing-zone marker -- the cube must be placed
# (grasped, moved, released) so it ends up resting somewhere within this
# radius of the sampled place target, measured on the table plane.
DEFAULT_PLACE_RADIUS = 0.07


class PG3DPickXArm7GripperEnv(PG3DReachXArm7GripperEnv):
    """xArm7 + gripper reach env with one dynamic, graspable cube added.

    The cube is a real ``body_type="dynamic"`` SAPIEN actor with collision --
    it can be pushed, grasped, and lifted like any rigid body, and its pose
    can be read back (``self.cube.pose``) as ground truth for grasp-success
    checking. Standalone (no external override) it randomizes to a fresh XY
    position on the table every reset; scripts/eval_pose_variety_pick_and_place.py
    overrides that per-episode via the same batched ``Pose.create_from_pq``
    technique dataset-gen/eval scripts already use for start_site/goal_site,
    for full control over which position is tested.
    """

    def __init__(
        self,
        *args: Any,
        cube_half_size: float = DEFAULT_CUBE_HALF_SIZE,
        place_radius: float = DEFAULT_PLACE_RADIUS,
        **kwargs: Any,
    ) -> None:
        self.cube_half_size = float(cube_half_size)
        self.place_radius = float(place_radius)
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict[str, Any]) -> None:
        super()._load_scene(options)
        # Rest pose is overwritten every episode in _initialize_episode below;
        # this initial_pose only matters for the very first scene build.
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[0.85, 0.15, 0.15, 1.0],
            name="pick_cube",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0.4, 0.0, self.cube_half_size]),
        )
        # Place-target landing zone: a thin, flat, kinematic disk resting on
        # the table (no collision -- purely a visual/goal marker, same as
        # start_site/goal_site; actual place SUCCESS is checked numerically
        # by the eval script via the cube's own final XY distance to this
        # marker's center, not by any collision/contact against this actor).
        # SAPIEN's cylinder primitive is defined with its length along local
        # +Z by default, so a small half_length here + identity orientation
        # gives a flat disk with its circular face up, as seen from above.
        self.place_site = actors.build_cylinder(
            self.scene,
            radius=self.place_radius,
            half_length=0.002,
            color=[0.2, 0.55, 0.9, 0.45],
            name="place_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0.4, 0.2, 0.0]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict[str, Any]) -> None:
        super()._initialize_episode(env_idx, options)
        # Table surface is at the same world Z as the robot base -- see the
        # "height (dz) above the base/table surface" convention documented on
        # XARM7_REACH_BOX_BASE in reach_config.py (dz_lo=0.05 there means "5cm
        # above the table", i.e. base_z IS table_z). Cube center sits
        # half_size above that so its bottom face rests exactly on the table.
        table_z = float(ROBOT_BASE_POSE.p[2])
        rest_z = table_z + self.cube_half_size
        base_xy = np.asarray(ROBOT_BASE_POSE.p[:2], dtype=np.float32)
        dx_lo, dx_hi = XARM7_REACH_BOX_BASE[0]
        dy_lo, dy_hi = XARM7_REACH_BOX_BASE[1]
        batch_size = len(env_idx)
        with torch.device(self.device):
            xy = torch.rand((batch_size, 2)) * torch.tensor(
                [dx_hi - dx_lo, dy_hi - dy_lo]
            ) + torch.tensor([dx_lo, dy_lo])
            xy = xy + torch.tensor(base_xy.tolist())
            z = torch.full((batch_size, 1), rest_z)
            cube_p = torch.cat([xy, z], dim=1)
        self.cube.set_pose(Pose.create_from_pq(cube_p))


def set_cube_pose(env: Any, position: np.ndarray) -> None:
    """Explicitly place the cube at ``position`` (world frame), identity yaw.

    Uses the same batched ``Pose.create_from_pq`` form the reach eval scripts
    already rely on for start_site/goal_site -- a raw ``sapien.Pose(p=...)``
    does not reliably land in the GPU rigid-body buffer on this sim backend
    (see ``_set_site_pose`` in eval_pose_variety_checkpoint_reachability.py).
    """
    cube = getattr(env.unwrapped, "cube", None)
    if cube is None:
        return
    cube.set_pose(Pose.create_from_pq(np.asarray(position, dtype=np.float32).reshape(1, 3)))


@register_env("PG3DPick-XArm7-Gripper-Workspace-v0", max_episode_steps=200)
class PG3DPickXArm7GripperWorkspaceEnv(PG3DPickXArm7GripperEnv):
    """Broad workspace goal distribution, xArm7 + gripper + graspable cube."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        spec = REACH_TASK_SPECS["PG3DReach-Workspace-v0"]
        kwargs.setdefault("goal_center", spec.goal_center)
        kwargs.setdefault("goal_half_extents", spec.goal_half_extents)
        super().__init__(*args, **kwargs)


def register_pg3d_xarm7_pick_envs() -> None:
    """Register ``PG3DPick-XArm7-Gripper-*`` env ids. Safe to call repeatedly.

    Mirrors pg3d.envs.xarm_adapter.__init__'s register_* functions exactly:
    the @register_env decorator above only runs once no matter how many times
    this function is called, because Python only executes a module's
    top-level code on its FIRST import -- every call after that just returns
    the cached module object.
    """
    from pg3d.envs.xarm_adapter import pick_env  # noqa: F401
