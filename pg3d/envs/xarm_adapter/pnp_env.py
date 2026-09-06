"""Standalone pick-and-place scene for xArm7 + gripper: one red cube, one green goal marker.

Deliberately a FRESH scene contract, not a variant of ``pick_env.py`` -- that
one grew out of the reach eval and carries its start/goal-pair marker
conventions with it. Here the scene is exactly two visible things:

* a RED CUBE (``--cube-edge``, default 4cm on a side -- the gripper's
  measured ~5.47cm rest opening can't fit a wider cube) -- a real dynamic
  SAPIEN body with an explicit mass and a high-friction contact material, the
  object to be picked up;
* a GREEN GOAL MARKER (the inherited ``goal_site`` sphere) -- where that cube
  is supposed to end up.

``start_site`` (the red marker sphere the reach envs park at the TCP) is
pushed out of sight below the table on every reset: the cube is the red thing
in this scene, and a second red sphere floating at the gripper only made the
recorded videos harder to read.

Why goal_site stays even though this scene has "no markers": it is NOT
cosmetic. ``PG3DReachEnv._get_obs_extra`` publishes its pose as ``goal_pos``,
which is the goal-conditioning signal the trained reach checkpoint actually
consumes -- deleting it would leave the policy unconditioned. So it stays,
and doubles as the visible green destination marker. Its radius is
``goal_thresh`` (see ``--goal-marker-radius`` on the script).

Both the cube and the goal are resampled every reset, independently, within
the arm's own reach box (inset by ``spawn_margin``) and forced at least
``min_object_separation`` apart so "pick here, place there" is always a real
relocation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import sapien
import torch
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from pg3d.envs.maniskill_adapter.reach_config import REACH_TASK_SPECS
from pg3d.envs.xarm_adapter.reach_config import XARM7_REACH_BOX_BASE
from pg3d.envs.xarm_adapter.reach_env import PG3DReachXArm7GripperEnv, ROBOT_BASE_POSE

# 4cm on a side -> 0.02 half-extent. Was 7cm; the gripper's measured rest
# opening is ~5.47cm (see scripts/pnp_xarm7.py's tune-gripper finger-
# separation check), so a 7cm cube could never physically fit between the
# jaws -- every force/stiffness sweep against it was measuring that
# collision fight, not the gripper's actual clamp behaviour.
DEFAULT_CUBE_EDGE = 0.04

# Cube physics. Mass is set EXPLICITLY rather than left to density: a solid
# 7cm box at water density would come out at ~0.34kg, which needs far more
# clamp force to hold than this gripper can deliver. 0.1kg matches what
# ManiSkill uses for its own build_colorful_cube -- the one cube in their
# codebase actually meant to be grasped -- which sets `builder._mass = 0.1`
# alongside the same high-friction material used below.
DEFAULT_CUBE_MASS = 0.1
CUBE_STATIC_FRICTION = 5.0
CUBE_DYNAMIC_FRICTION = 3.0

CUBE_COLOR = [0.85, 0.12, 0.12, 1.0]  # red -- the object to pick
GOAL_COLOR = [0.10, 0.80, 0.15, 1.0]  # green -- where it goes

# Radius of the green goal marker sphere. Also the base env's success
# threshold (they are the same `goal_thresh` field upstream), but success in
# this scene is judged by the cube's own pose, not by that, so this value is
# chosen purely for visibility.
DEFAULT_GOAL_MARKER_RADIUS = 0.03

# Keep spawns off the very edge of the reach box: the extreme corners are
# near-singular for the arm and leave no room for a tilted approach.
DEFAULT_SPAWN_MARGIN = 0.06

# Minimum XY separation between the cube and the goal, so every episode is a
# genuine relocation rather than "put it back where it already is".
DEFAULT_MIN_OBJECT_SEPARATION = 0.20

# Where start_site gets parked so it is not visible in renders.
_HIDDEN_SITE_POSITION = (0.0, 0.0, -1.0)


class PG3DPnPXArm7GripperEnv(PG3DReachXArm7GripperEnv):
    """xArm7 + gripper, one graspable red cube and one green goal marker."""

    def __init__(
        self,
        *args: Any,
        cube_edge: float = DEFAULT_CUBE_EDGE,
        cube_mass: float = DEFAULT_CUBE_MASS,
        goal_marker_radius: float = DEFAULT_GOAL_MARKER_RADIUS,
        spawn_margin: float = DEFAULT_SPAWN_MARGIN,
        min_object_separation: float = DEFAULT_MIN_OBJECT_SEPARATION,
        **kwargs: Any,
    ) -> None:
        self.cube_edge = float(cube_edge)
        self.cube_half_size = self.cube_edge / 2.0
        self.cube_mass = float(cube_mass)
        self.spawn_margin = float(spawn_margin)
        self.min_object_separation = float(min_object_separation)
        # goal_thresh IS the goal_site sphere's radius upstream -- setting it
        # here is how the green marker gets its size.
        kwargs.setdefault("goal_thresh", float(goal_marker_radius))
        super().__init__(*args, **kwargs)

    # ---------------------------------------------------------------- scene

    def _load_scene(self, options: dict[str, Any]) -> None:
        # Table + goal_site (green) + start_site (red) come from the reach env.
        super()._load_scene(options)

        # The cube is built by hand rather than via actors.build_cube because
        # that helper exposes neither a physics material nor a mass -- it
        # calls add_box_collision with no material at all, so the cube would
        # silently fall back to SAPIEN's default (low) friction and a
        # density-derived mass. Both matter here: this cube has to survive
        # being clamped and lifted.
        builder = self.scene.create_actor_builder()
        cube_material = sapien.pysapien.physx.PhysxMaterial(
            static_friction=CUBE_STATIC_FRICTION,
            dynamic_friction=CUBE_DYNAMIC_FRICTION,
            restitution=0.0,
        )
        builder.add_box_collision(
            half_size=[self.cube_half_size] * 3,
            material=cube_material,
        )
        builder.add_box_visual(
            half_size=[self.cube_half_size] * 3,
            material=sapien.render.RenderMaterial(base_color=CUBE_COLOR),
        )
        # Overwritten every reset in _initialize_episode; only matters for
        # the very first scene build.
        builder.set_initial_pose(sapien.Pose(p=[0.4, 0.0, self.cube_half_size]))
        builder._mass = self.cube_mass
        self.cube = builder.build(name="pnp_cube")

        # Recolour the inherited goal marker green (the reach env builds it
        # green already, but pin it here so this scene's colour contract --
        # red = pick, green = place -- does not depend on that staying true).
        _set_actor_color(self.goal_site, GOAL_COLOR)

    # -------------------------------------------------------------- episode

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict[str, Any]) -> None:
        # Resets the robot to rest_qpos and samples the parent's own goal;
        # both goal_site and start_site are overwritten below.
        super()._initialize_episode(env_idx, options)

        batch_size = len(env_idx)
        table_z = float(ROBOT_BASE_POSE.p[2])
        cube_rest_z = table_z + self.cube_half_size
        bounds = self._spawn_bounds_xy()

        # ManiSkill's own per-episode RNG, so spawns are reproducible from the
        # reset seed exactly like the rest of the episode's randomisation
        # (falls back to a fresh generator only if that attribute ever moves).
        rng = getattr(self, "_episode_rng", None)
        if rng is None:
            rng = np.random.default_rng()
        cube_xy = np.empty((batch_size, 2), dtype=np.float32)
        goal_xy = np.empty((batch_size, 2), dtype=np.float32)
        for i in range(batch_size):
            c, g = _sample_separated_pair(
                rng, bounds=bounds, min_separation=self.min_object_separation
            )
            cube_xy[i], goal_xy[i] = c, g

        with torch.device(self.device):
            cube_p = torch.cat(
                [
                    torch.as_tensor(cube_xy),
                    torch.full((batch_size, 1), cube_rest_z),
                ],
                dim=1,
            )
            # Goal marker sits where the cube's CENTRE should end up, i.e. at
            # the cube's own resting height -- so "cube on the marker" and
            # "cube centre at the marker" are the same statement.
            goal_p = torch.cat(
                [
                    torch.as_tensor(goal_xy),
                    torch.full((batch_size, 1), cube_rest_z),
                ],
                dim=1,
            )
            hidden_p = torch.tensor(_HIDDEN_SITE_POSITION).repeat(batch_size, 1)

        self.cube.set_pose(Pose.create_from_pq(cube_p))
        self.goal_site.set_pose(Pose.create_from_pq(goal_p))
        # Park the red start marker out of sight: the cube is this scene's
        # red object, and a second red sphere at the TCP only confuses the
        # recorded video.
        self.start_site.set_pose(Pose.create_from_pq(hidden_p))

    # --------------------------------------------------------------- helpers

    def _spawn_bounds_xy(self) -> np.ndarray:
        """World-frame [2,2] XY box objects may spawn in (reach box, inset)."""
        box = np.asarray(XARM7_REACH_BOX_BASE, dtype=np.float32)[:2].copy()
        box[:, 0] += self.spawn_margin
        box[:, 1] -= self.spawn_margin
        base_xy = np.asarray(ROBOT_BASE_POSE.p[:2], dtype=np.float32).reshape(2, 1)
        return (box + base_xy).astype(np.float32)


def _set_actor_color(actor: Any, color: list[float]) -> None:
    """Best-effort recolour of an already-built actor's render shapes."""
    try:
        for render_body in actor._objs:
            for part in render_body.find_component_by_type(
                sapien.render.RenderBodyComponent
            ).render_shapes:
                part.material.set_base_color(color)
    except Exception:  # noqa: BLE001 -- purely cosmetic, never fail a build over it
        pass


def _sample_separated_pair(
    rng: np.random.Generator,
    *,
    bounds: np.ndarray,
    min_separation: float,
    max_attempts: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Two XY points in `bounds`, at least `min_separation` apart.

    Falls back to the farthest pair seen rather than looping forever -- if
    `min_separation` is set larger than the box's own diagonal it is simply
    unsatisfiable, and silently returning the best available beats hanging.
    """
    best: tuple[np.ndarray, np.ndarray] | None = None
    best_distance = -1.0
    for _ in range(max_attempts):
        a = rng.uniform(bounds[:, 0], bounds[:, 1]).astype(np.float32)
        b = rng.uniform(bounds[:, 0], bounds[:, 1]).astype(np.float32)
        distance = float(np.linalg.norm(a - b))
        if distance >= min_separation:
            return a, b
        if distance > best_distance:
            best, best_distance = (a, b), distance
    assert best is not None
    return best


def cube_position(env: Any) -> np.ndarray:
    """Ground-truth world position of the cube (GPU/CPU-safe)."""
    from pg3d.utils.arrays import to_numpy

    cube = getattr(env.unwrapped, "cube", None)
    if cube is None:
        raise AttributeError("env.unwrapped has no 'cube' -- wrong env id?")
    return to_numpy(cube.pose.p).reshape(-1, 3)[0].astype(np.float32)


def goal_position(env: Any) -> np.ndarray:
    """Ground-truth world position of the green goal marker (GPU/CPU-safe)."""
    from pg3d.utils.arrays import to_numpy

    return to_numpy(env.unwrapped.goal_site.pose.p).reshape(-1, 3)[0].astype(np.float32)


@register_env("PG3DPnP-XArm7-Gripper-v0", max_episode_steps=400)
class PG3DPnPXArm7GripperWorkspaceEnv(PG3DPnPXArm7GripperEnv):
    """Registered scene: red cube + green goal marker, broad workspace."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        spec = REACH_TASK_SPECS["PG3DReach-Workspace-v0"]
        kwargs.setdefault("goal_center", spec.goal_center)
        kwargs.setdefault("goal_half_extents", spec.goal_half_extents)
        super().__init__(*args, **kwargs)


def register_pnp_envs() -> None:
    """Register ``PG3DPnP-XArm7-Gripper-v0``. Safe to call repeatedly."""
    from pg3d.envs.xarm_adapter import pnp_env  # noqa: F401
