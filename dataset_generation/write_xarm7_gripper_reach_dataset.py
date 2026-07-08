#!/usr/bin/env python
"""Write a pg3d reach dataset for the xArm7 with the xArm parallel-jaw gripper.

Mirrors ``write_xarm7_reach_dataset.py`` but targets the ``xarm7_gripper``
agent (TCP = ``link_tcp``, 172 mm past the arm flange) instead of the bare
no-gripper arm.

Key differences vs. the no-gripper script:
* env-id     → ``PG3DReach-XArm7-Gripper-Workspace-v0``
* robot-uid  → ``xarm7_gripper``
* planner    → ``XArm7GripperMotionPlanningSolver`` (MOVE_GROUP = link_tcp, 7-dim
  arm-only waypoints; the gripper's 8th action dim is filled in by
  ``--gripper-open`` below, not planned)
* action dim → 8 (7 arm + 1 mimic-controlled gripper DOF, ``drive_joint``)
* workspace  → same FK-verified xArm7 bounds (re-verify if TCP offset matters)

Usage:
    python dataset_generation/write_xarm7_gripper_reach_dataset.py \\
        --num-demos 24 --output /scratch2/skills/pg3d_xarm7_gripper_reach.zarr
"""

from __future__ import annotations

import sys

from pg3d.envs.xarm_adapter import register_pg3d_xarm7_gripper_reach_envs
from pg3d.envs.xarm_adapter.reach_config import (
    XARM7_REACH_WORKSPACE_BOUNDS,
    XARM7_WORKSPACE_BOUNDS,
)

from dataset_generation.write_maniskill_reach_dataset import parse_args, run_generation

_XARM7_GRIPPER_DEFAULTS: list[str] = [
    "--env-id", "PG3DReach-XArm7-Gripper-Workspace-v0",
    "--robot-uid", "xarm7_gripper",
    "--table-margin", "0.0",
    "--lateral-z-offset", "0.10",
    "--vertical-lateral-offset", "0.10",
    "--min-curve-offset", "0.08",
    "--waypoint-xy-noise", "0.03",
    "--waypoint-z-noise", "0.02",
    "--min-start-goal-distance", "0.14",
    "--goal-marker-points", "192",
    "--tcp-marker-points", "0",
    "--robot-point-fraction", "1.0",
    # Fed straight into sim_action's gripper column pre-normalization (see
    # _format_sim_action's gripper_action param): this is the NORMALIZED
    # action value in [-1, 1], where +1.0 always means "this controller's
    # configured upper bound" (for XArm7Gripper's mimic controller, upper =
    # _GRIPPER_CLOSED ~0.84 rad) and -1.0 means its lower bound (0.0 rad,
    # fully open). 0.13571428 is drive_joint's real target 0.4769 rad
    # (a semi-open gripper, verified against artifacts/xarm7_gripper_0p4769*
    # -- all 6 gripper joints settle within ~0.001 rad of this target)
    # converted via (0.4769 - mid) / half_range with low=0.0, high=0.84:
    # mid=0.42, half_range=0.42 -> 0.057/0.42 = 0.13571428. Held constant
    # the whole episode by the planner, same as the old closed-gripper
    # behavior -- just a different constant. Do not copy Panda's default
    # (0.04) here: that value is only meaningful in Panda's own
    # [-0.01, 0.04] finger-joint range, not this controller's [0, 0.84].
    "--gripper-open", "0.13571428",
    "--output", "artifacts/pg3d_xarm7_gripper_reach.zarr",
]


def _bounds_flag(name: str, bounds) -> list[str]:
    flat = [f"{float(v):.6f}" for v in bounds.reshape(-1)]
    return [name, *flat]


def main(argv: list[str] | None = None) -> int:
    try:
        from pg3d.envs.xarm_adapter.motionplanner import (
            XArm7GripperMotionPlanningSolver,
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"Failed to import the xArm7 gripper motion planner: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2

    user_argv = list(sys.argv[1:] if argv is None else argv)
    injected = (
        _XARM7_GRIPPER_DEFAULTS
        + _bounds_flag("--reach-workspace-bounds", XARM7_REACH_WORKSPACE_BOUNDS)
        + _bounds_flag("--workspace-bounds", XARM7_WORKSPACE_BOUNDS)
    )
    args = parse_args(injected + user_argv)
    return run_generation(
        args,
        planner_cls=XArm7GripperMotionPlanningSolver,
        register_envs=register_pg3d_xarm7_gripper_reach_envs,
        planner_name="XArm7GripperMotionPlanningSolver.move_to_pose_with_screw",
    )


if __name__ == "__main__":
    raise SystemExit(main())
