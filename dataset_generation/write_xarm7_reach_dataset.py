#!/usr/bin/env python
"""Write a multimodal pg3d reach dataset for the no-gripper xArm7.

Thin entrypoint over the Panda writer: it reuses the entire sampling / motion-plan
replay / zarr-writing pipeline (``run_generation`` and helpers in
``write_maniskill_reach_dataset``) and only swaps the robot-specific pieces:

* motion planner -> ``XArm7NoGripperMotionPlanningSolver`` (mplib, gripper-agnostic),
* env / robot    -> ``PG3DReach-XArm7-Workspace-v0`` / ``xarm7_nogripper``,
* action dim     -> 7 (no gripper column; the shared action builders are already
  action-dim-aware, so ``--gripper-open`` is simply inert here),
* workspace      -> xArm7's FK-verified, base-relative, table-agnostic bounds
  (``pg3d.envs.xarm_adapter.reach_config``), symmetric left/right of the base,
* trajectory families -> detour magnitudes scaled down for xArm7's shallower /
  shorter reach box (overridable on the CLI).

xArm7-specific values are injected as *defaults* (prepended to argv) so any flag the
user passes still wins. Example:

    python dataset_generation/write_xarm7_reach_dataset.py \
        --num-demos 24 --output /scratch2/skills/pg3d_xarm7_reach_final.zarr
"""

from __future__ import annotations

import sys

from pg3d.envs.xarm_adapter import register_pg3d_xarm7_reach_envs
from pg3d.envs.xarm_adapter.reach_config import (
    XARM7_REACH_WORKSPACE_BOUNDS,
    XARM7_WORKSPACE_BOUNDS,
)

from dataset_generation.write_maniskill_reach_dataset import parse_args, run_generation

# xArm7 reach box is shallower in x (0.40 m vs Panda 0.84) and shorter in z (0.35 vs
# 0.52), same width in y. Detour offsets / noise are scaled down so waypoint families
# stay inside the reachable box; min start-goal distance trimmed for the shallower
# workspace. These are empirical starting points — tune from per-seed family yield.
_XARM7_DEFAULTS: list[str] = [
    "--env-id", "PG3DReach-XArm7-Workspace-v0",
    "--robot-uid", "xarm7_nogripper",
    "--table-margin", "0.0",          # table-agnostic: bounds anchored to the base, not table edges
    "--lateral-z-offset", "0.10",     # was 0.15 (less vertical room)
    "--vertical-lateral-offset", "0.10",
    "--min-curve-offset", "0.08",     # was 0.10
    "--waypoint-xy-noise", "0.03",    # was 0.04
    "--waypoint-z-noise", "0.02",     # was 0.025
    "--min-start-goal-distance", "0.14",  # was 0.16 (shallower workspace)
    "--output", "/scratch2/skills/pg3d_xarm7_reach_final.zarr",
]


def _bounds_flag(name: str, bounds) -> list[str]:
    """Flatten a [3,2] world-frame bounds box to the writer's nargs=6 flag order."""
    flat = [f"{float(v):.6f}" for v in bounds.reshape(-1)]
    return [name, *flat]


def main(argv: list[str] | None = None) -> int:
    try:
        from pg3d.envs.xarm_adapter.motionplanner import (
            XArm7NoGripperMotionPlanningSolver,
        )
    except Exception as exc:  # noqa: BLE001 - surface a clear install hint
        print(
            f"Failed to import the xArm7 motion planner (mplib/ManiSkill): "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2

    user_argv = list(sys.argv[1:] if argv is None else argv)
    injected = (
        _XARM7_DEFAULTS
        + _bounds_flag("--reach-workspace-bounds", XARM7_REACH_WORKSPACE_BOUNDS)
        + _bounds_flag("--workspace-bounds", XARM7_WORKSPACE_BOUNDS)
    )
    # User-supplied flags come last so argparse lets them override the xArm7 defaults.
    args = parse_args(injected + user_argv)
    return run_generation(
        args,
        planner_cls=XArm7NoGripperMotionPlanningSolver,
        register_envs=register_pg3d_xarm7_reach_envs,
        planner_name="XArm7NoGripperMotionPlanningSolver.move_to_pose_with_screw",
    )


if __name__ == "__main__":
    raise SystemExit(main())
