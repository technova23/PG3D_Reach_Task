#!/usr/bin/env python
"""Verify the xArm7 reach envelope with an mplib-IK reachability sweep.

The reach/crop boxes in ``pg3d.envs.xarm_adapter.reach_config`` are documented as
"FK-verified" — this script is the runnable artifact behind that claim. It stands up
the ``xarm7_nogripper`` env, builds the same mplib planner used for data generation,
and sweeps a grid of target positions (at the reach task's downward tabletop
orientation) through mplib IK. For each box it reports:

* corner reachability  — are all 8 corners of the sampling box IK-solvable?
* grid reachability     — fraction of an NxNxN interior grid that is IK-solvable.

Use it to (re)tune ``XARM7_REACH_BOX_BASE`` after any change to the base pose, the
rest keyframe, the URDF, or the TCP link: shrink the box until corners hit 100% and
the interior stays high, or grow it while corners stay reachable.

Reachability is defined against the *rest-pose TCP orientation* (z-axis down,
quaternion [0,1,0,0] wxyz) — the orientation every reach goal is planned to, so IK
feasibility here is a faithful proxy for "can the data-gen planner service a goal
placed at this point."

Usage:
    # Verify the shipped reach box + max envelope at a 5x5x5 interior grid
    python scripts/verify_xarm7_reachability.py

    # Denser grid, also dump per-point results to CSV
    python scripts/verify_xarm7_reachability.py --grid 9 --csv artifacts/xarm7_reach_sweep.csv

    # Sweep the gripper variant instead (TCP = link_tcp)
    python scripts/verify_xarm7_reachability.py --variant gripper
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Rest-pose TCP orientation (wxyz): 180 deg about x -> tool z-axis points down at the
# table. This is what agent.tcp reports at the rest keyframe and what every reach goal
# inherits, so we test IK feasibility against exactly this orientation.
_DOWN_QUAT_WXYZ = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)


def _corners(box: np.ndarray) -> np.ndarray:
    """8 corners of a base-relative [3,2] box, shape (8,3)."""
    xs, ys, zs = box
    grid = np.array(np.meshgrid(xs, ys, zs, indexing="ij")).reshape(3, -1).T
    return grid.astype(np.float64)


def _grid_points(box: np.ndarray, n: int) -> np.ndarray:
    """N x N x N interior grid over a base-relative [3,2] box, shape (n^3, 3)."""
    axes = [np.linspace(lo, hi, n) for lo, hi in box]
    grid = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
    return grid.astype(np.float64)


def _sweep(planner: Any, base_pos: np.ndarray, start_qpos: np.ndarray,
           pts_base: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """Return a bool mask: is each base-relative point IK-reachable (world frame)?

    mplib's base pose is already set on the planner, so IK goal poses are given in
    world frame = base_pos + base-relative offset.
    """
    reachable = np.zeros(len(pts_base), dtype=bool)
    for i, dp in enumerate(pts_base):
        world_p = base_pos + dp
        goal = np.hstack([world_p, quat_wxyz]).astype(np.float64)
        status, _ = planner.IK(goal, start_qpos, n_init_qpos=20, threshold=1e-3)
        reachable[i] = status == "Success"
    return reachable


def _build_planner(variant: str):
    """Register + make the env and return (planner, base_pos, start_qpos, boxes)."""
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    if variant == "gripper":
        from pg3d.envs.xarm_adapter import register_pg3d_xarm7_gripper_reach_envs
        from pg3d.envs.xarm_adapter.motionplanner import XArm7GripperMotionPlanningSolver as Solver
        register_pg3d_xarm7_gripper_reach_envs()
        env_id, robot_uid = "PG3DReach-XArm7-Gripper-Workspace-v0", "xarm7_gripper"
    else:
        from pg3d.envs.xarm_adapter import register_pg3d_xarm7_reach_envs
        from pg3d.envs.xarm_adapter.motionplanner import XArm7NoGripperMotionPlanningSolver as Solver
        register_pg3d_xarm7_reach_envs()
        env_id, robot_uid = "PG3DReach-XArm7-Workspace-v0", "xarm7_nogripper"

    env = gym.make(env_id, obs_mode="none", render_mode="rgb_array",
                   robot_uids=robot_uid, num_envs=1)
    env.reset(seed=0)
    u = env.unwrapped

    planner = Solver(
        env, debug=False, vis=False,
        base_pose=u.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
    )
    base_pos = np.asarray(u.agent.robot.pose.p).reshape(-1)[:3].astype(np.float64)
    # mplib plans over the arm's active joints; the rest keyframe is the natural IK seed.
    # planner.IK expects a qpos of len(move_group_joints); slice to the arm DOFs mplib
    # actually plans over (gripper joints, if any, are side branches not in the chain).
    n_ik = len(planner.planner.user_joint_names)
    start_qpos = np.asarray(u.agent.robot.get_qpos()).reshape(-1)[:n_ik].astype(np.float64)
    # Return the raw mplib planner (it owns .IK); the solver just wraps it.
    return env, planner, planner.planner, base_pos, start_qpos


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        from pg3d.envs.xarm_adapter.reach_config import (
            XARM7_MAX_ENVELOPE_BASE,
            XARM7_REACH_BOX_BASE,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to import reach_config: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    try:
        env, solver, planner, base_pos, start_qpos = _build_planner(args.variant)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to build planner: {type(exc).__name__}: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    boxes = {
        "REACH_BOX (sampling)": XARM7_REACH_BOX_BASE,
        "MAX_ENVELOPE (reference)": XARM7_MAX_ENVELOPE_BASE,
    }

    csv_rows: list[str] = ["box,dx,dy,dz,reachable"]
    try:
        print(f"\nxArm7 reachability sweep — variant={args.variant}, "
              f"base={base_pos.tolist()}, orient(wxyz)={_DOWN_QUAT_WXYZ.tolist()}")
        print(f"IK seed = rest qpos, grid = {args.grid}^3\n")

        for name, box in boxes.items():
            corners = _corners(box)
            corner_mask = _sweep(planner, base_pos, start_qpos, corners, _DOWN_QUAT_WXYZ)
            grid = _grid_points(box, args.grid)
            grid_mask = _sweep(planner, base_pos, start_qpos, grid, _DOWN_QUAT_WXYZ)

            print(f"── {name}")
            print(f"   box (base-rel): dx{box[0].tolist()} dy{box[1].tolist()} dz{box[2].tolist()}")
            print(f"   corners : {corner_mask.sum()}/8 reachable"
                  f"{'  ✓ all corners OK' if corner_mask.all() else '  ✗ some corners UNREACHABLE'}")
            print(f"   grid    : {grid_mask.sum()}/{len(grid_mask)} "
                  f"({100*grid_mask.mean():.1f}%) reachable\n")

            if args.csv:
                tag = name.split()[0]
                for dp, ok in zip(grid, grid_mask):
                    csv_rows.append(f"{tag},{dp[0]:.4f},{dp[1]:.4f},{dp[2]:.4f},{int(ok)}")
    finally:
        solver.close()
        env.close()

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(csv_rows) + "\n")
        print(f"wrote per-point results: {out}")

    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="mplib-IK reachability sweep for the xArm7 reach envelope.")
    p.add_argument("--variant", choices=["nogripper", "gripper"], default="nogripper",
                   help="Which xArm7 agent/TCP to sweep (default: nogripper, TCP=link_eef).")
    p.add_argument("--grid", type=int, default=5, help="Interior grid resolution per axis (N^3 points).")
    p.add_argument("--csv", type=str, default=None, help="Optional path to dump per-point reachability CSV.")
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
