#!/usr/bin/env python
"""Verify the Panda reach workspace with an mplib-IK reachability sweep.

Same methodology as ``scripts/verify_xarm7_reachability.py``, applied to the Panda
arm, so the two are directly comparable: it stands up the default Panda reach env,
builds the same ``PandaArmMotionPlanningSolver`` used for data generation, and
sweeps the box's 8 corners + an NxNxN interior grid through mplib IK at the reach
task's downward tabletop orientation (quaternion [0,1,0,0] wxyz).

The box checked is ``_default_reach_workspace_bounds``'s hardcoded numbers from
``dataset_generation/write_maniskill_reach_dataset.py`` (dx in [-0.42,0.42],
dy in [-0.45,0.45], dz in [0.20,0.72]) -- the box referenced throughout this repo
as "Panda-comparable" when sizing the xArm7 workspace. This script exists to check
whether that reference box itself is actually 8/8 corners + ~100% interior
reachable for Panda, the same bar xArm7's box is held to, rather than assuming it.

Unlike the xArm7 script, Panda has only one supported variant in this repo
(``SUPPORTED_ROBOTS = ["panda"]`` in reach_env.py), so there is no --variant flag.

Usage:
    python scripts/verify_panda_reachability.py
    python scripts/verify_panda_reachability.py --grid 9 --csv artifacts/panda_reach_sweep.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Rest-pose TCP orientation (wxyz): 180 deg about x -> tool z-axis points down at the
# table. Matches the orientation every reach goal is planned to (see
# verify_xarm7_reachability.py for the xArm7 equivalent of this same check).
_DOWN_QUAT_WXYZ = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)

# The "Panda-comparable" reference box, copied from
# dataset_generation/write_maniskill_reach_dataset.py::_default_reach_workspace_bounds.
# Treated as world-frame here (Panda's base sits at/near world origin in this repo's
# default env setup), unlike xArm7's base-relative XARM7_REACH_BOX_BASE.
PANDA_REACH_BOX_WORLD = np.array(
    [
        [-0.42, 0.42],
        [-0.45, 0.45],
        [0.20, 0.72],
    ],
    dtype=np.float32,
)


def _corners(box: np.ndarray) -> np.ndarray:
    """8 corners of a [3,2] box, shape (8,3)."""
    xs, ys, zs = box
    grid = np.array(np.meshgrid(xs, ys, zs, indexing="ij")).reshape(3, -1).T
    return grid.astype(np.float64)


def _grid_points(box: np.ndarray, n: int) -> np.ndarray:
    """N x N x N interior grid over a [3,2] box, shape (n^3, 3)."""
    axes = [np.linspace(lo, hi, n) for lo, hi in box]
    grid = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
    return grid.astype(np.float64)


def _sweep(
    planner: Any, start_qpos: np.ndarray, pts_world: np.ndarray, quat_wxyz: np.ndarray
) -> np.ndarray:
    """Return a bool mask: is each world-frame point IK-reachable?"""
    reachable = np.zeros(len(pts_world), dtype=bool)
    for i, world_p in enumerate(pts_world):
        goal = np.hstack([world_p, quat_wxyz]).astype(np.float64)
        status, _ = planner.IK(goal, start_qpos, n_init_qpos=20, threshold=1e-3)
        reachable[i] = status == "Success"
    return reachable


def _build_planner():
    """Register + make the Panda reach env and return (env, solver, planner, start_qpos)."""
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from mani_skill.examples.motionplanning.panda.motionplanner import (
        PandaArmMotionPlanningSolver,
    )

    from pg3d.envs.maniskill_adapter import register_pg3d_reach_envs

    register_pg3d_reach_envs()
    env = gym.make(
        "PG3DReach-BalancedWorkspace-v0",
        obs_mode="none",
        render_mode="rgb_array",
        robot_uids="panda",
        num_envs=1,
    )
    env.reset(seed=0)
    u = env.unwrapped

    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=False,
        base_pose=u.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
    )
    n_ik = len(planner.planner.user_joint_names)
    start_qpos = np.asarray(u.agent.robot.get_qpos()).reshape(-1)[:n_ik].astype(np.float64)
    return env, planner, planner.planner, start_qpos


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        env, solver, planner, start_qpos = _build_planner()
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to build planner: {type(exc).__name__}: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1

    box = PANDA_REACH_BOX_WORLD
    csv_rows: list[str] = ["box,x,y,z,reachable"]
    try:
        print(f"\nPanda reachability sweep — orient(wxyz)={_DOWN_QUAT_WXYZ.tolist()}")
        print(f"IK seed = rest qpos, grid = {args.grid}^3\n")

        corners = _corners(box)
        corner_mask = _sweep(planner, start_qpos, corners, _DOWN_QUAT_WXYZ)
        grid = _grid_points(box, args.grid)
        grid_mask = _sweep(planner, start_qpos, grid, _DOWN_QUAT_WXYZ)

        print("── PANDA_REACH_BOX (Panda-comparable reference)")
        print(f"   box (world): x{box[0].tolist()} y{box[1].tolist()} z{box[2].tolist()}")
        print(
            f"   corners : {corner_mask.sum()}/8 reachable"
            f"{'  ✓ all corners OK' if corner_mask.all() else '  ✗ some corners UNREACHABLE'}"
        )
        print(f"   grid    : {grid_mask.sum()}/{len(grid_mask)} ({100 * grid_mask.mean():.1f}%) reachable\n")

        if args.csv:
            for dp, ok in zip(grid, grid_mask):
                csv_rows.append(f"PANDA_REACH_BOX,{dp[0]:.4f},{dp[1]:.4f},{dp[2]:.4f},{int(ok)}")
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
    p = argparse.ArgumentParser(description="mplib-IK reachability sweep for the Panda reach box.")
    p.add_argument("--grid", type=int, default=5, help="Interior grid resolution per axis (N^3 points).")
    p.add_argument("--csv", type=str, default=None, help="Optional path to dump per-point reachability CSV.")
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
