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

    # Also empirically discover the true reachable envelope (broad grid, IK +
    # forward-kinematics re-check per point) before verifying the shipped boxes,
    # and get a suggested XARM7_REACH_BOX_BASE derived from what's actually reachable
    python scripts/verify_xarm7_reachability.py --variant gripper --discover-envelope
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


def _discover_reachable_envelope(
    planner: Any,
    env: Any,
    base_pos: np.ndarray,
    start_qpos: np.ndarray,
    quat_wxyz: np.ndarray,
    *,
    bounds: np.ndarray,
    grid_shape: tuple[int, int, int],
    min_base_clearance: float,
    position_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Broad grid-search reachability discovery over `bounds` (base-relative),
    the same analogy as a PyBullet-style "IK then re-verify with forward
    kinematics" sweep: solve IK for every candidate point, then actually set
    the resulting qpos on the simulated robot and check where its TCP really
    ended up, rather than trusting the solver's own status flag alone.

    Unlike a plain PyBullet `calculateInverseKinematics` sweep, this keeps two
    properties that matter for this repo's actual failure modes:
    - mplib's IK is collision-aware (rejects self-colliding solutions), which
      is what originally caught the xarm7_gripper wrist-fold blind spot (see
      reach_config.py docstring); a non-collision-aware solver would report
      false positives there.
    - IK is solved at the fixed downward TCP orientation every reach goal
      actually uses, not a free/unconstrained orientation, which would
      overstate reachability relative to what data-gen can actually use.
    """
    from dataset_generation.write_maniskill_reach_dataset import _set_robot_qpos, _tcp_pose

    axes = [np.linspace(lo, hi, n) for (lo, hi), n in zip(bounds, grid_shape, strict=True)]
    grid = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T.astype(np.float64)

    reachable_points: list[np.ndarray] = []
    unreachable_points: list[np.ndarray] = []
    for dp in grid:
        if float(np.linalg.norm(dp[:2])) < min_base_clearance:
            continue
        world_p = base_pos + dp
        goal = np.hstack([world_p, quat_wxyz]).astype(np.float64)
        status, ik_result = planner.IK(goal, start_qpos, n_init_qpos=20, threshold=1e-3)
        if status != "Success":
            unreachable_points.append(dp)
            continue
        # mplib's IK may return either a single qpos array or a list of
        # candidate qpos arrays depending on version; handle both.
        ik_qpos = ik_result[0] if isinstance(ik_result, (list, tuple)) else ik_result
        _set_robot_qpos(env, np.asarray(ik_qpos, dtype=np.float32))
        actual_tcp = _tcp_pose(env.unwrapped)
        distance = float(np.linalg.norm(actual_tcp[:3].astype(np.float64) - world_p))
        if distance <= position_tolerance:
            reachable_points.append(dp)
        else:
            unreachable_points.append(dp)
    _set_robot_qpos(env, start_qpos)
    return (
        np.asarray(reachable_points, dtype=np.float64).reshape(-1, 3),
        np.asarray(unreachable_points, dtype=np.float64).reshape(-1, 3),
    )


def _suggest_box_from_points(points: np.ndarray, *, percentile: float) -> np.ndarray:
    """Robust axis-aligned box [3,2] from a reachable point cloud: trims the
    outer `percentile` on each side per axis so a handful of sparse,
    isolated reachable points don't blow the suggested box out to an
    unrepresentative edge (the interior grid density is what should drive
    the box, not a lucky far-corner hit).
    """
    lo = np.percentile(points, percentile, axis=0)
    hi = np.percentile(points, 100.0 - percentile, axis=0)
    return np.stack([lo, hi], axis=1)


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

        if args.discover_envelope:
            print("── STAGE 1: envelope discovery (broad grid, IK + forward-kinematics re-check)")
            discover_bounds = np.asarray(args.discover_bounds, dtype=np.float64).reshape(3, 2)
            print(f"   scan bounds (base-rel): dx{discover_bounds[0].tolist()} "
                  f"dy{discover_bounds[1].tolist()} dz{discover_bounds[2].tolist()} "
                  f"grid={tuple(args.discover_grid)} "
                  f"min_base_clearance={args.discover_min_base_clearance} "
                  f"position_tolerance={args.discover_position_tolerance}")
            reachable_pts, unreachable_pts = _discover_reachable_envelope(
                planner,
                env,
                base_pos,
                start_qpos,
                _DOWN_QUAT_WXYZ,
                bounds=discover_bounds,
                grid_shape=tuple(args.discover_grid),
                min_base_clearance=args.discover_min_base_clearance,
                position_tolerance=args.discover_position_tolerance,
            )
            total_scanned = len(reachable_pts) + len(unreachable_pts)
            print(f"   reachable: {len(reachable_pts)}/{total_scanned} "
                  f"({100 * len(reachable_pts) / max(total_scanned, 1):.1f}%)")
            if len(reachable_pts) > 0:
                suggested_box = _suggest_box_from_points(
                    reachable_pts, percentile=args.discover_percentile
                )
                print(f"   suggested box (base-rel, {args.discover_percentile:.0f}th/"
                      f"{100 - args.discover_percentile:.0f}th percentile of reachable points):")
                print(f"     dx{suggested_box[0].tolist()} dy{suggested_box[1].tolist()} "
                      f"dz{suggested_box[2].tolist()}")
                print("   (STAGE 2 below verifies XARM7_REACH_BOX_BASE/MAX_ENVELOPE as shipped, "
                      "not this suggestion -- update reach_config.py and re-run to verify it.)\n")
            else:
                print("   no reachable points found in the scanned bounds\n")
            if args.csv:
                for dp in reachable_pts:
                    csv_rows.append(f"DISCOVERY,{dp[0]:.4f},{dp[1]:.4f},{dp[2]:.4f},1")
                for dp in unreachable_pts:
                    csv_rows.append(f"DISCOVERY,{dp[0]:.4f},{dp[1]:.4f},{dp[2]:.4f},0")
            print("── STAGE 2: verify current reach_config.py boxes")

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
    p.add_argument(
        "--discover-envelope",
        action="store_true",
        help=(
            "Before verifying the shipped boxes, run a broad exploratory grid sweep "
            "(over --discover-bounds) to empirically map the true reachable envelope, "
            "instead of only checking the already-chosen XARM7_REACH_BOX_BASE/"
            "MAX_ENVELOPE. Each candidate point's IK solution is re-verified by actually "
            "setting it on the simulated robot and checking the resulting TCP position "
            "via forward kinematics (within --discover-position-tolerance), not just "
            "trusting the IK solver's own status flag. Prints a suggested axis-aligned "
            "box derived from the discovered reachable points."
        ),
    )
    p.add_argument(
        "--discover-bounds",
        type=float,
        nargs=6,
        default=[-1.0, 1.0, -1.0, 1.0, 0.0, 1.2],
        metavar=("DX_MIN", "DX_MAX", "DY_MIN", "DY_MAX", "DZ_MIN", "DZ_MAX"),
        help=(
            "Base-relative bounds of the broad candidate region to scan when "
            "--discover-envelope is set: a natural symmetric box around the base "
            "(default +/-1.0m lateral, 0-1.2m vertical), not derived from any existing "
            "reach_config.py box -- the point is to independently discover the true "
            "reachable shape, not re-measure a pre-set guess."
        ),
    )
    p.add_argument(
        "--discover-grid",
        type=int,
        nargs=3,
        default=[12, 12, 10],
        metavar=("NX", "NY", "NZ"),
        help="Grid resolution per axis (x, y, z) for --discover-envelope.",
    )
    p.add_argument(
        "--discover-min-base-clearance",
        type=float,
        default=0.15,
        help="Skip candidate points with horizontal (xy) distance from the base below this, "
        "to avoid self-collision artifacts near the base column.",
    )
    p.add_argument(
        "--discover-position-tolerance",
        type=float,
        default=0.02,
        help="Max distance (m) between target and actual FK-derived TCP position to count "
        "an IK-solved point as genuinely reachable.",
    )
    p.add_argument(
        "--discover-percentile",
        type=float,
        default=5.0,
        help="Percentile (and its 100-p complement) used to derive the suggested box from "
        "the reachable point cloud, trimming sparse outlier points on each side.",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
