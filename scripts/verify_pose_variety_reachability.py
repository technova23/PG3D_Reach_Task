#!/usr/bin/env python
"""Randomized start/waypoint/goal pose-variety reachability test for xArm7.

Ground-truth check for the claim behind ``--randomize-start-goal-orientation``
data generation ("at inference time, whatever start/goal angle we're given, the
arm should be able to achieve it") -- independent of any trained policy or
GraspGen. No grasp model involved: for ``--trials`` random (position,
orientation) triples sampled across the tuned ``XARM7_REACH_BOX_BASE``
workspace and a broad orientation cone, this script:

1. IK-verifies each of the 3 poses (start / waypoint / goal) INDEPENDENTLY via
   mplib's collision-aware IK, using the same ``_resolve_reachable_orientation``
   retry-a-few-fresh-orientations logic that ``--randomize-start-goal-orientation``
   dataset generation already relies on -- a hard reachability gate before any
   motion planning is attempted. By default a trial with any unreachable pose
   is recorded as a failure and skipped (no resampling), so the reported
   success rate is an honest measure of the box's pose-variety coverage; pass
   ``--position-resample-attempts`` > 1 to instead resample just the failing
   position (not the box's reachability) if you want more videos rather than
   an unbiased rate.
2. For every trial where all 3 poses ARE IK-solvable, actually motion-plans
   (collision-aware RRT/screw, not just IK) a smooth multi-segment trajectory
   rest -> start -> waypoint -> goal, and if THAT succeeds too, replays it in
   sim and saves an MP4 of the gripper actually reaching through all three
   poses in sequence.

Usage:
    python scripts/verify_pose_variety_reachability.py --trials 20 \
        --video-dir artifacts/pose_variety_reachability

    # Wider orientation cone, more trials, denser video, gripper variant
    python scripts/verify_pose_variety_reachability.py --trials 50 \
        --orientation-cone-deg 150 --video-dir artifacts/pose_variety_reachability \
        --video-fps 15 --render-stride 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _build_env_and_planner(variant: str) -> tuple[Any, Any, Any, np.ndarray]:
    """Register + make the env; return (env, solver, planner, rest_qpos)."""
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    if variant == "gripper":
        from pg3d.envs.xarm_adapter import register_pg3d_xarm7_gripper_reach_envs
        from pg3d.envs.xarm_adapter.motionplanner import (
            XArm7GripperMotionPlanningSolver as Solver,
        )

        register_pg3d_xarm7_gripper_reach_envs()
        env_id, robot_uid = "PG3DReach-XArm7-Gripper-Workspace-v0", "xarm7_gripper"
    else:
        from pg3d.envs.xarm_adapter import register_pg3d_xarm7_reach_envs
        from pg3d.envs.xarm_adapter.motionplanner import (
            XArm7NoGripperMotionPlanningSolver as Solver,
        )

        register_pg3d_xarm7_reach_envs()
        env_id, robot_uid = "PG3DReach-XArm7-Workspace-v0", "xarm7_nogripper"

    env = gym.make(
        env_id, obs_mode="none", render_mode="rgb_array", robot_uids=robot_uid, num_envs=1
    )
    env.reset(seed=0)
    u = env.unwrapped

    solver = Solver(
        env,
        debug=False,
        vis=False,
        base_pose=u.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
    )
    n_ik = len(solver.planner.user_joint_names)
    rest_qpos = np.asarray(u.agent.robot.get_qpos()).reshape(-1)[:n_ik].astype(np.float32)
    return env, solver, solver.planner, rest_qpos


def _sample_workspace_triple(
    rng: np.random.Generator,
    *,
    bounds_world: np.ndarray,
    min_pairwise_distance: float,
    resample_attempts: int,
) -> list[np.ndarray]:
    """Sample 3 world-frame positions with a minimum pairwise separation.

    Pure geometry (no IK involved) -- resampling here only guards against a
    degenerate near-identical start/mid/goal triple, it never masks a real
    reachability failure.
    """
    bounds = np.asarray(bounds_world, dtype=np.float32).reshape(3, 2)

    def sample_one() -> np.ndarray:
        return rng.uniform(bounds[:, 0], bounds[:, 1]).astype(np.float32)

    for _ in range(max(resample_attempts, 1)):
        points = [sample_one() for _ in range(3)]
        ok = all(
            float(np.linalg.norm(points[i] - points[j])) >= min_pairwise_distance
            for i in range(3)
            for j in range(i + 1, 3)
        )
        if ok:
            return points
    return points  # last draw, even if a pair ended up close


def _render_trajectory_video(
    *,
    env: Any,
    positions: np.ndarray,
    render_stride: int,
    video_fps: int,
    video_path: Path,
) -> None:
    from pg3d.utils.arrays import frame_to_numpy as _frame_to_numpy

    from dataset_generation.write_maniskill_reach_dataset import _set_robot_qpos
    from scripts.rollout_dp3_reach_policy import save_video

    frames = []
    for step_idx, qpos in enumerate(positions):
        if step_idx % max(render_stride, 1) != 0 and step_idx != len(positions) - 1:
            continue
        _set_robot_qpos(env, qpos)
        frames.append(_frame_to_numpy(env.render()))
    save_video(video_path, frames, fps=video_fps)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from dataset_generation.write_maniskill_reach_dataset import (
        _pose_with_orientation,
        _plan_multisegment_trajectory,
        _resolve_reachable_orientation,
        _sample_broad_orientation_sapien,
        _set_robot_qpos,
    )
    from pg3d.envs.xarm_adapter.reach_config import XARM7_REACH_WORKSPACE_BOUNDS

    try:
        import sapien
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to import sapien: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    try:
        env, solver, planner, rest_qpos = _build_env_and_planner(args.variant)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to build env/planner: {type(exc).__name__}: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1

    rng = np.random.default_rng(args.seed)
    bounds_world = np.asarray(XARM7_REACH_WORKSPACE_BOUNDS, dtype=np.float32)

    if args.video_dir is not None:
        args.video_dir.mkdir(parents=True, exist_ok=True)

    trial_rows: list[dict[str, Any]] = []
    counts = {
        "total": 0,
        "start_ik_failed": 0,
        "waypoint_ik_failed": 0,
        "goal_ik_failed": 0,
        "motion_plan_failed": 0,
        "success": 0,
    }

    try:
        print(
            f"Pose-variety reachability test — variant={args.variant} trials={args.trials} "
            f"orientation_cone_deg={args.orientation_cone_deg} seed={args.seed}\n"
        )
        labels = ("start", "waypoint", "goal")
        for trial_idx in range(args.trials):
            positions = _sample_workspace_triple(
                rng,
                bounds_world=bounds_world,
                min_pairwise_distance=args.min_pairwise_distance,
                resample_attempts=args.position_resample_attempts,
            )

            resolved: list[tuple[np.ndarray, np.ndarray]] = []
            failure_label: str | None = None
            for label, position in zip(labels, positions, strict=True):
                primary_quat = _sample_broad_orientation_sapien(
                    rng, cone_deg=args.orientation_cone_deg
                )
                result = _resolve_reachable_orientation(
                    planner=planner,
                    env=env,
                    sapien=sapien,
                    position=position,
                    primary_quat=primary_quat,
                    seed_qpos=rest_qpos,
                    rng=rng,
                    cone_deg=args.orientation_cone_deg,
                    extra_attempts=args.extra_orientation_attempts,
                    suppress_planner_output=True,
                )
                if result is None:
                    failure_label = label
                    break
                resolved.append(result)

            row: dict[str, Any] = {
                "trial": trial_idx,
                "start_pos": positions[0].tolist(),
                "waypoint_pos": positions[1].tolist(),
                "goal_pos": positions[2].tolist(),
            }
            counts["total"] += 1

            if failure_label is not None:
                counts[f"{failure_label}_ik_failed"] += 1
                row["outcome"] = f"{failure_label}_ik_failed"
                trial_rows.append(row)
                print(f"[trial {trial_idx:03d}] FAIL — {failure_label} pose not IK-reachable")
                continue

            poses = [
                _pose_with_orientation(sapien, position=position, quat=quat)
                for position, (_qpos, quat) in zip(positions, resolved, strict=True)
            ]
            plan = _plan_multisegment_trajectory(
                planner=planner,
                env=env,
                poses=poses,
                start_qpos=rest_qpos,
                suppress_planner_output=True,
                smooth_trajectory=True,
            )
            if plan is None:
                counts["motion_plan_failed"] += 1
                row["outcome"] = "motion_plan_failed"
                trial_rows.append(row)
                print(
                    f"[trial {trial_idx:03d}] FAIL — all 3 poses IK-reachable but "
                    "no collision-free multi-segment plan found"
                )
                continue

            trajectory_positions, status = plan
            counts["success"] += 1
            row["outcome"] = "success"
            row["plan_status"] = status
            row["num_waypoints"] = int(trajectory_positions.shape[0])
            print(
                f"[trial {trial_idx:03d}] SUCCESS — {trajectory_positions.shape[0]} waypoints, "
                f"status={status}"
            )

            if args.video_dir is not None:
                video_path = args.video_dir / f"trial_{trial_idx:03d}.mp4"
                _render_trajectory_video(
                    env=env,
                    positions=trajectory_positions,
                    render_stride=args.render_stride,
                    video_fps=args.video_fps,
                    video_path=video_path,
                )
                row["video"] = str(video_path)
                print(f"              video: {video_path}")

            _set_robot_qpos(env, rest_qpos)
            trial_rows.append(row)
    finally:
        solver.close()
        env.close()

    print("\n── Summary")
    print(f"   total trials         : {counts['total']}")
    print(f"   start IK failed      : {counts['start_ik_failed']}")
    print(f"   waypoint IK failed   : {counts['waypoint_ik_failed']}")
    print(f"   goal IK failed       : {counts['goal_ik_failed']}")
    print(f"   motion plan failed   : {counts['motion_plan_failed']}")
    print(
        f"   success (video)      : {counts['success']} "
        f"({100 * counts['success'] / max(counts['total'], 1):.1f}%)"
    )

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps({"counts": counts, "trials": trial_rows}, indent=2)
        )
        print(f"\nwrote summary: {args.summary_json}")

    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Randomized start/waypoint/goal pose-variety reachability test for xArm7 "
            "(IK-verify each pose, then motion-plan + execute + video the reachable ones)."
        )
    )
    p.add_argument(
        "--variant",
        choices=["nogripper", "gripper"],
        default="gripper",
        help="Which xArm7 agent/TCP to test (default: gripper, matches the pose-variety datasets).",
    )
    p.add_argument("--trials", type=int, default=20, help="Number of random start/waypoint/goal triples.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--orientation-cone-deg",
        type=float,
        default=120.0,
        help="Half-angle (deg) of the broad orientation cone each pose's orientation is drawn from.",
    )
    p.add_argument(
        "--extra-orientation-attempts",
        type=int,
        default=3,
        help="Extra fresh orientation resamples tried at the SAME position before declaring it unreachable.",
    )
    p.add_argument(
        "--min-pairwise-distance",
        type=float,
        default=0.15,
        help="Minimum required distance (m) between each pair of start/waypoint/goal positions.",
    )
    p.add_argument(
        "--position-resample-attempts",
        type=int,
        default=1,
        help=(
            "Resample attempts for the 3 POSITIONS (pure geometry, not IK) to satisfy "
            "--min-pairwise-distance. Default 1 = no resampling beyond the separation check; "
            "does not affect the reported IK/motion-plan success rate."
        ),
    )
    p.add_argument("--video-dir", type=Path, default=None, help="Directory to save per-trial MP4s to.")
    p.add_argument("--video-fps", type=int, default=10)
    p.add_argument(
        "--render-stride",
        type=int,
        default=2,
        help="Render every Nth trajectory waypoint (plus the final one) to keep videos a reasonable length.",
    )
    p.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path to write a JSON summary (counts + per-trial rows) to.",
    )
    args = p.parse_args(argv)
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    if args.extra_orientation_attempts < 0:
        raise ValueError("--extra-orientation-attempts must be non-negative")
    if args.min_pairwise_distance < 0.0:
        raise ValueError("--min-pairwise-distance must be non-negative")
    if args.position_resample_attempts <= 0:
        raise ValueError("--position-resample-attempts must be positive")
    if args.video_fps <= 0:
        raise ValueError("--video-fps must be positive")
    if args.render_stride <= 0:
        raise ValueError("--render-stride must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(main())
