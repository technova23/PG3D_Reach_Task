#!/usr/bin/env python
"""Render a short video of the xArm7-gripper reach env (no zarr required).

Runs the motion planner for N episodes and saves MP4s so you can inspect
the scene layout, robot geometry, camera framing, and trajectory quality
before committing to a full dataset generation run.

Usage:
    # Preview 3 episodes, output to artifacts/preview/
    python scripts/preview_xarm7_gripper_env.py

    # No-gripper variant
    python scripts/preview_xarm7_gripper_env.py --variant nogripper

    # More episodes / custom seed
    python scripts/preview_xarm7_gripper_env.py --episodes 5 --seed 42

    # Skip motion planning, just show the static reset scene (fast)
    python scripts/preview_xarm7_gripper_env.py --static
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
    except Exception as exc:
        print(f"Failed to import ManiSkill: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    if args.variant == "gripper":
        from pg3d.envs.xarm_adapter import register_pg3d_xarm7_gripper_reach_envs
        register_pg3d_xarm7_gripper_reach_envs()
        env_id = "PG3DReach-XArm7-Gripper-Workspace-v0"
        robot_uid = "xarm7_gripper"
    elif args.variant == "robotiq":
        from pg3d.envs.xarm_adapter import register_pg3d_xarm7_robotiq_reach_envs
        register_pg3d_xarm7_robotiq_reach_envs()
        env_id = "PG3DReach-XArm7-Robotiq-Workspace-v0"
        robot_uid = "xarm7_robotiq"
    else:
        from pg3d.envs.xarm_adapter import register_pg3d_xarm7_reach_envs
        register_pg3d_xarm7_reach_envs()
        env_id = "PG3DReach-XArm7-Workspace-v0"
        robot_uid = "xarm7_nogripper"

    # Build a nice 3rd-person visualization camera.
    # Robot base = [-0.615, 0, 0]; workspace center ≈ [-0.315, 0, 0.22].
    # Camera placed at world [0.5, -0.9, 0.9] looking toward workspace.
    # ManiSkill's human_render_camera_configs expects a nested dict where
    # "pose" is a flat [x, y, z, qw, qx, qy, qz] list (7 elements).
    from mani_skill.utils import sapien_utils
    vis_pose = sapien_utils.look_at(eye=[0.7, 0.0, 0.45], target=[-0.315, 0.0, 0.22])
    p = np.asarray(vis_pose.p).reshape(-1)
    q = np.asarray(vis_pose.q).reshape(-1)  # [w, x, y, z]
    vis_cam_dict = {
        "pose": [float(p[0]), float(p[1]), float(p[2]),
                 float(q[0]), float(q[1]), float(q[2]), float(q[3])],
        "width": 640,
        "height": 480,
        "fov": float(np.deg2rad(60)),
        "near": 0.1,
        "far": 10.0,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    env: Any | None = None
    try:
        # human_render_camera_configs overrides the calibrated D455 render camera
        # with a proper 3rd-person visualization angle.
        env = gym.make(
            env_id,
            obs_mode="pointcloud",
            render_mode="rgb_array",
            robot_uids=robot_uid,
            num_envs=1,
            human_render_camera_configs={"render_camera": vis_cam_dict},
        )

        if args.static:
            _render_static(env, args)
        else:
            _render_episodes(env, args, env_id=env_id, robot_uid=robot_uid)

    except Exception as exc:
        print(f"Preview failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if env is not None:
            env.close()

    return 0


def _render_static(env: Any, args: argparse.Namespace) -> None:
    """Just reset the env and capture a single image — fastest sanity check."""
    import imageio.v2 as imageio

    env.reset(seed=args.seed)
    frame = _to_uint8(env.render())
    out = args.output_dir / "static_reset.png"
    imageio.imwrite(out, frame)
    print(f"saved: {out}")


def _goal_pose(unwrapped_env: Any, sapien: Any) -> Any:
    """Mirror dataset_generation/write_maniskill_reach_dataset.py's _goal_pose."""
    goal_pos = np.asarray(unwrapped_env.goal_site.pose.p).reshape(-1, 3)[0]
    tcp_pose = np.asarray(unwrapped_env.agent.tcp.pose.raw_pose).reshape(-1, 7)[0]
    return sapien.Pose(p=goal_pos, q=tcp_pose[3:])


def _render_episodes(env: Any, args: argparse.Namespace, *, env_id: str, robot_uid: str) -> None:
    """Run the motion planner for each episode and save MP4s.

    Mirrors dataset_generation/write_maniskill_reach_dataset.py: dry-run the
    planner to get a joint-position waypoint sequence, then step the env
    through those waypoints (capturing a render frame per step).
    """
    import imageio.v2 as imageio
    import sapien

    if args.variant == "gripper":
        from pg3d.envs.xarm_adapter.motionplanner import XArm7GripperMotionPlanningSolver
        planner_cls = XArm7GripperMotionPlanningSolver
    elif args.variant == "robotiq":
        from pg3d.envs.xarm_adapter.motionplanner import XArm7RobotiqMotionPlanningSolver
        planner_cls = XArm7RobotiqMotionPlanningSolver
    else:
        from pg3d.envs.xarm_adapter.motionplanner import XArm7NoGripperMotionPlanningSolver
        planner_cls = XArm7NoGripperMotionPlanningSolver

    for ep in range(args.episodes):
        seed = args.seed + ep
        env.reset(seed=seed, options={"reconfigure": True})
        unwrapped = env.unwrapped
        goal_pose = _goal_pose(unwrapped, sapien)

        planner = planner_cls(
            env,
            debug=False,
            vis=False,
            base_pose=unwrapped.agent.robot.pose,
            visualize_target_grasp_pose=False,
            print_env_info=False,
        )

        frames: list[np.ndarray] = [_to_uint8(env.render())]
        success = False
        try:
            plan = planner.move_to_pose_with_screw(goal_pose, dry_run=True)
            if plan != -1 and "position" in plan:
                positions = np.asarray(plan["position"], dtype=np.float32)
                for pos in positions:
                    env.step(pos)
                    frames.append(_to_uint8(env.render()))
                success = True
        except Exception as exc:
            print(f"  ep={ep} seed={seed} planner error: {exc}")
        finally:
            planner.close()

        out = args.output_dir / f"ep{ep:03d}_seed{seed}_{'ok' if success else 'fail'}.mp4"
        imageio.mimsave(str(out), frames, fps=args.fps)
        print(f"saved: {out}  (success={success}, {len(frames)} frames)")


def _to_uint8(frame: Any) -> np.ndarray:
    """Convert ManiSkill render output (NumPy or Torch tensor) to uint8 HWC."""
    import torch

    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()

    arr = np.asarray(frame)

    if arr.ndim == 4:
        arr = arr[0]  # (B,H,W,C) -> (H,W,C)

    if arr.dtype != np.uint8:
        # Handle float images in [0,1]
        if np.issubdtype(arr.dtype, np.floating):
            arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)

    return arr


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preview xArm7 reach env as video.")
    p.add_argument("--variant", choices=["gripper", "nogripper", "robotiq"], default="gripper")
    p.add_argument("--episodes", type=int, default=3, help="Episodes to render (ignored if --static)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--static", action="store_true", help="Just save a single reset-frame PNG (fast)")
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/preview"))
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
