from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from pg3d.envs.maniskill_adapter import adapt_observation
from pg3d.utils.arrays import frame_to_numpy


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
    except Exception as exc:
        print(
            f"Failed to import ManiSkill/Gymnasium: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print(
            "Install with: "
            "uv sync --extra cu129 --extra maniskill --group dev --group notebooks"
        )
        return 2

    render_mode = "rgb_array" if args.video_frames > 0 else None
    env: Any | None = None
    try:
        env = gym.make(
            args.env_id,
            obs_mode=args.obs_mode,
            render_mode=render_mode,
            robot_uids=args.robot_uid,
            num_envs=1,
        )
        obs, info = env.reset(seed=args.seed)
        adapted = adapt_observation(obs, info=info, env=env, task_name=args.env_id)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        _save_summary(args.output_dir / "summary.json", adapted, args=args)
        _save_npz(args.output_dir / "observation.npz", adapted)

        if args.video_frames > 0:
            _save_video(env, args.output_dir / "observation.mp4", frames=args.video_frames)
        if args.rerun_path is not None:
            _save_rerun(args.rerun_path, adapted)

        print(f"saved summary: {args.output_dir / 'summary.json'}")
        print(f"saved arrays: {args.output_dir / 'observation.npz'}")
        if args.video_frames > 0:
            print(f"saved video: {args.output_dir / 'observation.mp4'}")
        if args.rerun_path is not None:
            print(f"saved rerun: {args.rerun_path}")
        return 0
    except Exception as exc:
        print(f"Failed to save ManiSkill observation: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if env is not None:
            env.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save a single adapted ManiSkill observation artifact."
    )
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument("--robot-uid", default="panda")
    parser.add_argument("--obs-mode", default="state_dict", choices=["state_dict", "pointcloud"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/maniskill_observation"))
    parser.add_argument("--video-frames", type=int, default=0)
    parser.add_argument("--rerun-path", type=Path, default=None)
    return parser.parse_args(argv)


def _save_summary(path: Path, adapted: Any, *, args: argparse.Namespace) -> None:
    summary = adapted.summary()
    summary["command"] = {
        "env_id": args.env_id,
        "robot_uid": args.robot_uid,
        "obs_mode": args.obs_mode,
        "seed": args.seed,
        "video_frames": args.video_frames,
        "rerun_path": str(args.rerun_path) if args.rerun_path is not None else None,
    }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def _save_npz(path: Path, adapted: Any) -> None:
    arrays: dict[str, np.ndarray] = {
        "point_cloud": adapted.point_cloud,
        "agent_pos": adapted.robot_state.as_agent_pos(),
    }
    if adapted.robot_state.joint_velocities is not None:
        arrays["joint_velocities"] = adapted.robot_state.joint_velocities
    if adapted.robot_state.tcp_pose is not None:
        arrays["tcp_pose"] = adapted.robot_state.tcp_pose
    if adapted.robot_mask is not None:
        arrays["robot_mask"] = adapted.robot_mask
    if adapted.sim_gt is not None and adapted.sim_gt.target_position is not None:
        arrays["target_position"] = adapted.sim_gt.target_position
    for key, value in adapted.point_features.items():
        arrays[f"feature_{key}"] = value
    for key, value in adapted.object_masks.items():
        arrays[f"mask_{key}"] = value
    np.savez_compressed(path, **arrays)


def _save_video(env: Any, path: Path, *, frames: int) -> None:
    import imageio.v2 as imageio

    images = [frame_to_numpy(env.render())]
    for _ in range(max(frames - 1, 0)):
        env.step(env.action_space.sample())
        images.append(frame_to_numpy(env.render()))
    imageio.mimsave(path, images, fps=10)


def _save_rerun(path: Path, adapted: Any) -> None:
    try:
        import rerun as rr
    except Exception as exc:
        raise RuntimeError(
            "Rerun export requires: "
            "uv sync --extra cu129 --extra maniskill --extra viz --group dev --group notebooks"
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    rr.init("pg3d_maniskill_observation", spawn=False)
    rr.save(str(path))
    colors = adapted.point_features.get("rgb")
    rr.log("world/point_cloud", rr.Points3D(adapted.point_cloud, colors=colors))
    if adapted.robot_mask is not None and np.any(adapted.robot_mask):
        rr.log(
            "world/robot_points",
            rr.Points3D(adapted.point_cloud[adapted.robot_mask], colors=[0, 128, 255]),
        )
    if adapted.sim_gt is not None and adapted.sim_gt.target_position is not None:
        rr.log(
            "world/target",
            rr.Points3D(adapted.sim_gt.target_position.reshape(1, 3), colors=[0, 255, 0]),
        )


if __name__ == "__main__":
    raise SystemExit(main())
