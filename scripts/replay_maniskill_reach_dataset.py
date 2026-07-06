from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from pg3d.envs.maniskill_adapter import register_pg3d_reach_envs
from pg3d.envs.maniskill_adapter.dataset import load_reach_metadata
from pg3d.utils.arrays import (
    bool_any as _bool_any,
)
from pg3d.utils.arrays import (
    bool_info as _bool_info,
)
from pg3d.utils.arrays import (
    float_info as _float_info,
)
from pg3d.utils.arrays import (
    frame_to_numpy as _frame_to_numpy,
)


def _set_robot_qpos(env: Any, qpos: np.ndarray) -> None:
    """Teleport the robot to `qpos` with zero velocity, no physics stepping.

    Mirrors ``_set_robot_qpos`` in dataset_generation/write_maniskill_reach_dataset.py:
    data-gen's recorded episode begins exactly at this configuration (the randomize-start
    transit that got the robot here is never itself part of `sim_action`), so replay must
    place the robot here directly rather than let a fresh env.reset() (which lands on the
    agent's `rest` keyframe) and the first env.step() paper over the gap.
    """
    robot = env.unwrapped.agent.robot
    current = np.asarray(
        robot.get_qpos() if hasattr(robot, "get_qpos") else robot.qpos,
        dtype=np.float32,
    )
    current_shape = current.shape
    flat = current.reshape(-1).copy()
    qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
    flat[: qpos.shape[0]] = qpos
    next_qpos = flat.reshape(current_shape)
    if hasattr(robot, "set_qpos"):
        robot.set_qpos(next_qpos)
    else:
        robot.qpos = next_qpos.reshape(-1)
    if hasattr(robot, "set_qvel"):
        robot.set_qvel(np.zeros_like(next_qpos, dtype=np.float32))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
        import sapien
    except Exception as exc:
        print(
            f"Failed to import ManiSkill/Gymnasium: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print(
            "Install with: "
            "uv sync --extra cu129 --extra maniskill --group dev --group notebooks",
            file=sys.stderr,
        )
        return 2

    register_pg3d_reach_envs()
    metadata = load_reach_metadata(args.dataset)
    root = zarr.open_group(str(args.dataset), mode="r")
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    starts = np.concatenate([np.asarray([0], dtype=np.int64), episode_ends[:-1]])
    data = root["data"]
    sim_action = data["sim_action"]
    env_kwargs = dict(metadata["env_kwargs"])
    if args.video_dir is not None:
        env_kwargs["render_mode"] = "rgb_array"

    env: Any | None = None
    failures = 0
    try:
        env = gym.make(metadata["env_id"], **env_kwargs)
        max_episodes = (
            len(episode_ends)
            if args.episodes is None
            else min(args.episodes, len(episode_ends))
        )
        if args.video_dir is not None:
            args.video_dir.mkdir(parents=True, exist_ok=True)
        if args.rerun_dir is not None:
            args.rerun_dir.mkdir(parents=True, exist_ok=True)
        for episode_idx in range(max_episodes):
            episode_meta = metadata["episodes"][episode_idx]
            seed = int(episode_meta["seed"])
            env.reset(seed=seed, options={"reconfigure": True})
            # A bare reset lands the robot at its `rest` keyframe, not the recorded
            # episode's actual starting qpos: data-gen (with the default
            # --randomize-start) moves the robot to a sampled start pose via its own
            # planning phase *before* recording begins, so row 0 of `sim_action` is
            # already far from the reset pose. Without this, the first env.step below
            # asks the PD controller to close that whole gap in one 50ms tick -- a
            # large, unrecorded jump that looks like a jerky "rush" at episode start.
            _set_robot_qpos(env, sim_action[starts[episode_idx]])
            # `start_site` (the red marker) is set once inside the env's own
            # _initialize_episode -- during env.reset(), before the line above ever
            # runs -- to whatever the TCP happened to be at *that* moment (the rest
            # keyframe). It is never updated afterward, so it never reflected the
            # true randomized start pose, even during the original data-gen recording.
            # Re-point it at the actual recorded start TCP so the marker means what it
            # looks like it means.
            recorded_start_tcp = np.asarray(
                data["tcp_pose"][starts[episode_idx], :3], dtype=np.float32
            )
            env.unwrapped.start_site.set_pose(sapien.Pose(p=recorded_start_tcp))
            frames: list[np.ndarray] = []
            if args.video_dir is not None:
                frames.append(_frame_to_numpy(env.render()))
            info: dict[str, Any] = {}
            for action in sim_action[starts[episode_idx] : episode_ends[episode_idx]]:
                _obs, _reward, terminated, truncated, info = env.step(
                    np.asarray(action, dtype=np.float32)
                )
                if args.video_dir is not None:
                    frames.append(_frame_to_numpy(env.render()))
                if _bool_any(terminated) or _bool_any(truncated):
                    break
            success = _bool_info(info, "success")
            final_distance = _float_info(info, "tcp_to_goal_dist", default=float("nan"))
            failures += 0 if success else 1
            print(
                f"episode={episode_idx} seed={seed} success={success} "
                f"final_distance={final_distance:.4f}"
            )
            start = int(starts[episode_idx])
            end = int(episode_ends[episode_idx])
            if args.video_dir is not None:
                video_path = args.video_dir / f"episode_{episode_idx:03d}.mp4"
                _save_video(video_path, frames, fps=args.video_fps)
                print(f"saved video: {video_path}")
            if args.rerun_dir is not None:
                rerun_path = args.rerun_dir / f"episode_{episode_idx:03d}.rrd"
                _save_rerun_episode(
                    rerun_path,
                    point_cloud=np.asarray(data["point_cloud"][start:end]),
                    robot_mask=np.asarray(data["robot_mask"][start:end]),
                    point_valid_mask=np.asarray(data["point_valid_mask"][start:end]),
                    target_position=np.asarray(data["target_position"][start:end]),
                    tcp_pose=np.asarray(data["tcp_pose"][start:end]),
                )
                print(f"saved rerun: {rerun_path}")
    except Exception as exc:
        print(f"Failed to replay reach dataset: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if env is not None:
            env.close()
    if failures and not args.allow_failure:
        return 1
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a pg3d ManiSkill reach Zarr dataset.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--allow-failure", action="store_true")
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--rerun-dir", type=Path, default=None)
    return parser.parse_args(argv)


def _save_video(path: Path, frames: list[np.ndarray], *, fps: int) -> None:
    if not frames:
        raise RuntimeError("no frames were captured for video export")
    try:
        import imageio.v2 as imageio
    except Exception as exc:
        raise RuntimeError("video export requires imageio") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)


def _save_rerun_episode(
    path: Path,
    *,
    point_cloud: np.ndarray,
    robot_mask: np.ndarray,
    point_valid_mask: np.ndarray,
    target_position: np.ndarray,
    tcp_pose: np.ndarray,
) -> None:
    try:
        import rerun as rr
    except Exception as exc:
        raise RuntimeError(
            "Rerun export requires: "
            "uv sync --extra cu129 --extra maniskill --extra viz --group dev --group notebooks"
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    rr.init("pg3d_maniskill_reach_replay", spawn=False)
    rr.save(str(path))
    for step_idx in range(point_cloud.shape[0]):
        rr.set_time_sequence("step", step_idx)
        valid = np.asarray(point_valid_mask[step_idx], dtype=bool)
        points = np.asarray(point_cloud[step_idx], dtype=np.float32)[valid]
        if points.size:
            rr.log("world/point_cloud", rr.Points3D(points, colors=[180, 180, 180]))
            robot_points = points[np.asarray(robot_mask[step_idx], dtype=bool)[valid]]
            if robot_points.size:
                rr.log(
                    "world/robot_points",
                    rr.Points3D(robot_points, colors=[0, 128, 255]),
                )
        target = np.asarray(target_position[step_idx], dtype=np.float32).reshape(1, 3)
        if np.all(np.isfinite(target)):
            rr.log("world/target", rr.Points3D(target, colors=[0, 255, 0]))
        tcp = np.asarray(tcp_pose[step_idx, :3], dtype=np.float32).reshape(1, 3)
        if np.all(np.isfinite(tcp)):
            rr.log("world/tcp", rr.Points3D(tcp, colors=[255, 220, 0]))
    rr.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
