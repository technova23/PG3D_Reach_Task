from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from pg3d.envs.maniskill_adapter import (
    ManiSkillGhostPandaGeometryProvider,
    register_pg3d_reach_envs,
)
from pg3d.envs.maniskill_adapter.dataset import (
    PointCloudCropConfig,
    crop_point_cloud,
    load_reach_metadata,
)
from pg3d.envs.maniskill_adapter.types import Observation, RobotState, SimGroundTruth
from pg3d.policies.dp3 import SimpleDP3
from pg3d.policies.dp3.checkpoint import (
    latest_reach_checkpoint,
    load_reach_policy_from_checkpoint,
)
from pg3d.utils.arrays import (
    bool_any as _bool_any,
)
from pg3d.utils.arrays import (
    float_value as _float_value,
)
from pg3d.utils.arrays import (
    frame_to_numpy as _frame_to_numpy,
)
from pg3d.utils.devices import select_device
from pg3d.utils.serialization import jsonable as _jsonable
from pg3d.world_model import ActionChunk, GeometricWorldModel
from scripts.rollout_dp3_reach_policy import (
    ActionMode,
    RolloutSpec,
    append_obs_window,
    crop_config_from_metadata,
    make_initial_obs_window,
    obs_window_to_torch,
    policy_action_to_sim_action,
    rollout_observation_entry,
    save_video,
    select_rollout_specs,
)

Source = Literal["dataset", "fresh"]
Entry = dict[str, np.ndarray | bool | float]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint, args.checkpoint_dir)
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
            "uv sync --extra cu129 --extra maniskill --extra viz --group dev --group notebooks",
            file=sys.stderr,
        )
        return 2

    register_pg3d_reach_envs()
    device = select_device(args.device)
    policy = load_reach_policy_from_checkpoint(
        checkpoint_path,
        device=device,
        prefer_ema=args.checkpoint_model == "ema",
    )
    metadata = load_reach_metadata(args.dataset)
    crop_config = crop_config_from_metadata(metadata)
    action_mode = _action_mode(str(metadata.get("action_mode", "abs_joint")))
    goal_thresh = (
        float(args.goal_thresh)
        if args.goal_thresh is not None
        else float(dict(metadata.get("env_kwargs", {})).get("goal_thresh", 0.025))
    )
    dataset_episode_seeds = [
        int(episode["seed"]) for episode in metadata.get("episodes", []) if "seed" in episode
    ]
    episode_indices = (
        _unique_seed_episode_indices(dataset_episode_seeds, max_count=args.episodes)
        if args.unique_dataset_seeds
        else args.episode_indices
    )
    specs = select_rollout_specs(
        source=args.source,
        dataset_episode_seeds=dataset_episode_seeds,
        episodes=args.episodes,
        episode_indices=episode_indices,
        seed_start=args.seed_start,
    )
    if not specs:
        raise RuntimeError("no comparison episodes selected")
    if args.unique_dataset_seeds:
        print(
            "unique dataset seed selection: "
            f"selected={len(specs)} available_unique={len(set(dataset_episode_seeds))} "
            f"dataset_rows={len(dataset_episode_seeds)}",
            flush=True,
        )

    sim_env_kwargs = _env_kwargs(metadata, render_mode="rgb_array" if args.video else None)
    ghost_env_kwargs = _env_kwargs(metadata, render_mode=None)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sim_env: Any | None = None
    ghost_env: Any | None = None
    summaries: list[dict[str, Any]] = []
    metrics_path = args.output_dir / "metrics.jsonl"
    try:
        sim_env = gym.make(str(metadata["env_id"]), **sim_env_kwargs)
        ghost_env = gym.make(str(metadata["env_id"]), **ghost_env_kwargs)
        with metrics_path.open("w", encoding="utf-8") as metrics_file:
            for spec in specs:
                episode_timeline: list[dict[str, Any]] = []
                summary = run_comparison_episode(
                    sim_env=sim_env,
                    ghost_env=ghost_env,
                    policy=policy,
                    spec=spec,
                    action_mode=action_mode,
                    crop_config=crop_config,
                    goal_thresh=goal_thresh,
                    output_dir=args.output_dir,
                    device=device,
                    max_steps=args.max_steps,
                    replan_stride=(
                        args.replan_stride
                        if args.replan_stride is not None
                        else int(policy.n_action_steps)
                    ),
                    gripper_open=args.gripper_open,
                    match_current_robot_points=args.match_current_robot_points,
                    video=args.video,
                    video_fps=args.video_fps,
                    metrics_file=metrics_file,
                    timeline=episode_timeline,
                )
                if args.rerun:
                    rerun_path = rerun_path_for_episode(args.output_dir, spec.output_index)
                    save_rerun_comparison(rerun_path, episode_timeline)
                    summary["rerun"] = str(rerun_path)
                summaries.append(summary)
                print(
                    f"episode={spec.output_index} "
                    f"dataset_episode={spec.dataset_episode_index} seed={spec.seed} "
                    f"sim_success={summary['sim_success']} "
                    f"wm_success={summary['world_model_success']} "
                    f"sim_final={summary['sim_final_distance']:.4f} "
                    f"wm_final={summary['world_model_final_distance']:.4f} "
                    f"steps={summary['steps']}"
                )
    except Exception as exc:
        print(
            f"Failed to compare world-model and ManiSkill rollout: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    finally:
        if sim_env is not None:
            sim_env.close()
        if ghost_env is not None:
            ghost_env.close()

    summary = {
        "checkpoint": str(checkpoint_path),
        "dataset": str(args.dataset),
        "source": args.source,
        "env_id": metadata["env_id"],
        "env_kwargs": sim_env_kwargs,
        "ghost_env_kwargs": ghost_env_kwargs,
        "action_mode": action_mode,
        "goal_thresh": goal_thresh,
        "rerun_files": [episode.get("rerun") for episode in summaries if episode.get("rerun")],
        "episodes": summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    failures = sum(
        0 if episode["sim_success"] and episode["world_model_success"] else 1
        for episode in summaries
    )
    return 0 if args.allow_failure or failures == 0 else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a DP3 reach policy rollout in the P07 world model and ManiSkill."
    )
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument("--checkpoint", type=Path, default=None)
    checkpoint_group.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-model", choices=["ema", "raw"], default="ema")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("artifacts/reach-datasets/pg3d-reach-narrow-100.zarr"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reach-datasets/world-model-vs-sim"),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--source", choices=["dataset", "fresh"], default="dataset")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--episode-indices", type=int, nargs="+", default=None)
    parser.add_argument(
        "--unique-dataset-seeds",
        action="store_true",
        help=(
            "With --source dataset, compare only the first dataset row for each unique "
            "episode seed, capped by --episodes."
        ),
    )
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--replan-stride", type=int, default=None)
    parser.add_argument("--goal-thresh", type=float, default=None)
    parser.add_argument("--gripper-open", type=float, default=0.04)
    parser.add_argument(
        "--match-current-robot-points",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cap ghost robot clouds to the current cropped robot-mask count.",
    )
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--allow-failure", action="store_true")
    args = parser.parse_args(argv)
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.unique_dataset_seeds and args.source != "dataset":
        raise ValueError("--unique-dataset-seeds requires --source dataset")
    if args.unique_dataset_seeds and args.episode_indices is not None:
        raise ValueError("--unique-dataset-seeds cannot be combined with --episode-indices")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.replan_stride is not None and args.replan_stride <= 0:
        raise ValueError("--replan-stride must be positive")
    if args.goal_thresh is not None and args.goal_thresh <= 0.0:
        raise ValueError("--goal-thresh must be positive")
    if args.video_fps <= 0:
        raise ValueError("--video-fps must be positive")
    return args


def resolve_checkpoint_path(checkpoint: Path | None, checkpoint_dir: Path | None) -> Path:
    """Resolve either an explicit checkpoint path or the latest checkpoint in a directory."""
    if checkpoint is not None:
        return checkpoint
    if checkpoint_dir is None:
        raise ValueError("checkpoint or checkpoint_dir is required")
    return latest_reach_checkpoint(checkpoint_dir)


def rerun_path_for_episode(output_dir: Path, episode_index: int) -> Path:
    """Return the per-episode Rerun comparison path."""
    if episode_index < 0:
        raise ValueError("episode_index must be non-negative")
    return output_dir / f"episode_{episode_index:03d}_comparison.rrd"


def _unique_seed_episode_indices(
    dataset_episode_seeds: list[int],
    *,
    max_count: int,
) -> list[int]:
    if max_count <= 0:
        raise ValueError("max_count must be positive")
    seen: set[int] = set()
    indices: list[int] = []
    for dataset_idx, seed in enumerate(dataset_episode_seeds):
        if seed in seen:
            continue
        seen.add(seed)
        indices.append(dataset_idx)
        if len(indices) >= max_count:
            break
    if not indices:
        raise ValueError("dataset metadata did not contain any episode seeds")
    return indices


def run_comparison_episode(
    *,
    sim_env: Any,
    ghost_env: Any,
    policy: SimpleDP3,
    spec: RolloutSpec,
    action_mode: ActionMode,
    crop_config: PointCloudCropConfig,
    goal_thresh: float,
    output_dir: Path,
    device: torch.device,
    max_steps: int,
    replan_stride: int,
    gripper_open: float,
    match_current_robot_points: bool,
    video: bool,
    video_fps: int,
    metrics_file: Any,
    timeline: list[dict[str, Any]],
) -> dict[str, Any]:
    sim_obs, sim_info = sim_env.reset(seed=spec.seed, options={"reconfigure": True})
    provider = ManiSkillGhostPandaGeometryProvider(
        ghost_env,
        task_name=_env_task_name(sim_env),
        crop_bounds=crop_config.bounds,
    )
    provider.reset(seed=spec.seed, options={"reconfigure": True})
    world_model = GeometricWorldModel(provider)

    sim_entry = rollout_observation_entry(sim_obs, sim_info, env=sim_env, crop_config=crop_config)
    wm_entry = _copy_entry(sim_entry)
    obs_window = make_initial_obs_window(wm_entry, n_obs_steps=int(policy.n_obs_steps))
    frames = [_frame_to_numpy(sim_env.render())] if video else []
    steps = 0
    policy_steps = 0
    sim_first_success_step: int | None = None
    wm_first_success_step: int | None = None
    sim_final_distance = _entry_distance(sim_entry)
    wm_final_distance = _entry_distance(wm_entry)
    sim_min_distance = sim_final_distance if np.isfinite(sim_final_distance) else float("inf")
    wm_min_distance = wm_final_distance if np.isfinite(wm_final_distance) else float("inf")

    _append_timeline_pair(
        timeline,
        episode=spec.output_index,
        step=0,
        policy_step=0,
        world_model_entry=wm_entry,
        sim_entry=sim_entry,
    )

    while steps < max_steps:
        if sim_first_success_step is not None and wm_first_success_step is not None:
            break
        with torch.no_grad():
            policy_input = obs_window_to_torch(
                obs_window,
                device=device,
                goal_marker_points=int(policy.goal_marker_points),
                goal_marker_radius=float(policy.goal_marker_radius),
            )
            policy_output = policy.predict_action(policy_input)
            policy_actions = policy_output["action"][0].detach().cpu().numpy()

        stride = min(replan_stride, int(policy_actions.shape[0]), max_steps - steps)
        if stride <= 0:
            break
        if match_current_robot_points:
            provider.set_robot_point_budget_from_mask(
                np.asarray(wm_entry["robot_mask"], dtype=bool),
                point_valid_mask=np.asarray(wm_entry["point_valid_mask"], dtype=bool),
            )
        wm_observation = entry_to_world_model_observation(wm_entry)
        chunk = ActionChunk(
            actions=policy_actions[:stride],
            action_mode=action_mode,
            dt=1.0,
            metadata={"policy_step": policy_steps},
        )
        imagined = world_model.imagine(wm_observation, chunk)

        for chunk_step, policy_action in enumerate(policy_actions[:stride]):
            wm_entry = world_model_entry_from_rollout_step(
                imagined,
                chunk_step,
                previous_entry=wm_entry,
                crop_config=crop_config,
                goal_thresh=goal_thresh,
            )
            sim_action = policy_action_to_sim_action(
                policy_action,
                np.asarray(sim_entry["agent_pos"], dtype=np.float32),
                action_mode=action_mode,
                sim_action_dim=int(np.prod(sim_env.action_space.shape)),
                low=getattr(sim_env.action_space, "low", None),
                high=getattr(sim_env.action_space, "high", None),
                gripper_open=gripper_open,
            )
            sim_obs, reward, terminated, truncated, sim_info = sim_env.step(sim_action)
            steps += 1
            if video:
                frames.append(_frame_to_numpy(sim_env.render()))
            sim_entry = rollout_observation_entry(
                sim_obs,
                sim_info,
                env=sim_env,
                crop_config=crop_config,
            )
            obs_window = append_obs_window(
                obs_window,
                wm_entry,
                n_obs_steps=int(policy.n_obs_steps),
            )

            sim_final_distance = _entry_distance(sim_entry)
            wm_final_distance = _entry_distance(wm_entry)
            if np.isfinite(sim_final_distance):
                sim_min_distance = min(sim_min_distance, sim_final_distance)
            if np.isfinite(wm_final_distance):
                wm_min_distance = min(wm_min_distance, wm_final_distance)
            sim_success = bool(sim_entry["success"])
            wm_success = bool(wm_entry["success"])
            if sim_success and sim_first_success_step is None:
                sim_first_success_step = steps
            if wm_success and wm_first_success_step is None:
                wm_first_success_step = steps

            _append_timeline_pair(
                timeline,
                episode=spec.output_index,
                step=steps,
                policy_step=policy_steps,
                world_model_entry=wm_entry,
                sim_entry=sim_entry,
            )
            metrics_file.write(
                json.dumps(
                    _jsonable(
                        {
                            "episode": spec.output_index,
                            "seed": spec.seed,
                            "source": spec.source,
                            "step": steps,
                            "policy_step": policy_steps,
                            "chunk_step": chunk_step,
                            "reward": _float_value(reward),
                            "sim_success": sim_success,
                            "world_model_success": wm_success,
                            "sim_first_success_step": sim_first_success_step,
                            "world_model_first_success_step": wm_first_success_step,
                            "sim_final_distance": sim_final_distance,
                            "world_model_final_distance": wm_final_distance,
                            "sim_robot_points": _valid_robot_count(sim_entry),
                            "world_model_robot_points": _valid_robot_count(wm_entry),
                        }
                    ),
                    sort_keys=True,
                )
                + "\n"
            )
            metrics_file.flush()
            if _bool_any(truncated) or _bool_any(terminated):
                break
            if sim_first_success_step is not None and wm_first_success_step is not None:
                break
        policy_steps += 1
        if _bool_any(truncated) or _bool_any(terminated):
            break

    video_path = None
    if video:
        video_path = output_dir / f"episode_{spec.output_index:03d}.mp4"
        save_video(video_path, frames, fps=video_fps)

    return {
        "episode": spec.output_index,
        "seed": spec.seed,
        "source": spec.source,
        "dataset_episode_index": spec.dataset_episode_index,
        "steps": steps,
        "policy_steps": policy_steps,
        "sim_success": sim_first_success_step is not None,
        "world_model_success": wm_first_success_step is not None,
        "sim_first_success_step": sim_first_success_step,
        "world_model_first_success_step": wm_first_success_step,
        "sim_final_distance": sim_final_distance,
        "world_model_final_distance": wm_final_distance,
        "sim_min_distance": sim_min_distance if np.isfinite(sim_min_distance) else None,
        "world_model_min_distance": wm_min_distance if np.isfinite(wm_min_distance) else None,
        "video": str(video_path) if video_path is not None else None,
    }


def entry_to_world_model_observation(entry: Entry) -> Observation:
    """Convert a fixed-size policy entry into a valid-point world-model observation."""
    valid = np.asarray(entry["point_valid_mask"], dtype=bool)
    points = np.asarray(entry["point_cloud"], dtype=np.float32)[valid]
    robot_mask = np.asarray(entry["robot_mask"], dtype=bool)[valid]
    return Observation(
        point_cloud=points,
        point_features={},
        robot_mask=robot_mask,
        robot_state=RobotState(
            joint_positions=np.asarray(entry["agent_pos"], dtype=np.float32),
            tcp_pose=np.asarray(entry["tcp_pose"], dtype=np.float32),
        ),
        sim_gt=SimGroundTruth(
            task_name="PG3DReach",
            target_position=np.asarray(entry["target_position"], dtype=np.float32),
            success=bool(entry["success"]),
        ),
    )


def world_model_entry_from_rollout_step(
    rollout: Any,
    step_index: int,
    *,
    previous_entry: Entry,
    crop_config: PointCloudCropConfig,
    goal_thresh: float,
) -> Entry:
    """Convert one imagined rollout step into the fixed-size DP3 policy entry shape."""
    scene = rollout.scene_point_clouds[step_index]
    robot_mask = rollout.robot_masks[step_index]
    cropped = crop_point_cloud(scene, robot_mask=robot_mask, config=crop_config)
    target = np.asarray(previous_entry["target_position"], dtype=np.float32).reshape(3)
    eef = np.asarray(rollout.eef_path[step_index], dtype=np.float32).reshape(3)
    tcp_pose = np.asarray(previous_entry["tcp_pose"], dtype=np.float32).copy()
    if tcp_pose.shape[0] < 7:
        tcp_pose = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    tcp_pose[:3] = eef
    final_distance = float(np.linalg.norm(eef - target))
    return {
        "point_cloud": cropped["point_cloud"],
        "robot_mask": cropped["robot_mask"],
        "point_valid_mask": cropped["point_valid_mask"],
        "agent_pos": np.asarray(rollout.q[step_index], dtype=np.float32).copy(),
        "target_position": target.copy(),
        "tcp_pose": tcp_pose.astype(np.float32, copy=False),
        "success": final_distance <= goal_thresh,
        "final_distance": final_distance,
    }


def save_rerun_comparison(path: Path, timeline: list[dict[str, Any]]) -> None:
    """Write a Rerun overlay comparing world-model and simulator point-cloud timelines."""
    try:
        import rerun as rr
    except Exception as exc:
        raise RuntimeError(
            "Rerun export requires: "
            "uv sync --extra cu129 --extra maniskill --extra viz --group dev --group notebooks"
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    rr.init("pg3d_world_model_vs_maniskill", spawn=False)
    rr.save(str(path))
    for frame_idx, item in enumerate(timeline):
        rr.set_time_sequence("frame", frame_idx)
        _log_entry_points(rr, "world_model", item["world_model"], [0, 160, 255])
        _log_entry_points(rr, "sim", item["sim"], [255, 120, 0])
        target = np.asarray(item["target_position"], dtype=np.float32).reshape(1, 3)
        if np.all(np.isfinite(target)):
            rr.log("goal", rr.Points3D(target, colors=[0, 255, 0]))
        wm_tcp = np.asarray(item["world_model_tcp"], dtype=np.float32).reshape(1, 3)
        sim_tcp = np.asarray(item["sim_tcp"], dtype=np.float32).reshape(1, 3)
        if np.all(np.isfinite(wm_tcp)):
            rr.log("world_model/tcp", rr.Points3D(wm_tcp, colors=[0, 220, 255]))
        if np.all(np.isfinite(sim_tcp)):
            rr.log("sim/tcp", rr.Points3D(sim_tcp, colors=[255, 220, 0]))
    rr.disconnect()


def _log_entry_points(rr: Any, prefix: str, entry: Entry, robot_color: list[int]) -> None:
    valid = np.asarray(entry["point_valid_mask"], dtype=bool)
    points = np.asarray(entry["point_cloud"], dtype=np.float32)[valid]
    if points.size == 0:
        return
    robot_mask = np.asarray(entry["robot_mask"], dtype=bool)[valid]
    scene_points = points[~robot_mask]
    robot_points = points[robot_mask]
    if scene_points.size:
        rr.log(f"{prefix}/scene_points", rr.Points3D(scene_points, colors=[120, 120, 120]))
    if robot_points.size:
        rr.log(f"{prefix}/robot_points", rr.Points3D(robot_points, colors=robot_color))


def _append_timeline_pair(
    timeline: list[dict[str, Any]],
    *,
    episode: int,
    step: int,
    policy_step: int,
    world_model_entry: Entry,
    sim_entry: Entry,
) -> None:
    timeline.append(
        {
            "episode": episode,
            "step": step,
            "policy_step": policy_step,
            "world_model": _copy_entry(world_model_entry),
            "sim": _copy_entry(sim_entry),
            "target_position": np.asarray(world_model_entry["target_position"], dtype=np.float32),
            "world_model_tcp": np.asarray(world_model_entry["tcp_pose"], dtype=np.float32)[:3],
            "sim_tcp": np.asarray(sim_entry["tcp_pose"], dtype=np.float32)[:3],
        }
    )


def _env_kwargs(metadata: dict[str, Any], *, render_mode: str | None) -> dict[str, Any]:
    env_kwargs = dict(metadata["env_kwargs"])
    env_kwargs["obs_mode"] = "pointcloud"
    env_kwargs["num_envs"] = 1
    if render_mode is None:
        env_kwargs.pop("render_mode", None)
    else:
        env_kwargs["render_mode"] = render_mode
    return env_kwargs


def _copy_entry(entry: Entry) -> Entry:
    return {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in entry.items()
    }


def _entry_distance(entry: Entry) -> float:
    return float(np.asarray(entry["final_distance"], dtype=np.float32).reshape(-1)[0])


def _valid_robot_count(entry: Entry) -> int:
    robot = np.asarray(entry["robot_mask"], dtype=bool)
    valid = np.asarray(entry["point_valid_mask"], dtype=bool)
    return int(np.count_nonzero(robot & valid))


def _env_task_name(env: Any) -> str:
    unwrapped = getattr(env, "unwrapped", env)
    spec = getattr(unwrapped, "spec", None)
    return str(getattr(spec, "id", "unknown"))


def _action_mode(value: str) -> ActionMode:
    if value not in {"abs_joint", "delta_joint"}:
        raise ValueError(f"unsupported action_mode {value!r}")
    return value  # type: ignore[return-value]


if __name__ == "__main__":
    raise SystemExit(main())
