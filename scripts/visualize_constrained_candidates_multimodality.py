from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr

from pg3d.envs.maniskill_adapter import register_pg3d_reach_envs
from pg3d.envs.maniskill_adapter.dataset import load_reach_metadata
from pg3d.policies.dp3.checkpoint import load_reach_policy_from_checkpoint
from pg3d.utils.arrays import bool_any as _bool_any
from pg3d.utils.arrays import bool_info as _bool_info
from pg3d.utils.arrays import float_info as _float_info
from pg3d.utils.devices import select_device
from pg3d.utils.serialization import jsonable as _jsonable
from scripts.rollout_dp3_reach_policy import (
    _action_mode,
    append_obs_window,
    crop_config_from_metadata,
    make_initial_obs_window,
    obs_window_to_torch,
    policy_action_to_sim_action,
    rollout_observation_entry,
)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
        import rerun as rr
    except Exception as exc:
        print(
            f"Failed to import visualization stack: {type(exc).__name__}: {exc}",
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
        args.checkpoint,
        device=device,
        prefer_ema=args.checkpoint_model == "ema",
    )
    if args.scheduler_type != "ddim" or args.num_inference_steps is not None:
        policy.set_scheduler(args.scheduler_type, num_inference_steps=args.num_inference_steps)
        print(
            f"scheduler overridden: type={args.scheduler_type} "
            f"num_inference_steps={policy.num_inference_steps}"
        )

    metadata = load_reach_metadata(args.dataset)
    zarr_root = zarr.open_group(str(args.dataset), mode="r")
    metadata_episodes = metadata.get("episodes", [])
    if not metadata_episodes:
        raise ValueError("dataset metadata has no episode seeds")
    if any("seed" not in episode for episode in metadata_episodes):
        raise ValueError("every dataset metadata episode must contain a seed")
    episode_seeds = [int(episode["seed"]) for episode in metadata_episodes]
    _validate_episode_index(args.episode_index, episode_seeds)
    crop_config = crop_config_from_metadata(metadata)
    action_mode = _action_mode(str(metadata.get("action_mode", "abs_joint")))
    env_kwargs = dict(metadata["env_kwargs"])
    env_kwargs["obs_mode"] = "pointcloud"
    env_kwargs["render_mode"] = "rgb_array"
    env_kwargs["num_envs"] = 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    env: Any | None = None
    try:
        env = gym.make(str(metadata["env_id"]), **env_kwargs)
        if args.search_episodes > 1:
            result = _find_most_diverse_episode(
                args=args,
                env=env,
                policy=policy,
                episode_seeds=episode_seeds,
                action_mode=action_mode,
                crop_config=crop_config,
                device=device,
                zarr_root=zarr_root,
            )
        else:
            zarr_context = _zarr_episode_context(zarr_root, args.episode_index)
            result = _evaluate_episode_candidates(
                env=env,
                policy=policy,
                rollout_seed=episode_seeds[args.episode_index],
                episode_index=args.episode_index,
                action_mode=action_mode,
                crop_config=crop_config,
                device=device,
                zarr_context=zarr_context,
                args=args,
            )
    finally:
        if env is not None:
            env.close()

    candidates = result["candidates"]
    candidate_paths = [candidate["tcp_path"] for candidate in candidates]
    successful_count = len(result["successful_candidates"])
    print(
        "writing all sampled policy candidates: "
        f"episode_index={result['episode_index']} rollout_seed={result['rollout_seed']} "
        f"plotted={len(candidate_paths)}/{result['sampled_count']} "
        f"successful={successful_count}/{result['sampled_count']} "
        f"diversity_score={result['diversity']['score']:.5f} "
        f"total_path_points={sum(int(path.shape[0]) for path in candidate_paths)} "
        f"clusters={_format_trajectory_cluster_counts(result['trajectory_clusters'])}",
        flush=True,
    )
    _write_rerun(
        rr=rr,
        output=args.output,
        initial_entry=result["initial_entry"],
        candidate_paths=candidate_paths,
        candidate_summaries=candidates,
    )
    if args.video is not None:
        _write_matplotlib_video(
            output=args.video,
            initial_entry=result["initial_entry"],
            candidate_paths=candidate_paths,
            candidate_summaries=candidates,
            fps=args.video_fps,
        )

    summary = {
        "dataset": str(args.dataset),
        "checkpoint": str(args.checkpoint),
        "episode_index": result["episode_index"],
        "rollout_seed": result["rollout_seed"],
        "zarr_episode_start": result["zarr_episode_context"],
        "candidate_source": "policy_unconditioned",
        "sample_seed_start": args.seed,
        "sampled_count": result["sampled_count"],
        "plotted_count": len(candidates),
        "successful_count": successful_count,
        "search_episodes": args.search_episodes,
        "search_start_episode_index": args.episode_index,
        "search_results": result.get("search_results", []),
        "diversity": result["diversity"],
        "trajectory_clustering": result["trajectory_clustering"],
        "trajectory_clusters": result["trajectory_clusters"],
        "candidates": candidates,
        "rerun": str(args.output),
        "video": str(args.video) if args.video is not None else None,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"saved rerun: {args.output}")
    if args.video is not None:
        print(f"saved video: {args.video}")
    print(f"saved summary: {summary_path}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize natural stochastic DP3 checkpoint rollout candidates. "
            "This script does not use trajectory-family conditioning."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-model", choices=["ema", "raw"], default="ema")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument(
        "--search-episodes",
        type=int,
        default=1,
        help="number of consecutive dataset episodes to try; saves the one with max diversity",
    )
    parser.add_argument(
        "--min-successful-candidates",
        type=int,
        default=2,
        help="minimum successful candidates preferred during diversity search",
    )
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument(
        "--success-distance",
        type=float,
        default=0.03,
        help="mark a candidate as successful if env success is true or min distance is within this",
    )
    parser.add_argument(
        "--stop-on-success-distance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="stop a candidate rollout once it reaches --success-distance",
    )
    parser.add_argument(
        "--stagnation-patience",
        type=int,
        default=48,
        help=(
            "stop a candidate after this many steps without improving min distance; "
            "set 0 to disable"
        ),
    )
    parser.add_argument(
        "--stagnation-min-improvement",
        type=float,
        default=1e-3,
        help="minimum distance improvement in meters that resets stagnation patience",
    )
    parser.add_argument(
        "--min-rollout-steps",
        type=int,
        default=16,
        help="minimum sim steps before success-distance or stagnation early stop can trigger",
    )
    parser.add_argument("--replan-stride", type=int, default=None)
    parser.add_argument(
        "--fixed-replan-seed",
        type=int,
        default=None,
        help=(
            "if set, reset the PyTorch RNG to this seed before every predict_action call "
            "within each candidate rollout. Forces the same DDPM noise initialization at "
            "every replan step so the policy always picks the same trajectory mode. "
            "Use to test whether mode-switching between replans is causing stalls."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gripper-open", type=float, default=0.04)
    parser.add_argument(
        "--cluster-distance-threshold",
        type=float,
        default=None,
        help=(
            "RMS distance threshold in meters for trajectory clustering; defaults to an "
            "automatic threshold from nearest-neighbor trajectory distances"
        ),
    )
    parser.add_argument(
        "--cluster-resampled-points",
        type=int,
        default=32,
        help="number of time-normalized points used for trajectory distance clustering",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/constrained_candidates/natural_policy_candidates.rrd"),
    )
    parser.add_argument(
        "--video",
        type=str,
        default="artifacts/constrained_candidates/natural_policy_candidates.mp4",
        help="optional matplotlib MP4 output; pass --video none to disable",
    )
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=10,
        help="print progress every N sampled candidates / video frames",
    )
    parser.add_argument(
        "--scheduler-type",
        choices=["ddim", "ddpm"],
        default="ddim",
        help=(
            "Diffusion scheduler for inference. 'ddpm' is stochastic — each candidate draws "
            "a different mode, which restores multimodality and reduces saddle-point stalls. "
            "'ddim' (default) is deterministic and faster but can average symmetric modes."
        ),
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help=(
            "Override the number of denoising steps at inference. "
            "ddpm typically needs more steps (e.g. 100) than ddim (default is num_train_timesteps)."
        ),
    )
    args = parser.parse_args(argv)
    args.video = None if args.video.lower() in {"", "none", "null", "off"} else Path(args.video)
    if args.search_episodes <= 0:
        raise ValueError("--search-episodes must be positive")
    if args.min_successful_candidates <= 0:
        raise ValueError("--min-successful-candidates must be positive")
    if args.candidates <= 0:
        raise ValueError("--candidates must be positive")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.success_distance <= 0:
        raise ValueError("--success-distance must be positive")
    if args.stagnation_patience < 0:
        raise ValueError("--stagnation-patience must be non-negative")
    if args.stagnation_min_improvement < 0:
        raise ValueError("--stagnation-min-improvement must be non-negative")
    if args.min_rollout_steps < 0:
        raise ValueError("--min-rollout-steps must be non-negative")
    if args.replan_stride is not None and args.replan_stride <= 0:
        raise ValueError("--replan-stride must be positive")
    if args.video_fps <= 0:
        raise ValueError("--video-fps must be positive")
    if args.progress_interval <= 0:
        raise ValueError("--progress-interval must be positive")
    if args.cluster_distance_threshold is not None and args.cluster_distance_threshold <= 0:
        raise ValueError("--cluster-distance-threshold must be positive when set")
    if args.cluster_resampled_points <= 1:
        raise ValueError("--cluster-resampled-points must be greater than 1")
    return args


def _validate_episode_index(index: int, episode_seeds: list[int]) -> None:
    if index < 0 or index >= len(episode_seeds):
        raise IndexError(
            f"--episode-index {index} is outside dataset episode range "
            f"[0, {len(episode_seeds) - 1}]"
        )


def _zarr_episode_context(zarr_root: Any, episode_index: int) -> dict[str, Any]:
    """Read the exact initial state and goal for one Zarr episode."""
    episode_ends = np.asarray(zarr_root["meta"]["episode_ends"][:], dtype=np.int64)
    if episode_index < 0 or episode_index >= len(episode_ends):
        raise IndexError(
            f"episode index {episode_index} is outside Zarr episode range "
            f"[0, {len(episode_ends) - 1}]"
        )
    episode_start = 0 if episode_index == 0 else int(episode_ends[episode_index - 1])
    episode_end = int(episode_ends[episode_index])
    if episode_start >= episode_end:
        raise ValueError(f"Zarr episode {episode_index} is empty")

    data = zarr_root["data"]
    required = (
        "state",
        "target_position",
        "tcp_pose",
        "point_cloud",
        "robot_mask",
        "point_valid_mask",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"dataset is missing required Zarr arrays: {missing}")
    return {
        "episode_index": int(episode_index),
        "episode_start": episode_start,
        "episode_end": episode_end,
        "state": np.asarray(data["state"][episode_start], dtype=np.float32).copy(),
        "target_position": np.asarray(
            data["target_position"][episode_start], dtype=np.float32
        ).reshape(3).copy(),
        "tcp_pose": np.asarray(data["tcp_pose"][episode_start], dtype=np.float32).copy(),
        "point_cloud": np.asarray(data["point_cloud"][episode_start], dtype=np.float32).copy(),
        "robot_mask": np.asarray(data["robot_mask"][episode_start], dtype=bool).copy(),
        "point_valid_mask": np.asarray(
            data["point_valid_mask"][episode_start], dtype=bool
        ).copy(),
    }


def _reset_to_zarr_episode(
    env: Any,
    *,
    rollout_seed: int,
    zarr_context: dict[str, Any],
) -> tuple[Any, Any]:
    """Reset simulator bookkeeping, then restore the selected Zarr start and goal."""
    env.reset(seed=rollout_seed, options={"reconfigure": True})
    unwrapped = env.unwrapped
    robot = unwrapped.agent.robot
    current_qpos = np.asarray(robot.get_qpos(), dtype=np.float32)
    stored_qpos = np.asarray(zarr_context["state"], dtype=np.float32).reshape(-1)
    if stored_qpos.size > current_qpos.size:
        raise ValueError(
            f"stored state has {stored_qpos.size} joints, simulator has {current_qpos.size}"
        )
    qpos = current_qpos.copy().reshape(-1)
    qpos[: stored_qpos.size] = stored_qpos
    qpos = qpos.reshape(current_qpos.shape)
    robot.set_qpos(qpos)
    if hasattr(robot, "set_qvel"):
        robot.set_qvel(np.zeros_like(qpos, dtype=np.float32))

    from mani_skill.utils.structs.pose import Pose

    target = np.asarray(zarr_context["target_position"], dtype=np.float32).reshape(1, 3)
    start = np.asarray(zarr_context["tcp_pose"], dtype=np.float32).reshape(-1)[:3].reshape(1, 3)
    unwrapped.goal_site.set_pose(Pose.create_from_pq(target))
    if getattr(unwrapped, "start_site", None) is not None:
        unwrapped.start_site.set_pose(Pose.create_from_pq(start))

    info = unwrapped.get_info()
    obs = unwrapped.get_obs(info)
    return obs, info


def _apply_zarr_initial_entry(
    entry: dict[str, Any], zarr_context: dict[str, Any]
) -> dict[str, Any]:
    """Make the first policy/visualization entry byte-for-byte episode-indexed."""
    result = dict(entry)
    for key in (
        "point_cloud",
        "robot_mask",
        "point_valid_mask",
        "target_position",
        "tcp_pose",
    ):
        result[key] = np.asarray(zarr_context[key]).copy()
    result["agent_pos"] = np.asarray(zarr_context["state"], dtype=np.float32).copy()
    result["final_distance"] = float(
        np.linalg.norm(
            np.asarray(result["tcp_pose"], dtype=np.float32).reshape(-1)[:3]
            - np.asarray(result["target_position"], dtype=np.float32).reshape(3)
        )
    )
    result["success"] = False
    return result


def _zarr_context_summary(zarr_context: dict[str, Any]) -> dict[str, Any]:
    return {
        "episode_index": int(zarr_context["episode_index"]),
        "episode_start": int(zarr_context["episode_start"]),
        "episode_end": int(zarr_context["episode_end"]),
        "start_tcp": np.asarray(zarr_context["tcp_pose"], dtype=np.float32)[:3].tolist(),
        "target_position": np.asarray(
            zarr_context["target_position"], dtype=np.float32
        ).tolist(),
    }


def _find_most_diverse_episode(
    *,
    args: argparse.Namespace,
    env: Any,
    policy: Any,
    episode_seeds: list[int],
    action_mode: str,
    crop_config: Any,
    device: torch.device,
    zarr_root: Any,
) -> dict[str, Any]:
    end_index = min(args.episode_index + args.search_episodes, len(episode_seeds))
    search_results: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for episode_index in range(args.episode_index, end_index):
        zarr_context = _zarr_episode_context(zarr_root, episode_index)
        result = _evaluate_episode_candidates(
            env=env,
            policy=policy,
            rollout_seed=episode_seeds[episode_index],
            episode_index=episode_index,
            action_mode=action_mode,
            crop_config=crop_config,
            device=device,
            zarr_context=zarr_context,
            args=args,
        )
        search_row = {
            "episode_index": episode_index,
            "rollout_seed": result["rollout_seed"],
            "sampled_count": result["sampled_count"],
            "successful_count": len(result["successful_candidates"]),
            "diversity": result["diversity"],
            "trajectory_clustering": result["trajectory_clustering"],
            "trajectory_clusters": result["trajectory_clusters"],
        }
        search_results.append(search_row)
        print(
            "search episode "
            f"{episode_index}: successes={search_row['successful_count']}/"
            f"{search_row['sampled_count']} "
            f"score={search_row['diversity']['score']:.5f} "
            f"mean_pairwise={search_row['diversity']['mean_pairwise_distance']:.5f}",
            flush=True,
        )
        if _is_better_diversity_result(result, best, args=args):
            best = result
    if best is None:
        raise RuntimeError("diversity search produced no candidate results")
    best["search_results"] = search_results
    if len(best["successful_candidates"]) < args.min_successful_candidates:
        print(
            "warning: best episode has fewer successful candidates than requested: "
            f"{len(best['successful_candidates'])}/{args.min_successful_candidates}",
            file=sys.stderr,
        )
    return best


def _is_better_diversity_result(
    candidate: dict[str, Any],
    current: dict[str, Any] | None,
    *,
    args: argparse.Namespace,
) -> bool:
    if current is None:
        return True
    candidate_successes = len(candidate["successful_candidates"])
    current_successes = len(current["successful_candidates"])
    candidate_meets = candidate_successes >= args.min_successful_candidates
    current_meets = current_successes >= args.min_successful_candidates
    if candidate_meets != current_meets:
        return candidate_meets
    if candidate["diversity"]["score"] != current["diversity"]["score"]:
        return candidate["diversity"]["score"] > current["diversity"]["score"]
    return candidate_successes > current_successes


def _evaluate_episode_candidates(
    *,
    env: Any,
    policy: Any,
    rollout_seed: int,
    episode_index: int,
    action_mode: str,
    crop_config: Any,
    device: torch.device,
    zarr_context: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    obs, info = _reset_to_zarr_episode(
        env,
        rollout_seed=rollout_seed,
        zarr_context=zarr_context,
    )
    initial_entry = rollout_observation_entry(obs, info, env=env, crop_config=crop_config)
    initial_entry = _apply_zarr_initial_entry(initial_entry, zarr_context)
    print(
        f"episode {episode_index}: sampling {args.candidates} candidates "
        f"rollout_seed={rollout_seed} max_steps={args.steps} "
        f"zarr_row={zarr_context['episode_start']} "
        f"start={np.asarray(zarr_context['tcp_pose'])[:3].tolist()} "
        f"target={np.asarray(zarr_context['target_position']).tolist()}",
        flush=True,
    )
    candidates: list[dict[str, Any]] = []
    for idx in range(args.candidates):
        candidate = _sample_candidate_path(
            env=env,
            policy=policy,
            rollout_seed=rollout_seed,
            sample_seed=int(args.seed + episode_index * args.candidates + idx),
            action_mode=action_mode,
            crop_config=crop_config,
            device=device,
            max_steps=args.steps,
            replan_stride=args.replan_stride,
            fixed_replan_seed=args.fixed_replan_seed,
            gripper_open=args.gripper_open,
            candidate_index=idx,
            success_distance=args.success_distance,
            stop_on_success_distance=args.stop_on_success_distance,
            stagnation_patience=args.stagnation_patience,
            stagnation_min_improvement=args.stagnation_min_improvement,
            min_rollout_steps=args.min_rollout_steps,
            zarr_context=zarr_context,
        )
        candidates.append(candidate)
        if (
            (idx + 1) % args.progress_interval == 0
            or idx == 0
            or idx + 1 == args.candidates
        ):
            success_count = sum(1 for item in candidates if bool(item["success"]))
            print(
                f"episode {episode_index}: sampled {idx + 1}/{args.candidates} "
                f"success={success_count} latest_steps={candidate['steps']} "
                f"latest_min_dist={candidate['min_distance']:.4f}",
                flush=True,
            )
    clustering = _annotate_candidate_trajectory_clusters(
        candidates,
        distance_threshold=args.cluster_distance_threshold,
        resampled_points=args.cluster_resampled_points,
    )
    trajectory_clusters = _trajectory_cluster_counts(candidates)
    print(
        f"episode {episode_index}: trajectory clusters "
        f"{_format_trajectory_cluster_counts(trajectory_clusters)} "
        f"threshold={clustering['distance_threshold']:.5f}m",
        flush=True,
    )
    successful = [candidate for candidate in candidates if bool(candidate["success"])]
    return {
        "episode_index": episode_index,
        "rollout_seed": rollout_seed,
        "zarr_episode_context": _zarr_context_summary(zarr_context),
        "initial_entry": initial_entry,
        "sampled_count": len(candidates),
        "candidates": candidates,
        "successful_candidates": successful,
        "trajectory_clustering": clustering,
        "trajectory_clusters": trajectory_clusters,
        "diversity": _trajectory_diversity([candidate["tcp_path"] for candidate in successful]),
    }


def _trajectory_diversity(paths: list[np.ndarray]) -> dict[str, Any]:
    if len(paths) < 2:
        return {
            "score": 0.0,
            "mean_pairwise_distance": 0.0,
            "max_pairwise_distance": 0.0,
            "path_count": len(paths),
            "resampled_points": 0,
        }
    resampled = np.stack([_resample_path(path, points=32) for path in paths], axis=0)
    flat = resampled.reshape(resampled.shape[0], -1)
    distances: list[float] = []
    for idx in range(flat.shape[0]):
        diff = flat[idx + 1 :] - flat[idx]
        if diff.size:
            distances.extend(np.sqrt(np.mean(diff * diff, axis=1)).astype(float).tolist())
    distances_array = np.asarray(distances, dtype=np.float32)
    mean_pairwise = float(np.mean(distances_array)) if distances_array.size else 0.0
    max_pairwise = float(np.max(distances_array)) if distances_array.size else 0.0
    return {
        "score": mean_pairwise,
        "mean_pairwise_distance": mean_pairwise,
        "max_pairwise_distance": max_pairwise,
        "path_count": len(paths),
        "resampled_points": int(resampled.shape[1]),
    }


def _resample_path(path: np.ndarray, *, points: int) -> np.ndarray:
    path = np.asarray(path, dtype=np.float32).reshape(-1, 3)
    if path.shape[0] == 0:
        return np.zeros((points, 3), dtype=np.float32)
    if path.shape[0] == 1:
        return np.repeat(path, points, axis=0)
    source = np.linspace(0.0, 1.0, path.shape[0], dtype=np.float32)
    target = np.linspace(0.0, 1.0, points, dtype=np.float32)
    return np.stack(
        [np.interp(target, source, path[:, dim]) for dim in range(3)],
        axis=1,
    ).astype(np.float32)


def _annotate_candidate_trajectory_clusters(
    candidates: list[dict[str, Any]],
    *,
    distance_threshold: float | None,
    resampled_points: int,
) -> dict[str, Any]:
    paths = [
        np.asarray(candidate["tcp_path"], dtype=np.float32).reshape(-1, 3)
        for candidate in candidates
    ]
    if not paths:
        return {
            "method": "connected_components_pairwise_rms",
            "cluster_count": 0,
            "distance_threshold": 0.0,
            "distance_threshold_source": "empty",
            "resampled_points": int(resampled_points),
            "pairwise_distance": _distance_stats(np.zeros(0, dtype=np.float32)),
            "nearest_neighbor_distance": _distance_stats(np.zeros(0, dtype=np.float32)),
        }

    resampled = np.stack([_resample_path(path, points=resampled_points) for path in paths], axis=0)
    flat = resampled.reshape(resampled.shape[0], -1)
    distances = _pairwise_rms_distances(flat)
    upper = distances[np.triu_indices(distances.shape[0], k=1)]
    if distance_threshold is None:
        threshold = _auto_trajectory_cluster_threshold(distances)
        threshold_source = "auto_nearest_neighbor"
    else:
        threshold = float(distance_threshold)
        threshold_source = "argument"
    labels = _connected_components_from_threshold(distances, threshold=threshold)
    labels = _renumber_cluster_labels(labels)

    cluster_means: dict[int, np.ndarray] = {}
    cluster_sizes: dict[int, int] = {}
    for cluster_id in sorted(set(labels)):
        indices = np.flatnonzero(labels == cluster_id)
        cluster_sizes[int(cluster_id)] = int(indices.size)
        cluster_means[int(cluster_id)] = flat[indices].mean(axis=0)

    for idx, candidate in enumerate(candidates):
        cluster_id = int(labels[idx])
        centroid_distance = float(
            np.sqrt(np.mean((flat[idx] - cluster_means[cluster_id]) ** 2))
        )
        candidate.update(
            {
                "trajectory_cluster_id": cluster_id,
                "trajectory_cluster_name": f"cluster_{cluster_id:02d}",
                "trajectory_cluster_size": cluster_sizes[cluster_id],
                "trajectory_cluster_centroid_distance": centroid_distance,
                "trajectory_cluster_distance_threshold": threshold,
            }
        )

    nearest = _nearest_neighbor_distances(distances)
    return {
        "method": "connected_components_pairwise_rms",
        "cluster_count": int(len(set(labels))),
        "distance_threshold": threshold,
        "distance_threshold_source": threshold_source,
        "resampled_points": int(resampled_points),
        "pairwise_distance": _distance_stats(upper),
        "nearest_neighbor_distance": _distance_stats(nearest),
    }


def _pairwise_rms_distances(flat_paths: np.ndarray) -> np.ndarray:
    flat_paths = np.asarray(flat_paths, dtype=np.float32)
    count = flat_paths.shape[0]
    distances = np.zeros((count, count), dtype=np.float32)
    for idx in range(count):
        diff = flat_paths[idx + 1 :] - flat_paths[idx]
        if diff.size:
            values = np.sqrt(np.mean(diff * diff, axis=1)).astype(np.float32)
            distances[idx, idx + 1 :] = values
            distances[idx + 1 :, idx] = values
    return distances


def _auto_trajectory_cluster_threshold(distances: np.ndarray) -> float:
    nearest = _nearest_neighbor_distances(distances)
    if nearest.size == 0:
        return 0.0
    nearest = nearest[np.isfinite(nearest)]
    if nearest.size == 0:
        return 0.0
    if float(np.max(nearest)) <= 1e-8:
        return 1e-6
    return float(max(np.percentile(nearest, 75) * 1.75, np.percentile(nearest, 90)))


def _nearest_neighbor_distances(distances: np.ndarray) -> np.ndarray:
    distances = np.asarray(distances, dtype=np.float32)
    if distances.shape[0] < 2:
        return np.zeros(0, dtype=np.float32)
    masked = distances.copy()
    np.fill_diagonal(masked, np.inf)
    return np.min(masked, axis=1).astype(np.float32)


def _connected_components_from_threshold(distances: np.ndarray, *, threshold: float) -> np.ndarray:
    count = distances.shape[0]
    labels = np.full(count, -1, dtype=np.int64)
    cluster_id = 0
    for start_idx in range(count):
        if labels[start_idx] >= 0:
            continue
        stack = [start_idx]
        labels[start_idx] = cluster_id
        while stack:
            idx = stack.pop()
            neighbors = np.flatnonzero((distances[idx] <= threshold) & (labels < 0))
            for neighbor in neighbors:
                labels[neighbor] = cluster_id
                stack.append(int(neighbor))
        cluster_id += 1
    return labels


def _renumber_cluster_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    rows: list[tuple[int, int, int]] = []
    for cluster_id in sorted(set(labels.tolist())):
        indices = np.flatnonzero(labels == cluster_id)
        rows.append((-int(indices.size), int(indices[0]), int(cluster_id)))
    mapping = {old: new for new, (_neg_size, _first_idx, old) in enumerate(sorted(rows))}
    return np.asarray([mapping[int(label)] for label in labels], dtype=np.int64)


def _distance_stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "min": 0.0, "p25": 0.0, "mean": 0.0, "p50": 0.0, "p75": 0.0, "max": 0.0}
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "p25": float(np.percentile(values, 25)),
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "max": float(np.max(values)),
    }


def _trajectory_cluster_counts(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    counts: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        cluster_id = int(candidate.get("trajectory_cluster_id", -1))
        cluster_name = str(candidate.get("trajectory_cluster_name", "unknown"))
        key = f"{cluster_id}:{cluster_name}"
        row = counts.setdefault(
            key,
            {
                "cluster_id": cluster_id,
                "cluster_name": cluster_name,
                "count": 0,
                "successful_count": 0,
            },
        )
        row["count"] += 1
        row["successful_count"] += int(bool(candidate.get("success", False)))
    return dict(sorted(counts.items(), key=lambda item: (item[1]["cluster_id"], item[0])))


def _format_trajectory_cluster_counts(counts: dict[str, dict[str, Any]]) -> str:
    if not counts:
        return "none"
    return ", ".join(
        f"{row['cluster_id']}:{row['cluster_name']}={row['count']}({row['successful_count']} ok)"
        for row in counts.values()
    )


def _sample_candidate_path(
    *,
    env: Any,
    policy: Any,
    rollout_seed: int,
    sample_seed: int,
    action_mode: str,
    crop_config: Any,
    device: torch.device,
    max_steps: int,
    replan_stride: int | None,
    fixed_replan_seed: int | None,
    gripper_open: float,
    candidate_index: int,
    success_distance: float,
    stop_on_success_distance: bool,
    stagnation_patience: int,
    stagnation_min_improvement: float,
    min_rollout_steps: int,
    zarr_context: dict[str, Any],
) -> dict[str, Any]:
    torch.manual_seed(sample_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sample_seed)
    obs, info = _reset_to_zarr_episode(
        env,
        rollout_seed=rollout_seed,
        zarr_context=zarr_context,
    )
    entry = rollout_observation_entry(obs, info, env=env, crop_config=crop_config)
    entry = _apply_zarr_initial_entry(entry, zarr_context)
    obs_window = make_initial_obs_window(entry, n_obs_steps=int(policy.n_obs_steps))
    tcp_path = [np.asarray(entry["tcp_pose"], dtype=np.float32).reshape(-1)[:3]]
    distances = [float(np.asarray(entry["final_distance"], dtype=np.float32).reshape(-1)[0])]
    steps = 0
    env_success = False
    first_success_step: int | None = None
    best_distance = float(np.nanmin(np.asarray(distances, dtype=np.float32)))
    steps_since_best = 0
    early_stop_reason: str | None = None
    while steps < max_steps:
        if fixed_replan_seed is not None:
            torch.manual_seed(fixed_replan_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(fixed_replan_seed)
        with torch.no_grad():
            policy_input = obs_window_to_torch(
                obs_window,
                device=device,
                goal_marker_points=int(policy.goal_marker_points),
                goal_marker_radius=float(policy.goal_marker_radius),
            )
            action_chunk = policy.predict_action(policy_input)["action"][0].detach().cpu().numpy()

        stride = replan_stride if replan_stride is not None else int(policy.n_action_steps)
        stop_rollout = False
        for policy_action in action_chunk[: min(stride, max_steps - steps)]:
            current_state = obs_window[-1]["agent_pos"]
            sim_action = policy_action_to_sim_action(
                policy_action,
                current_state,
                action_mode=action_mode,
                sim_action_dim=int(np.prod(env.action_space.shape)),
                low=getattr(env.action_space, "low", None),
                high=getattr(env.action_space, "high", None),
                gripper_open=gripper_open,
            )
            obs, _reward, terminated, truncated, info = env.step(sim_action)
            steps += 1
            entry = rollout_observation_entry(obs, info, env=env, crop_config=crop_config)
            obs_window = append_obs_window(obs_window, entry, n_obs_steps=int(policy.n_obs_steps))
            tcp_path.append(np.asarray(entry["tcp_pose"], dtype=np.float32).reshape(-1)[:3])
            distance = _float_info(info, "tcp_to_goal_dist", default=float("nan"))
            if not np.isfinite(distance):
                distance = float(
                    np.asarray(entry["final_distance"], dtype=np.float32).reshape(-1)[0]
                )
            distances.append(distance)
            if np.isfinite(distance) and distance < best_distance - stagnation_min_improvement:
                best_distance = float(distance)
                steps_since_best = 0
            else:
                steps_since_best += 1
            env_success = _bool_info(info, "success")
            if env_success and first_success_step is None:
                first_success_step = steps
            if env_success:
                early_stop_reason = "env_success"
            elif (
                stop_on_success_distance
                and steps >= min_rollout_steps
                and np.isfinite(best_distance)
                and best_distance <= success_distance
            ):
                early_stop_reason = "success_distance"
            elif (
                stagnation_patience > 0
                and steps >= min_rollout_steps
                and steps_since_best >= stagnation_patience
            ):
                early_stop_reason = "stagnation"
            elif _bool_any(terminated) or _bool_any(truncated):
                early_stop_reason = "terminated_or_truncated"
            if early_stop_reason is not None:
                stop_rollout = True
                break
        if stop_rollout:
            break

    path = np.asarray(tcp_path, dtype=np.float32)
    min_distance = float(np.nanmin(np.asarray(distances, dtype=np.float32)))
    accepted_success = bool(env_success or min_distance <= success_distance)
    return {
        "candidate_index": candidate_index,
        "sample_seed": sample_seed,
        "steps": steps,
        "success": accepted_success,
        "env_success": env_success,
        "first_success_step": first_success_step,
        "early_stop_reason": early_stop_reason,
        "tcp_path": path,
        "final_distance": float(distances[-1]) if distances else None,
        "min_distance": min_distance,
        "success_distance": success_distance,
        "source": "policy_unconditioned",
        "fixed_replan_seed": fixed_replan_seed,
    }


def _write_rerun(
    *,
    rr: Any,
    output: Path,
    initial_entry: dict[str, Any],
    candidate_paths: list[np.ndarray],
    candidate_summaries: list[dict[str, Any]],
) -> None:
    rr.init("pg3d_natural_policy_candidate_trajectories", spawn=False)
    rr.save(str(output))
    rr.set_time_sequence("step", 0)

    valid = np.asarray(initial_entry["point_valid_mask"], dtype=bool)
    points = np.asarray(initial_entry["point_cloud"], dtype=np.float32)[valid]
    robot_mask = np.asarray(initial_entry["robot_mask"], dtype=bool)[valid]
    scene_points = points[~robot_mask]
    if scene_points.size:
        rr.log(
            "world/point_cloud",
            rr.Points3D(scene_points, colors=[150, 150, 150], radii=0.003),
        )

    target = np.asarray(initial_entry["target_position"], dtype=np.float32).reshape(1, 3)
    start = np.asarray(initial_entry["tcp_pose"], dtype=np.float32).reshape(-1)[:3].reshape(1, 3)
    rr.log("world/goal", rr.Points3D(target, colors=[0, 255, 0], radii=0.018))
    rr.log("world/start_tcp", rr.Points3D(start, colors=[255, 220, 0], radii=0.014))
    for idx, path in enumerate(candidate_paths):
        if path.shape[0] < 2:
            continue
        success = bool(candidate_summaries[idx].get("success", False))
        color = _candidate_cluster_color(candidate_summaries[idx], dim_failed=True)
        cluster_id = int(candidate_summaries[idx].get("trajectory_cluster_id", -1))
        cluster_name = str(candidate_summaries[idx].get("trajectory_cluster_name", "unknown"))
        prefix = "success" if success else "failed"
        rr.log(
            f"world/candidates/{cluster_id:02d}_{cluster_name}/{prefix}_{idx:02d}",
            rr.LineStrips3D([path], colors=color),
            static=True,
        )
        rr.log(
            f"world/candidates/{cluster_id:02d}_{cluster_name}/end_{idx:02d}",
            rr.Points3D(path[-1:].astype(np.float32), colors=color, radii=0.01),
            static=True,
        )
    rr.disconnect()


def _write_matplotlib_video(
    *,
    output: Path,
    initial_entry: dict[str, Any],
    candidate_paths: list[np.ndarray],
    candidate_summaries: list[dict[str, Any]],
    fps: int,
) -> None:
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"writing video frames: output={output} fps={fps}", flush=True)
    valid = np.asarray(initial_entry["point_valid_mask"], dtype=bool)
    points = np.asarray(initial_entry["point_cloud"], dtype=np.float32)[valid]
    robot_mask = np.asarray(initial_entry["robot_mask"], dtype=bool)[valid]
    target = np.asarray(initial_entry["target_position"], dtype=np.float32).reshape(3)
    start = np.asarray(initial_entry["tcp_pose"], dtype=np.float32).reshape(-1)[:3]
    all_points = [target.reshape(1, 3), start.reshape(1, 3)]
    all_points.extend(path for path in candidate_paths if path.size)
    bounds = np.concatenate(all_points, axis=0)
    mins = np.min(bounds, axis=0)
    maxs = np.max(bounds, axis=0)
    mid = (mins + maxs) * 0.5
    span = max(float(np.max(maxs - mins)) * 1.15, 0.16)

    frames = []
    frame_count = max(24, fps * 4)
    for frame_idx in range(frame_count):
        fig = plt.figure(figsize=(8.0, 6.0), dpi=140)
        ax = fig.add_subplot(111, projection="3d")
        if points.size:
            scene = points[~robot_mask]
            if scene.size:
                ax.scatter(scene[:, 0], scene[:, 1], scene[:, 2], s=2, c="#9ca3af", alpha=0.16)
        ax.scatter([start[0]], [start[1]], [start[2]], c="gold", s=55, edgecolors="black")
        ax.scatter([target[0]], [target[1]], [target[2]], c="limegreen", s=70, edgecolors="black")
        for idx, path in enumerate(candidate_paths):
            if path.shape[0] < 2:
                continue
            success = bool(candidate_summaries[idx].get("success", False))
            color = _candidate_cluster_color(candidate_summaries[idx], dim_failed=False)
            color = tuple(np.asarray(color, dtype=np.float32) / 255.0)
            linewidth = 2.0 if success else 1.0
            alpha = 0.95 if success else 0.38
            ax.plot(
                path[:, 0],
                path[:, 1],
                path[:, 2],
                color=color,
                linewidth=linewidth,
                alpha=alpha,
            )
            ax.scatter(path[-1:, 0], path[-1:, 1], path[-1:, 2], color=color, s=22, alpha=alpha)
        legend_handles = _trajectory_cluster_legend_handles(candidate_summaries, Line2D)
        if legend_handles:
            ax.legend(handles=legend_handles, loc="upper left", fontsize=6, framealpha=0.72)
        ax.set_xlim(mid[0] - span * 0.5, mid[0] + span * 0.5)
        ax.set_ylim(mid[1] - span * 0.5, mid[1] + span * 0.5)
        ax.set_zlim(mid[2] - span * 0.5, mid[2] + span * 0.5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title("Natural checkpoint candidates")
        ax.view_init(elev=24, azim=-70 + 360.0 * frame_idx / frame_count)
        fig.tight_layout()
        fig.canvas.draw()
        image = _canvas_rgb_array(fig.canvas)
        frames.append(image.copy())
        plt.close(fig)
        if (
            (frame_idx + 1) % max(1, min(10, frame_count)) == 0
            or frame_idx == 0
            or frame_idx + 1 == frame_count
        ):
            print(f"video frame {frame_idx + 1}/{frame_count}", flush=True)
    print(f"encoding video: {output}", flush=True)
    imageio.mimsave(output, frames, fps=fps, macro_block_size=16)
    print(f"encoded video: {output}", flush=True)


def _trajectory_cluster_palette() -> list[tuple[int, int, int]]:
    return [
        (70, 130, 255),
        (255, 150, 40),
        (80, 210, 120),
        (210, 90, 255),
        (255, 90, 120),
        (80, 220, 220),
        (235, 205, 70),
        (130, 110, 255),
        (255, 105, 210),
        (105, 190, 90),
        (245, 120, 80),
        (120, 220, 170),
    ]


def _trajectory_cluster_color(cluster_id: int) -> tuple[int, int, int]:
    palette = _trajectory_cluster_palette()
    if cluster_id < 0:
        return (150, 150, 150)
    return palette[cluster_id % len(palette)]


def _candidate_cluster_color(
    candidate: dict[str, Any],
    *,
    dim_failed: bool,
) -> tuple[int, int, int]:
    cluster_id = int(candidate.get("trajectory_cluster_id", -1))
    color = _trajectory_cluster_color(cluster_id)
    if dim_failed and not bool(candidate.get("success", False)):
        return tuple(int(round(channel * 0.55)) for channel in color)
    return color


def _trajectory_cluster_legend_handles(
    candidates: list[dict[str, Any]],
    line_cls: Any,
) -> list[Any]:
    seen: dict[int, str] = {}
    for candidate in candidates:
        cluster_id = int(candidate.get("trajectory_cluster_id", -1))
        if cluster_id not in seen:
            seen[cluster_id] = str(candidate.get("trajectory_cluster_name", "unknown"))
    handles = []
    for cluster_id, cluster_name in sorted(seen.items()):
        color = np.asarray(_trajectory_cluster_color(cluster_id))
        handles.append(
            line_cls(
                [0],
                [0],
                color=tuple(color.astype(np.float32) / 255.0),
                lw=2.0,
                label=f"{cluster_id}:{cluster_name}",
            )
        )
    return handles


def _canvas_rgb_array(canvas: Any) -> np.ndarray:
    """Return an RGB image from Matplotlib canvases across backend versions."""
    width, height = canvas.get_width_height()
    if hasattr(canvas, "buffer_rgba"):
        rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
        return rgba[:, :, :3].copy()
    if hasattr(canvas, "tostring_rgb"):
        return np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
    if hasattr(canvas, "tostring_argb"):
        argb = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8).reshape(height, width, 4)
        return argb[:, :, [1, 2, 3]].copy()
    raise AttributeError("Matplotlib canvas cannot export RGB pixels")


if __name__ == "__main__":
    raise SystemExit(main())
