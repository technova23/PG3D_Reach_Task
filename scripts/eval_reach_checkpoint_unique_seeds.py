from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr

from pg3d.envs.maniskill_adapter import register_pg3d_reach_envs
from pg3d.envs.maniskill_adapter.dataset import load_reach_metadata
from pg3d.policies.dp3.checkpoint import (
    latest_reach_checkpoint,
    load_reach_policy_from_checkpoint,
)
from pg3d.utils.arrays import bool_any as _bool_any
from pg3d.utils.arrays import bool_info as _bool_info
from pg3d.utils.arrays import frame_to_numpy as _frame_to_numpy
from pg3d.utils.devices import select_device
from pg3d.utils.serialization import jsonable as _jsonable
from scripts.rollout_dp3_reach_policy import (
    ActionMode,
    PointCloudCropConfig,
    RolloutSpec,
    append_obs_window,
    crop_config_from_metadata,
    make_initial_obs_window,
    obs_window_to_torch,
    policy_action_to_sim_action,
    rollout_observation_entry,
    save_video,
)


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
        return 2

    register_pg3d_reach_envs()
    if args.dataset is not None:
        metadata = load_reach_metadata(args.dataset)
    else:
        # fresh mode without a dataset: minimal metadata from CLI args; crop_config will use defaults
        metadata = {
            "env_id": args.env_id,
            "env_kwargs": {"obs_mode": "pointcloud", "num_envs": 1},
            "action_mode": "abs_joint",
            "goal_thresh": 0.025,
            "episodes": [],
        }
    device = select_device(args.device)
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    policy = load_reach_policy_from_checkpoint(
        checkpoint_path,
        device=device,
        prefer_ema=args.checkpoint_model == "ema",
    )
    checkpoint_info = _checkpoint_diagnostics(checkpoint_path)
    action_mode = _action_mode(str(metadata.get("action_mode", "abs_joint")))
    crop_config = crop_config_from_metadata(metadata)
    # When --dataset is absent (fresh mode with fallback metadata), num_points defaults to 512.
    # Auto-correct it from the policy's actual input shape so the crop matches training.
    if args.dataset is None and args.crop_num_points is None:
        obs_encoder = getattr(policy, "obs_encoder", None)
        pc_shape = getattr(obs_encoder, "point_cloud_shape", None)
        goal_pts = int(getattr(policy, "goal_marker_points", 0))
        if pc_shape is not None:
            inferred_num_points = int(pc_shape[0]) - goal_pts
            if inferred_num_points > 0 and inferred_num_points != crop_config.num_points:
                print(
                    f"info: auto-correcting crop num_points from {crop_config.num_points} "
                    f"to {inferred_num_points} (= policy point_cloud_shape[0]={pc_shape[0]} "
                    f"- goal_marker_points={goal_pts})",
                    flush=True,
                )
                crop_config = PointCloudCropConfig(
                    bounds=crop_config.bounds,
                    num_points=inferred_num_points,
                    robot_point_fraction=crop_config.robot_point_fraction,
                )
    elif args.crop_num_points is not None:
        crop_config = PointCloudCropConfig(
            bounds=crop_config.bounds,
            num_points=args.crop_num_points,
            robot_point_fraction=crop_config.robot_point_fraction,
        )
    goal_thresh = (
        float(args.goal_thresh)
        if args.goal_thresh is not None
        else float(dict(metadata.get("env_kwargs", {})).get("goal_thresh", 0.025))
    )
    dataset_episode_seeds = [
        int(episode["seed"]) for episode in metadata.get("episodes", []) if "seed" in episode
    ]
    if args.source == "dataset":
        unique_indices = _unique_seed_indices(
            dataset_episode_seeds,
            count=args.episodes,
            start_index=args.start_index,
        )
        if not unique_indices:
            raise RuntimeError("no unique-seed dataset episodes selected")
    else:
        unique_indices = []
    run_config = _run_config_summary(
        metadata=metadata,
        policy=policy,
        checkpoint_info=checkpoint_info,
        action_mode=action_mode,
        crop_config=crop_config,
        max_steps=args.max_steps,
        execution_horizon_chunks=args.execution_horizon_chunks,
    )
    print("eval config: " + json.dumps(_jsonable(run_config), sort_keys=True), flush=True)
    if args.print_config_only:
        return 0

    zarr_root = (
        zarr.open_group(str(args.dataset), mode="r") if args.source == "dataset" else None
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    metrics_path = args.output_dir / "metrics.jsonl"
    env: Any | None = None
    try:
        env_make_kwargs = _env_kwargs(metadata)
        if args.max_episode_steps is not None:
            env_make_kwargs["max_episode_steps"] = args.max_episode_steps
        if args.video:
            env_make_kwargs["render_mode"] = "rgb_array"
        env = gym.make(str(metadata["env_id"]), **env_make_kwargs)
        video_dir = (args.output_dir / "videos") if args.video else None
        if video_dir is not None:
            video_dir.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", encoding="utf-8") as metrics_file:
            episode_counter = 0
            if args.source == "dataset":
                for seed_rank, dataset_idx in enumerate(unique_indices):
                    seed = int(dataset_episode_seeds[dataset_idx])
                    episode_number = seed_rank + 1
                    viz_this = _should_viz(args, episode_number)
                    seed_paths: list[dict[str, Any]] = []
                    for sample_idx in range(args.samples_per_seed):
                        spec = RolloutSpec(
                            output_index=seed_rank * args.samples_per_seed + sample_idx,
                            seed=seed,
                            source="dataset",
                            dataset_episode_index=int(dataset_idx),
                        )
                        # Distinct, reproducible diffusion noise per (seed, sample).
                        # _seed_policy_noise seeds the global RNG as a backup;
                        # gen is the primary fix — it bypasses the global RNG entirely
                        # and is safe on both CPU and CUDA.
                        _seed_policy_noise(seed, sample_idx)
                        gen = torch.Generator(device=device)
                        gen.manual_seed(
                            (int(seed) * 1_000_003 + int(sample_idx) * 97) % (2**31 - 1)
                        )
                        row = run_reach_episode(
                            env=env,
                            policy=policy,
                            spec=spec,
                            zarr_root=zarr_root,
                            action_mode=action_mode,
                            crop_config=crop_config,
                            goal_thresh=goal_thresh,
                            max_steps=args.max_steps,
                            post_success_steps=args.post_success_steps,
                            execution_horizon_chunks=args.execution_horizon_chunks,
                            gripper_open=args.gripper_open,
                            device=device,
                            generator=gen,
                            video_path=(
                                video_dir
                                / f"dataset_ep{dataset_idx:03d}_seed{seed}_s{sample_idx}.mp4"
                                if video_dir is not None
                                else None
                            ),
                            video_fps=args.video_fps,
                        )
                        row["sample_index"] = sample_idx
                        tcp_path = row.pop("tcp_path", None)
                        if viz_this and tcp_path is not None:
                            seed_paths.append(
                                {
                                    "sample_index": sample_idx,
                                    "tcp_path": np.asarray(tcp_path, dtype=np.float32),
                                    "success": bool(row["success"]),
                                }
                            )
                        rows.append(row)
                        metrics_file.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
                        metrics_file.flush()
                        episode_counter += 1
                        print(
                            f"episode={episode_counter} dataset_episode={row['dataset_episode_index']} "
                            f"seed={row['seed']} sample={sample_idx + 1}/{args.samples_per_seed} "
                            f"success={row['success']} "
                            f"final={_format_optional(row['final_distance'])} "
                            f"min={_format_optional(row['min_distance'])} "
                            f"steps={row['steps']} "
                            f"start={_format_xyz(row.get('start_tcp'))} "
                            f"goal={_format_xyz(row.get('target_position'))}"
                            + (f" video={row['video']}" if row.get("video") else ""),
                            flush=True,
                        )
                        # When visualizing multimodality we want every sample drawn, so
                        # do not stop early on the first success.
                        if row["success"] and not viz_this:
                            break  # got success — skip remaining samples for this seed
                    if viz_this and seed_paths:
                        _plot_multimodality(
                            output_dir=args.output_dir,
                            episode_number=episode_number,
                            seed=seed,
                            target=np.asarray(row["target_position"], dtype=np.float32),
                            samples=seed_paths,
                            goal_thresh=goal_thresh,
                        )
            else:
                for ep_idx in range(args.episodes):
                    fresh_seed = args.seed + ep_idx
                    episode_number = ep_idx + 1
                    viz_this = _should_viz(args, episode_number)
                    seed_paths = []
                    for sample_idx in range(args.samples_per_seed):
                        spec = RolloutSpec(
                            output_index=ep_idx * args.samples_per_seed + sample_idx,
                            seed=fresh_seed,
                            source="fresh",
                            dataset_episode_index=None,
                        )
                        # Reseed the start-sampling RNG per seed so every sample of a
                        # given seed gets the SAME start/goal; only policy noise varies.
                        sample_rng = np.random.default_rng(fresh_seed)
                        # Explicit generator per (seed, sample) — primary fix for CUDA.
                        _seed_policy_noise(fresh_seed, sample_idx)
                        gen = torch.Generator(device=device)
                        gen.manual_seed(
                            (int(fresh_seed) * 1_000_003 + int(sample_idx) * 97) % (2**31 - 1)
                        )
                        row = run_fresh_episode(
                            env=env,
                            policy=policy,
                            spec=spec,
                            rng=sample_rng,
                            action_mode=action_mode,
                            crop_config=crop_config,
                            goal_thresh=goal_thresh,
                            success_radius=args.fresh_success_radius,
                            max_steps=args.max_steps,
                            post_success_steps=args.post_success_steps,
                            execution_horizon_chunks=args.execution_horizon_chunks,
                            gripper_open=args.gripper_open,
                            device=device,
                            fresh_bounds=args.fresh_bounds,
                            fresh_start_attempts=args.fresh_start_attempts,
                            fresh_base_clearance=args.fresh_base_clearance,
                            fresh_min_start_goal_distance=args.fresh_min_start_goal_distance,
                            fresh_start_mode=args.fresh_start_mode,
                            generator=gen,
                            video_path=(
                                video_dir
                                / f"fresh_ep{ep_idx:03d}_seed{fresh_seed}_s{sample_idx}.mp4"
                                if video_dir is not None
                                else None
                            ),
                            video_fps=args.video_fps,
                        )
                        row["sample_index"] = sample_idx
                        tcp_path = row.pop("tcp_path", None)
                        if viz_this and tcp_path is not None:
                            seed_paths.append(
                                {
                                    "sample_index": sample_idx,
                                    "tcp_path": np.asarray(tcp_path, dtype=np.float32),
                                    "success": bool(row["success"]),
                                }
                            )
                        rows.append(row)
                        metrics_file.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
                        metrics_file.flush()
                        episode_counter += 1
                        print(
                            f"episode={episode_counter} seed={row['seed']} "
                            f"sample={sample_idx + 1}/{args.samples_per_seed} "
                            f"success={row['success']} "
                            f"final={_format_optional(row['final_distance'])} "
                            f"min={_format_optional(row['min_distance'])} "
                            f"steps={row['steps']} "
                            f"start={_format_xyz(row.get('start_tcp'))} "
                            f"goal={_format_xyz(row.get('target_position'))}"
                            + (f" video={row['video']}" if row.get("video") else ""),
                            flush=True,
                        )
                        # When visualizing multimodality we want every sample drawn, so
                        # do not stop early on the first success.
                        if row["success"] and not viz_this:
                            break  # got success — skip remaining samples for this start/goal
                    if viz_this and seed_paths:
                        _plot_multimodality(
                            output_dir=args.output_dir,
                            episode_number=episode_number,
                            seed=fresh_seed,
                            target=np.asarray(row["target_position"], dtype=np.float32),
                            samples=seed_paths,
                            goal_thresh=goal_thresh,
                        )
    except Exception as exc:
        print(f"Failed reach checkpoint eval: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if env is not None:
            env.close()

    summary = {
        "checkpoint": str(checkpoint_path),
        "source": args.source,
        "dataset": str(args.dataset) if args.dataset is not None else None,
        "env_id": metadata["env_id"],
        "env_kwargs": _env_kwargs(metadata),
        "action_mode": action_mode,
        "config_diagnostics": run_config,
        "goal_thresh": goal_thresh,
        "episodes_requested": args.episodes,
        "samples_per_seed": args.samples_per_seed,
        "episodes_completed": len(rows),
        "unique_dataset_seed_count": len({row["seed"] for row in rows}),
        "success_rate": _success_rate(rows),
        "seed_any_success_rate": _seed_any_success_rate(rows),
        "seed_all_success_rate": _seed_all_success_rate(rows, args.samples_per_seed),
        "final_distance_mean": _mean(row.get("final_distance") for row in rows),
        "final_distance_median": _median(row.get("final_distance") for row in rows),
        "min_distance_mean": _mean(row.get("min_distance") for row in rows),
        "min_distance_median": _median(row.get("min_distance") for row in rows),
        "episodes": rows,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        "summary: "
        f"episodes={len(rows)} samples_per_seed={args.samples_per_seed} "
        f"success_rate={_format_optional(summary['success_rate'])} "
        f"seed_any_success={_format_optional(summary['seed_any_success_rate'])} "
        f"seed_all_success={_format_optional(summary['seed_all_success_rate'])} "
        f"final_distance_mean={_format_optional(summary['final_distance_mean'])} "
        f"path={summary_path}",
        flush=True,
    )
    failures = sum(0 if row["success"] else 1 for row in rows)
    return 0 if args.allow_failure or failures == 0 else 1


# Conservative Panda-reachable table box (mirrors _default_reach_workspace_bounds in write_maniskill)
_DEFAULT_FRESH_BOUNDS = np.asarray(
    [[-0.42, 0.42], [-0.45, 0.45], [0.20, 0.72]], dtype=np.float32
)

# Neutral "arm-over-table" qpos used as the IK planning base in fresh mode.
# The env rest pose ([0, π/8, 0, -5π/8, 0, 3π/4, π/4]) has the arm partially
# upright; plan_screw from there fails for many reachable Cartesian targets.
# This neutral pose keeps the arm hovering over the workspace in a task-like
# configuration, matching what the policy actually sees during episodes.
_NEUTRAL_IK_QPOS = np.array([0.0, 0.5, 0.0, -2.5, 0.0, 3.0, 0.8], dtype=np.float32)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a plain DP3 reach checkpoint on first-occurrence unique dataset seeds "
            "or on fresh random start/goal pairs. "
            "Normal policy rollout only: no constraints, rejection, reranking, video, or Rerun."
        )
    )
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument("--checkpoint", type=Path, default=None)
    checkpoint_group.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-model", choices=["ema", "raw"], default="ema")
    parser.add_argument(
        "--source",
        choices=["dataset", "fresh"],
        default="dataset",
        help=(
            "dataset: replay start/goal from zarr (default). "
            "fresh: sample new random start/goal pairs within --fresh-bounds."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to reach zarr dataset. Required when --source dataset.",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default=None,
        help=(
            "ManiSkill env ID to use when --source fresh and --dataset is not provided. "
            "E.g. PG3DReach-BalancedWorkspace-v0. Ignored when --dataset is given "
            "(env ID is read from dataset metadata)."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--episodes", type=int, default=366)
    parser.add_argument(
        "--samples-per-seed",
        type=int,
        default=1,
        help=(
            "number of independent policy rollouts to run per unique seed "
            "(default: 1). The environment resets to the same start/goal each time "
            "but the policy samples different DDPM noise, so outcomes vary. "
            "Pass 3 to get 366*3=1098 total rollouts."
        ),
    )
    parser.add_argument(
        "--candidates-per-sample",
        type=int,
        default=None,
        help=(
            "Alias for --samples-per-seed (especially useful in --source fresh mode). "
            "If set, overrides --samples-per-seed. "
            "Each sample/episode gets N independent rollouts with different policy noise."
        ),
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Skip this many unique dataset seeds before selecting episodes.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument(
        "--crop-num-points",
        type=int,
        default=None,
        help=(
            "Override the number of scene points in the point-cloud crop. "
            "When --dataset is absent (fresh mode), this is auto-inferred from the checkpoint "
            "as policy.point_cloud_shape[0] - goal_marker_points. "
            "Only set manually if auto-inference is wrong."
        ),
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help=(
            "Override the ManiSkill env's built-in episode length limit (default: 100). "
            "Set higher than --max-steps to let your step budget control termination. "
            "Forwarded directly to gym.make(..., max_episode_steps=N)."
        ),
    )
    parser.add_argument("--post-success-steps", type=int, default=8)
    parser.add_argument("--execution-horizon-chunks", type=int, default=1)
    parser.add_argument("--goal-thresh", type=float, default=None)
    parser.add_argument(
        "--fresh-success-radius",
        type=float,
        default=0.03,
        help=(
            "Fresh mode only: count a reach as successful when the TCP comes within this "
            "radius (meters) of the goal, in addition to the env's built-in success flag. "
            "Default 0.03 (3cm), looser than the env goal_thresh (~2.5cm)."
        ),
    )
    parser.add_argument("--gripper-open", type=float, default=0.04)
    # --- fresh-mode options ---
    parser.add_argument(
        "--fresh-bounds",
        type=float,
        nargs=6,
        default=_DEFAULT_FRESH_BOUNDS.reshape(-1).tolist(),
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX", "Z_MIN", "Z_MAX"),
        help=(
            "Cartesian TCP workspace bounds for fresh start/goal sampling. "
            "Defaults to the conservative Panda-reachable table box used in write_maniskill: "
            "[-0.42, 0.42, -0.45, 0.45, 0.20, 0.72]."
        ),
    )
    parser.add_argument(
        "--fresh-start-mode",
        choices=["ik_cartesian", "uniform_joint"],
        default="ik_cartesian",
        help=(
            "How to sample fresh start poses. "
            "'ik_cartesian' (default): sample Cartesian TCP target and solve IK with rest "
            "orientation via PandaArmMotionPlanningSolver — matches data-gen behavior. "
            "'uniform_joint': sample joint angles uniformly across URDF limits (OOD, legacy)."
        ),
    )
    parser.add_argument(
        "--fresh-start-attempts",
        type=int,
        default=100,
        help="Max attempts per episode to find a valid start pose.",
    )
    parser.add_argument(
        "--fresh-base-clearance",
        type=float,
        default=0.06,
        help="Min horizontal XY distance from robot base for sampled TCP start positions.",
    )
    parser.add_argument(
        "--fresh-min-start-goal-distance",
        type=float,
        default=0.16,
        help="Min Euclidean distance between sampled TCP start and goal in meters.",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help=(
            "Render an MP4 per episode into <output-dir>/videos/ so you can visualize "
            "the rollout. Forces the env render_mode to rgb_array."
        ),
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=10,
        help="Frames per second for the per-episode MP4 videos (default: 10).",
    )
    parser.add_argument(
        "--viz-trajectories",
        action="store_true",
        help=(
            "Save a per-episode multimodality plot into <output-dir>/multimodality/ that "
            "overlays the TCP path of every sample for that seed (3D + top-down XY views), "
            "so you can see how the diffusion policy fans out into distinct modes. "
            "When on, the early-stop-on-first-success is disabled for the plotted episodes "
            "so all --samples-per-seed rollouts are drawn."
        ),
    )
    parser.add_argument(
        "--viz-episodes",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help=(
            "1-based episode (unique-seed) indices to plot when --viz-trajectories is set. "
            "E.g. --viz-episodes 1 plots the first episode. Default: all episodes."
        ),
    )
    parser.add_argument(
        "--print-config-only",
        action="store_true",
        help="Print metadata/checkpoint/eval diagnostics and exit before running episodes.",
    )
    parser.add_argument("--allow-failure", action="store_true")
    args = parser.parse_args(argv)
    if args.candidates_per_sample is not None:
        args.samples_per_seed = args.candidates_per_sample
    if args.source == "dataset" and args.dataset is None:
        parser.error("--dataset is required when --source dataset")
    if args.source == "fresh" and args.dataset is None and args.env_id is None:
        parser.error("--source fresh without --dataset requires --env-id")
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.samples_per_seed <= 0:
        raise ValueError("--samples-per-seed must be positive")
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.post_success_steps < 0:
        raise ValueError("--post-success-steps must be non-negative")
    if args.execution_horizon_chunks <= 0:
        raise ValueError("--execution-horizon-chunks must be positive")
    if args.fresh_start_attempts <= 0:
        raise ValueError("--fresh-start-attempts must be positive")
    if args.fresh_base_clearance < 0:
        raise ValueError("--fresh-base-clearance must be non-negative")
    if args.fresh_min_start_goal_distance < 0:
        raise ValueError("--fresh-min-start-goal-distance must be non-negative")
    args.fresh_bounds = np.asarray(args.fresh_bounds, dtype=np.float32).reshape(3, 2)
    return args


def resolve_checkpoint_path(checkpoint: Path | None, checkpoint_dir: Path | None) -> Path:
    if checkpoint is not None:
        return checkpoint
    if checkpoint_dir is None:
        raise ValueError("checkpoint or checkpoint_dir is required")
    return latest_reach_checkpoint(checkpoint_dir)


def run_reach_episode(
    *,
    env: Any,
    policy: Any,
    spec: RolloutSpec,
    zarr_root: Any,
    action_mode: ActionMode,
    crop_config: Any,
    goal_thresh: float,
    max_steps: int,
    post_success_steps: int,
    execution_horizon_chunks: int,
    gripper_open: float,
    device: torch.device,
    generator: torch.Generator | None = None,
    video_path: Path | None = None,
    video_fps: int = 10,
) -> dict[str, Any]:
    if spec.dataset_episode_index is None:
        raise ValueError("run_reach_episode requires a dataset_episode_index to restore Zarr state")
    zarr_context = _zarr_episode_context(zarr_root, spec.dataset_episode_index)
    obs, info = _reset_to_zarr_episode(env, rollout_seed=spec.seed, zarr_context=zarr_context)
    frames: list[np.ndarray] | None = (
        [_frame_to_numpy(env.render())] if video_path is not None else None
    )
    entry = rollout_observation_entry(obs, info, env=env, crop_config=crop_config)
    entry = _apply_zarr_initial_entry(entry, zarr_context)
    obs_window = make_initial_obs_window(entry, n_obs_steps=int(policy.n_obs_steps))
    start_tcp = np.asarray(zarr_context["tcp_pose"], dtype=np.float32).reshape(-1)[:3]
    target = np.asarray(zarr_context["target_position"], dtype=np.float32).reshape(3)
    tcp_path = [start_tcp.copy()]
    distances = [_entry_distance(entry)]
    first_success_step: int | None = None
    observed_post_success_steps = 0
    steps = 0
    replans = 0
    terminated_or_truncated = False
    was_training = bool(getattr(policy, "training", False))
    policy.eval()
    try:
        while steps < max_steps:
            if first_success_step is not None and observed_post_success_steps >= post_success_steps:
                break
            with torch.inference_mode():
                policy_input = obs_window_to_torch(
                    obs_window,
                    device=device,
                    goal_marker_points=int(getattr(policy, "goal_marker_points", 0)),
                    goal_marker_radius=float(getattr(policy, "goal_marker_radius", 0.045)),
                )
                output = policy.predict_action(policy_input, generator=generator)
                action_chunk = output["action"][0].detach().cpu().numpy()
            replans += 1
            steps_to_execute = min(
                action_chunk.shape[0],
                int(policy.n_action_steps) * execution_horizon_chunks,
                max_steps - steps,
            )
            for policy_action in action_chunk[:steps_to_execute]:
                sim_action = policy_action_to_sim_action(
                    policy_action,
                    np.asarray(entry["agent_pos"], dtype=np.float32),
                    action_mode=action_mode,
                    sim_action_dim=int(np.prod(env.action_space.shape)),
                    low=getattr(env.action_space, "low", None),
                    high=getattr(env.action_space, "high", None),
                    gripper_open=gripper_open,
                )
                obs, _reward, terminated, truncated, info = env.step(sim_action)
                steps += 1
                if frames is not None:
                    frames.append(_frame_to_numpy(env.render()))
                entry = rollout_observation_entry(obs, info, env=env, crop_config=crop_config)
                obs_window = append_obs_window(
                    obs_window,
                    entry,
                    n_obs_steps=int(policy.n_obs_steps),
                )
                tcp = np.asarray(entry["tcp_pose"], dtype=np.float32).reshape(-1)[:3]
                tcp_path.append(tcp.copy())
                distance = _entry_distance(entry)
                distances.append(distance)
                success = _bool_info(info, "success") or (
                    np.isfinite(distance) and distance <= goal_thresh
                )
                if success and first_success_step is None:
                    first_success_step = steps
                elif first_success_step is not None:
                    observed_post_success_steps += 1
                terminated_or_truncated = _bool_any(terminated) or _bool_any(truncated)
                if (
                    terminated_or_truncated
                    or steps >= max_steps
                    or (
                        first_success_step is not None
                        and observed_post_success_steps >= post_success_steps
                    )
                ):
                    break
            if terminated_or_truncated:
                break
    finally:
        if was_training:
            policy.train()
    saved_video = _save_episode_video(frames, video_path, video_fps)
    finite_distances = [distance for distance in distances if np.isfinite(distance)]
    final_tcp = tcp_path[-1]
    return {
        "episode": spec.output_index,
        "video": saved_video,
        "dataset_episode_index": spec.dataset_episode_index,
        "seed": spec.seed,
        "source": spec.source,
        "success": first_success_step is not None,
        "first_success_step": first_success_step,
        "steps": steps,
        "replans": replans,
        "terminated_or_truncated": terminated_or_truncated,
        "start_tcp": start_tcp.tolist(),
        "target_position": target.tolist(),
        "final_tcp": final_tcp.tolist(),
        "start_distance": finite_distances[0] if finite_distances else None,
        "final_distance": finite_distances[-1] if finite_distances else None,
        "min_distance": min(finite_distances) if finite_distances else None,
        "path_length": _path_length(np.stack(tcp_path, axis=0)),
        # Full TCP trajectory (N, 3); popped before metrics/summary serialization,
        # used only for the optional multimodality plots.
        "tcp_path": np.stack(tcp_path, axis=0).astype(np.float32),
    }


def _zarr_episode_context(zarr_root: Any, episode_index: int) -> dict[str, Any]:
    episode_ends = np.asarray(zarr_root["meta"]["episode_ends"][:], dtype=np.int64)
    if episode_index < 0 or episode_index >= len(episode_ends):
        raise IndexError(
            f"episode index {episode_index} is outside Zarr range [0, {len(episode_ends) - 1}]"
        )
    episode_start = 0 if episode_index == 0 else int(episode_ends[episode_index - 1])
    episode_end = int(episode_ends[episode_index])
    if episode_start >= episode_end:
        raise ValueError(f"Zarr episode {episode_index} is empty")
    data = zarr_root["data"]
    required = (
        "state", "target_position", "tcp_pose", "point_cloud", "robot_mask", "point_valid_mask"
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"dataset missing required Zarr arrays: {missing}")
    row = episode_start
    return {
        "episode_index": int(episode_index),
        "episode_start": episode_start,
        "state": np.asarray(data["state"][row], dtype=np.float32).copy(),
        "target_position": np.asarray(
            data["target_position"][row], dtype=np.float32
        ).reshape(3).copy(),
        "tcp_pose": np.asarray(data["tcp_pose"][row], dtype=np.float32).copy(),
        "point_cloud": np.asarray(data["point_cloud"][row], dtype=np.float32).copy(),
        "robot_mask": np.asarray(data["robot_mask"][row], dtype=bool).copy(),
        "point_valid_mask": np.asarray(data["point_valid_mask"][row], dtype=bool).copy(),
    }


def _reset_to_zarr_episode(
    env: Any,
    *,
    rollout_seed: int,
    zarr_context: dict[str, Any],
) -> tuple[Any, Any]:
    env.reset(seed=rollout_seed, options={"reconfigure": True})
    unwrapped = env.unwrapped
    robot = unwrapped.agent.robot
    current_qpos = np.asarray(robot.get_qpos(), dtype=np.float32)
    stored_qpos = np.asarray(zarr_context["state"], dtype=np.float32).reshape(-1)
    qpos = current_qpos.copy().reshape(-1)
    qpos[: stored_qpos.size] = stored_qpos
    robot.set_qpos(qpos.reshape(current_qpos.shape))
    if hasattr(robot, "set_qvel"):
        robot.set_qvel(np.zeros_like(qpos.reshape(current_qpos.shape), dtype=np.float32))
    from mani_skill.utils.structs.pose import Pose

    target = np.asarray(zarr_context["target_position"], dtype=np.float32).reshape(1, 3)
    start = np.asarray(zarr_context["tcp_pose"], dtype=np.float32)[:3].reshape(1, 3)
    unwrapped.goal_site.set_pose(Pose.create_from_pq(target))
    if getattr(unwrapped, "start_site", None) is not None:
        unwrapped.start_site.set_pose(Pose.create_from_pq(start))
    info = unwrapped.get_info()
    obs = unwrapped.get_obs(info)
    return obs, info


def _apply_zarr_initial_entry(
    entry: dict[str, Any],
    zarr_context: dict[str, Any],
) -> dict[str, Any]:
    result = dict(entry)
    for key in ("point_cloud", "robot_mask", "point_valid_mask", "target_position", "tcp_pose"):
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


def run_fresh_episode(
    *,
    env: Any,
    policy: Any,
    spec: RolloutSpec,
    rng: np.random.Generator,
    action_mode: ActionMode,
    crop_config: Any,
    goal_thresh: float,
    success_radius: float | None,
    max_steps: int,
    post_success_steps: int,
    execution_horizon_chunks: int,
    gripper_open: float,
    device: torch.device,
    fresh_bounds: np.ndarray,
    fresh_start_attempts: int,
    fresh_base_clearance: float,
    fresh_min_start_goal_distance: float,
    fresh_start_mode: str = "ik_cartesian",
    generator: torch.Generator | None = None,
    video_path: Path | None = None,
    video_fps: int = 10,
) -> dict[str, Any]:
    obs, info, start_tcp, target = _fresh_reset(
        env,
        seed=spec.seed,
        rng=rng,
        bounds=fresh_bounds,
        start_attempts=fresh_start_attempts,
        base_clearance=fresh_base_clearance,
        min_start_goal_distance=fresh_min_start_goal_distance,
        start_mode=fresh_start_mode,
    )
    frames: list[np.ndarray] | None = (
        [_frame_to_numpy(env.render())] if video_path is not None else None
    )
    # In fresh mode, accept a reach as successful within success_radius (default 3cm),
    # which is looser than the env's built-in goal_thresh (2.5cm).
    effective_thresh = (
        max(float(goal_thresh), float(success_radius))
        if success_radius is not None
        else float(goal_thresh)
    )
    entry = rollout_observation_entry(obs, info, env=env, crop_config=crop_config)
    obs_window = make_initial_obs_window(entry, n_obs_steps=int(policy.n_obs_steps))
    tcp_path = [start_tcp.copy()]
    distances = [_entry_distance(entry)]
    first_success_step: int | None = None
    observed_post_success_steps = 0
    steps = 0
    replans = 0
    terminated_or_truncated = False
    was_training = bool(getattr(policy, "training", False))
    policy.eval()
    try:
        while steps < max_steps:
            if first_success_step is not None and observed_post_success_steps >= post_success_steps:
                break
            with torch.inference_mode():
                policy_input = obs_window_to_torch(
                    obs_window,
                    device=device,
                    goal_marker_points=int(getattr(policy, "goal_marker_points", 0)),
                    goal_marker_radius=float(getattr(policy, "goal_marker_radius", 0.045)),
                )
                output = policy.predict_action(policy_input, generator=generator)
                action_chunk = output["action"][0].detach().cpu().numpy()
            replans += 1
            steps_to_execute = min(
                action_chunk.shape[0],
                int(policy.n_action_steps) * execution_horizon_chunks,
                max_steps - steps,
            )
            for policy_action in action_chunk[:steps_to_execute]:
                sim_action = policy_action_to_sim_action(
                    policy_action,
                    np.asarray(entry["agent_pos"], dtype=np.float32),
                    action_mode=action_mode,
                    sim_action_dim=int(np.prod(env.action_space.shape)),
                    low=getattr(env.action_space, "low", None),
                    high=getattr(env.action_space, "high", None),
                    gripper_open=gripper_open,
                )
                obs, _reward, terminated, truncated, info = env.step(sim_action)
                steps += 1
                if frames is not None:
                    frames.append(_frame_to_numpy(env.render()))
                entry = rollout_observation_entry(obs, info, env=env, crop_config=crop_config)
                obs_window = append_obs_window(
                    obs_window,
                    entry,
                    n_obs_steps=int(policy.n_obs_steps),
                )
                tcp = np.asarray(entry["tcp_pose"], dtype=np.float32).reshape(-1)[:3]
                tcp_path.append(tcp.copy())
                distance = _entry_distance(entry)
                distances.append(distance)
                success = _bool_info(info, "success") or (
                    np.isfinite(distance) and distance <= effective_thresh
                )
                if success and first_success_step is None:
                    first_success_step = steps
                elif first_success_step is not None:
                    observed_post_success_steps += 1
                terminated_or_truncated = _bool_any(terminated) or _bool_any(truncated)
                if (
                    terminated_or_truncated
                    or steps >= max_steps
                    or (
                        first_success_step is not None
                        and observed_post_success_steps >= post_success_steps
                    )
                ):
                    break
            if terminated_or_truncated:
                break
    finally:
        if was_training:
            policy.train()
    saved_video = _save_episode_video(frames, video_path, video_fps)
    finite_distances = [d for d in distances if np.isfinite(d)]
    final_tcp = tcp_path[-1]
    return {
        "episode": spec.output_index,
        "video": saved_video,
        "dataset_episode_index": None,
        "seed": spec.seed,
        "source": "fresh",
        "success": first_success_step is not None,
        "first_success_step": first_success_step,
        "steps": steps,
        "replans": replans,
        "terminated_or_truncated": terminated_or_truncated,
        "start_tcp": start_tcp.tolist(),
        "target_position": target.tolist(),
        "final_tcp": final_tcp.tolist(),
        "start_distance": finite_distances[0] if finite_distances else None,
        "final_distance": finite_distances[-1] if finite_distances else None,
        "min_distance": min(finite_distances) if finite_distances else None,
        "path_length": _path_length(np.stack(tcp_path, axis=0)),
        # Full TCP trajectory (N, 3); popped before metrics/summary serialization,
        # used only for the optional multimodality plots.
        "tcp_path": np.stack(tcp_path, axis=0).astype(np.float32),
    }


def _fresh_reset(
    env: Any,
    *,
    seed: int,
    rng: np.random.Generator,
    bounds: np.ndarray,
    start_attempts: int,
    base_clearance: float,
    min_start_goal_distance: float,
    start_mode: str = "ik_cartesian",
) -> tuple[Any, Any, np.ndarray, np.ndarray]:
    """Reset env for a fresh goal and sample a start pose.

    start_mode='ik_cartesian' (default): sample Cartesian TCP target and solve IK with
    rest-pose orientation via PandaArmMotionPlanningSolver — mirrors data-gen behavior.
    start_mode='uniform_joint': sample arm joint angles uniformly (legacy, produces OOD poses).

    Returns (obs, info, start_tcp_xyz, target_xyz).
    """
    obs, info = env.reset(seed=seed, options={"reconfigure": True})
    unwrapped = env.unwrapped
    robot = unwrapped.agent.robot

    target = np.asarray(unwrapped.goal_site.pose.p, dtype=np.float32).reshape(-1, 3)[0]

    # Full robot qpos after reset (rest pose, arm + gripper)
    current_qpos = _to_numpy_safe(
        robot.get_qpos() if hasattr(robot, "get_qpos") else robot.qpos
    ).reshape(-1).astype(np.float32)

    found_qpos: np.ndarray | None = None
    start_tcp_xyz: np.ndarray | None = None

    if start_mode == "ik_cartesian":
        # Extract rest-pose TCP orientation to keep it fixed during IK
        tcp_raw = getattr(unwrapped.agent, "tcp_pose", None)
        if tcp_raw is not None and hasattr(tcp_raw, "raw_pose"):
            reset_tcp_full = _to_numpy_safe(tcp_raw.raw_pose).reshape(-1, 7)[0].astype(np.float32)
        else:
            reset_tcp_full = _to_numpy_safe(unwrapped.agent.tcp.pose.raw_pose).reshape(-1, 7)[0].astype(np.float32)
        reset_quat = reset_tcp_full[3:7]

        found_qpos, start_tcp_xyz = _sample_start_ik_cartesian(
            env=env,
            rng=rng,
            seed=seed,
            bounds=bounds,
            reset_qpos=current_qpos,
            reset_quat=reset_quat,
            target=target,
            start_attempts=start_attempts,
            base_clearance=base_clearance,
            min_start_goal_distance=min_start_goal_distance,
            ik_plan_qpos=_NEUTRAL_IK_QPOS,
        )
        # _sample_start_ik_cartesian leaves robot at rest qpos on both success and failure.
        # Apply the found qpos now.
        if found_qpos is not None:
            _set_qpos_safe(robot, found_qpos)

    else:
        # Legacy: sample joint angles uniformly across URDF limits
        qlimits = _to_numpy_safe(robot.get_qlimits() if hasattr(robot, "get_qlimits") else robot.qlimits)
        qlimits = qlimits.reshape(-1, 2)
        arm_dof = min(7, qlimits.shape[0])
        q_low = qlimits[:arm_dof, 0].astype(np.float64)
        q_high = qlimits[:arm_dof, 1].astype(np.float64)
        robot_base_xy = _to_numpy_safe(robot.pose.p).reshape(-1)[:2].astype(np.float64)
        bounds_f64 = np.asarray(bounds, dtype=np.float64).reshape(3, 2)

        for _ in range(start_attempts):
            q_arm = rng.uniform(q_low, q_high).astype(np.float32)
            candidate_qpos = current_qpos.copy()
            candidate_qpos[:arm_dof] = q_arm
            robot.set_qpos(candidate_qpos.reshape(current_qpos.shape))
            tcp_raw = getattr(unwrapped.agent, "tcp_pose", None)
            if tcp_raw is not None and hasattr(tcp_raw, "raw_pose"):
                tcp_xyz = _to_numpy_safe(tcp_raw.raw_pose).reshape(-1, 7)[0, :3].astype(np.float64)
            else:
                tcp_xyz = _to_numpy_safe(unwrapped.agent.tcp.pose.raw_pose).reshape(-1, 7)[0, :3].astype(np.float64)
            if not np.all((tcp_xyz >= bounds_f64[:, 0]) & (tcp_xyz <= bounds_f64[:, 1])):
                continue
            if float(np.linalg.norm(tcp_xyz[:2] - robot_base_xy)) < base_clearance:
                continue
            if float(np.linalg.norm(tcp_xyz - target.astype(np.float64))) < min_start_goal_distance:
                continue
            found_qpos = candidate_qpos
            start_tcp_xyz = tcp_xyz.astype(np.float32)
            break

    if found_qpos is None:
        # Fallback: rest pose
        _set_qpos_safe(robot, current_qpos)
        tcp_raw = getattr(unwrapped.agent, "tcp_pose", None)
        if tcp_raw is not None and hasattr(tcp_raw, "raw_pose"):
            start_tcp_xyz = _to_numpy_safe(tcp_raw.raw_pose).reshape(-1, 7)[0, :3].astype(np.float32)
        else:
            start_tcp_xyz = _to_numpy_safe(unwrapped.agent.tcp.pose.raw_pose).reshape(-1, 7)[0, :3].astype(np.float32)
        print(
            f"warning: fresh_reset seed={seed} mode={start_mode!r} could not find an "
            f"in-bounds start after {start_attempts} attempts; using default reset qpos",
            flush=True,
        )

    if start_tcp_xyz is None:
        start_tcp_xyz = current_qpos[:3].copy()

    # Zero velocities after any qpos change
    if hasattr(robot, "set_qvel"):
        current_raw = robot.get_qpos() if hasattr(robot, "get_qpos") else robot.qpos
        robot.set_qvel(np.zeros(_to_numpy_safe(current_raw).shape, dtype=np.float32))

    # Move the red start_site marker to the actual sampled start TCP
    start_site = getattr(unwrapped, "start_site", None)
    if start_site is not None:
        from mani_skill.utils.structs.pose import Pose

        start_site.set_pose(Pose.create_from_pq(start_tcp_xyz.reshape(1, 3)))
    update_render = getattr(unwrapped.scene, "update_render", None)
    if callable(update_render):
        update_render()

    info = unwrapped.get_info()
    obs = unwrapped.get_obs(info)
    return obs, info, start_tcp_xyz, target


def _ik_plan_valid(plan: Any) -> bool:
    """Return True if the mplib plan succeeded and contains finite joint positions."""
    if plan == -1 or not isinstance(plan, dict) or "position" not in plan:
        return False
    positions = np.asarray(plan["position"], dtype=np.float32)
    if positions.ndim != 2 or positions.shape[0] == 0 or not np.all(np.isfinite(positions)):
        return False
    status = str(plan.get("status", "")).lower()
    return "fail" not in status and "error" not in status


def _set_qpos_safe(robot: Any, qpos: np.ndarray) -> None:
    """Set robot qpos, padding with current values for extra DOF (gripper fingers)."""
    qpos_flat = np.asarray(qpos, dtype=np.float32).reshape(-1)
    current_raw = robot.get_qpos() if hasattr(robot, "get_qpos") else robot.qpos
    current = _to_numpy_safe(current_raw).astype(np.float32, copy=True)
    current_shape = current.shape
    current = current.reshape(-1)
    next_qpos = current.copy()
    next_qpos[: qpos_flat.shape[0]] = qpos_flat
    if hasattr(robot, "set_qpos"):
        robot.set_qpos(next_qpos.reshape(current_shape))
    else:
        robot.qpos = next_qpos.reshape(-1)


def _sample_start_ik_cartesian(
    *,
    env: Any,
    rng: np.random.Generator,
    seed: int,
    bounds: np.ndarray,
    reset_qpos: np.ndarray,
    reset_quat: np.ndarray,
    target: np.ndarray,
    start_attempts: int,
    base_clearance: float,
    min_start_goal_distance: float,
    ik_plan_qpos: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Sample a start qpos by solving IK from a Cartesian target, matching data-gen.

    Mirrors _sample_reachable_start in write_maniskill_reach_dataset.py:
      1. Sample Cartesian TCP position uniformly in bounds.
      2. Solve IK via PandaArmMotionPlanningSolver with rest-pose orientation.
      3. Apply candidate, read actual FK TCP, validate, restore.
    Returns (start_qpos, start_tcp_xyz) or (None, None) on failure.
    Robot is left at reset_qpos on both success and failure exit.

    ik_plan_qpos: optional qpos to set the robot to before calling the planner
    (the planner reads robot.get_qpos() as its start config). Defaults to
    reset_qpos. Pass _NEUTRAL_IK_QPOS (padded) so the screw planner starts from
    an arm-over-table pose where plan_screw succeeds for far more targets.
    """
    import sapien
    from mani_skill.examples.motionplanning.panda.motionplanner import (
        PandaArmMotionPlanningSolver,
    )

    unwrapped = env.unwrapped
    robot = unwrapped.agent.robot
    robot_base_xy = _to_numpy_safe(robot.pose.p).reshape(-1)[:2].astype(np.float64)
    bounds_f64 = np.asarray(bounds, dtype=np.float64).reshape(3, 2)
    target_f64 = target.astype(np.float64)

    # Build the qpos the planner will start from. If caller provides a neutral
    # pose, pad it to full robot DOF using reset_qpos for the finger joints.
    if ik_plan_qpos is not None:
        plan_start_qpos = reset_qpos.copy()
        n = min(ik_plan_qpos.shape[0], plan_start_qpos.shape[0])
        plan_start_qpos[:n] = ik_plan_qpos[:n]
    else:
        plan_start_qpos = reset_qpos

    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=False,
        base_pose=robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
    )

    reject_counts = {k: 0 for k in ("geometry", "too_close_sampled", "planner", "too_close_actual", "geometry_actual")}
    for _ in range(start_attempts):
        sampled_xyz = rng.uniform(bounds_f64[:, 0], bounds_f64[:, 1]).astype(np.float64)

        # Bounds + clearance checks on the sampled Cartesian position
        if not np.all((sampled_xyz >= bounds_f64[:, 0]) & (sampled_xyz <= bounds_f64[:, 1])):
            reject_counts["geometry"] += 1
            continue
        if float(np.linalg.norm(sampled_xyz[:2] - robot_base_xy)) < base_clearance:
            reject_counts["geometry"] += 1
            continue
        if float(np.linalg.norm(sampled_xyz - target_f64)) < min_start_goal_distance:
            reject_counts["too_close_sampled"] += 1
            continue

        # Solve IK: set robot to neutral planning pose, then plan to target.
        # move_to_pose_with_screw reads robot.get_qpos() as its start config.
        _set_qpos_safe(robot, plan_start_qpos)
        target_pose = sapien.Pose(p=sampled_xyz.astype(np.float32), q=reset_quat)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            plan = planner.move_to_pose_with_screw(target_pose, dry_run=True)
        _set_qpos_safe(robot, reset_qpos)  # always restore to env rest pose

        if not _ik_plan_valid(plan):
            reject_counts["planner"] += 1
            continue

        start_qpos_candidate = np.asarray(plan["position"], dtype=np.float32)[-1].copy()

        # Apply candidate and read actual FK TCP position
        _set_qpos_safe(robot, start_qpos_candidate)
        tcp_raw = getattr(unwrapped.agent, "tcp_pose", None)
        if tcp_raw is not None and hasattr(tcp_raw, "raw_pose"):
            actual_xyz = _to_numpy_safe(tcp_raw.raw_pose).reshape(-1, 7)[0, :3].astype(np.float64)
        else:
            actual_xyz = _to_numpy_safe(unwrapped.agent.tcp.pose.raw_pose).reshape(-1, 7)[0, :3].astype(np.float64)
        _set_qpos_safe(robot, reset_qpos)  # restore

        # Validate actual TCP (same checks as data-gen)
        if float(np.linalg.norm(actual_xyz - target_f64)) < min_start_goal_distance:
            reject_counts["too_close_actual"] += 1
            continue
        if not np.all((actual_xyz >= bounds_f64[:, 0]) & (actual_xyz <= bounds_f64[:, 1])):
            reject_counts["geometry_actual"] += 1
            continue
        if float(np.linalg.norm(actual_xyz[:2] - robot_base_xy)) < base_clearance:
            reject_counts["geometry_actual"] += 1
            continue

        return start_qpos_candidate, actual_xyz.astype(np.float32)

    print(
        f"warning: IK start sampling seed={seed} exhausted {start_attempts} attempts "
        f"geometry={reject_counts['geometry']} too_close_sampled={reject_counts['too_close_sampled']} "
        f"planner={reject_counts['planner']} too_close_actual={reject_counts['too_close_actual']} "
        f"geometry_actual={reject_counts['geometry_actual']}",
        flush=True,
    )
    return None, None


def _seed_policy_noise(seed: int, sample_idx: int) -> None:
    """Seed torch so each (seed, sample) rollout draws a distinct, reproducible noise vector.

    The DP3 policy samples its initial diffusion latent via torch.randn from the global
    torch RNG (no explicit generator). Reseeding here with a value that depends on BOTH
    the episode seed and the sample index guarantees every sample of the same start/goal
    gets a different diffusion trajectory, while keeping the whole run reproducible.
    """
    combined = (int(seed) * 1_000_003 + int(sample_idx) * 97) % (2**31 - 1)
    torch.manual_seed(combined)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(combined)


def _to_numpy_safe(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    try:
        import torch as _torch
        if isinstance(x, _torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def _unique_seed_indices(
    dataset_episode_seeds: list[int],
    *,
    count: int,
    start_index: int,
) -> list[int]:
    """Return dataset indices of the first occurrence of each unique seed."""
    seen: set[int] = set()
    unique_indices: list[int] = []
    for dataset_idx, seed in enumerate(dataset_episode_seeds):
        if seed in seen:
            continue
        seen.add(seed)
        unique_indices.append(dataset_idx)
    return unique_indices[start_index : start_index + count]


def _unique_seed_specs(
    dataset_episode_seeds: list[int],
    *,
    count: int,
    start_index: int,
    samples_per_seed: int = 1,
) -> list[RolloutSpec]:
    selected = _unique_seed_indices(
        dataset_episode_seeds, count=count, start_index=start_index
    )
    specs: list[RolloutSpec] = []
    for seed_rank, dataset_idx in enumerate(selected):
        seed = int(dataset_episode_seeds[dataset_idx])
        for sample_idx in range(samples_per_seed):
            specs.append(
                RolloutSpec(
                    output_index=seed_rank * samples_per_seed + sample_idx,
                    seed=seed,
                    source="dataset",
                    dataset_episode_index=int(dataset_idx),
                )
            )
    return specs


def _env_kwargs(metadata: dict[str, Any]) -> dict[str, Any]:
    env_kwargs = dict(metadata["env_kwargs"])
    env_kwargs["obs_mode"] = "pointcloud"
    env_kwargs["num_envs"] = 1
    env_kwargs.pop("render_mode", None)
    return env_kwargs


def _checkpoint_diagnostics(checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    policy_kwargs = dict(checkpoint.get("policy_kwargs", {}))
    args = dict(checkpoint.get("args", {}))
    normalizer = checkpoint.get("normalizer", {})
    return {
        "checkpoint_version": checkpoint.get("checkpoint_version"),
        "step": checkpoint.get("step"),
        "has_ema_model": checkpoint.get("ema_model") is not None,
        "policy_kwargs": {
            "horizon": policy_kwargs.get("horizon"),
            "n_obs_steps": policy_kwargs.get("n_obs_steps"),
            "n_action_steps": policy_kwargs.get("n_action_steps"),
            "goal_marker_points": policy_kwargs.get("goal_marker_points"),
            "goal_marker_radius": policy_kwargs.get("goal_marker_radius"),
            "use_goal_encoder": policy_kwargs.get("use_goal_encoder", False),
            "condition_type": policy_kwargs.get("condition_type"),
            "encoder_output_dim": policy_kwargs.get("encoder_output_dim"),
            "num_inference_steps": policy_kwargs.get("num_inference_steps"),
            "num_train_timesteps": policy_kwargs.get("num_train_timesteps"),
        },
        "train_args": {
            "normalizer_max_steps": args.get("normalizer_max_steps"),
            "max_steps": args.get("max_steps"),
            "dataset": args.get("dataset"),
            "goal_marker_points": args.get("goal_marker_points"),
            "goal_marker_radius": args.get("goal_marker_radius"),
            "use_goal_encoder": args.get("use_goal_encoder", False),
        },
        "normalizer_fields": sorted(
            {
                parts[1]
                for key in normalizer
                if (parts := str(key).split(".")) and len(parts) >= 3 and parts[0] == "fields"
            }
        ),
    }


def _run_config_summary(
    *,
    metadata: dict[str, Any],
    policy: Any,
    checkpoint_info: dict[str, Any],
    action_mode: ActionMode,
    crop_config: Any,
    max_steps: int,
    execution_horizon_chunks: int,
) -> dict[str, Any]:
    obs_encoder = getattr(policy, "obs_encoder", None)
    point_cloud_shape = getattr(obs_encoder, "point_cloud_shape", None)
    state_shape = getattr(obs_encoder, "state_shape", None)
    policy_goal_points = int(getattr(policy, "goal_marker_points", 0))
    checkpoint_goal_points = checkpoint_info["policy_kwargs"].get("goal_marker_points")
    warnings: list[str] = []
    if str(metadata.get("action_mode", "abs_joint")) != action_mode:
        warnings.append("metadata action_mode was normalized by parser")
    if checkpoint_goal_points is not None and int(checkpoint_goal_points) != policy_goal_points:
        warnings.append("loaded policy goal_marker_points differs from checkpoint policy_kwargs")
    if point_cloud_shape is not None and int(point_cloud_shape[0]) != int(crop_config.num_points):
        warnings.append("policy point-cloud shape differs from metadata crop num_points")
    return {
        "metadata_action_mode": metadata.get("action_mode", "abs_joint"),
        "used_action_mode": action_mode,
        "metadata_crop": {
            "bounds": np.asarray(crop_config.bounds, dtype=np.float32).tolist(),
            "num_points": int(crop_config.num_points),
            "robot_point_fraction": float(getattr(crop_config, "robot_point_fraction", 0.25)),
        },
        "policy": {
            "horizon": int(getattr(policy, "horizon", -1)),
            "n_obs_steps": int(getattr(policy, "n_obs_steps", -1)),
            "n_action_steps": int(getattr(policy, "n_action_steps", -1)),
            "goal_marker_points": policy_goal_points,
            "goal_marker_radius": float(getattr(policy, "goal_marker_radius", 0.0)),
            "use_goal_encoder": bool(getattr(policy, "use_goal_encoder", False)),
            "point_cloud_shape": list(point_cloud_shape) if point_cloud_shape is not None else None,
            "state_shape": list(state_shape) if state_shape is not None else None,
        },
        "checkpoint": checkpoint_info,
        "eval": {
            "max_steps": int(max_steps),
            "execution_horizon_chunks": int(execution_horizon_chunks),
        },
        "warnings": warnings,
    }


def _entry_distance(entry: dict[str, Any]) -> float:
    return float(np.asarray(entry["final_distance"], dtype=np.float32).reshape(-1)[0])


def _path_length(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _should_viz(args: Any, episode_number: int) -> bool:
    """Return True when episode_number (1-based) should get a multimodality plot."""
    if not getattr(args, "viz_trajectories", False):
        return False
    if args.viz_episodes is None:
        return True
    return episode_number in set(args.viz_episodes)


def _plot_multimodality(
    *,
    output_dir: Path,
    episode_number: int,
    seed: int,
    target: np.ndarray,
    samples: list[dict[str, Any]],
    goal_thresh: float,
) -> Path | None:
    """Overlay every sample's TCP path for one seed to visualize policy multimodality.

    Draws a 3D view and a top-down XY view side by side: each diffusion sample is one
    coloured polyline (solid if it reached the goal, dashed if not), with a shared start
    marker, the goal point, and the goal-threshold circle. Saved as a PNG and returned.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception as exc:  # pragma: no cover - optional dependency
        print(
            f"warning: matplotlib unavailable, skipping multimodality plot: {exc}",
            file=sys.stderr,
        )
        return None

    if not samples:
        return None

    target = np.asarray(target, dtype=np.float32).reshape(3)
    start = np.asarray(samples[0]["tcp_path"], dtype=np.float32).reshape(-1, 3)[0]
    try:
        cmap = plt.get_cmap("turbo")
    except Exception:
        cmap = plt.get_cmap("viridis")
    n = len(samples)
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]
    n_success = sum(1 for s in samples if s["success"])

    fig = plt.figure(figsize=(13, 5.5))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    axxy = fig.add_subplot(1, 2, 2)

    for s, color in zip(samples, colors):
        path = np.asarray(s["tcp_path"], dtype=np.float32).reshape(-1, 3)
        style = "-" if s["success"] else "--"
        label = f"s{s['sample_index']} {'reach' if s['success'] else 'miss'}"
        ax3d.plot(
            path[:, 0], path[:, 1], path[:, 2],
            style, color=color, linewidth=1.8, alpha=0.9, label=label,
        )
        axxy.plot(path[:, 0], path[:, 1], style, color=color, linewidth=1.8, alpha=0.9, label=label)

    # Shared start (red) and goal (green) markers.
    ax3d.scatter(*start, color="red", s=55, marker="o", label="start")
    ax3d.scatter(*target, color="green", s=80, marker="*", label="goal")
    axxy.scatter(start[0], start[1], color="red", s=55, marker="o")
    axxy.scatter(target[0], target[1], color="green", s=110, marker="*")
    circle = plt.Circle(
        (float(target[0]), float(target[1])),
        float(goal_thresh),
        color="green", fill=False, linestyle=":", alpha=0.7,
    )
    axxy.add_patch(circle)

    ax3d.set_title(f"3D TCP paths  (ep {episode_number}, seed {seed})")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    axxy.set_title(f"top-down XY  ({n_success}/{n} reached goal)")
    axxy.set_xlabel("x")
    axxy.set_ylabel("y")
    axxy.set_aspect("equal", adjustable="datalim")
    axxy.grid(True, alpha=0.3)
    handles, labels = axxy.get_legend_handles_labels()
    handles += [
        Line2D([0], [0], color="black", linestyle="-", label="reached"),
        Line2D([0], [0], color="black", linestyle="--", label="missed"),
    ]
    axxy.legend(handles=handles, fontsize=7, loc="best")

    fig.suptitle(
        f"Diffusion multimodality — episode {episode_number}, seed {seed}, {n} samples",
        fontsize=12,
    )
    fig.tight_layout()

    plot_dir = output_dir / "multimodality"
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = plot_dir / f"episode_{episode_number:03d}_seed{seed}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"multimodality plot: episode={episode_number} seed={seed} -> {out_path}", flush=True)
    return out_path


def _seed_any_success_rate(rows: list[dict[str, Any]]) -> float | None:
    """Fraction of unique seeds where at least one sample succeeded."""
    if not rows:
        return None
    by_seed: dict[int, list[bool]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), []).append(bool(row["success"]))
    return float(sum(1 for results in by_seed.values() if any(results)) / len(by_seed))


def _seed_all_success_rate(rows: list[dict[str, Any]], samples_per_seed: int) -> float | None:
    """Fraction of unique seeds where every sample succeeded."""
    if not rows:
        return None
    by_seed: dict[int, list[bool]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), []).append(bool(row["success"]))
    return float(sum(1 for results in by_seed.values() if all(results)) / len(by_seed))


def _success_rate(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return float(sum(1 for row in rows if row["success"]) / len(rows))


def _mean(values: Any) -> float | None:
    array = _finite_array(values)
    return None if array.size == 0 else float(np.mean(array))


def _median(values: Any) -> float | None:
    array = _finite_array(values)
    return None if array.size == 0 else float(np.median(array))


def _finite_array(values: Any) -> np.ndarray:
    numbers: list[float] = []
    for value in values:
        if value is None:
            continue
        numeric = float(value)
        if math.isfinite(numeric):
            numbers.append(numeric)
    return np.asarray(numbers, dtype=np.float64)


def _save_episode_video(
    frames: list[np.ndarray] | None,
    video_path: Path | None,
    video_fps: int,
) -> str | None:
    if frames is None or video_path is None:
        return None
    try:
        save_video(video_path, frames, fps=int(video_fps))
    except Exception as exc:  # pragma: no cover - rendering/codec issues are env-specific
        print(
            f"warning: failed to write video {video_path}: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return None
    return str(video_path)


def _format_xyz(xyz: Any) -> str:
    if xyz is None:
        return "(nan,nan,nan)"
    try:
        arr = np.asarray(xyz, dtype=np.float32).reshape(-1)
        return f"({arr[0]:.3f},{arr[1]:.3f},{arr[2]:.3f})"
    except Exception:
        return "(nan,nan,nan)"


def _format_optional(value: Any) -> str:
    if value is None:
        return "nan"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(numeric):
        return "nan"
    return f"{numeric:.4f}"


def _action_mode(value: str) -> ActionMode:
    if value not in {"abs_joint", "delta_joint"}:
        raise ValueError(f"unsupported action_mode {value!r}")
    return value  # type: ignore[return-value]


if __name__ == "__main__":
    raise SystemExit(main())
