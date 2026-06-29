from __future__ import annotations

import argparse
import json
import math
import secrets
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
from pg3d.utils.devices import select_device
from pg3d.utils.serialization import jsonable as _jsonable
from scripts.rollout_dp3_reach_policy_nitin import (
    ActionMode,
    RolloutSpec,
    append_obs_window,
    crop_config_from_metadata,
    make_initial_obs_window,
    obs_window_to_torch,
    policy_action_to_sim_action,
    rollout_observation_entry,
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
    metadata = load_reach_metadata(args.dataset)
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
    action_mode = _action_mode(str(metadata.get("action_mode", "delta_joint")))
    crop_config = crop_config_from_metadata(metadata)
    goal_thresh = (
        float(args.goal_thresh)
        if args.goal_thresh is not None
        else float(dict(metadata.get("env_kwargs", {})).get("goal_thresh", 0.025))
    )
    dataset_episode_seeds = _resolve_dataset_episode_seeds(metadata)
    specs = _unique_seed_specs(
        dataset_episode_seeds,
        count=args.episodes,
        start_index=args.start_index,
        samples_per_seed=args.samples_per_seed,
    )
    if not specs:
        raise RuntimeError("no unique-seed dataset episodes selected")
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

    zarr_root = zarr.open_group(str(args.dataset), mode="r")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rerun_enabled = not args.no_rerun
    rerun_path = args.rerun_path or (args.output_dir / "accuracy_chk_nitin.rrd")
    rerun_logger: Any | None = None
    if rerun_enabled:
        try:
            import rerun as rr
        except Exception as exc:
            print(f"[WARN] rerun-sdk unavailable, skipping rerun export: {exc}", file=sys.stderr)
            rerun_enabled = False
        else:
            rerun_logger = rr
            rerun_path.parent.mkdir(parents=True, exist_ok=True)
            rr.init("pg3d_dp3_reach_policy_accuracy", spawn=False)
            rr.save(str(rerun_path))
    rows: list[dict[str, Any]] = []
    metrics_path = args.output_dir / "metrics.jsonl"
    env: Any | None = None
    try:
        env = gym.make(str(metadata["env_id"]), **_env_kwargs(metadata))
        with metrics_path.open("w", encoding="utf-8") as metrics_file:
            for spec in specs:
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
                    sample_noise_mode=args.sample_noise_mode,
                    sample_noise_seed=_sample_noise_seed(
                        mode=args.sample_noise_mode,
                        base_seed=args.seed,
                        sample_index=spec.output_index,
                    ),
                    rerun_logger=rerun_logger,
                )
                row["sample_index"] = spec.output_index % args.samples_per_seed
                rows.append(row)
                metrics_file.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
                metrics_file.flush()
                print(
                    f"episode={row['episode']} dataset_episode={row['dataset_episode_index']} "
                    f"seed={row['seed']} sample={row['sample_index']}/{args.samples_per_seed} "
                    f"success={row['success']} "
                    f"final={_format_optional(row['final_distance'])} "
                    f"min={_format_optional(row['min_distance'])} "
                    f"steps={row['steps']}",
                    flush=True,
                )
    except Exception as exc:
        print(f"Failed reach checkpoint eval: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if rerun_logger is not None:
            rerun_logger.disconnect()
        if env is not None:
            env.close()

    summary = {
        "checkpoint": str(checkpoint_path),
        "dataset": str(args.dataset),
        "env_id": metadata["env_id"],
        "env_kwargs": _env_kwargs(metadata),
        "action_mode": action_mode,
        "config_diagnostics": run_config,
        "goal_thresh": goal_thresh,
        "rerun_enabled": rerun_enabled,
        "rerun_path": str(rerun_path) if rerun_enabled else None,
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a plain DP3 reach checkpoint on first-occurrence unique dataset seeds. "
            "Normal policy rollout only: no constraints, rejection, reranking, or video."
        )
    )
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument("--checkpoint", type=Path, default=None)
    checkpoint_group.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-model", choices=["ema", "raw"], default="ema")
    parser.add_argument("--dataset", type=Path, required=True)
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
        "--start-index",
        type=int,
        default=0,
        help="Skip this many unique dataset seeds before selecting episodes.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sample-noise-mode",
        choices=["random", "seeded"],
        default="random",
        help=(
            "how to seed the diffusion latent for each sample; "
            "'random' uses fresh entropy per rollout, 'seeded' is reproducible"
        ),
    )
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--post-success-steps", type=int, default=8)
    parser.add_argument("--execution-horizon-chunks", type=int, default=1)
    parser.add_argument("--goal-thresh", type=float, default=None)
    parser.add_argument("--gripper-open", type=float, default=0.04)
    parser.add_argument(
        "--rerun-path",
        type=Path,
        default=None,
        help=(
            "single Rerun file for the full evaluation run; defaults to "
            "<output-dir>/accuracy_chk_nitin.rrd"
        ),
    )
    parser.add_argument(
        "--no-rerun",
        action="store_true",
        help="disable per-episode Rerun export",
    )
    parser.add_argument(
        "--print-config-only",
        action="store_true",
        help="Print metadata/checkpoint/eval diagnostics and exit before running episodes.",
    )
    parser.add_argument("--allow-failure", action="store_true")
    args = parser.parse_args(argv)
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
    sample_noise_mode: str,
    sample_noise_seed: int,
    rerun_logger: Any | None,
) -> dict[str, Any]:
    if spec.dataset_episode_index is None:
        raise ValueError("run_reach_episode requires a dataset_episode_index to restore Zarr state")
    zarr_context = _zarr_episode_context(zarr_root, spec.dataset_episode_index)
    obs, info = _reset_to_zarr_episode(env, rollout_seed=spec.seed, zarr_context=zarr_context)
    entry = rollout_observation_entry(obs, info, env=env, crop_config=crop_config)
    entry = _apply_zarr_initial_entry(entry, zarr_context)
    initial_entry = dict(entry)
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
    sample_generator = _make_sample_generator(
        device=device,
        sample_noise_mode=sample_noise_mode,
        seed=sample_noise_seed,
    )
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
                output = _predict_action_with_generator(
                    policy=policy,
                    obs_dict=policy_input,
                    generator=sample_generator,
                )
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
    finite_distances = [distance for distance in distances if np.isfinite(distance)]
    final_tcp = tcp_path[-1]
    success = first_success_step is not None
    if rerun_logger is not None:
        _log_episode_to_rerun(
            rr=rerun_logger,
            episode_index=spec.output_index,
            initial_entry=initial_entry,
            tcp_path=np.asarray(tcp_path, dtype=np.float32),
            success=success,
        )
    return {
        "episode": spec.output_index,
        "dataset_episode_index": spec.dataset_episode_index,
        "seed": spec.seed,
        "source": spec.source,
        "success": success,
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
    }


def _log_episode_to_rerun(
    *,
    rr: Any,
    episode_index: int,
    initial_entry: dict[str, Any],
    tcp_path: np.ndarray,
    success: bool,
) -> None:
    rr.set_time_sequence("step", 0)

    valid = np.asarray(initial_entry["point_valid_mask"], dtype=bool)
    points = np.asarray(initial_entry["point_cloud"], dtype=np.float32)[valid]
    robot_mask = np.asarray(initial_entry["robot_mask"], dtype=bool)[valid]
    scene_points = points[~robot_mask]
    robot_points = points[robot_mask]
    if scene_points.size:
        rr.log(
            f"episodes/{episode_index:04d}/point_cloud/scene",
            rr.Points3D(scene_points, colors=[[170, 170, 170]] * len(scene_points), radii=0.0035),
            static=True,
        )
    if robot_points.size:
        rr.log(
            f"episodes/{episode_index:04d}/point_cloud/robot",
            rr.Points3D(robot_points, colors=[[255, 90, 90]] * len(robot_points), radii=0.005),
            static=True,
        )

    target = np.asarray(initial_entry["target_position"], dtype=np.float32).reshape(1, 3)
    start = np.asarray(initial_entry["tcp_pose"], dtype=np.float32).reshape(-1)[:3].reshape(1, 3)
    rr.log(
        f"episodes/{episode_index:04d}/world/target",
        rr.Points3D(target, colors=[[0, 255, 0]], radii=0.018),
        static=True,
    )
    rr.log(
        f"episodes/{episode_index:04d}/world/start_tcp",
        rr.Points3D(start, colors=[[0, 120, 255]], radii=0.014),
        static=True,
    )

    path_color = [[0, 150, 255]] if success else [[255, 60, 60]]
    rr.log(
        f"episodes/{episode_index:04d}/world/tcp_path",
        rr.LineStrips3D([np.asarray(tcp_path, dtype=np.float32)], colors=path_color, radii=0.003),
        static=True,
    )
    rr.log(
        f"episodes/{episode_index:04d}/world/tcp_end",
        rr.Points3D([np.asarray(tcp_path[-1], dtype=np.float32)], colors=path_color, radii=0.012),
        static=True,
    )


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


def _resolve_dataset_episode_seeds(metadata: dict[str, Any]) -> list[int]:
    episodes = metadata.get("episodes", [])
    if episodes and all(isinstance(episode, dict) and "seed" in episode for episode in episodes):
        return [int(episode["seed"]) for episode in episodes]

    collection = metadata.get("collection", {})
    seed_start = collection.get("seed_start", metadata.get("seed_start"))
    if seed_start is None:
        raise ValueError("dataset metadata has no episode seeds and no seed_start fallback")

    num_episodes = (
        metadata.get("dataset_stats", {}).get("num_episodes")
        or metadata.get("summary", {}).get("num_episodes")
        or len(episodes)
    )
    if not num_episodes:
        raise ValueError("dataset metadata has no episode seeds and no num_episodes fallback")

    return [int(seed_start) + idx for idx in range(int(num_episodes))]


def _make_sample_generator(
    *,
    device: torch.device,
    sample_noise_mode: str,
    seed: int,
) -> torch.Generator:
    generator = torch.Generator(device=device)
    if sample_noise_mode == "seeded":
        generator.manual_seed(int(seed))
    else:
        generator.manual_seed(int(secrets.randbits(63)))
    return generator


def _sample_noise_seed(*, mode: str, base_seed: int, sample_index: int) -> int:
    if mode == "seeded":
        return int(base_seed + sample_index)
    return int(secrets.randbits(63))


def _predict_action_with_generator(
    *,
    policy: Any,
    obs_dict: dict[str, torch.Tensor],
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    nobs = policy.normalizer.normalize(obs_dict)
    assert isinstance(nobs, dict)
    if getattr(policy, "use_goal_encoder", False):
        nobs["goal_rel"] = policy._goal_rel(obs_dict)
    if not policy.use_pc_color:
        nobs["point_cloud"] = nobs["point_cloud"][..., :3]

    value = next(iter(nobs.values()))
    batch_size = value.shape[0]
    horizon = policy.horizon
    action_dim = policy.action_dim
    obs_steps = policy.n_obs_steps
    device = policy.device
    dtype = policy.dtype

    global_cond = None
    if policy.obs_as_global_cond:
        this_nobs = {
            key: tensor[:, :obs_steps, ...].reshape(-1, *tensor.shape[2:])
            for key, tensor in nobs.items()
        }
        nobs_features = policy.obs_encoder(this_nobs)
        if "cross_attention" in policy.condition_type:
            global_cond = nobs_features.reshape(batch_size, obs_steps, -1)
        else:
            global_cond = nobs_features.reshape(batch_size, -1)
        cond_data = torch.zeros(
            size=(batch_size, horizon, action_dim),
            device=device,
            dtype=dtype,
        )
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
    else:
        this_nobs = {
            key: tensor[:, :horizon, ...].reshape(-1, *tensor.shape[2:])
            for key, tensor in nobs.items()
        }
        nobs_features = policy.obs_encoder(this_nobs).reshape(batch_size, horizon, -1)
        cond_data = torch.zeros(
            size=(batch_size, horizon, action_dim + policy.obs_feature_dim),
            device=device,
            dtype=dtype,
        )
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        cond_data[:, :obs_steps, action_dim:] = nobs_features[:, :obs_steps]
        cond_mask[:, :obs_steps, action_dim:] = True

    trajectory = torch.randn(
        size=cond_data.shape,
        dtype=cond_data.dtype,
        device=cond_data.device,
        generator=generator,
    )
    policy.noise_scheduler.set_timesteps(policy.num_inference_steps)
    trajectory[cond_mask] = cond_data[cond_mask]
    for timestep in policy.noise_scheduler.timesteps:
        trajectory[cond_mask] = cond_data[cond_mask]
        model_output = policy.model(sample=trajectory, timestep=timestep, global_cond=global_cond)
        trajectory = policy.noise_scheduler.step(model_output, timestep, trajectory).prev_sample
    trajectory[cond_mask] = cond_data[cond_mask]

    naction_pred = trajectory[..., :action_dim]
    action_pred = policy.normalizer["action"].unnormalize(naction_pred)
    start = obs_steps - 1
    end = start + policy.n_action_steps
    return {
        "action": action_pred[:, start:end],
        "action_pred": action_pred,
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


def _unique_seed_specs(
    dataset_episode_seeds: list[int],
    *,
    count: int,
    start_index: int,
    samples_per_seed: int = 1,
) -> list[RolloutSpec]:
    seen: set[int] = set()
    unique_indices: list[int] = []
    for dataset_idx, seed in enumerate(dataset_episode_seeds):
        if seed in seen:
            continue
        seen.add(seed)
        unique_indices.append(dataset_idx)
    selected = unique_indices[start_index : start_index + count]
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
    if str(metadata.get("action_mode", "delta_joint")) != action_mode:
        warnings.append("metadata action_mode was normalized by parser")
    if checkpoint_goal_points is not None and int(checkpoint_goal_points) != policy_goal_points:
        warnings.append("loaded policy goal_marker_points differs from checkpoint policy_kwargs")
    if point_cloud_shape is not None and int(point_cloud_shape[0]) != int(crop_config.num_points):
        warnings.append("policy point-cloud shape differs from metadata crop num_points")
    return {
        "metadata_action_mode": metadata.get("action_mode", "delta_joint"),
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
