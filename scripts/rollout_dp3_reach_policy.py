from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from pg3d.envs.maniskill_adapter import adapt_observation, register_pg3d_reach_envs
from pg3d.envs.maniskill_adapter.dataset import (
    DEFAULT_WORKSPACE_BOUNDS,
    PointCloudCropConfig,
    crop_point_cloud,
    load_reach_metadata,
)
from pg3d.envs.xarm_adapter import register_pg3d_xarm7_gripper_reach_envs
from pg3d.policies.dp3 import SimpleDP3
from pg3d.policies.dp3.checkpoint import load_reach_policy_from_checkpoint
from pg3d.policies.dp3.goal_markers import (
    DEFAULT_GOAL_MARKER_RADIUS,
    insert_goal_marker_points,
)
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
    float_value as _float_value,
)
from pg3d.utils.arrays import (
    frame_to_numpy as _frame_to_numpy,
)
from pg3d.utils.devices import select_device
from pg3d.utils.serialization import jsonable as _jsonable

Source = Literal["dataset", "fresh"]
ActionMode = Literal["abs_joint", "delta_joint"]


@dataclass(frozen=True)
class RolloutSpec:
    """One policy rollout seed and optional source episode."""

    output_index: int
    seed: int
    source: Source
    dataset_episode_index: int | None = None


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
            "uv sync --extra cu129 --extra maniskill --extra viz --group dev --group notebooks",
            file=sys.stderr,
        )
        return 2

    register_pg3d_reach_envs()
    register_pg3d_xarm7_gripper_reach_envs()
    device = select_device(args.device)
    policy = load_reach_policy_from_checkpoint(
        args.checkpoint,
        device=device,
        prefer_ema=args.checkpoint_model == "ema",
    )
    trajectory_family_count = _policy_trajectory_family_count(policy)
    trajectory_family_id = args.trajectory_family_id
    if trajectory_family_count is not None:
        if trajectory_family_id is None:
            trajectory_family_id = min(10, trajectory_family_count - 1)
        if not 0 <= trajectory_family_id < trajectory_family_count:
            raise ValueError(
                "--trajectory-family-id must be in "
                f"[0, {trajectory_family_count - 1}], got {trajectory_family_id}"
            )
        print(
            "rollout family conditioning: "
            f"id={trajectory_family_id} count={trajectory_family_count}",
            flush=True,
        )
    metadata = load_reach_metadata(args.dataset)
    dataset_episode_seeds = [
        int(episode["seed"]) for episode in metadata.get("episodes", []) if "seed" in episode
    ]
    specs = select_rollout_specs(
        source=args.source,
        dataset_episode_seeds=dataset_episode_seeds,
        episodes=args.episodes,
        episode_indices=args.episode_indices,
        seed_start=args.seed_start,
    )
    if not specs:
        raise RuntimeError("no rollout episodes selected")

    crop_config = crop_config_from_metadata(metadata)
    action_mode = _action_mode(str(metadata.get("action_mode", "abs_joint")))
    env_kwargs = dict(metadata["env_kwargs"])
    env_kwargs["render_mode"] = "rgb_array"
    env_kwargs.setdefault("obs_mode", "pointcloud")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env: Any | None = None
    summaries: list[dict[str, Any]] = []
    metrics_path = args.output_dir / "metrics.jsonl"
    try:
        env = gym.make(str(metadata["env_id"]), **env_kwargs)
        with metrics_path.open("w", encoding="utf-8") as metrics_file:
            for spec in specs:
                summary = run_policy_rollout(
                    env=env,
                    policy=policy,
                    spec=spec,
                    action_mode=action_mode,
                    crop_config=crop_config,
                    output_dir=args.output_dir,
                    device=device,
                    max_steps=args.max_steps,
                    replan_stride=(
                        args.replan_stride
                        if args.replan_stride is not None
                        else int(policy.n_action_steps)
                    ),
                    post_success_steps=args.post_success_steps,
                    gripper_open=args.gripper_open,
                    video_fps=args.video_fps,
                    metrics_file=metrics_file,
                    trajectory_family_id=trajectory_family_id,
                    trajectory_family_count=trajectory_family_count,
                )
                summaries.append(summary)
                print(
                    f"episode={spec.output_index} seed={spec.seed} "
                    f"success={summary['success']} final_distance={summary['final_distance']:.4f} "
                    f"steps={summary['steps']}"
                )
    except Exception as exc:
        print(f"Failed to roll out DP3 reach policy: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if env is not None:
            env.close()

    summary = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "source": args.source,
        "env_id": metadata["env_id"],
        "env_kwargs": env_kwargs,
        "action_mode": action_mode,
        "episodes": summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    failures = sum(0 if episode["success"] else 1 for episode in summaries)
    return 0 if args.allow_failure or failures == 0 else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Roll out a trained pg3d-native DP3 reach policy in ManiSkill."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-model", choices=["ema", "raw"], default="ema")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("artifacts/reach-dataset-smoke/pg3d-reach-smoke.zarr"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reach-policy-rollouts"),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--source", choices=["dataset", "fresh"], default="dataset")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--episode-indices", type=int, nargs="+", default=None)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--replan-stride", type=int, default=None)
    parser.add_argument("--post-success-steps", type=int, default=8)
    parser.add_argument("--gripper-open", type=float, default=0.04)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument(
        "--trajectory-family-id",
        type=int,
        default=None,
        help=(
            "family id to inject for checkpoints trained with trajectory_family_onehot; "
            "defaults to shallow_direct id 10 when available"
        ),
    )
    parser.add_argument("--allow-failure", action="store_true")
    args = parser.parse_args(argv)
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.replan_stride is not None and args.replan_stride <= 0:
        raise ValueError("--replan-stride must be positive")
    if args.post_success_steps < 0:
        raise ValueError("--post-success-steps must be non-negative")
    return args


def run_policy_rollout(
    *,
    env: Any,
    policy: SimpleDP3,
    spec: RolloutSpec,
    action_mode: ActionMode,
    crop_config: PointCloudCropConfig,
    output_dir: Path,
    device: torch.device,
    max_steps: int,
    replan_stride: int,
    post_success_steps: int,
    gripper_open: float,
    video_fps: int,
    metrics_file: Any | None,
    video_path: Path | None = None,
    write_rerun: bool = True,
    trajectory_family_id: int | None = None,
    trajectory_family_count: int | None = None,
) -> dict[str, Any]:
    obs, info = env.reset(seed=spec.seed, options={"reconfigure": True})
    frames = [_frame_to_numpy(env.render())]
    timeline: list[dict[str, np.ndarray | bool | float]] = []
    first_entry = rollout_observation_entry(obs, info, env=env, crop_config=crop_config)
    obs_window = make_initial_obs_window(first_entry, n_obs_steps=int(policy.n_obs_steps))
    timeline.append(first_entry)
    steps = 0
    success = False
    first_success_step: int | None = None
    final_distance = float("nan")
    min_distance = float("inf")
    observed_post_success_steps = 0
    action_norms: list[float] = []
    post_success_action_norms: list[float] = []
    post_success_distances: list[float] = []

    while steps < max_steps:
        with torch.no_grad():
            policy_input = obs_window_to_torch(
                obs_window,
                device=device,
                goal_marker_points=int(policy.goal_marker_points),
                goal_marker_radius=float(policy.goal_marker_radius),
                trajectory_family_id=trajectory_family_id,
                trajectory_family_count=trajectory_family_count,
            )
            policy_output = policy.predict_action(policy_input)
            action_chunk = policy_output["action"][0].detach().cpu().numpy()

        for policy_action in action_chunk[:replan_stride]:
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
            obs, reward, terminated, truncated, info = env.step(sim_action)
            steps += 1
            frames.append(_frame_to_numpy(env.render()))
            entry = rollout_observation_entry(obs, info, env=env, crop_config=crop_config)
            obs_window = append_obs_window(obs_window, entry, n_obs_steps=int(policy.n_obs_steps))
            timeline.append(entry)
            success = _bool_info(info, "success")
            final_distance = _float_info(info, "tcp_to_goal_dist", default=float("nan"))
            if np.isfinite(final_distance):
                min_distance = min(min_distance, final_distance)
            action_norm = float(np.linalg.norm(sim_action))
            action_norms.append(action_norm)
            if success:
                if first_success_step is None:
                    first_success_step = steps
                    if np.isfinite(final_distance):
                        post_success_distances.append(final_distance)
                else:
                    observed_post_success_steps += 1
                    post_success_action_norms.append(action_norm)
                    if np.isfinite(final_distance):
                        post_success_distances.append(final_distance)
            post_success_done = (
                first_success_step is not None and observed_post_success_steps >= post_success_steps
            )
            if metrics_file is not None:
                metrics_file.write(
                    json.dumps(
                        _jsonable(
                            {
                                "episode": spec.output_index,
                                "seed": spec.seed,
                                "step": steps,
                                "reward": _float_value(reward),
                                "success": success,
                                "first_success_step": first_success_step,
                                "final_distance": final_distance,
                                "min_distance": (
                                    min_distance if np.isfinite(min_distance) else None
                                ),
                                "action_norm": action_norm,
                                "post_success": bool(first_success_step is not None),
                            }
                        ),
                        sort_keys=True,
                    )
                    + "\n"
                )
                metrics_file.flush()
            if (
                post_success_done
                or (first_success_step is None and _bool_any(terminated))
                or _bool_any(truncated)
                or steps >= max_steps
            ):
                break
        if first_success_step is not None and observed_post_success_steps >= post_success_steps:
            break
        if steps >= max_steps:
            break

    video_path = video_path or output_dir / f"episode_{spec.output_index:03d}.mp4"
    save_video(video_path, frames, fps=video_fps)
    rerun_path = output_dir / f"episode_{spec.output_index:03d}.rrd" if write_rerun else None
    if rerun_path is not None:
        save_rerun_timeline(rerun_path, timeline)
    return {
        "episode": spec.output_index,
        "seed": spec.seed,
        "source": spec.source,
        "dataset_episode_index": spec.dataset_episode_index,
        "steps": steps,
        "success": first_success_step is not None,
        "first_success_step": first_success_step,
        "final_distance": final_distance,
        "min_distance": min_distance if np.isfinite(min_distance) else None,
        "post_success_distance_drift": _distance_drift(post_success_distances),
        "mean_action_norm": float(np.mean(action_norms)) if action_norms else 0.0,
        "mean_post_success_action_norm": (
            float(np.mean(post_success_action_norms)) if post_success_action_norms else 0.0
        ),
        "video": str(video_path),
        "rerun": str(rerun_path) if rerun_path is not None else None,
    }


def crop_config_from_metadata(metadata: dict[str, Any]) -> PointCloudCropConfig:
    crop = metadata.get("crop", {})
    bounds = np.asarray(crop.get("bounds", DEFAULT_WORKSPACE_BOUNDS), dtype=np.float32)
    num_points = int(crop.get("num_points", 512))
    robot_point_fraction = float(crop.get("robot_point_fraction", 0.25))
    obstacle_point_quota = int(crop.get("obstacle_point_quota", 0))
    return PointCloudCropConfig(
        bounds=bounds,
        num_points=num_points,
        robot_point_fraction=robot_point_fraction,
        obstacle_point_quota=obstacle_point_quota,
    )


def rollout_observation_entry(
    obs: Any,
    info: Any,
    *,
    env: Any,
    crop_config: PointCloudCropConfig,
) -> dict[str, np.ndarray | bool | float]:
    adapted = adapt_observation(obs, info=info, env=env, task_name=_env_task_name(env))
    cropped = crop_point_cloud(
        adapted.point_cloud,
        robot_mask=adapted.robot_mask,
        aligned_masks={"obstacle_mask": adapted.object_masks["pg3d_obstacle"]}
        if "pg3d_obstacle" in adapted.object_masks
        else None,
        config=crop_config,
    )
    target_position = (
        np.zeros((3,), dtype=np.float32)
        if adapted.sim_gt is None or adapted.sim_gt.target_position is None
        else adapted.sim_gt.target_position.astype(np.float32, copy=True)
    )
    tcp_pose = (
        np.zeros((7,), dtype=np.float32)
        if adapted.robot_state.tcp_pose is None
        else adapted.robot_state.tcp_pose.astype(np.float32, copy=True)
    )
    entry = {
        "point_cloud": cropped["point_cloud"],
        "robot_mask": cropped["robot_mask"],
        "point_valid_mask": cropped["point_valid_mask"],
        "agent_pos": adapted.robot_state.as_agent_pos(),
        "target_position": target_position,
        "tcp_pose": tcp_pose,
        "success": bool(adapted.sim_gt.success) if adapted.sim_gt is not None else False,
        "final_distance": _float_info(info, "tcp_to_goal_dist", default=float("nan")),
    }
    if "obstacle_mask" in cropped:
        entry["obstacle_mask"] = cropped["obstacle_mask"]
        entry["obstacle_points_raw"] = int(np.count_nonzero(adapted.object_masks["pg3d_obstacle"]))
        entry["obstacle_points_cropped"] = int(np.count_nonzero(cropped["obstacle_mask"]))
    return entry


def make_initial_obs_window(
    entry: dict[str, np.ndarray | bool | float],
    *,
    n_obs_steps: int,
) -> list[dict[str, np.ndarray | bool | float]]:
    """Pad the initial policy observation window by repeating the first observation."""
    if n_obs_steps <= 0:
        raise ValueError("n_obs_steps must be positive")
    return [_copy_entry(entry) for _ in range(n_obs_steps)]


def append_obs_window(
    window: list[dict[str, np.ndarray | bool | float]],
    entry: dict[str, np.ndarray | bool | float],
    *,
    n_obs_steps: int,
) -> list[dict[str, np.ndarray | bool | float]]:
    """Append one observation and keep the most recent policy-visible window."""
    if n_obs_steps <= 0:
        raise ValueError("n_obs_steps must be positive")
    return [*_copy_window(window), _copy_entry(entry)][-n_obs_steps:]


def obs_window_to_torch(
    window: list[dict[str, np.ndarray | bool | float]],
    *,
    device: torch.device,
    goal_marker_points: int = 0,
    goal_marker_radius: float = DEFAULT_GOAL_MARKER_RADIUS,
    trajectory_family_id: int | None = None,
    trajectory_family_count: int | None = None,
) -> dict[str, torch.Tensor]:
    """Convert a rolling observation window into a batched DP3 observation dict."""
    point_cloud = np.stack([entry["point_cloud"] for entry in window], axis=0)
    if goal_marker_points:
        target_position = np.stack([entry["target_position"] for entry in window], axis=0)
        point_cloud = insert_goal_marker_points(
            point_cloud,
            target_position,
            num_points=goal_marker_points,
            radius=goal_marker_radius,
        )
    agent_pos = np.stack([entry["agent_pos"] for entry in window], axis=0)
    goal_xyz = np.stack([entry["target_position"] for entry in window], axis=0)
    ee_position = np.stack(
        [np.asarray(entry["tcp_pose"], dtype=np.float32).reshape(-1)[:3] for entry in window],
        axis=0,
    )
    batch = {
        "point_cloud": torch.from_numpy(point_cloud.astype(np.float32)).unsqueeze(0).to(device),
        "agent_pos": torch.from_numpy(agent_pos.astype(np.float32)).unsqueeze(0).to(device),
        "goal_xyz": torch.from_numpy(goal_xyz.astype(np.float32)).unsqueeze(0).to(device),
        "ee_position": torch.from_numpy(ee_position.astype(np.float32)).unsqueeze(0).to(device),
    }
    if trajectory_family_count is not None:
        if trajectory_family_count <= 0:
            raise ValueError("trajectory_family_count must be positive")
        family_id = 0 if trajectory_family_id is None else int(trajectory_family_id)
        if not 0 <= family_id < trajectory_family_count:
            raise ValueError(
                f"trajectory_family_id={family_id} is outside [0, {trajectory_family_count - 1}]"
            )
        family = np.zeros((len(window), trajectory_family_count), dtype=np.float32)
        family[:, family_id] = 1.0
        batch["trajectory_family_onehot"] = torch.from_numpy(family).unsqueeze(0).to(device)
    return batch


def _policy_trajectory_family_count(policy: SimpleDP3) -> int | None:
    family_shape = getattr(policy.obs_encoder, "family_shape", None)
    if family_shape is None:
        return None
    if len(family_shape) != 1:
        raise ValueError(f"unsupported trajectory family shape: {family_shape}")
    return int(family_shape[0])


def policy_action_to_sim_action(
    policy_action: np.ndarray,
    current_state: np.ndarray,
    *,
    action_mode: ActionMode,
    sim_action_dim: int,
    low: np.ndarray | None = None,
    high: np.ndarray | None = None,
    gripper_open: float = 0.04,
) -> np.ndarray:
    """Convert a 7D DP3 arm label into the ManiSkill Panda simulator action."""
    action = np.asarray(policy_action, dtype=np.float32).reshape(-1)
    state = np.asarray(current_state, dtype=np.float32).reshape(-1)
    if action.shape[0] != 7:
        raise ValueError(f"policy action must have shape [7], got {action.shape}")
    if state.shape[0] < 7:
        raise ValueError(f"current state must have at least 7 values, got {state.shape}")
    if action_mode == "abs_joint":
        arm_action = action
    elif action_mode == "delta_joint":
        arm_action = state[:7] + action
    else:
        raise ValueError(f"unsupported action_mode {action_mode!r}")

    if sim_action_dim == 7:
        sim_action = arm_action.astype(np.float32, copy=True)
    elif sim_action_dim == 8:
        sim_action = np.concatenate([arm_action, [gripper_open]]).astype(np.float32)
    else:
        raise ValueError(f"unsupported simulator action dimension {sim_action_dim}")
    if low is not None and high is not None:
        sim_action = np.clip(sim_action, np.asarray(low), np.asarray(high)).astype(np.float32)
    return sim_action


def select_rollout_specs(
    *,
    source: Source,
    dataset_episode_seeds: list[int],
    episodes: int,
    episode_indices: list[int] | None = None,
    seed_start: int = 10000,
) -> list[RolloutSpec]:
    """Select dataset or fresh rollout seeds, skipping training seeds for fresh rollouts."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if source == "dataset":
        if episode_indices is None:
            episode_indices = list(range(min(episodes, len(dataset_episode_seeds))))
        specs = []
        for output_idx, dataset_idx in enumerate(episode_indices):
            if dataset_idx < 0 or dataset_idx >= len(dataset_episode_seeds):
                raise IndexError(f"dataset episode index {dataset_idx} is out of range")
            specs.append(
                RolloutSpec(
                    output_index=output_idx,
                    seed=dataset_episode_seeds[dataset_idx],
                    source="dataset",
                    dataset_episode_index=dataset_idx,
                )
            )
        return specs
    if source == "fresh":
        if episode_indices is not None:
            raise ValueError("--episode-indices is only valid with --source dataset")
        training_seeds = set(dataset_episode_seeds)
        specs = []
        candidate = seed_start
        while len(specs) < episodes:
            if candidate not in training_seeds:
                specs.append(
                    RolloutSpec(
                        output_index=len(specs),
                        seed=candidate,
                        source="fresh",
                    )
                )
            candidate += 1
        return specs
    raise ValueError(f"unsupported source {source!r}")


def select_mixed_rollout_specs(
    *,
    dataset_episode_seeds: list[int],
    total_count: int,
    seed_start: int = 10000,
    selection_seed: int | None = None,
) -> list[RolloutSpec]:
    """Select the trainer's default mixed dataset/fresh checkpoint rollout set.

    When ``selection_seed`` is ``None`` the dataset episodes are the first
    ``dataset_count`` indices (legacy deterministic behaviour). When a seed is
    given the dataset episodes are drawn as a seeded random subset spread across
    the WHOLE dataset, and the fresh seeds are offset by the same seed, so each
    checkpoint exercises diverse goals instead of always the first few episodes.
    """
    if total_count <= 0:
        raise ValueError("total_count must be positive")
    dataset_count = min(len(dataset_episode_seeds), math.ceil(total_count * 0.6))
    fresh_count = total_count - dataset_count
    if selection_seed is None or dataset_count == 0:
        dataset_indices = list(range(dataset_count))
        fresh_offset = 0
    else:
        rng = np.random.default_rng(selection_seed)
        dataset_indices = sorted(
            int(idx)
            for idx in rng.choice(len(dataset_episode_seeds), size=dataset_count, replace=False)
        )
        fresh_offset = int(np.random.default_rng(selection_seed + 1).integers(0, 10000))
    specs = []
    for dataset_idx in dataset_indices:
        specs.append(
            RolloutSpec(
                output_index=len(specs),
                seed=dataset_episode_seeds[dataset_idx],
                source="dataset",
                dataset_episode_index=dataset_idx,
            )
        )
    training_seeds = set(dataset_episode_seeds)
    candidate = seed_start + fresh_offset
    while len(specs) < dataset_count + fresh_count:
        if candidate not in training_seeds:
            specs.append(
                RolloutSpec(
                    output_index=len(specs),
                    seed=candidate,
                    source="fresh",
                )
            )
        candidate += 1
    return specs


def select_random_dataset_rollout_specs(
    *,
    dataset_episode_seeds: list[int],
    total_count: int,
    seed: int,
) -> list[RolloutSpec]:
    """Select a deterministic random subset of dataset episodes for validation rollouts."""
    if total_count <= 0:
        raise ValueError("total_count must be positive")
    if not dataset_episode_seeds:
        return []
    count = min(total_count, len(dataset_episode_seeds))
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(dataset_episode_seeds), size=count, replace=False))
    return [
        RolloutSpec(
            output_index=output_idx,
            seed=int(dataset_episode_seeds[int(dataset_idx)]),
            source="dataset",
            dataset_episode_index=int(dataset_idx),
        )
        for output_idx, dataset_idx in enumerate(selected)
    ]


def rollout_spec_video_stem(spec: RolloutSpec, *, validation: bool = False) -> str:
    """Return a stable video stem that includes dataset identity when available."""
    if validation and spec.dataset_episode_index is not None:
        return f"validation_episode_{spec.dataset_episode_index:03d}_seed_{spec.seed}"
    if spec.dataset_episode_index is not None:
        return f"{spec.source}_{spec.dataset_episode_index:03d}_seed_{spec.seed}"
    return f"{spec.source}_{spec.output_index:03d}_seed_{spec.seed}"


def save_video(path: Path, frames: list[np.ndarray], *, fps: int) -> None:
    if not frames:
        raise RuntimeError("no frames were captured for video export")
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)


def save_rerun_timeline(
    path: Path,
    timeline: list[dict[str, np.ndarray | bool | float]],
    *,
    constraints: list[object] | None = None,
    replans: list[dict[str, Any]] | None = None,
    goal_marker_points: int = 0,
    goal_marker_radius: float = DEFAULT_GOAL_MARKER_RADIUS,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path = path.with_suffix(".policy_input.npz")
    metadata_path = path.with_suffix(".policy_input.json")
    semantics = [
        _policy_point_cloud_semantics(
            entry,
            goal_marker_points=goal_marker_points,
            goal_marker_radius=goal_marker_radius,
        )
        for entry in timeline
    ]
    np.savez_compressed(
        bundle_path,
        point_cloud=np.stack([item["all_points"] for item in semantics]),
        colors=np.stack([item["all_colors"] for item in semantics]),
        valid_mask=np.stack([item["all_valid_mask"] for item in semantics]),
        robot_mask=np.stack([item["all_robot_mask"] for item in semantics]),
        obstacle_mask=np.stack([item["all_obstacle_mask"] for item in semantics]),
        scene_mask=np.stack([item["all_scene_mask"] for item in semantics]),
        goal_mask=np.stack([item["all_goal_mask"] for item in semantics]),
        target_position=np.stack(
            [np.asarray(entry["target_position"], dtype=np.float32) for entry in timeline]
        ),
        tcp_position=np.stack(
            [np.asarray(entry["tcp_pose"], dtype=np.float32).reshape(-1)[:3] for entry in timeline]
        ),
        tcp_clearance=np.asarray(
            [
                _rerun_tcp_clearance(
                    np.asarray(entry["tcp_pose"], dtype=np.float32).reshape(-1)[:3],
                    constraints or [],
                )
                for entry in timeline
            ],
            dtype=np.float32,
        ),
    )
    constraint_visuals: list[dict[str, Any]] = []
    if constraints:
        from pg3d.viz.constraints import avoid_region_line_visuals

        for visual in avoid_region_line_visuals(constraints):
            constraint_visuals.append(
                {
                    "name": visual.name,
                    "line_strips": [line.tolist() for line in visual.line_strips],
                    "color": list(visual.color),
                }
            )
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": "pg3d.policy_pointcloud_bundle.v1",
                "rerun_writer_version": "0.35.0",
                "constraint_visuals": constraint_visuals,
                "replans": _jsonable(replans or []),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    exporter_python = _rerun35_exporter_python()
    exporter = Path(__file__).with_name("export_policy_pointcloud_rerun35.py")
    subprocess.run(
        [
            str(exporter_python),
            str(exporter),
            "--bundle",
            str(bundle_path),
            "--metadata",
            str(metadata_path),
            "--output",
            str(path),
        ],
        check=True,
    )


def _rerun35_exporter_python() -> Path:
    configured = os.environ.get("PG3D_RERUN35_PYTHON")
    candidate = (
        Path(configured)
        if configured
        else Path(__file__).resolve().parents[1] / ".venv-rerun35" / "bin" / "python"
    )
    if not candidate.is_file():
        raise RuntimeError(
            "Rerun 0.35 export requires an isolated exporter environment. "
            "Create .venv-rerun35 as documented in docs/runbooks/commands.md or set "
            "PG3D_RERUN35_PYTHON to its Python executable."
        )
    return candidate


def _policy_point_cloud_semantics(
    entry: dict[str, Any],
    *,
    goal_marker_points: int,
    goal_marker_radius: float,
) -> dict[str, np.ndarray]:
    point_cloud = np.asarray(entry["point_cloud"], dtype=np.float32).copy()
    if goal_marker_points:
        point_cloud = insert_goal_marker_points(
            point_cloud,
            np.asarray(entry["target_position"], dtype=np.float32),
            num_points=goal_marker_points,
            radius=goal_marker_radius,
        )
    valid = np.asarray(entry["point_valid_mask"], dtype=bool).copy()
    robot = np.asarray(entry["robot_mask"], dtype=bool).copy()
    obstacle = np.asarray(entry.get("obstacle_mask", np.zeros_like(valid)), dtype=bool).copy()
    goal = np.zeros_like(valid)
    if goal_marker_points:
        marker_count = min(goal_marker_points, goal.size)
        goal[-marker_count:] = True
        valid[-marker_count:] = True
        robot[-marker_count:] = False
        obstacle[-marker_count:] = False
    colors = np.full((point_cloud.shape[0], 3), [160, 160, 160], dtype=np.uint8)
    colors[~valid] = [40, 40, 40]
    colors[robot] = [0, 128, 255]
    colors[obstacle] = [180, 90, 20]
    colors[goal] = [0, 255, 0]
    scene = valid & ~robot & ~obstacle & ~goal
    return {
        "all_points": point_cloud,
        "all_colors": colors,
        "all_valid_mask": valid,
        "all_robot_mask": robot,
        "all_obstacle_mask": obstacle,
        "all_scene_mask": scene,
        "all_goal_mask": goal,
        "points": point_cloud[valid],
        "colors": colors[valid],
        "robot_mask": robot[valid],
        "obstacle_mask": obstacle[valid],
        "goal_mask": goal[valid],
    }


def _rerun_tcp_clearance(tcp: np.ndarray, constraints: list[object]) -> float | None:
    clearances: list[float] = []
    for constraint in constraints:
        region = getattr(constraint, "region", None)
        if region is None or not hasattr(region, "signed_distance"):
            continue
        distance = float(region.signed_distance(tcp.reshape(1, 3))[0])
        clearances.append(distance - float(getattr(constraint, "margin", 0.0)))
    return min(clearances) if clearances else None


def _copy_window(
    window: list[dict[str, np.ndarray | bool | float]],
) -> list[dict[str, np.ndarray | bool | float]]:
    return [_copy_entry(entry) for entry in window]


def _copy_entry(
    entry: dict[str, np.ndarray | bool | float],
) -> dict[str, np.ndarray | bool | float]:
    return {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in entry.items()
    }


def _env_task_name(env: Any) -> str:
    unwrapped = getattr(env, "unwrapped", env)
    spec = getattr(unwrapped, "spec", None)
    return str(getattr(spec, "id", "unknown"))


def _action_mode(value: str) -> ActionMode:
    if value not in {"abs_joint", "delta_joint"}:
        raise ValueError(f"unsupported action_mode {value!r}")
    return value  # type: ignore[return-value]


def _distance_drift(distances: list[float]) -> float:
    finite = np.asarray([value for value in distances if np.isfinite(value)], dtype=np.float32)
    if finite.size <= 1:
        return 0.0
    return float(np.max(finite) - np.min(finite))


if __name__ == "__main__":
    raise SystemExit(main())
