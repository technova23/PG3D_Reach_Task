from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pg3d.envs.maniskill_adapter import register_pg3d_reach_envs
from pg3d.envs.maniskill_adapter.dataset import load_reach_metadata
from pg3d.eval import (
    NominalPathAvoidConfig,
    min_constraint_clearance,
    nominal_path_avoid_region,
    save_episode_constraints,
)
from pg3d.policies.dp3.checkpoint import load_reach_policy_from_checkpoint
from pg3d.utils.arrays import (
    bool_any as _bool_any,
)
from pg3d.utils.arrays import (
    bool_info as _bool_info,
)
from pg3d.utils.arrays import (
    float_info as _float_info,
)
from pg3d.utils.devices import select_device
from pg3d.utils.serialization import jsonable as _jsonable
from scripts.rollout_dp3_reach_policy import (
    ActionMode,
    RolloutSpec,
    append_obs_window,
    crop_config_from_metadata,
    make_initial_obs_window,
    obs_window_to_torch,
    policy_action_to_sim_action,
    rollout_observation_entry,
    select_rollout_specs,
)


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
    metadata = load_reach_metadata(args.dataset)
    dataset_episode_seeds = [
        int(episode["seed"]) for episode in metadata.get("episodes", []) if "seed" in episode
    ]
    specs = select_rollout_specs(
        source="dataset",
        dataset_episode_seeds=dataset_episode_seeds,
        episodes=args.episodes,
        episode_indices=args.episode_indices,
        seed_start=args.seed_start,
    )
    if not specs:
        raise RuntimeError("no dataset episodes selected")

    device = select_device(args.device)
    policy = load_reach_policy_from_checkpoint(
        args.checkpoint,
        device=device,
        prefer_ema=args.checkpoint_model == "ema",
    )
    crop_config = crop_config_from_metadata(metadata)
    action_mode = _action_mode(str(metadata.get("action_mode", "abs_joint")))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    constraints_dir = args.output_dir / "constraints"
    paths_dir = args.output_dir / "paths"
    constraints_dir.mkdir(parents=True, exist_ok=True)
    paths_dir.mkdir(parents=True, exist_ok=True)

    env: Any | None = None
    attempts: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    try:
        env = gym.make(str(metadata["env_id"]), **_env_kwargs(metadata))
        for spec in specs:
            row = _rollout_base_episode(
                env=env,
                policy=policy,
                spec=spec,
                action_mode=action_mode,
                crop_config=crop_config,
                device=device,
                max_steps=args.max_steps,
                replan_stride=(
                    args.replan_stride
                    if args.replan_stride is not None
                    else int(policy.n_action_steps)
                ),
                post_success_steps=args.post_success_steps,
                gripper_open=args.gripper_open,
            )
            attempts.append(_attempt_summary(row))
            print(
                f"attempt={spec.output_index} dataset_episode={spec.dataset_episode_index} "
                f"seed={spec.seed} success={row['success']} "
                f"final={_format_optional(row['final_distance'])} steps={row['steps']}"
            )
            if not bool(row["success"]):
                continue
            selected_output_index = len(selected)
            tcp_path = _constraint_path(row)
            constraint = nominal_path_avoid_region(
                tcp_path,
                config=NominalPathAvoidConfig(
                    radius=args.avoid_radius,
                    path_fraction=args.path_fraction,
                    margin=args.avoid_margin,
                    weight=args.avoid_weight,
                    tolerance=args.avoid_tolerance,
                ),
            )
            constraint_path = constraints_dir / f"episode_{selected_output_index:03d}.json"
            save_episode_constraints(constraint_path, [constraint])
            path_path = paths_dir / f"episode_{selected_output_index:03d}.npy"
            np.save(path_path, tcp_path.astype(np.float32, copy=False))
            selected.append(
                {
                    "output_index": selected_output_index,
                    "attempt_output_index": spec.output_index,
                    "dataset_episode_index": spec.dataset_episode_index,
                    "seed": spec.seed,
                    "constraint": str(constraint_path.relative_to(args.output_dir)),
                    "tcp_path": str(path_path.relative_to(args.output_dir)),
                    "tcp_path_points": int(tcp_path.shape[0]),
                    "tcp_path_length": _path_length(tcp_path),
                    "center": constraint.region.center.tolist(),
                    "radius": float(constraint.region.radius),
                    "discrete_min_clearance": min_constraint_clearance(tcp_path, [constraint]),
                    "final_distance": row["final_distance"],
                    "min_distance": row["min_distance"],
                    "first_success_step": row["first_success_step"],
                }
            )
    except Exception as exc:
        print(
            f"Failed to build nominal-path constraints: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    finally:
        if env is not None:
            env.close()

    episode_indices_path = args.output_dir / "episode_indices.txt"
    episode_indices_path.write_text(
        "".join(f"{int(row['dataset_episode_index'])}\n" for row in selected),
        encoding="utf-8",
    )
    manifest = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_model": args.checkpoint_model,
        "dataset": str(args.dataset),
        "source": "dataset",
        "env_id": metadata["env_id"],
        "env_kwargs": _env_kwargs(metadata),
        "attempted_episodes": len(attempts),
        "selected_episodes": len(selected),
        "min_successes": args.min_successes,
        "constraints_dir": "constraints",
        "episode_indices_file": "episode_indices.txt",
        "constraint_config": {
            "type": "nominal_path",
            "avoid_radius": args.avoid_radius,
            "path_fraction": args.path_fraction,
            "avoid_margin": args.avoid_margin,
            "avoid_weight": args.avoid_weight,
            "avoid_tolerance": args.avoid_tolerance,
        },
        "attempts": attempts,
        "selected": selected,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(_jsonable(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if len(selected) < args.min_successes and not args.allow_too_few_successes:
        print(
            f"only {len(selected)} base-success episodes selected; "
            f"required at least {args.min_successes}",
            file=sys.stderr,
        )
        return 1
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build small avoid-region constraints on successful nominal DP3 reach paths."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-model", choices=["ema", "raw"], default="ema")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--episode-indices", type=int, nargs="+", default=None)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--replan-stride", type=int, default=None)
    parser.add_argument("--post-success-steps", type=int, default=8)
    parser.add_argument("--gripper-open", type=float, default=0.04)
    parser.add_argument("--avoid-radius", type=float, default=0.03)
    parser.add_argument("--path-fraction", type=float, default=0.5)
    parser.add_argument("--avoid-margin", type=float, default=0.0)
    parser.add_argument("--avoid-weight", type=float, default=1.0)
    parser.add_argument("--avoid-tolerance", type=float, default=1e-6)
    parser.add_argument("--min-successes", type=int, default=15)
    parser.add_argument("--allow-too-few-successes", action="store_true")
    args = parser.parse_args(argv)
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.replan_stride is not None and args.replan_stride <= 0:
        raise ValueError("--replan-stride must be positive")
    if args.post_success_steps < 0:
        raise ValueError("--post-success-steps must be non-negative")
    if args.avoid_radius <= 0.0:
        raise ValueError("--avoid-radius must be positive")
    if not 0.0 <= args.path_fraction <= 1.0:
        raise ValueError("--path-fraction must be in [0, 1]")
    if args.avoid_margin < 0.0:
        raise ValueError("--avoid-margin must be non-negative")
    if args.avoid_tolerance < 0.0:
        raise ValueError("--avoid-tolerance must be non-negative")
    if args.min_successes < 0:
        raise ValueError("--min-successes must be non-negative")
    return args


def _rollout_base_episode(
    *,
    env: Any,
    policy: Any,
    spec: RolloutSpec,
    action_mode: ActionMode,
    crop_config: Any,
    device: torch.device,
    max_steps: int,
    replan_stride: int,
    post_success_steps: int,
    gripper_open: float,
) -> dict[str, Any]:
    obs, info = env.reset(seed=spec.seed, options={"reconfigure": True})
    first_entry = rollout_observation_entry(obs, info, env=env, crop_config=crop_config)
    obs_window = make_initial_obs_window(first_entry, n_obs_steps=int(policy.n_obs_steps))
    tcp_positions = [np.asarray(first_entry["tcp_pose"], dtype=np.float32).reshape(-1)[:3]]
    target_position = np.asarray(first_entry["target_position"], dtype=np.float32).reshape(3)
    steps = 0
    first_success_step: int | None = None
    observed_post_success_steps = 0
    final_distance = float(
        np.asarray(first_entry["final_distance"], dtype=np.float32).reshape(-1)[0]
    )
    min_distance = final_distance if np.isfinite(final_distance) else float("inf")
    terminated_or_truncated = False
    was_training = policy.training
    policy.eval()
    try:
        while steps < max_steps:
            with torch.inference_mode():
                policy_input = obs_window_to_torch(
                    obs_window,
                    device=device,
                    goal_marker_points=int(policy.goal_marker_points),
                    goal_marker_radius=float(policy.goal_marker_radius),
                )
                policy_output = policy.predict_action(policy_input)
                action_chunk = policy_output["action"][0].detach().cpu().numpy()
            for policy_action in action_chunk[:replan_stride]:
                sim_action = policy_action_to_sim_action(
                    policy_action,
                    np.asarray(obs_window[-1]["agent_pos"], dtype=np.float32),
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
                tcp_positions.append(
                    np.asarray(entry["tcp_pose"], dtype=np.float32).reshape(-1)[:3]
                )
                final_distance = _float_info(info, "tcp_to_goal_dist", default=float("nan"))
                if np.isfinite(final_distance):
                    min_distance = min(min_distance, final_distance)
                success = _bool_info(info, "success")
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
            if (
                terminated_or_truncated
                or steps >= max_steps
                or (
                    first_success_step is not None
                    and observed_post_success_steps >= post_success_steps
                )
            ):
                break
    finally:
        if was_training:
            policy.train()
    return {
        "output_index": spec.output_index,
        "seed": spec.seed,
        "source": spec.source,
        "dataset_episode_index": spec.dataset_episode_index,
        "steps": steps,
        "success": first_success_step is not None,
        "first_success_step": first_success_step,
        "final_distance": final_distance if np.isfinite(final_distance) else None,
        "min_distance": min_distance if np.isfinite(min_distance) else None,
        "target_position": target_position,
        "tcp_positions": np.stack(tcp_positions, axis=0).astype(np.float32, copy=False),
    }


def _constraint_path(row: dict[str, Any]) -> np.ndarray:
    tcp = np.asarray(row["tcp_positions"], dtype=np.float32)
    first_success_step = row.get("first_success_step")
    if first_success_step is None:
        return tcp
    return tcp[: int(first_success_step) + 1]


def _attempt_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "output_index": row["output_index"],
        "dataset_episode_index": row["dataset_episode_index"],
        "seed": row["seed"],
        "success": row["success"],
        "first_success_step": row["first_success_step"],
        "steps": row["steps"],
        "final_distance": row["final_distance"],
        "min_distance": row["min_distance"],
    }


def _env_kwargs(metadata: dict[str, Any]) -> dict[str, Any]:
    env_kwargs = dict(metadata["env_kwargs"])
    env_kwargs["obs_mode"] = "pointcloud"
    env_kwargs["num_envs"] = 1
    env_kwargs.pop("render_mode", None)
    return env_kwargs


def _action_mode(value: str) -> ActionMode:
    if value not in {"abs_joint", "delta_joint"}:
        raise ValueError(f"unsupported action_mode {value!r}")
    return value  # type: ignore[return-value]


def _path_length(points: np.ndarray) -> float:
    if points.shape[0] <= 1:
        return 0.0
    return float(np.sum(np.linalg.norm(points[1:] - points[:-1], axis=1)))


def _format_optional(value: Any) -> str:
    if value is None:
        return "nan"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(numeric):
        return "nan"
    return f"{numeric:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
