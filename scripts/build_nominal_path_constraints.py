from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr

from pg3d.constraints import BoxRegion
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
    metadata = load_reach_metadata(args.dataset)
    dataset_episode_seeds = [
        int(episode["seed"]) for episode in metadata.get("episodes", []) if "seed" in episode
    ]
    specs = select_rollout_specs(
        source="dataset",
        dataset_episode_seeds=dataset_episode_seeds,
        episodes=args.episodes,
        episode_indices=(
            _read_episode_indices_file(args.episode_indices_file)
            if args.episode_indices_file is not None
            else args.episode_indices
        ),
        seed_start=args.seed_start,
    )
    if not specs:
        raise RuntimeError("no dataset episodes selected")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    constraints_dir = args.output_dir / "constraints"
    paths_dir = args.output_dir / "paths"
    constraints_dir.mkdir(parents=True, exist_ok=True)
    paths_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    if args.path_source == "dataset_demo":
        root = zarr.open_group(str(args.dataset), mode="r")
        rows = [_dataset_demo_episode(root, spec=spec) for spec in specs]
    else:
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
            args.checkpoint,
            device=device,
            prefer_ema=args.checkpoint_model == "ema",
        )
        crop_config = crop_config_from_metadata(metadata)
        action_mode = _action_mode(str(metadata.get("action_mode", "abs_joint")))
        env: Any | None = None
        try:
            env = gym.make(str(metadata["env_id"]), **_env_kwargs(metadata))
            for spec in specs:
                rows.append(
                    _rollout_base_episode(
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
                )
        except Exception as exc:
            print(
                f"Failed to collect nominal policy paths: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 1
        finally:
            if env is not None:
                env.close()

    attempts = [_attempt_summary(row) for row in rows]
    eligible_rows = (
        rows
        if args.path_source == "dataset_demo"
        else [row for row in rows if bool(row["success"])]
    )
    resolved_geometry = _resolve_shared_grounded_geometry(args, eligible_rows)
    selected: list[dict[str, Any]] = []
    try:
        for row in rows:
            spec = row["spec"]
            print(
                f"attempt={spec.output_index} dataset_episode={spec.dataset_episode_index} "
                f"seed={spec.seed} path_source={args.path_source} success={row['success']} "
                f"final={_format_optional(row['final_distance'])} steps={row['steps']}"
            )
            if args.path_source == "policy_success" and not bool(row["success"]):
                continue
            selected_output_index = len(selected)
            tcp_path = _constraint_path(row)
            constraint, placement = _build_constraint(
                args,
                row=row,
                tcp_path=tcp_path,
                resolved_geometry=resolved_geometry,
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
                    "collision_geometry": constraint.region.to_json(),
                    "discrete_min_clearance": min_constraint_clearance(tcp_path, [constraint]),
                    "final_distance": row["final_distance"],
                    "min_distance": row["min_distance"],
                    "first_success_step": row["first_success_step"],
                    "path_source": args.path_source,
                    "resolved_path_fraction": placement["path_fraction"],
                    "anchor_local_offset_fraction": placement["anchor_local_offset_fraction"],
                    "initial_robot_clearance": placement["initial_robot_clearance"],
                }
            )
    except Exception as exc:
        print(
            f"Failed to build nominal-path constraints: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

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
        "path_source": args.path_source,
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
            "avoid_clearance_scale": args.avoid_clearance_scale,
            "avoid_tolerance": args.avoid_tolerance,
            "avoid_shape": args.avoid_shape,
            "avoid_box_half_extents": args.avoid_box_half_extents,
            "avoid_cylinder_half_length": args.avoid_cylinder_half_length,
            "obstacle_yaw_deg": args.obstacle_yaw_deg,
            "support_plane_z": args.support_plane_z,
            "path_height_margin": args.path_height_margin,
            "initial_robot_clearance_margin": args.initial_robot_clearance_margin,
            "path_fraction_search_min": args.path_fraction_search_min,
            "path_fraction_search_max": args.path_fraction_search_max,
            "path_fraction_search_step": args.path_fraction_search_step,
            "anchor_offset_max_fraction": args.anchor_offset_max_fraction,
            "anchor_offset_step_fraction": args.anchor_offset_step_fraction,
            "resolved_box_half_extents": resolved_geometry["box_half_extents"],
            "resolved_cylinder_half_length": resolved_geometry["cylinder_half_length"],
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
        description=(
            "Build fixed avoid-region constraints from successful nominal-policy "
            "paths or stored dataset demonstration paths."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-model", choices=["ema", "raw"], default="ema")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--path-source",
        choices=["policy_success", "dataset_demo"],
        default="policy_success",
        help=(
            "policy_success keeps only successful checkpoint rollouts; dataset_demo "
            "uses every selected episode's stored TCP path for full-distribution eval."
        ),
    )
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--episode-indices", type=int, nargs="+", default=None)
    parser.add_argument("--episode-indices-file", type=Path, default=None)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--replan-stride", type=int, default=None)
    parser.add_argument("--post-success-steps", type=int, default=8)
    parser.add_argument("--gripper-open", type=float, default=0.04)
    parser.add_argument("--avoid-radius", type=float, default=0.03)
    parser.add_argument("--path-fraction", type=float, default=0.5)
    parser.add_argument("--avoid-margin", type=float, default=0.0)
    parser.add_argument("--avoid-weight", type=float, default=1.0)
    parser.add_argument(
        "--avoid-clearance-scale",
        type=float,
        default=0.05,
        help=(
            "Soft-clearance decay scale in meters used to distinguish feasible "
            "reranking candidates; set to 0 for hinge-only avoidance."
        ),
    )
    parser.add_argument("--avoid-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--avoid-shape",
        choices=["sphere", "box", "cuboid", "cylinder"],
        default="sphere",
    )
    parser.add_argument("--avoid-box-half-extents", type=float, nargs=3, default=None)
    parser.add_argument("--avoid-cylinder-half-length", type=float, default=None)
    parser.add_argument("--obstacle-yaw-deg", type=float, default=0.0)
    parser.add_argument(
        "--support-plane-z",
        type=float,
        default=None,
        help=(
            "Ground the obstacle on this world-z support plane while retaining the "
            "nominal path point's x/y position. ManiSkill's tabletop is z=0."
        ),
    )
    parser.add_argument(
        "--path-height-margin",
        type=float,
        default=0.02,
        help=(
            "For a grounded box/cylinder, extend one shared obstacle top this far "
            "above the highest selected path anchor."
        ),
    )
    parser.add_argument(
        "--initial-robot-clearance-margin",
        type=float,
        default=None,
        help=(
            "When set, move the path-intersecting anchor deterministically until the "
            "stored initial robot cloud has at least this signed clearance."
        ),
    )
    parser.add_argument("--path-fraction-search-min", type=float, default=0.2)
    parser.add_argument("--path-fraction-search-max", type=float, default=0.8)
    parser.add_argument("--path-fraction-search-step", type=float, default=0.05)
    parser.add_argument("--anchor-offset-max-fraction", type=float, default=0.9)
    parser.add_argument("--anchor-offset-step-fraction", type=float, default=0.15)
    parser.add_argument("--min-successes", type=int, default=15)
    parser.add_argument("--allow-too-few-successes", action="store_true")
    args = parser.parse_args(argv)
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.episode_indices is not None and args.episode_indices_file is not None:
        raise ValueError("--episode-indices and --episode-indices-file are mutually exclusive")
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
    if not np.isfinite(args.avoid_clearance_scale) or args.avoid_clearance_scale < 0.0:
        raise ValueError("--avoid-clearance-scale must be finite and non-negative")
    if args.avoid_tolerance < 0.0:
        raise ValueError("--avoid-tolerance must be non-negative")
    if args.avoid_box_half_extents is not None and (
        not np.all(np.isfinite(args.avoid_box_half_extents))
        or np.any(np.asarray(args.avoid_box_half_extents) <= 0.0)
    ):
        raise ValueError("--avoid-box-half-extents must contain three positive values")
    if args.avoid_cylinder_half_length is not None and (
        not np.isfinite(args.avoid_cylinder_half_length) or args.avoid_cylinder_half_length <= 0.0
    ):
        raise ValueError("--avoid-cylinder-half-length must be positive")
    if not np.isfinite(args.obstacle_yaw_deg):
        raise ValueError("--obstacle-yaw-deg must be finite")
    if args.support_plane_z is not None and not np.isfinite(args.support_plane_z):
        raise ValueError("--support-plane-z must be finite")
    if not np.isfinite(args.path_height_margin) or args.path_height_margin < 0.0:
        raise ValueError("--path-height-margin must be finite and non-negative")
    if args.initial_robot_clearance_margin is not None and (
        not np.isfinite(args.initial_robot_clearance_margin)
        or args.initial_robot_clearance_margin < 0.0
    ):
        raise ValueError("--initial-robot-clearance-margin must be finite and non-negative")
    if not (
        0.0
        <= args.path_fraction_search_min
        <= args.path_fraction
        <= args.path_fraction_search_max
        <= 1.0
    ):
        raise ValueError("path fraction search bounds must contain --path-fraction within [0, 1]")
    if args.path_fraction_search_step <= 0.0:
        raise ValueError("--path-fraction-search-step must be positive")
    if not 0.0 <= args.anchor_offset_max_fraction < 1.0:
        raise ValueError("--anchor-offset-max-fraction must be in [0, 1)")
    if args.anchor_offset_step_fraction <= 0.0:
        raise ValueError("--anchor-offset-step-fraction must be positive")
    if args.min_successes < 0:
        raise ValueError("--min-successes must be non-negative")
    return args


def _read_episode_indices_file(path: Path) -> list[int]:
    indices: list[int] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            value = int(line)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number} is not an integer episode index") from exc
        if value < 0:
            raise ValueError(f"{path}:{line_number} episode index must be non-negative")
        indices.append(value)
    if not indices:
        raise ValueError(f"{path} does not contain any episode indices")
    return indices


def _dataset_demo_episode(root: Any, *, spec: RolloutSpec) -> dict[str, Any]:
    """Load one complete stored demonstration path without consulting policy outcomes."""
    if spec.dataset_episode_index is None:
        raise ValueError("dataset_demo path source requires dataset episode indices")
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    episode_index = int(spec.dataset_episode_index)
    if not 0 <= episode_index < episode_ends.size:
        raise IndexError(f"dataset episode index {episode_index} outside [0, {episode_ends.size})")
    start = 0 if episode_index == 0 else int(episode_ends[episode_index - 1])
    end = int(episode_ends[episode_index])
    tcp = np.asarray(root["data"]["tcp_pose"][start:end, :3], dtype=np.float32)
    target = np.asarray(root["data"]["target_position"][start:end], dtype=np.float32)
    success = np.asarray(root["data"]["success"][start:end], dtype=bool).reshape(-1)
    point_cloud = np.asarray(root["data"]["point_cloud"][start], dtype=np.float32)
    robot_mask = np.asarray(root["data"]["robot_mask"][start], dtype=bool).reshape(-1)
    point_valid_mask = np.asarray(
        root["data"]["point_valid_mask"][start],
        dtype=bool,
    ).reshape(-1)
    if tcp.ndim != 2 or tcp.shape[1] != 3 or tcp.shape[0] < 2:
        raise ValueError(f"dataset episode {episode_index} has invalid TCP path shape {tcp.shape}")
    if target.shape != tcp.shape:
        raise ValueError(
            f"dataset episode {episode_index} target shape {target.shape} != {tcp.shape}"
        )
    if success.shape != (tcp.shape[0],):
        raise ValueError(
            f"dataset episode {episode_index} success shape {success.shape} != {(tcp.shape[0],)}"
        )
    if point_cloud.ndim != 2 or point_cloud.shape[1] != 3:
        raise ValueError(
            f"dataset episode {episode_index} point cloud has invalid shape {point_cloud.shape}"
        )
    if robot_mask.shape != (point_cloud.shape[0],) or point_valid_mask.shape != (
        point_cloud.shape[0],
    ):
        raise ValueError(
            f"dataset episode {episode_index} robot/valid masks do not match {point_cloud.shape}"
        )
    first_success_indices = np.flatnonzero(success)
    first_success_step = int(first_success_indices[0]) if first_success_indices.size else None
    distances = np.linalg.norm(tcp - target, axis=1)
    return {
        "spec": spec,
        "output_index": spec.output_index,
        "seed": spec.seed,
        "source": spec.source,
        "dataset_episode_index": spec.dataset_episode_index,
        "steps": int(tcp.shape[0] - 1),
        "success": first_success_step is not None,
        "first_success_step": first_success_step,
        "final_distance": float(distances[-1]),
        "min_distance": float(np.min(distances)),
        "target_position": target[0],
        "tcp_positions": tcp,
        "initial_robot_points": point_cloud[robot_mask & point_valid_mask],
    }


def _resolve_shared_grounded_geometry(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
) -> dict[str, tuple[float, float, float] | float | None]:
    """Resolve one actor height shared by every selected precomputed constraint."""
    box_half_extents = (
        tuple(float(value) for value in args.avoid_box_half_extents)
        if args.avoid_box_half_extents is not None
        else None
    )
    cylinder_half_length = (
        float(args.avoid_cylinder_half_length)
        if args.avoid_cylinder_half_length is not None
        else None
    )
    if args.support_plane_z is None or args.avoid_shape == "sphere":
        return {
            "box_half_extents": box_half_extents,
            "cylinder_half_length": cylinder_half_length,
        }
    if not rows:
        raise ValueError("cannot resolve grounded geometry without selected paths")
    fractions = (
        _candidate_path_fractions(args)
        if args.initial_robot_clearance_margin is not None
        else [float(args.path_fraction)]
    )
    anchor_heights = [
        float(_point_at_arc_fraction(_constraint_path(row), fraction=fraction)[2])
        for row in rows
        for fraction in fractions
    ]
    required_half_height = 0.5 * (
        max(anchor_heights) + float(args.path_height_margin) - float(args.support_plane_z)
    )
    if required_half_height <= 0.0:
        raise ValueError("resolved grounded obstacle height must be positive")
    if args.avoid_shape in ("box", "cuboid"):
        dimensions = np.asarray(
            box_half_extents
            if box_half_extents is not None
            else (args.avoid_radius, args.avoid_radius, args.avoid_radius),
            dtype=np.float32,
        )
        dimensions[2] = max(float(dimensions[2]), required_half_height)
        box_half_extents = tuple(float(value) for value in dimensions)
    elif args.avoid_shape == "cylinder":
        cylinder_half_length = max(
            float(cylinder_half_length or args.avoid_radius),
            required_half_height,
        )
    return {
        "box_half_extents": box_half_extents,
        "cylinder_half_length": cylinder_half_length,
    }


def _build_constraint(
    args: argparse.Namespace,
    *,
    row: dict[str, Any],
    tcp_path: np.ndarray,
    resolved_geometry: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    config = NominalPathAvoidConfig(
        radius=args.avoid_radius,
        path_fraction=args.path_fraction,
        margin=args.avoid_margin,
        weight=args.avoid_weight,
        clearance_scale=args.avoid_clearance_scale,
        tolerance=args.avoid_tolerance,
        shape=args.avoid_shape,
        box_half_extents=resolved_geometry["box_half_extents"],
        yaw=np.deg2rad(float(args.obstacle_yaw_deg)),
        cylinder_half_length=resolved_geometry["cylinder_half_length"],
        support_plane_z=args.support_plane_z,
    )
    if args.initial_robot_clearance_margin is None:
        constraint = nominal_path_avoid_region(tcp_path, config=config)
        return constraint, {
            "path_fraction": float(args.path_fraction),
            "anchor_local_offset_fraction": [0.0, 0.0],
            "initial_robot_clearance": None,
        }
    if args.avoid_shape not in ("box", "cuboid") or args.support_plane_z is None:
        raise ValueError("initial robot clearance placement currently requires a grounded box")
    robot_points = np.asarray(row.get("initial_robot_points"), dtype=np.float32)
    if robot_points.ndim != 2 or robot_points.shape[1] != 3 or not len(robot_points):
        raise ValueError("initial robot clearance placement requires robot points")
    yaw = np.deg2rad(float(args.obstacle_yaw_deg))
    rotation = np.asarray(
        [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]],
        dtype=np.float32,
    )
    for fraction in _candidate_path_fractions(args):
        base = nominal_path_avoid_region(
            tcp_path,
            config=replace(config, path_fraction=fraction),
        )
        if not isinstance(base.region, BoxRegion):
            raise TypeError("clearance-aware placement expected a BoxRegion")
        half_extents = np.asarray(base.region.half_extents, dtype=np.float32)
        for local_offset in _candidate_anchor_offsets(args):
            center = np.asarray(base.region.center, dtype=np.float32).copy()
            scaled_offset = np.asarray(local_offset, dtype=np.float32) * half_extents[:2]
            center[:2] += rotation @ scaled_offset
            candidate = replace(
                base,
                region=BoxRegion(
                    center=center,
                    half_extents=half_extents,
                    yaw=base.region.yaw,
                ),
            )
            clearance = min_constraint_clearance(robot_points, [candidate])
            if clearance < float(args.initial_robot_clearance_margin):
                continue
            path_clearance = min_constraint_clearance(tcp_path, [candidate])
            if path_clearance >= -float(args.avoid_tolerance):
                continue
            return candidate, {
                "path_fraction": float(fraction),
                "anchor_local_offset_fraction": [
                    float(local_offset[0]),
                    float(local_offset[1]),
                ],
                "initial_robot_clearance": float(clearance),
            }
    raise ValueError(
        f"episode {row['dataset_episode_index']} has no path-intersecting placement "
        f"with {float(args.initial_robot_clearance_margin):.3f} m initial clearance"
    )


def _candidate_path_fractions(args: argparse.Namespace) -> list[float]:
    preferred = float(args.path_fraction)
    step = float(args.path_fraction_search_step)
    lower = float(args.path_fraction_search_min)
    upper = float(args.path_fraction_search_max)
    fractions = [preferred]
    index = 1
    while preferred + index * step <= upper + 1e-9 or preferred - index * step >= lower - 1e-9:
        high = preferred + index * step
        low = preferred - index * step
        if high <= upper + 1e-9:
            fractions.append(float(round(high, 10)))
        if low >= lower - 1e-9:
            fractions.append(float(round(low, 10)))
        index += 1
    return fractions


def _candidate_anchor_offsets(args: argparse.Namespace) -> list[tuple[float, float]]:
    maximum = float(args.anchor_offset_max_fraction)
    step = float(args.anchor_offset_step_fraction)
    count = int(np.floor(maximum / step + 1e-9))
    values = [index * step for index in range(-count, count + 1)]
    offsets = [(x, y) for x in values for y in values]
    return sorted(
        offsets,
        key=lambda offset: (
            round(offset[0] ** 2 + offset[1] ** 2, 10),
            abs(offset[0]) + abs(offset[1]),
            offset[0],
            offset[1],
        ),
    )


def _point_at_arc_fraction(points: Any, *, fraction: float) -> np.ndarray:
    path = np.asarray(points, dtype=np.float32)
    if path.ndim != 2 or path.shape[1] != 3 or path.shape[0] == 0:
        raise ValueError(f"path must have shape [T, 3], got {path.shape}")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("path fraction must be in [0, 1]")
    if path.shape[0] == 1:
        return path[0].copy()
    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    total = float(np.sum(segment_lengths))
    if total <= 1e-8:
        return path[0].copy()
    target = fraction * total
    cumulative = np.concatenate([np.zeros((1,), dtype=np.float32), np.cumsum(segment_lengths)])
    upper = min(int(np.searchsorted(cumulative, target, side="right")), path.shape[0] - 1)
    lower = max(0, upper - 1)
    span = float(cumulative[upper] - cumulative[lower])
    alpha = 0.0 if span <= 1e-8 else (target - float(cumulative[lower])) / span
    return (path[lower] + alpha * (path[upper] - path[lower])).astype(np.float32)


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
        "spec": spec,
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
        "initial_robot_points": _entry_robot_points(first_entry),
    }


def _entry_robot_points(entry: dict[str, Any]) -> np.ndarray:
    points = np.asarray(entry["point_cloud"], dtype=np.float32).reshape(-1, 3)
    mask = np.asarray(entry["robot_mask"], dtype=bool).reshape(-1)
    valid = entry.get("point_valid_mask")
    if valid is not None:
        mask = mask & np.asarray(valid, dtype=bool).reshape(-1)
    if mask.shape != (points.shape[0],):
        raise ValueError("initial robot mask does not match point cloud")
    return points[mask].astype(np.float32, copy=False)


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
