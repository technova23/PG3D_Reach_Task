"""Build the ten-episode E10 U-shape placement review fixture.

The builder uses only the locked nominal policy and geometry/planner validity. It
does not run rejection, reranking, or ITPS, so compared-method outcomes cannot
influence placement.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr

from pg3d.constraints import AvoidRegion
from pg3d.envs.maniskill_adapter import register_pg3d_reach_envs
from pg3d.envs.maniskill_adapter.dataset import (
    PointCloudCropConfig,
    git_commit_info,
    load_reach_metadata,
)
from pg3d.envs.maniskill_adapter.panda_collision import (
    load_panda_collision_point_template,
)
from pg3d.envs.obstacles import transform_box_component, u_shape_components
from pg3d.eval import save_episode_constraints
from pg3d.eval.u_shape_placement import (
    UShapeCandidate,
    UShapeSearchConfig,
    ValidatedUShapeCandidate,
    component_geometry,
    enumerate_u_shape_trajectory_candidates,
    evaluate_u_shape_geometry,
    u_shape_constraints,
    validated_candidate_sort_key,
)
from pg3d.policies.dp3.checkpoint import load_reach_policy_from_checkpoint
from pg3d.utils.arrays import bool_any, bool_info, to_numpy
from pg3d.utils.devices import select_device
from pg3d.utils.serialization import jsonable
from pg3d.world_model.panda_collision import DifferentiablePandaCollisionPoints
from scripts.eval_constrained_reach import (
    _policy_obstacle_point_count,
    _robot_obstacle_contact_pairs,
)
from scripts.eval_reach_checkpoint_unique_seeds import (
    _apply_zarr_initial_entry,
    _reset_to_zarr_episode,
    _zarr_episode_context,
)
from scripts.rollout_dp3_reach_policy import (
    append_obs_window,
    crop_config_from_metadata,
    make_initial_obs_window,
    obs_window_to_torch,
    policy_action_to_sim_action,
    rollout_observation_entry,
    save_rerun_timeline,
)
from scripts.write_maniskill_reach_dataset import _format_sim_action, _hold_sim_action

DEFAULT_SOURCE_FIXTURE = Path("configs/eval/e3_candidate_midpath_75cm_frozen_v1/fixture.json")
DEFAULT_CONFIG_DIR = Path("configs/eval/e10_u_shape_10ep_review_v1")
DEFAULT_ARTIFACT_DIR = Path("artifacts/e10-u-shape-10ep-review-v1")
EXPECTED_DATASET_EPISODES = (305, 317, 974, 986, 1010, 1034, 1069, 1117, 1129, 1138)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = json.loads(args.source_fixture.read_text(encoding="utf-8"))
    episodes = list(source["episodes"])
    _validate_source_episode_order(episodes)
    if args.episode_start:
        episodes = episodes[args.episode_start :]
    if args.episode_limit is not None:
        episodes = episodes[: args.episode_limit]
    if not episodes:
        raise ValueError("source fixture contains no selected episodes")
    _prepare_output(args.config_dir, overwrite=args.overwrite)
    _prepare_output(args.artifact_dir, overwrite=args.overwrite)

    metadata = load_reach_metadata(Path(source["dataset"]))
    crop_config = replace(crop_config_from_metadata(metadata), obstacle_point_quota=64)
    action_mode = str(metadata.get("action_mode", "abs_joint"))
    if action_mode not in {"abs_joint", "delta_joint"}:
        raise ValueError(f"unsupported action mode {action_mode!r}")
    device = select_device(args.device)
    policy = load_reach_policy_from_checkpoint(
        Path(source["checkpoint"]),
        device=device,
        prefer_ema=source.get("checkpoint_model", "ema") == "ema",
    )
    zarr_root = zarr.open_group(str(source["dataset"]), mode="r")

    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
    except Exception as exc:
        raise RuntimeError("ManiSkill and Gymnasium are required for U-shape placement") from exc
    register_pg3d_reach_envs()
    nominal_env = gym.make(str(metadata["env_id"]), **_env_kwargs(metadata))
    try:
        collision_model, world_from_base = _collision_model(nominal_env)
        records = []
        for episode in episodes:
            output_index = int(episode["output_index"])
            print(
                f"[u-shape] episode={output_index} dataset={episode['dataset_episode_index']} "
                "collecting locked nominal path",
                flush=True,
            )
            context = _zarr_episode_context(zarr_root, int(episode["dataset_episode_index"]))
            if args.nominal_source == "locked-policy":
                nominal = _rollout_nominal(
                    nominal_env,
                    policy=policy,
                    context=context,
                    simulator_seed=int(episode["simulator_seed"]),
                    policy_seed=int(episode["policy_seed"]),
                    crop_config=crop_config,
                    action_mode=action_mode,
                    device=device,
                    max_steps=args.max_steps,
                    hold_steps=args.hold_steps,
                    gripper_open=args.gripper_open,
                )
            else:
                nominal = _prequalified_dataset_nominal(
                    zarr_root,
                    context=context,
                    episode_index=int(episode["dataset_episode_index"]),
                )
            if not nominal["stable_success"]:
                raise RuntimeError(
                    f"episode {output_index} locked base policy did not achieve stable success"
                )
            _write_nominal_cache(args.artifact_dir, output_index, nominal)
            record = _select_episode_placement(
                gym=gym,
                metadata=metadata,
                crop_config=crop_config,
                context=context,
                episode=episode,
                nominal=nominal,
                collision_model=collision_model,
                world_from_base=world_from_base,
                search_config=UShapeSearchConfig(),
                max_planner_candidates=args.max_planner_candidates,
                max_steps=args.max_steps,
                hold_steps=args.hold_steps,
                gripper_open=args.gripper_open,
                rejection_path=args.artifact_dir
                / "rejections"
                / f"episode_{output_index:03d}.json",
            )
            _write_episode_outputs(
                args=args,
                episode=episode,
                context=context,
                nominal=nominal,
                record=record,
                crop_config=crop_config,
                policy=policy,
                collision_model=collision_model,
            )
            records.append(record["fixture_record"])
    finally:
        nominal_env.close()

    if args.episode_limit is None:
        if len(records) != len(EXPECTED_DATASET_EPISODES):
            raise RuntimeError(
                f"refusing to serialize a partial review fixture with {len(records)} episodes"
            )
        _write_suite_outputs(
            args=args,
            source=source,
            episodes=records,
            metadata=metadata,
        )
    else:
        print(
            "diagnostic --episode-limit run: review fixture metadata was not serialized",
            flush=True,
        )
    print(
        f"built {len(records)} U-shape review episodes in {args.config_dir} and "
        f"{args.artifact_dir}",
        flush=True,
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-fixture", type=Path, default=DEFAULT_SOURCE_FIXTURE)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--nominal-source",
        choices=["locked-policy", "prequalified-dataset"],
        default="locked-policy",
        help=(
            "Use a fresh locked-policy rollout, or the successful dataset demonstration after "
            "the fixed episodes have independently been prequalified for locked-policy success."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--hold-steps", type=int, default=16)
    parser.add_argument("--max-planner-candidates", type=int, default=24)
    parser.add_argument("--gripper-open", type=float, default=0.04)
    parser.add_argument("--episode-limit", type=int, default=None)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.max_steps <= 0 or args.hold_steps <= 0 or args.max_planner_candidates <= 0:
        raise ValueError("step, hold, and planner-candidate limits must be positive")
    if args.episode_limit is not None and args.episode_limit <= 0:
        raise ValueError("--episode-limit must be positive")
    if not 0 <= args.episode_start < 10:
        raise ValueError("--episode-start must be within [0, 9]")
    if not 0.0 <= args.gripper_open <= 0.04:
        raise ValueError("--gripper-open must be within [0, 0.04]")
    return args


def _prepare_output(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {path}; pass --overwrite")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _validate_source_episode_order(episodes: list[dict[str, Any]]) -> None:
    indices = tuple(int(episode["dataset_episode_index"]) for episode in episodes)
    if indices != EXPECTED_DATASET_EPISODES:
        raise ValueError(
            "source fixture episode order changed: "
            f"expected {EXPECTED_DATASET_EPISODES}, got {indices}"
        )
    if tuple(int(episode["output_index"]) for episode in episodes) != tuple(range(10)):
        raise ValueError("source fixture output indices must be exactly 0 through 9")
    for seed_name in ("simulator_seed", "policy_seed"):
        seeds = [int(episode[seed_name]) for episode in episodes]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"source fixture contains duplicate {seed_name} values")


def _write_nominal_cache(artifact_dir: Path, output_index: int, nominal: dict[str, Any]) -> None:
    path = artifact_dir / "nominal" / f"episode_{output_index:03d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        tcp_path=np.asarray(nominal["tcp_path"], dtype=np.float32),
        start=np.asarray(nominal["start"], dtype=np.float32),
        goal=np.asarray(nominal["goal"], dtype=np.float32),
        first_chunk_distance=np.asarray(nominal["first_chunk_distance"], dtype=np.float32),
        stable_success=np.asarray(nominal["stable_success"], dtype=np.bool_),
        first_success_step=np.asarray(
            -1 if nominal["first_success_step"] is None else nominal["first_success_step"],
            dtype=np.int32,
        ),
        steps=np.asarray(nominal["steps"], dtype=np.int32),
    )


def _env_kwargs(
    metadata: dict[str, Any],
    *,
    obstacle_half_extents: np.ndarray | None = None,
) -> dict[str, Any]:
    kwargs = dict(metadata["env_kwargs"])
    kwargs["obs_mode"] = "pointcloud"
    kwargs["num_envs"] = 1
    kwargs.pop("render_mode", None)
    if obstacle_half_extents is not None:
        kwargs["pg3d_obstacle_half_extents"] = tuple(float(x) for x in obstacle_half_extents)
        kwargs["pg3d_obstacle_family"] = "u_shape"
    return kwargs


def _rollout_nominal(
    env: Any,
    *,
    policy: Any,
    context: dict[str, Any],
    simulator_seed: int,
    policy_seed: int,
    crop_config: PointCloudCropConfig,
    action_mode: str,
    device: torch.device,
    max_steps: int,
    hold_steps: int,
    gripper_open: float,
) -> dict[str, Any]:
    obs, info = _reset_to_zarr_episode(
        env,
        rollout_seed=simulator_seed,
        zarr_context=context,
    )
    entry = rollout_observation_entry(obs, info, env=env, crop_config=crop_config)
    entry = _apply_zarr_initial_entry(entry, context)
    window = make_initial_obs_window(entry, n_obs_steps=int(policy.n_obs_steps))
    path = [np.asarray(entry["tcp_pose"], dtype=np.float32).reshape(-1)[:3].copy()]
    consecutive_success = 0
    first_success_step: int | None = None
    steps = 0
    was_training = bool(policy.training)
    policy.eval()
    torch.manual_seed(policy_seed)
    generator = torch.Generator(device=device).manual_seed(policy_seed)
    try:
        while steps < max_steps and consecutive_success < hold_steps:
            with torch.inference_mode():
                batch = obs_window_to_torch(
                    window,
                    device=device,
                    goal_marker_points=int(getattr(policy, "goal_marker_points", 0)),
                    goal_marker_radius=float(getattr(policy, "goal_marker_radius", 0.045)),
                )
                output = policy.predict_action(batch, generator=generator)
                chunk = output["action"][0].detach().cpu().numpy()
            for policy_action in chunk[: int(policy.n_action_steps)]:
                sim_action = policy_action_to_sim_action(
                    policy_action,
                    np.asarray(entry["agent_pos"], dtype=np.float32),
                    action_mode=action_mode,  # type: ignore[arg-type]
                    sim_action_dim=int(np.prod(env.action_space.shape)),
                    low=getattr(env.action_space, "low", None),
                    high=getattr(env.action_space, "high", None),
                    gripper_open=gripper_open,
                )
                obs, _reward, terminated, truncated, info = env.step(sim_action)
                steps += 1
                entry = rollout_observation_entry(obs, info, env=env, crop_config=crop_config)
                window = append_obs_window(window, entry, n_obs_steps=int(policy.n_obs_steps))
                path.append(np.asarray(entry["tcp_pose"], dtype=np.float32).reshape(-1)[:3].copy())
                success = bool_info(info, "success")
                if success:
                    if first_success_step is None:
                        first_success_step = steps
                    consecutive_success += 1
                else:
                    consecutive_success = 0
                if bool_any(truncated) or (bool_any(terminated) and not success):
                    break
                if steps >= max_steps or consecutive_success >= hold_steps:
                    break
            if bool_any(truncated) or (bool_any(terminated) and not bool_info(info, "success")):
                break
    finally:
        if was_training:
            policy.train()
    tcp_path = np.asarray(path, dtype=np.float32)
    first_chunk_segments = np.linalg.norm(np.diff(tcp_path[:9], axis=0), axis=1)
    return {
        "stable_success": consecutive_success >= hold_steps,
        "first_success_step": first_success_step,
        "steps": steps,
        "tcp_path": tcp_path,
        "first_chunk_distance": float(np.sum(first_chunk_segments)),
        "start": tcp_path[0],
        "goal": np.asarray(context["target_position"], dtype=np.float32).reshape(3),
        "goal_qpos": np.asarray(entry["agent_pos"], dtype=np.float32).copy(),
        "goal_pose": np.asarray(entry["tcp_pose"], dtype=np.float32).copy(),
        "source": "locked_policy_obstacle_free",
    }


def _prequalified_dataset_nominal(
    zarr_root: Any,
    *,
    context: dict[str, Any],
    episode_index: int,
) -> dict[str, Any]:
    """Load a successful demonstration after base success was fixed externally.

    This mode avoids rerunning the expensive base policy solely for placement
    calibration.  It is explicit in fixture provenance and must only be used for
    episodes already accepted by an outcome-blind nominal-success gate.
    """
    ends = np.asarray(zarr_root["meta"]["episode_ends"][:], dtype=np.int64)
    start = 0 if episode_index == 0 else int(ends[episode_index - 1])
    end = int(ends[episode_index])
    path = np.asarray(zarr_root["data"]["eef_pos"][start:end], dtype=np.float32)
    success = np.asarray(zarr_root["data"]["success"][start:end], dtype=bool)
    if len(path) < 25 or not bool(np.any(success)):
        raise RuntimeError(f"dataset episode {episode_index} is not a complete successful nominal")
    first_success = int(np.flatnonzero(success)[0]) + 1
    return {
        "stable_success": True,
        "first_success_step": first_success,
        "steps": len(path) - 1,
        "tcp_path": path,
        "first_chunk_distance": float(np.sum(np.linalg.norm(np.diff(path[:9], axis=0), axis=1))),
        "start": path[0].copy(),
        "goal": np.asarray(context["target_position"], dtype=np.float32).reshape(3),
        "goal_qpos": np.asarray(zarr_root["data"]["state"][end - 1], dtype=np.float32),
        "goal_pose": np.asarray(zarr_root["data"]["tcp_pose"][end - 1], dtype=np.float32),
        "source": "prequalified_successful_dataset_demonstration",
    }


def _collision_model(env: Any) -> tuple[DifferentiablePandaCollisionPoints, torch.Tensor]:
    unwrapped = env.unwrapped
    template = load_panda_collision_point_template(
        unwrapped.agent.urdf_path,
        point_count=1024,
        sample_seed=0,
    )
    model = DifferentiablePandaCollisionPoints(template, gripper_open=0.04).cpu()
    base = to_numpy(unwrapped.agent.robot.pose.to_transformation_matrix()).astype(np.float32)
    if base.ndim == 3:
        base = base[0]
    return model, torch.as_tensor(base, dtype=torch.float32)


def _robot_clouds(
    model: DifferentiablePandaCollisionPoints,
    world_from_base: torch.Tensor,
    qpos: np.ndarray,
) -> np.ndarray:
    q = np.asarray(qpos, dtype=np.float32)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    with torch.inference_mode():
        points = model(torch.as_tensor(q[:, :7]), world_from_base)
    return points.detach().cpu().numpy().astype(np.float32)


def _select_episode_placement(
    *,
    gym: Any,
    metadata: dict[str, Any],
    crop_config: PointCloudCropConfig,
    context: dict[str, Any],
    episode: dict[str, Any],
    nominal: dict[str, Any],
    collision_model: DifferentiablePandaCollisionPoints,
    world_from_base: torch.Tensor,
    search_config: UShapeSearchConfig,
    max_planner_candidates: int,
    max_steps: int,
    hold_steps: int,
    gripper_open: float,
    rejection_path: Path,
) -> dict[str, Any]:
    initial_qpos = np.asarray(context["state"], dtype=np.float32)
    start_exact = _robot_clouds(collision_model, world_from_base, initial_qpos)[0]
    start_observed = np.asarray(context["point_cloud"], dtype=np.float32)[
        np.asarray(context["point_valid_mask"], dtype=bool)
        & np.asarray(context["robot_mask"], dtype=bool)
    ]
    # Exact movable-link surfaces drive path validation.  The recorded camera
    # robot mask additionally covers fixed/base geometry at the initial gate.
    # The goal gate is evaluated on the live planner witness, because a stored
    # demonstration's terminal IK branch is not intrinsic to the goal pose.
    start_cloud = np.concatenate((start_exact, start_observed), axis=0)
    all_rejections: list[dict[str, Any]] = []
    for fallback in (False, True):
        candidates = enumerate_u_shape_trajectory_candidates(
            nominal["start"],
            nominal["goal"],
            nominal["tcp_path"],
            config=search_config,
            fallback=fallback,
        )
        geometric: list[tuple[UShapeCandidate, Any]] = []
        for candidate in candidates:
            check = evaluate_u_shape_geometry(
                candidate,
                start_robot_points=start_cloud,
                # A demonstration's terminal IK branch is not a hard property of
                # the goal.  The actual whole-arm goal gate is applied to the
                # planner witness and its live stable-hold replay below.
                goal_robot_points=np.asarray(nominal["goal"], dtype=np.float32).reshape(1, 3),
                nominal_tcp_path=nominal["tcp_path"],
                goal=nominal["goal"],
                config=search_config,
            )
            if check.accepted:
                geometric.append((candidate, check))
            else:
                all_rejections.append(
                    {
                        "candidate": _candidate_json(candidate),
                        "reasons": list(check.reasons),
                        "geometry": asdict(check),
                    }
                )
        geometric.sort(
            key=lambda item: (
                abs(item[0].back_chunk_ratio - search_config.target_back_chunks),
                -item[1].initial_clearance,
                item[0].footprint_area,
                abs(item[0].lateral_offset),
                item[0].mouth_width,
            )
        )
        validated: list[dict[str, Any]] = []
        planner_shortlist = _round_robin_back_shortlist(geometric, max_planner_candidates)
        for candidate, check in planner_shortlist:
            result, live_diagnostics = _validate_candidate_live(
                gym=gym,
                metadata=metadata,
                crop_config=crop_config,
                context=context,
                episode=episode,
                nominal=nominal,
                candidate=candidate,
                geometry=check,
                collision_model=collision_model,
                world_from_base=world_from_base,
                min_clearance=search_config.min_clearance,
                max_steps=max_steps,
                hold_steps=hold_steps,
                gripper_open=gripper_open,
            )
            if result is not None:
                validated.append(result)
            else:
                all_rejections.append(
                    {
                        "candidate": _candidate_json(candidate),
                        "reasons": ["no_valid_live_witness"],
                        "geometry": asdict(check),
                        "live_diagnostics": live_diagnostics,
                    }
                )
        if validated:
            validated.sort(key=lambda item: validated_candidate_sort_key(item["validated"]))
            return validated[0]
    rejection_path.parent.mkdir(parents=True, exist_ok=True)
    rejection_path.write_text(json.dumps(jsonable(all_rejections), indent=2), encoding="utf-8")
    raise RuntimeError(
        f"episode {episode['output_index']} has no valid U-shape candidate; "
        f"rejections written to {rejection_path}"
    )


def _round_robin_back_shortlist(
    candidates: list[tuple[UShapeCandidate, Any]],
    limit: int,
) -> list[tuple[UShapeCandidate, Any]]:
    """Preserve target preference while ensuring later safe boundaries are tried."""
    groups: dict[float, list[tuple[UShapeCandidate, Any]]] = {}
    for item in candidates:
        groups.setdefault(float(item[0].back_chunk_ratio), []).append(item)
    ordered_ratios = sorted(groups, key=lambda ratio: (abs(ratio - 2.5), ratio))
    selected: list[tuple[UShapeCandidate, Any]] = []
    depth = 0
    while len(selected) < limit:
        added = False
        for ratio in ordered_ratios:
            group = groups[ratio]
            if depth < len(group):
                selected.append(group[depth])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        depth += 1
    return selected


def _validate_candidate_live(
    *,
    gym: Any,
    metadata: dict[str, Any],
    crop_config: PointCloudCropConfig,
    context: dict[str, Any],
    episode: dict[str, Any],
    nominal: dict[str, Any],
    candidate: UShapeCandidate,
    geometry: Any,
    collision_model: DifferentiablePandaCollisionPoints,
    world_from_base: torch.Tensor,
    min_clearance: float,
    max_steps: int,
    hold_steps: int,
    gripper_open: float,
    minimum_obstacle_points: int = 64,
    planner_margins: tuple[float, ...] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    env = gym.make(
        str(metadata["env_id"]),
        **_env_kwargs(metadata, obstacle_half_extents=candidate.half_extents),
    )
    constraints = u_shape_constraints(candidate, target="robot")
    reset_options = {
        "pg3d_obstacle_center": candidate.root_center.astype(float).tolist(),
        "pg3d_obstacle_yaw": float(candidate.yaw),
    }
    diagnostics: dict[str, Any] = {"sides": {}}
    try:
        obs, info = _reset_to_zarr_episode(
            env,
            rollout_seed=int(episode["simulator_seed"]),
            zarr_context=context,
            reset_options=reset_options,
        )
        entry = rollout_observation_entry(obs, info, env=env, crop_config=crop_config)
        obstacle_points = _policy_obstacle_point_count(entry, goal_marker_points=0)
        diagnostics["obstacle_points_policy_input_reset"] = int(obstacle_points)
        if obstacle_points < minimum_obstacle_points:
            diagnostics["failure"] = "visibility_below_minimum"
            diagnostics["minimum_obstacle_points"] = int(minimum_obstacle_points)
            return None, diagnostics
        witnesses = []
        margins = (
            planner_margins
            if planner_margins is not None
            else (min_clearance + 0.01, min_clearance, 0.02, 0.0)
        )
        for side in ("left", "right"):
            side_diagnostics: dict[str, Any] = {"planner_attempts": []}
            diagnostics["sides"][side] = side_diagnostics
            qpath = None
            clouds = None
            clearance = None
            for planner_margin in margins:
                planned, planner_diagnostics = _plan_witness(
                    env,
                    candidate=candidate,
                    context=context,
                    side=side,
                    clearance_margin=planner_margin,
                    goal_quaternion=np.asarray(nominal["goal_pose"], dtype=np.float32)[3:7],
                )
                attempt = {
                    "planner_margin_m": planner_margin,
                    **planner_diagnostics,
                }
                side_diagnostics["planner_attempts"].append(attempt)
                if planned is None:
                    attempt["failure"] = "planner_failed"
                    continue
                attempt["planned_steps_raw"] = int(len(planned))
                planned_clouds = _robot_clouds(collision_model, world_from_base, planned)
                planned_clearance = _clearance_series(planned_clouds, constraints)
                attempt["planned_min_clearance_m"] = float(np.min(planned_clearance))
                if float(np.min(planned_clearance)) < min_clearance:
                    attempt["failure"] = "planned_clearance_below_gate"
                    continue
                qpath = planned
                clouds = planned_clouds
                clearance = planned_clearance
                side_diagnostics.update(attempt)
                break
            if qpath is None or clouds is None or clearance is None:
                side_diagnostics["failure"] = "all_planner_margins_failed"
                continue
            execution_budget = max_steps - hold_steps
            if len(qpath) > execution_budget:
                indices = np.rint(np.linspace(0, len(qpath) - 1, execution_budget)).astype(int)
                qpath = qpath[indices]
                side_diagnostics["time_resampled"] = True
            side_diagnostics["planned_steps"] = int(len(qpath))
            tcp_path = _tcp_path_from_qpos(env, qpath)
            witnesses.append(
                {
                    "side": side,
                    "qpos": qpath,
                    "tcp": tcp_path,
                    "clouds": clouds,
                    "clearance": clearance,
                    "path_length": _path_length(tcp_path),
                }
            )
        if not witnesses:
            diagnostics["failure"] = "no_geometrically_valid_planner_path"
            return None, diagnostics
        witnesses.sort(
            key=lambda item: (
                -float(np.min(item["clearance"])),
                item["path_length"],
                item["side"],
            )
        )
        for witness in witnesses:
            replay = _replay_witness(
                env,
                candidate=candidate,
                constraints=constraints,
                context=context,
                simulator_seed=int(episode["simulator_seed"]),
                qpath=witness["qpos"],
                crop_config=crop_config,
                collision_model=collision_model,
                world_from_base=world_from_base,
                min_clearance=min_clearance,
                max_steps=max_steps,
                hold_steps=hold_steps,
                gripper_open=gripper_open,
            )
            if replay is None:
                diagnostics["sides"][witness["side"]]["failure"] = "live_replay_failed"
                continue
            stable_hold_steps = int(replay["stable_hold_steps"])
            goal_clearance = float(np.min(replay["clearance"][-stable_hold_steps:]))
            validated_geometry = replace(geometry, goal_clearance=goal_clearance)
            validated = ValidatedUShapeCandidate(
                candidate=candidate,
                geometry=validated_geometry,
                witness_clearance=float(np.min(replay["clearance"])),
                witness_path_length=_path_length(replay["tcp"]),
                witness_steps=int(replay["steps"]),
                witness_side=str(witness["side"]),
                obstacle_points=int(obstacle_points),
            )
            fixture_record = _fixture_episode_record(
                episode=episode,
                nominal=nominal,
                validated=validated,
                replay=replay,
            )
            return (
                {
                    "validated": validated,
                    "replay": replay,
                    "fixture_record": fixture_record,
                    "constraints": constraints,
                },
                diagnostics,
            )
        diagnostics["failure"] = "all_live_replays_failed"
        return None, diagnostics
    finally:
        env.close()


def _plan_witness(
    env: Any,
    *,
    candidate: UShapeCandidate,
    context: dict[str, Any],
    side: str,
    clearance_margin: float,
    goal_quaternion: np.ndarray,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    import sapien
    from mani_skill.examples.motionplanning.panda.motionplanner import (
        PandaArmMotionPlanningSolver,
    )

    if side not in {"left", "right"}:
        raise ValueError("witness side must be left or right")
    unwrapped = env.unwrapped
    robot = unwrapped.agent.robot
    start_qpos = to_numpy(robot.get_qpos()).astype(np.float32).reshape(-1)
    start_pose = to_numpy(unwrapped.agent.tcp_pose.raw_pose).astype(np.float32).reshape(-1, 7)[0]
    goal_xyz = np.asarray(context["target_position"], dtype=np.float32).reshape(3)
    goal_quat = np.asarray(goal_quaternion, dtype=np.float32).reshape(4)
    sign = -1.0 if side == "left" else 1.0
    local_points = np.asarray(
        [
            [
                sign * (candidate.half_extents[0] + clearance_margin + 0.03),
                0.0,
            ],
        ],
        dtype=np.float32,
    )
    cos_yaw, sin_yaw = math.cos(candidate.yaw), math.sin(candidate.yaw)
    rotation = np.asarray([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float32)
    world_xy = local_points @ rotation.T + candidate.root_center[:2]
    z_mid = 0.5 * (float(start_pose[2]) + float(goal_xyz[2]))
    poses = [
        sapien.Pose(p=[*world_xy[0], z_mid], q=start_pose[3:7]),
        sapien.Pose(p=goal_xyz, q=goal_quat),
    ]
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=False,
        base_pose=robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
        joint_vel_limits=3.0,
        joint_acc_limits=3.0,
    )
    diagnostics: dict[str, Any] = {"segments": []}
    planner_started = time.perf_counter()
    try:
        for component in u_shape_components(candidate.half_extents):
            center, yaw = transform_box_component(
                component,
                center=candidate.root_center,
                yaw=candidate.yaw,
            )
            planner.add_box_collision(
                extents=2.0
                * (np.asarray(component.half_extents, dtype=np.float32) + float(clearance_margin)),
                pose=sapien.Pose(
                    p=center,
                    q=[math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)],
                ),
            )
        segments = []
        for segment_index, pose in enumerate(poses):
            segment_started = time.perf_counter()
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                plan = planner.move_to_pose_with_screw(pose, dry_run=True)
                planner_kind = "screw"
                if plan == -1 or "position" not in plan:
                    plan = planner.move_to_pose_with_RRTConnect(pose, dry_run=True)
                    planner_kind = "rrt_connect"
            if plan == -1 or "position" not in plan:
                if segment_index == 0:
                    with (
                        contextlib.redirect_stdout(io.StringIO()),
                        contextlib.redirect_stderr(io.StringIO()),
                    ):
                        direct_plan = planner.move_to_pose_with_RRTConnect(
                            sapien.Pose(p=goal_xyz, q=goal_quat),
                            dry_run=True,
                        )
                    if direct_plan != -1 and "position" in direct_plan:
                        direct_positions = np.asarray(direct_plan["position"], dtype=np.float32)
                        if (
                            direct_positions.ndim == 2
                            and len(direct_positions)
                            and np.isfinite(direct_positions).all()
                        ):
                            diagnostics["segments"].append(
                                {
                                    "segment_index": segment_index,
                                    "planner": "direct_goal_rrt_connect",
                                    "steps": int(len(direct_positions)),
                                    "elapsed_seconds": time.perf_counter() - segment_started,
                                }
                            )
                            return direct_positions, diagnostics
                diagnostics["segments"].append(
                    {
                        "segment_index": segment_index,
                        "failure": "screw_and_rrt_connect_failed",
                        "elapsed_seconds": time.perf_counter() - segment_started,
                    }
                )
                return None, diagnostics
            positions = np.asarray(plan["position"], dtype=np.float32)
            if positions.ndim != 2 or not len(positions) or not np.isfinite(positions).all():
                diagnostics["segments"].append(
                    {
                        "segment_index": segment_index,
                        "failure": "invalid_planner_positions",
                        "elapsed_seconds": time.perf_counter() - segment_started,
                    }
                )
                return None, diagnostics
            diagnostics["segments"].append(
                {
                    "segment_index": segment_index,
                    "planner": planner_kind,
                    "steps": int(len(positions)),
                    "elapsed_seconds": time.perf_counter() - segment_started,
                }
            )
            segments.append(positions)
            _set_robot_qpos(robot, positions[-1])
        combined = np.concatenate([segments[0], *[segment[1:] for segment in segments[1:]]], axis=0)
        return combined, diagnostics
    finally:
        diagnostics["elapsed_seconds"] = time.perf_counter() - planner_started
        _set_robot_qpos(robot, start_qpos)
        planner.close()


def _set_robot_qpos(robot: Any, qpos: np.ndarray) -> None:
    current = to_numpy(robot.get_qpos()).astype(np.float32)
    flat = current.reshape(-1)
    q = np.asarray(qpos, dtype=np.float32).reshape(-1)
    flat[: min(len(flat), len(q))] = q[: min(len(flat), len(q))]
    robot.set_qpos(flat.reshape(current.shape))
    robot.set_qvel(np.zeros_like(current))


def _tcp_path_from_qpos(env: Any, qpath: np.ndarray) -> np.ndarray:
    robot = env.unwrapped.agent.robot
    original = to_numpy(robot.get_qpos()).astype(np.float32)
    points = []
    try:
        for qpos in qpath:
            _set_robot_qpos(robot, qpos)
            tcp = to_numpy(env.unwrapped.agent.tcp_pose.p).astype(np.float32).reshape(-1, 3)[0]
            points.append(tcp.copy())
    finally:
        robot.set_qpos(original)
        robot.set_qvel(np.zeros_like(original))
    return np.asarray(points, dtype=np.float32)


def _replay_witness(
    env: Any,
    *,
    candidate: UShapeCandidate,
    constraints: list[AvoidRegion],
    context: dict[str, Any],
    simulator_seed: int,
    qpath: np.ndarray,
    crop_config: PointCloudCropConfig,
    collision_model: DifferentiablePandaCollisionPoints,
    world_from_base: torch.Tensor,
    min_clearance: float,
    max_steps: int,
    hold_steps: int,
    gripper_open: float,
) -> dict[str, Any] | None:
    reset_options = {
        "pg3d_obstacle_center": candidate.root_center.astype(float).tolist(),
        "pg3d_obstacle_yaw": float(candidate.yaw),
    }
    obs, info = _reset_to_zarr_episode(
        env,
        rollout_seed=simulator_seed,
        zarr_context=context,
        reset_options=reset_options,
    )
    timeline = [rollout_observation_entry(obs, info, env=env, crop_config=crop_config)]
    q_actual = [np.asarray(timeline[0]["agent_pos"], dtype=np.float32)]
    tcp = [np.asarray(timeline[0]["tcp_pose"], dtype=np.float32).reshape(-1)[:3]]
    stable = 0
    contacts: list[list[str]] = []
    for qpos in qpath:
        if len(timeline) - 1 >= max_steps:
            return None
        action = _format_sim_action(env, qpos)
        obs, _reward, terminated, truncated, info = env.step(action)
        entry = rollout_observation_entry(obs, info, env=env, crop_config=crop_config)
        timeline.append(entry)
        q_actual.append(np.asarray(entry["agent_pos"], dtype=np.float32))
        tcp.append(np.asarray(entry["tcp_pose"], dtype=np.float32).reshape(-1)[:3])
        contacts.extend(_robot_obstacle_contact_pairs(env))
        if (
            contacts
            or bool_any(truncated)
            or (bool_any(terminated) and not bool_info(info, "success"))
        ):
            return None
    while stable < hold_steps and len(timeline) - 1 < max_steps:
        obs, _reward, terminated, truncated, info = env.step(
            _hold_sim_action(env, gripper_open=gripper_open)
        )
        entry = rollout_observation_entry(obs, info, env=env, crop_config=crop_config)
        timeline.append(entry)
        q_actual.append(np.asarray(entry["agent_pos"], dtype=np.float32))
        tcp.append(np.asarray(entry["tcp_pose"], dtype=np.float32).reshape(-1)[:3])
        contacts.extend(_robot_obstacle_contact_pairs(env))
        stable = stable + 1 if bool_info(info, "success") else 0
        if (
            contacts
            or bool_any(truncated)
            or (bool_any(terminated) and not bool_info(info, "success"))
        ):
            return None
    if stable < hold_steps:
        return None
    clouds = _robot_clouds(collision_model, world_from_base, np.asarray(q_actual))
    clearance = _clearance_series(clouds, constraints)
    if float(np.min(clearance)) < min_clearance:
        return None
    return {
        "timeline": timeline,
        "qpos": np.asarray(q_actual, dtype=np.float32),
        "tcp": np.asarray(tcp, dtype=np.float32),
        "clouds": clouds,
        "clearance": clearance,
        "steps": len(timeline) - 1,
        "stable_hold_steps": stable,
        "physical_contacts": contacts,
    }


def _clearance_series(clouds: np.ndarray, constraints: list[AvoidRegion]) -> np.ndarray:
    values = []
    for cloud in clouds:
        values.append(
            min(
                float(np.min(constraint.region.signed_distance(cloud)))
                for constraint in constraints
            )
        )
    return np.asarray(values, dtype=np.float32)


def _path_length(points: np.ndarray) -> float:
    path = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))) if len(path) > 1 else 0.0


def _fixture_episode_record(
    *,
    episode: dict[str, Any],
    nominal: dict[str, Any],
    validated: ValidatedUShapeCandidate,
    replay: dict[str, Any],
) -> dict[str, Any]:
    candidate = validated.candidate
    geometry = validated.geometry
    output_index = int(episode["output_index"])
    return {
        "output_index": output_index,
        "dataset_episode_index": int(episode["dataset_episode_index"]),
        "simulator_seed": int(episode["simulator_seed"]),
        "policy_seed": int(episode["policy_seed"]),
        "source_pool_index": int(episode["source_pool_index"]),
        "placement_policy_seed": int(episode["placement_policy_seed"]),
        "root_center": candidate.root_center.astype(float).tolist(),
        "yaw_rad": float(candidate.yaw),
        "yaw_deg": float(np.rad2deg(candidate.yaw)),
        "envelope_half_extents_m": candidate.half_extents.astype(float).tolist(),
        "full_size_m": candidate.full_size.astype(float).tolist(),
        "mouth_width_m": candidate.mouth_width,
        "mouth_distance_m": candidate.mouth_distance,
        "mouth_first_chunk_ratio": candidate.mouth_chunk_fraction,
        "back_distance_m": candidate.back_distance,
        "back_chunk_ratio": candidate.back_chunk_ratio,
        "back_nominal_step": int(round(8 * candidate.back_chunk_ratio)),
        "usable_cavity_depth_m": candidate.back_distance - candidate.mouth_distance,
        "lateral_offset_m": candidate.lateral_offset,
        "first_chunk_distance_m": candidate.chunk_distance,
        "later_chunk_fallback_search": bool(candidate.fallback),
        "fallback_width": candidate.mouth_width >= 0.26,
        "initial_robot_clearance_m": geometry.initial_clearance,
        "goal_robot_clearance_m": geometry.goal_clearance,
        "witness_robot_clearance_m": validated.witness_clearance,
        "mouth_clearance_m": geometry.mouth_clearance,
        "goal_beyond_back_m": geometry.goal_beyond_back,
        "direct_back_clearance_m": geometry.direct_back_clearance,
        "witness_side": validated.witness_side,
        "witness_steps": validated.witness_steps,
        "witness_tcp_path_length_m": validated.witness_path_length,
        "stable_hold_steps": int(replay["stable_hold_steps"]),
        "obstacle_points_policy_input_reset": validated.obstacle_points,
        "nominal_stable_success": bool(nominal["stable_success"]),
        "nominal_first_success_step": nominal["first_success_step"],
        "nominal_source": nominal["source"],
        "components": component_geometry(candidate),
        "eef_constraint_file": f"constraints/eef/episode_{output_index:03d}.json",
        "robot_constraint_file": f"constraints/robot/episode_{output_index:03d}.json",
        "witness_file": f"witnesses/episode_{output_index:03d}.npz",
        "rerun_file": f"rerun/episode_{output_index:03d}.rrd",
        "topdown_file": f"topdown/episode_{output_index:03d}.png",
    }


def _write_episode_outputs(
    *,
    args: argparse.Namespace,
    episode: dict[str, Any],
    context: dict[str, Any],
    nominal: dict[str, Any],
    record: dict[str, Any],
    crop_config: PointCloudCropConfig,
    policy: Any,
    collision_model: DifferentiablePandaCollisionPoints,
) -> None:
    output_index = int(episode["output_index"])
    candidate = record["validated"].candidate
    replay = record["replay"]
    eef_constraints = u_shape_constraints(candidate, target="eef")
    robot_constraints = u_shape_constraints(candidate, target="robot")
    save_episode_constraints(
        args.config_dir / "constraints" / "eef" / f"episode_{output_index:03d}.json",
        eef_constraints,
    )
    save_episode_constraints(
        args.config_dir / "constraints" / "robot" / f"episode_{output_index:03d}.json",
        robot_constraints,
    )
    witness_path = args.config_dir / "witnesses" / f"episode_{output_index:03d}.npz"
    witness_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        witness_path,
        qpos=replay["qpos"],
        tcp_path=replay["tcp"],
        robot_points=replay["clouds"],
        clearance=replay["clearance"],
    )
    rerun_path = args.artifact_dir / "rerun" / f"episode_{output_index:03d}.rrd"
    inspection = {
        "nominal_tcp_path": nominal["tcp_path"],
        "witness_tcp_path": replay["tcp"],
        "witness_robot_points": replay["clouds"],
        "witness_robot_link_indices": collision_model.link_indices.detach().cpu().numpy(),
        "witness_clearance": replay["clearance"],
        "start_position": nominal["start"],
        "goal_position": nominal["goal"],
        "witness_side": record["validated"].witness_side,
        "summary": record["fixture_record"],
    }
    save_rerun_timeline(
        rerun_path,
        replay["timeline"],
        constraints=robot_constraints,
        goal_marker_points=int(getattr(policy, "goal_marker_points", 0)),
        goal_marker_radius=float(getattr(policy, "goal_marker_radius", 0.045)),
        recording_identity={
            "fixture_id": "e10-u-shape-10ep-review-v1",
            "episode": output_index,
            "dataset_episode_index": int(episode["dataset_episode_index"]),
            "simulator_seed": int(episode["simulator_seed"]),
            "policy_seed": int(episode["policy_seed"]),
            "status": "inspection_candidate",
        },
        inspection=inspection,
    )
    _save_topdown(
        args.artifact_dir / "topdown" / f"episode_{output_index:03d}.png",
        candidate=candidate,
        nominal=nominal["tcp_path"],
        witness=replay["tcp"],
        start=np.asarray(context["tcp_pose"], dtype=np.float32)[:3],
        goal=np.asarray(context["target_position"], dtype=np.float32),
        title=f"E10 U-shape review — episode {output_index:03d}",
    )


def _save_topdown(
    path: Path,
    *,
    candidate: UShapeCandidate,
    nominal: np.ndarray,
    witness: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    title: str,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    for component in u_shape_components(candidate.half_extents):
        center, yaw = transform_box_component(
            component,
            center=candidate.root_center,
            yaw=candidate.yaw,
        )
        hx, hy = component.half_extents[:2]
        local = np.asarray([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]], dtype=np.float32)
        rotation = np.asarray(
            [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
            dtype=np.float32,
        )
        world = local @ rotation.T + center[:2]
        ax.add_patch(Polygon(world, closed=True, facecolor="#5b8ecb", edgecolor="#173b68"))
    ax.plot(nominal[:, 0], nominal[:, 1], "--", color="gray", label="nominal base")
    ax.plot(witness[:, 0], witness[:, 1], color="cyan", linewidth=2, label="witness")
    ax.scatter(start[0], start[1], color="red", label="start", zorder=4)
    ax.scatter(goal[0], goal[1], color="green", label="goal", zorder=4)
    ax.set_title(title)
    ax.set_xlabel("world X (m)")
    ax.set_ylabel("world Y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_suite_outputs(
    *,
    args: argparse.Namespace,
    source: dict[str, Any],
    episodes: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    indices = [int(episode["dataset_episode_index"]) for episode in episodes]
    (args.config_dir / "episode_indices.txt").write_text(
        "# E10 U-shape review suite v1. Local output order 0--9.\n"
        + "\n".join(str(index) for index in indices)
        + "\n",
        encoding="utf-8",
    )
    fixture = {
        "schema_version": "pg3d.e10_u_shape_review_fixture.v1",
        "fixture_id": "e10-u-shape-10ep-review-v1",
        "status": "inspection_candidate",
        "source_fixture": str(args.source_fixture),
        "dataset": source["dataset"],
        "checkpoint": source["checkpoint"],
        "checkpoint_model": source.get("checkpoint_model", "ema"),
        "selection_policy": {
            "outcome_blind": True,
            "compared_methods_run": [],
            "target_back_chunks": 2.5,
            "primary_back_chunks": [2.0, 2.5, 3.0],
            "clearance_fallback_back_chunks": [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            "usable_cavity_depths_m": [0.08, 0.10, 0.12, 0.14, 0.16],
            "nominal_source": args.nominal_source,
            "minimum_whole_robot_clearance_m": 0.03,
            "minimum_policy_obstacle_points": 64,
            "full_height_m": 0.75,
            "maximum_steps": int(args.max_steps),
            "stable_hold_steps": int(args.hold_steps),
        },
        "episodes": episodes,
        "git": git_commit_info(Path(__file__).resolve().parents[1]),
    }
    (args.config_dir / "fixture.json").write_text(
        json.dumps(jsonable(fixture), indent=2, sort_keys=True), encoding="utf-8"
    )
    report = {
        "schema_version": "pg3d.e10_u_shape_placement_report.v1",
        "fixture": str(args.config_dir / "fixture.json"),
        "episode_count": len(episodes),
        "episodes": episodes,
    }
    (args.artifact_dir / "placement_report.json").write_text(
        json.dumps(jsonable(report), indent=2, sort_keys=True), encoding="utf-8"
    )
    readme = """# E10 ten-episode U-shape review fixture

This inspection candidate derives from the frozen E3 ten-episode suite. Geometry was selected
without running rejection, reranking, or ITPS. Every episode retains the original dataset,
simulator, and policy identities and includes a 3 cm-clear whole-arm motion-planner witness.

Open the review timelines with `rerun artifacts/e10-u-shape-10ep-review-v1/rerun/episode_XXX.rrd`.
Do not use this fixture for definitive trials until visual inspection approves it and an unchanged
copy is promoted to a separately versioned frozen fixture.
"""
    (args.config_dir / "README.md").write_text(readme, encoding="utf-8")


def _candidate_json(candidate: UShapeCandidate) -> dict[str, Any]:
    result = asdict(candidate)
    result["root_center"] = candidate.root_center.astype(float).tolist()
    result["half_extents"] = candidate.half_extents.astype(float).tolist()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
