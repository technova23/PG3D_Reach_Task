"""Execute saved U-shape planner paths in ManiSkill and record MP4/Rerun artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from pg3d.envs.maniskill_adapter import register_pg3d_reach_envs
from pg3d.eval.u_shape_placement import u_shape_constraints
from pg3d.utils.arrays import bool_any, frame_to_numpy, to_numpy
from scripts.build_u_shape_fixture import (
    _clearance_series,
    _collision_model,
    _env_kwargs,
    _robot_clouds,
)
from scripts.eval_reach_checkpoint_unique_seeds import (
    _reset_to_zarr_episode,
    _zarr_episode_context,
)
from scripts.rollout_dp3_reach_policy import (
    crop_config_from_metadata,
    policy_action_to_sim_action,
    rollout_observation_entry,
    save_rerun_timeline,
    save_video,
)
from scripts.validate_u_shape_fixture_planner import _candidate_from_record

DEFAULT_FIXTURE = Path("configs/eval/e10_u_shape_box_derived_review_v1/fixture.json")
DEFAULT_PLANNER_DIR = Path("artifacts/e10-u-shape-conventional-position-ik-margin3cm-extended-v1")
DEFAULT_OUTPUT_DIR = Path("artifacts/e10-u-shape-planner-path-executions-v1")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--planner-dir", type=Path, default=DEFAULT_PLANNER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--episodes", type=int, nargs="+", default=None)
    parser.add_argument("--safety-margin", type=float, default=0.03)
    parser.add_argument("--gripper-open", type=float, default=0.04)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--contact-force-tolerance", type=float, default=1e-6)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.safety_margin < 0.0 or not np.isfinite(args.safety_margin):
        raise ValueError("--safety-margin must be finite and non-negative")
    if args.video_fps <= 0:
        raise ValueError("--video-fps must be positive")
    if args.contact_force_tolerance < 0.0:
        raise ValueError("--contact-force-tolerance must be non-negative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    episodes = list(fixture["episodes"])
    requested = set(args.episodes) if args.episodes is not None else None
    if requested is not None:
        known = {int(item["output_index"]) for item in episodes}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"unknown fixture episode indices: {unknown}")
        episodes = [item for item in episodes if int(item["output_index"]) in requested]

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = _load_metadata(Path(fixture["dataset"]))
    crop_config = crop_config_from_metadata(metadata)
    zarr_root = zarr.open_group(str(fixture["dataset"]), mode="r")

    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
    except Exception as exc:
        raise RuntimeError("ManiSkill is required to execute planner paths") from exc

    register_pg3d_reach_envs()
    results: list[dict[str, Any]] = []
    for episode in episodes:
        output_index = int(episode["output_index"])
        path_file = args.planner_dir / "paths" / f"episode_{output_index:03d}.npz"
        if not path_file.exists():
            result = {
                "output_index": output_index,
                "dataset_episode_index": int(episode["dataset_episode_index"]),
                "status": "missing_saved_path",
                "path_file": str(path_file),
            }
            results.append(result)
            _write_json(args.output_dir / "results" / f"episode_{output_index:03d}.json", result)
            print(f"[replay] episode={output_index:03d} missing saved path", flush=True)
            continue
        print(f"[replay] episode={output_index:03d} path={path_file}", flush=True)
        result = _execute_episode(
            gym=gym,
            metadata=metadata,
            crop_config=crop_config,
            zarr_root=zarr_root,
            episode=episode,
            fixture=fixture,
            path_file=path_file,
            args=args,
        )
        results.append(result)
        _write_json(args.output_dir / "results" / f"episode_{output_index:03d}.json", result)

    report = {
        "schema_version": "pg3d.u_shape_planner_execution.v1",
        "fixture": str(args.fixture),
        "planner_dir": str(args.planner_dir),
        "output_dir": str(args.output_dir),
        "safety_margin_m": float(args.safety_margin),
        "requested_episodes": sorted(requested) if requested is not None else "all",
        "executed_count": sum(item["status"] == "executed" for item in results),
        "missing_path_count": sum(item["status"] == "missing_saved_path" for item in results),
        "results": results,
    }
    _write_json(args.output_dir / "report.json", report)
    return 0


def _load_metadata(dataset: Path) -> dict[str, Any]:
    from pg3d.envs.maniskill_adapter.dataset import load_reach_metadata

    return load_reach_metadata(dataset)


def _execute_episode(
    *,
    gym: Any,
    metadata: dict[str, Any],
    crop_config: Any,
    zarr_root: Any,
    episode: dict[str, Any],
    fixture: dict[str, Any],
    path_file: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_index = int(episode["output_index"])
    candidate = _candidate_from_record(episode)
    constraints = u_shape_constraints(candidate, target="robot", name_prefix="u_shape_box_derived")
    saved = np.load(path_file)
    planned_qpos = np.asarray(saved["qpos"], dtype=np.float32)
    if planned_qpos.ndim != 2 or planned_qpos.shape[1] != 7 or not len(planned_qpos):
        raise ValueError(f"invalid planned qpos path in {path_file}: {planned_qpos.shape}")

    context = _zarr_episode_context(zarr_root, int(episode["dataset_episode_index"]))
    env = gym.make(
        str(metadata["env_id"]),
        **_replay_env_kwargs(metadata, obstacle_half_extents=candidate.half_extents),
    )
    try:
        reset_options = {
            "pg3d_obstacle_center": candidate.root_center.astype(float).tolist(),
            "pg3d_obstacle_yaw": float(candidate.yaw),
        }
        obs, info = _reset_to_zarr_episode(
            env,
            rollout_seed=int(episode["simulator_seed"]),
            zarr_context=context,
            reset_options=reset_options,
        )
        timeline = [rollout_observation_entry(obs, info, env=env, crop_config=crop_config)]
        frames = [frame_to_numpy(env.render())]
        actual_qpos = [np.asarray(timeline[-1]["agent_pos"], dtype=np.float32)[:7]]
        tcp_path = [np.asarray(timeline[-1]["tcp_pose"], dtype=np.float32)[:3]]
        contact_force_norms = [0.0]
        terminated_steps: list[int] = []

        for step_index, q_target in enumerate(planned_qpos[1:], start=1):
            sim_action = policy_action_to_sim_action(
                q_target,
                np.asarray(timeline[-1]["agent_pos"], dtype=np.float32),
                action_mode="abs_joint",
                sim_action_dim=int(np.prod(env.action_space.shape)),
                low=getattr(env.action_space, "low", None),
                high=getattr(env.action_space, "high", None),
                gripper_open=float(args.gripper_open),
            )
            obs, _reward, terminated, truncated, info = env.step(sim_action)
            entry = rollout_observation_entry(obs, info, env=env, crop_config=crop_config)
            timeline.append(entry)
            frames.append(frame_to_numpy(env.render()))
            actual_qpos.append(np.asarray(entry["agent_pos"], dtype=np.float32)[:7])
            tcp_path.append(np.asarray(entry["tcp_pose"], dtype=np.float32)[:3])
            contact_force_norms.append(_robot_obstacle_contact_force_norm(env))
            if bool_any(terminated) or bool_any(truncated):
                terminated_steps.append(step_index)

        actual_qpos_array = np.asarray(actual_qpos, dtype=np.float32)
        tcp_path_array = np.asarray(tcp_path, dtype=np.float32)
        collision_model, world_from_base = _collision_model(env)
        actual_clouds = _robot_clouds(collision_model, world_from_base, actual_qpos_array)
        clearance = _clearance_series(actual_clouds, constraints)
        target = np.asarray(timeline[-1]["target_position"], dtype=np.float32).reshape(3)
        final_distance = float(np.linalg.norm(tcp_path_array[-1] - target))
        tracking_error = np.linalg.norm(actual_qpos_array - planned_qpos, axis=1)
        force_array = np.asarray(contact_force_norms, dtype=np.float32)
        physical_contact = bool(np.any(force_array > float(args.contact_force_tolerance)))
        raw_min_clearance = float(np.min(clearance))

        video_path = args.output_dir / "videos" / f"episode_{output_index:03d}.mp4"
        rerun_path = args.output_dir / "rerun" / f"episode_{output_index:03d}.rrd"
        result = {
            "output_index": output_index,
            "dataset_episode_index": int(episode["dataset_episode_index"]),
            "status": "executed",
            "execution_status": "completed_saved_path",
            "path_file": str(path_file),
            "planned_steps": int(len(planned_qpos)),
            "executed_steps": int(len(actual_qpos_array)),
            "final_target_distance_m": final_distance,
            "goal_reached_2p5cm": final_distance <= 0.025,
            "raw_min_clearance_m": raw_min_clearance,
            "three_cm_clear_all_states": raw_min_clearance >= float(args.safety_margin),
            "physical_contact": physical_contact,
            "max_pairwise_contact_force_norm": float(np.max(force_array)),
            "first_physical_contact_step": (
                int(np.flatnonzero(force_array > float(args.contact_force_tolerance))[0])
                if physical_contact
                else None
            ),
            "simulator_terminated_or_truncated_steps": terminated_steps,
            "joint_tracking_error_mean": float(np.mean(tracking_error)),
            "joint_tracking_error_max": float(np.max(tracking_error)),
            "video": str(video_path),
            "rerun": str(rerun_path),
        }
        save_video(video_path, frames, fps=int(args.video_fps))
        save_rerun_timeline(
            rerun_path,
            timeline,
            constraints=constraints,
            recording_identity={
                "fixture_id": fixture["fixture_id"],
                "episode": output_index,
                "dataset_episode_index": int(episode["dataset_episode_index"]),
                "status": "executed_saved_planner_path",
                "physical_contact": physical_contact,
                "raw_min_clearance_m": raw_min_clearance,
                "final_target_distance_m": final_distance,
            },
            inspection={
                "nominal_tcp_path": np.asarray(saved["tcp_path"], dtype=np.float32),
                "witness_tcp_path": tcp_path_array,
                "witness_robot_points": actual_clouds,
                "witness_robot_link_indices": collision_model.link_indices.detach().cpu().numpy(),
                "witness_clearance": clearance,
                "witness_side": "executed_saved_planner_path",
                "executed_tcp_path": tcp_path_array,
                "executed_robot_points": actual_clouds,
                "executed_clearance": clearance,
                "contact_force_norm": force_array,
                "start_position": tcp_path_array[0],
                "goal_position": target,
                "summary": result,
            },
        )
        return result
    finally:
        env.close()


def _robot_obstacle_contact_force_norm(env: Any) -> float:
    unwrapped = env.unwrapped
    robot = unwrapped.agent.robot
    actors = list(getattr(unwrapped, "pg3d_obstacle_actors", ()))
    maximum = 0.0
    for link in robot.links:
        for actor in actors:
            force = to_numpy(unwrapped.scene.get_pairwise_contact_forces(link, actor))
            maximum = max(maximum, float(np.linalg.norm(force)))
    return maximum


def _replay_env_kwargs(
    metadata: dict[str, Any],
    *,
    obstacle_half_extents: np.ndarray,
) -> dict[str, Any]:
    """Return U-shape environment options with RGB rendering enabled."""
    kwargs = _env_kwargs(metadata, obstacle_half_extents=obstacle_half_extents)
    kwargs["render_mode"] = "rgb_array"
    return kwargs


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
