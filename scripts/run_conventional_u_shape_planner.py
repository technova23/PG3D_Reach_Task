"""Run bounded direct configuration-space MPLib tests on fixed U scenes.

Each episode plans directly from its recorded start joint configuration to a
selected joint, fixed-pose, or position-only Cartesian goal. No hand-built
Cartesian waypoint or initial-clearance prefilter is used. The report separates
"MPLib returned a path" from exact whole-robot collision and 3 cm clearance.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from pg3d.envs.maniskill_adapter import register_pg3d_reach_envs
from pg3d.envs.maniskill_adapter.dataset import load_reach_metadata
from pg3d.envs.obstacles import transform_box_component, u_shape_components
from pg3d.eval.u_shape_placement import u_shape_constraints
from pg3d.utils.arrays import to_numpy
from scripts.build_u_shape_fixture import (
    _clearance_series,
    _collision_model,
    _env_kwargs,
    _path_length,
    _prequalified_dataset_nominal,
    _robot_clouds,
    _save_topdown,
    _tcp_path_from_qpos,
)
from scripts.eval_reach_checkpoint_unique_seeds import (
    _reset_to_zarr_episode,
    _zarr_episode_context,
)
from scripts.rollout_dp3_reach_policy import (
    crop_config_from_metadata,
    rollout_observation_entry,
    save_rerun_timeline,
)
from scripts.validate_u_shape_fixture_planner import (
    _candidate_from_record,
    _parent_failure_result,
    _prepare_output,
    _write_json,
)

DEFAULT_FIXTURE = Path("configs/eval/e10_u_shape_box_derived_review_v1/fixture.json")
DEFAULT_OUTPUT_DIR = Path("artifacts/e10-u-shape-conventional-planner-v1")


def _quaternion_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.asarray(left, dtype=np.float64)
    rw, rx, ry, rz = np.asarray(right, dtype=np.float64)
    result = np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )
    return (result / np.linalg.norm(result)).astype(np.float32)


def _axis_angle_quaternion(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    xyz = np.asarray(axis, dtype=np.float64)
    xyz /= np.linalg.norm(xyz)
    return np.asarray(
        [math.cos(0.5 * angle), *(xyz * math.sin(0.5 * angle))],
        dtype=np.float32,
    )


def _position_goal_orientations(reference: np.ndarray) -> list[np.ndarray]:
    """Return deterministic orientation samples for a position-only reach goal."""
    reference = np.asarray(reference, dtype=np.float32).copy()
    reference /= np.linalg.norm(reference)
    tilts = [
        _axis_angle_quaternion((1.0, 0.0, 0.0), angle)
        for angle in (0.0, math.radians(30.0), math.radians(-30.0))
    ]
    tilts.extend(
        _axis_angle_quaternion((0.0, 1.0, 0.0), angle)
        for angle in (math.radians(30.0), math.radians(-30.0))
    )
    orientations = []
    for yaw in np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False):
        yaw_quaternion = _axis_angle_quaternion((0.0, 0.0, 1.0), float(yaw))
        for tilt in tilts:
            orientations.append(
                _quaternion_multiply_wxyz(
                    yaw_quaternion,
                    _quaternion_multiply_wxyz(tilt, reference),
                )
            )
    return orientations


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker_episode is not None:
        return _run_worker(args)
    return _run_parent(args)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--episode-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--rrt-planning-time-seconds", type=float, default=5.0)
    parser.add_argument("--rrt-range", type=float, default=0.1)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--collision-margin", type=float, default=0.0)
    parser.add_argument("--minimum-clearance", type=float, default=0.03)
    parser.add_argument(
        "--goal-mode",
        choices=("dataset_qpos", "dataset_pose", "position_ik"),
        default="dataset_qpos",
        help=(
            "Plan to the recorded terminal joints, solve IK for the Cartesian goal "
            "with the demonstration's terminal orientation, or sample orientations for "
            "a position-only goal."
        ),
    )
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episode-limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--worker-episode", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (
        args.episode_timeout_seconds <= 0.0
        or args.rrt_planning_time_seconds <= 0.0
        or args.rrt_range <= 0.0
    ):
        raise ValueError("planner time limits must be positive")
    if args.attempts <= 0:
        raise ValueError("--attempts must be positive")
    if args.minimum_clearance <= 0.0 or args.collision_margin < 0.0:
        raise ValueError("clearance must be positive and collision margin non-negative")
    if not 0 <= args.episode_start < 10:
        raise ValueError("--episode-start must be in [0, 9]")
    if args.episode_limit is not None and args.episode_limit <= 0:
        raise ValueError("--episode-limit must be positive")
    return args


def _run_parent(args: argparse.Namespace) -> int:
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    episodes = list(fixture["episodes"])[args.episode_start :]
    if args.episode_limit is not None:
        episodes = episodes[: args.episode_limit]
    _prepare_output(args.output_dir, overwrite=args.overwrite)
    results: list[dict[str, Any]] = []
    script = Path(__file__).resolve()
    for episode in episodes:
        output_index = int(episode["output_index"])
        command = [
            sys.executable,
            str(script),
            "--fixture",
            str(args.fixture),
            "--output-dir",
            str(args.output_dir),
            "--episode-timeout-seconds",
            str(args.episode_timeout_seconds),
            "--rrt-planning-time-seconds",
            str(args.rrt_planning_time_seconds),
            "--rrt-range",
            str(args.rrt_range),
            "--attempts",
            str(args.attempts),
            "--collision-margin",
            str(args.collision_margin),
            "--minimum-clearance",
            str(args.minimum_clearance),
            "--goal-mode",
            args.goal_mode,
            "--worker-episode",
            str(output_index),
        ]
        print(
            f"[direct-rrt-parent] episode={output_index:03d} "
            f"timeout={args.episode_timeout_seconds:.1f}s",
            flush=True,
        )
        started = time.perf_counter()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        timed_out = False
        try:
            output, _ = process.communicate(timeout=args.episode_timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
        elapsed = time.perf_counter() - started
        log_path = args.output_dir / "logs" / f"episode_{output_index:03d}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(output, encoding="utf-8")
        result_path = args.output_dir / "results" / f"episode_{output_index:03d}.json"
        if timed_out:
            result = _parent_failure_result(
                episode,
                status="timeout",
                elapsed_seconds=elapsed,
                message=f"worker exceeded {args.episode_timeout_seconds:.1f}s hard limit",
            )
            _write_json(result_path, result)
        elif process.returncode != 0:
            result = _parent_failure_result(
                episode,
                status="worker_error",
                elapsed_seconds=elapsed,
                message=f"worker exited with code {process.returncode}",
            )
            result["log_tail"] = output[-4000:]
            _write_json(result_path, result)
        elif not result_path.is_file():
            result = _parent_failure_result(
                episode,
                status="missing_result",
                elapsed_seconds=elapsed,
                message="worker exited successfully without a result file",
            )
            _write_json(result_path, result)
        else:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["parent_wall_seconds"] = elapsed
            _write_json(result_path, result)
        results.append(result)
        print(
            f"[direct-rrt-parent] episode={output_index:03d} status={result['status']} "
            f"wall={elapsed:.2f}s",
            flush=True,
        )
    report = {
        "schema_version": "pg3d.e10_u_shape_direct_rrt_validation.v1",
        "fixture": str(args.fixture),
        "geometry_modified": False,
        "planner": {
            "library": "MPLib",
            "algorithm": "RRTConnect",
            "endpoint": {
                "dataset_qpos": "recorded successful dataset terminal joint configuration",
                "dataset_pose": (
                    "Cartesian task goal with dataset terminal end-effector orientation"
                ),
                "position_ik": "Cartesian task goal with sampled end-effector orientations",
            }[args.goal_mode],
            "goal_mode": args.goal_mode,
            "cartesian_waypoints": 0,
            "attempts": args.attempts,
            "rrt_planning_time_seconds_per_attempt": args.rrt_planning_time_seconds,
            "rrt_range": args.rrt_range,
            "episode_hard_timeout_seconds": args.episode_timeout_seconds,
            "initial_clearance_prefilter": False,
            "collision_margin_m": args.collision_margin,
        },
        "episode_count": len(results),
        "planner_path_count": sum(item.get("planner_path_found", False) for item in results),
        "exact_contact_free_count": sum(
            item.get("exact_contact_free_all_states", False) for item in results
        ),
        "exact_three_cm_count": sum(
            item.get("exact_three_cm_clear_all_states", False) for item in results
        ),
        "timeout_count": sum(item["status"] == "timeout" for item in results),
        "results": results,
    }
    _write_json(args.output_dir / "report.json", report)
    print(
        f"[direct-rrt-parent] complete paths={report['planner_path_count']}/{len(results)} "
        f"contact_free={report['exact_contact_free_count']} "
        f"three_cm={report['exact_three_cm_count']} timeouts={report['timeout_count']}",
        flush=True,
    )
    return 0


def _run_worker(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    output_index = int(args.worker_episode)
    episode = next(
        item for item in fixture["episodes"] if int(item["output_index"]) == output_index
    )
    metadata = load_reach_metadata(Path(fixture["dataset"]))
    crop_config = crop_config_from_metadata(metadata)
    zarr_root = zarr.open_group(str(fixture["dataset"]), mode="r")
    context = _zarr_episode_context(zarr_root, int(episode["dataset_episode_index"]))
    nominal = _prequalified_dataset_nominal(
        zarr_root,
        context=context,
        episode_index=int(episode["dataset_episode_index"]),
    )
    candidate = _candidate_from_record(episode)
    constraints = u_shape_constraints(candidate, target="robot", name_prefix="u_shape_box_derived")

    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
        import sapien
        from mani_skill.examples.motionplanning.panda.motionplanner import (
            PandaArmMotionPlanningSolver,
        )
    except Exception as exc:
        raise RuntimeError("ManiSkill, SAPIEN, and MPLib are required") from exc
    register_pg3d_reach_envs()
    env = gym.make(
        str(metadata["env_id"]),
        **_env_kwargs(metadata, obstacle_half_extents=candidate.half_extents),
    )
    planner = None
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
        collision_model, world_from_base = _collision_model(env)
        robot = env.unwrapped.agent.robot
        start_qpos = to_numpy(robot.get_qpos()).astype(np.float32).reshape(-1)
        goal_qpos = np.asarray(nominal["goal_qpos"], dtype=np.float32).reshape(-1)
        endpoint_clouds = _robot_clouds(
            collision_model,
            world_from_base,
            np.stack((start_qpos, goal_qpos)),
        )
        initial_clearance = float(
            min(
                np.min(constraint.region.signed_distance(endpoint_clouds[0]))
                for constraint in constraints
            )
        )
        goal_clearance = float(
            min(
                np.min(constraint.region.signed_distance(endpoint_clouds[1]))
                for constraint in constraints
            )
        )
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
        np.random.seed(int(episode["policy_seed"]) % (2**32))
        for component in u_shape_components(candidate.half_extents):
            center, yaw = transform_box_component(
                component,
                center=candidate.root_center,
                yaw=candidate.yaw,
            )
            planner.add_box_collision(
                extents=2.0
                * (np.asarray(component.half_extents, dtype=np.float32) + args.collision_margin),
                pose=sapien.Pose(
                    p=center,
                    q=[math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)],
                ),
            )
        attempts: list[dict[str, Any]] = []
        paths: list[np.ndarray] = []
        goal_pose = np.concatenate(
            (
                np.asarray(nominal["goal"], dtype=np.float32),
                np.asarray(nominal["goal_pose"], dtype=np.float32)[3:7],
            )
        )
        position_goal_qposes: list[np.ndarray] = []
        if args.goal_mode == "position_ik":
            for orientation in _position_goal_orientations(goal_pose[3:7]):
                world_goal_pose = np.concatenate((goal_pose[:3], orientation))
                base_goal_pose = planner.planner.transform_goal_to_wrt_base(world_goal_pose)
                ik_status, ik_qposes = planner.planner.IK(
                    base_goal_pose,
                    start_qpos.copy(),
                    n_init_qpos=20,
                    threshold=1e-3,
                )
                if ik_status != "Success":
                    continue
                for ik_qpos in ik_qposes:
                    qpos = np.asarray(ik_qpos, dtype=np.float32)
                    if not any(
                        np.linalg.norm(qpos[:7] - existing[:7]) < 0.05
                        for existing in position_goal_qposes
                    ):
                        position_goal_qposes.append(qpos)
        for attempt_index in range(args.attempts):
            attempt_started = time.perf_counter()
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                common_plan_kwargs = {
                    "time_step": env.unwrapped.control_timestep,
                    "rrt_range": args.rrt_range,
                    "planning_time": args.rrt_planning_time_seconds,
                    "use_point_cloud": True,
                    "planner_name": "RRTConnect",
                    "verbose": False,
                }
                if args.goal_mode == "position_ik" and not position_goal_qposes:
                    plan = {"status": "IK Failed! No collision-free position-goal solutions."}
                elif args.goal_mode == "position_ik":
                    plan = planner.planner.plan_qpos_to_qpos(
                        position_goal_qposes,
                        start_qpos.copy(),
                        **common_plan_kwargs,
                    )
                elif args.goal_mode == "dataset_pose":
                    plan = planner.planner.plan_qpos_to_pose(
                        goal_pose,
                        start_qpos.copy(),
                        **common_plan_kwargs,
                    )
                else:
                    plan = planner.planner.plan_qpos_to_qpos(
                        [goal_qpos.copy()],
                        start_qpos.copy(),
                        **common_plan_kwargs,
                    )
            attempt = {
                "attempt_index": attempt_index,
                "elapsed_seconds": time.perf_counter() - attempt_started,
                "planner_status": str(plan.get("status", "unknown")),
            }
            if args.goal_mode == "position_ik":
                attempt["collision_free_ik_goal_count"] = len(position_goal_qposes)
            positions = plan.get("position")
            if plan.get("status") == "Success" and positions is not None:
                qpath = np.asarray(positions, dtype=np.float32)
                attempt["raw_steps"] = int(len(qpath))
                paths.append(qpath)
            attempts.append(attempt)
        result_path = args.output_dir / "results" / f"episode_{output_index:03d}.json"
        if not paths:
            result = {
                "output_index": output_index,
                "dataset_episode_index": int(episode["dataset_episode_index"]),
                "status": "no_path",
                "planner_path_found": False,
                "goal_mode": args.goal_mode,
                "collision_free_ik_goal_count": len(position_goal_qposes),
                "initial_robot_clearance_m": initial_clearance,
                "goal_robot_clearance_m": goal_clearance,
                "attempts": attempts,
                "worker_wall_seconds": time.perf_counter() - started,
            }
            _write_json(result_path, result)
            return 0

        evaluated = []
        for path_index, qpath in enumerate(paths):
            clouds = _robot_clouds(collision_model, world_from_base, qpath)
            clearance = _clearance_series(clouds, constraints)
            tcp = _tcp_path_from_qpos(env, qpath)
            evaluated.append(
                {
                    "path_index": path_index,
                    "qpos": qpath,
                    "clouds": clouds,
                    "clearance": clearance,
                    "tcp": tcp,
                    "min_clearance": float(np.min(clearance)),
                    "min_clearance_after_start": float(np.min(clearance[1:]))
                    if len(clearance) > 1
                    else float(clearance[0]),
                    "path_length": _path_length(tcp),
                }
            )
        evaluated.sort(
            key=lambda item: (
                item["min_clearance"] < 0.0,
                item["min_clearance"] < args.minimum_clearance,
                -item["min_clearance"],
                item["path_length"],
            )
        )
        selected = evaluated[0]
        witness_path = args.output_dir / "paths" / f"episode_{output_index:03d}.npz"
        witness_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            witness_path,
            qpos=selected["qpos"],
            tcp_path=selected["tcp"],
            robot_points=selected["clouds"],
            clearance=selected["clearance"],
        )
        result = {
            "output_index": output_index,
            "dataset_episode_index": int(episode["dataset_episode_index"]),
            "status": "path_found",
            "planner_path_found": True,
            "goal_mode": args.goal_mode,
            "collision_free_ik_goal_count": len(position_goal_qposes),
            "geometry_modified": False,
            "initial_robot_clearance_m": initial_clearance,
            "goal_robot_clearance_m": goal_clearance,
            "exact_min_clearance_all_states_m": selected["min_clearance"],
            "exact_min_clearance_after_start_m": selected["min_clearance_after_start"],
            "exact_contact_free_all_states": selected["min_clearance"] >= 0.0,
            "exact_three_cm_clear_all_states": selected["min_clearance"] >= args.minimum_clearance,
            "planned_steps": int(len(selected["qpos"])),
            "planned_tcp_path_length_m": selected["path_length"],
            "attempts": attempts,
            "successful_attempt_count": len(paths),
            "worker_wall_seconds": time.perf_counter() - started,
            "path_file": str(witness_path),
            "rerun_file": str(args.output_dir / "rerun" / f"episode_{output_index:03d}.rrd"),
            "topdown_file": str(args.output_dir / "topdown" / f"episode_{output_index:03d}.png"),
            "execution_status": "not_executed_planner_diagnostic",
        }
        save_rerun_timeline(
            Path(result["rerun_file"]),
            timeline,
            constraints=constraints,
            recording_identity={
                "fixture_id": fixture["fixture_id"],
                "episode": output_index,
                "dataset_episode_index": int(episode["dataset_episode_index"]),
                "status": "direct_rrt_planned_unexecuted",
            },
            inspection={
                "nominal_tcp_path": nominal["tcp_path"],
                "witness_tcp_path": selected["tcp"],
                "witness_robot_points": selected["clouds"],
                "witness_robot_link_indices": collision_model.link_indices.detach().cpu().numpy(),
                "witness_clearance": selected["clearance"],
                "start_position": nominal["start"],
                "goal_position": nominal["goal"],
                "witness_side": "direct_configuration_space",
                "summary": result,
            },
        )
        _save_topdown(
            Path(result["topdown_file"]),
            candidate=candidate,
            nominal=nominal["tcp_path"],
            witness=selected["tcp"],
            start=nominal["start"],
            goal=nominal["goal"],
            title=f"Direct RRTConnect — episode {output_index:03d}",
        )
        result["worker_wall_seconds"] = time.perf_counter() - started
        _write_json(result_path, result)
        return 0
    finally:
        if planner is not None:
            planner.close()
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
