"""Validate fixed U-shape placements with bounded MPLib workers.

The parent process launches one fresh worker per episode and enforces a hard
wall-clock timeout. Workers never modify geometry: they attempt left/right
MPLib routes, apply the exact 3 cm whole-robot gate, replay viable paths in
PhysX, and save cyan-path Reruns only for successful witnesses.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
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
from pg3d.eval.u_shape_placement import UShapeCandidate, UShapeGeometryCheck, u_shape_constraints
from pg3d.utils.serialization import jsonable
from scripts.build_u_shape_fixture import (
    _collision_model,
    _env_kwargs,
    _path_length,
    _prequalified_dataset_nominal,
    _robot_clouds,
    _save_topdown,
    _validate_candidate_live,
)
from scripts.eval_reach_checkpoint_unique_seeds import _zarr_episode_context
from scripts.rollout_dp3_reach_policy import crop_config_from_metadata, save_rerun_timeline

DEFAULT_FIXTURE = Path("configs/eval/e10_u_shape_box_derived_review_v1/fixture.json")
DEFAULT_OUTPUT_DIR = Path("artifacts/e10-u-shape-box-derived-planner-v1")


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
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--hold-steps", type=int, default=16)
    parser.add_argument("--minimum-clearance", type=float, default=0.03)
    parser.add_argument("--gripper-open", type=float, default=0.04)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episode-limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--worker-episode", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.episode_timeout_seconds <= 0.0:
        raise ValueError("--episode-timeout-seconds must be positive")
    if args.max_steps <= 0 or args.hold_steps <= 0 or args.hold_steps >= args.max_steps:
        raise ValueError("step limits must be positive and hold steps must be below max steps")
    if args.minimum_clearance <= 0.0:
        raise ValueError("--minimum-clearance must be positive")
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
            "--max-steps",
            str(args.max_steps),
            "--hold-steps",
            str(args.hold_steps),
            "--minimum-clearance",
            str(args.minimum_clearance),
            "--gripper-open",
            str(args.gripper_open),
            "--worker-episode",
            str(output_index),
        ]
        print(
            f"[mplib-parent] episode={output_index:03d} "
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
            f"[mplib-parent] episode={output_index:03d} status={result['status']} "
            f"wall={elapsed:.2f}s",
            flush=True,
        )
    report = {
        "schema_version": "pg3d.e10_u_shape_planner_validation.v1",
        "fixture": str(args.fixture),
        "geometry_modified": False,
        "planner": {
            "library": "MPLib via ManiSkill PandaArmMotionPlanningSolver",
            "algorithms": ["screw", "RRTConnect"],
            "mplib_rrt_planning_time_seconds_per_call": 1.0,
            "episode_hard_timeout_seconds": args.episode_timeout_seconds,
            "sides": ["left", "right"],
            "planner_margins_m": [args.minimum_clearance, 0.0],
        },
        "validation": {
            "minimum_exact_whole_robot_clearance_m": args.minimum_clearance,
            "maximum_live_steps": args.max_steps,
            "stable_hold_steps": args.hold_steps,
        },
        "episode_count": len(results),
        "success_count": sum(item["status"] == "success" for item in results),
        "timeout_count": sum(item["status"] == "timeout" for item in results),
        "results": results,
    }
    _write_json(args.output_dir / "report.json", report)
    print(
        f"[mplib-parent] complete successes={report['success_count']}/{len(results)} "
        f"timeouts={report['timeout_count']}",
        flush=True,
    )
    return 0


def _run_worker(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    output_index = int(args.worker_episode)
    review_episode = next(
        item for item in fixture["episodes"] if int(item["output_index"]) == output_index
    )
    source_fixture = json.loads(Path(fixture["source_fixture"]).read_text(encoding="utf-8"))
    source_episode = next(
        item for item in source_fixture["episodes"] if int(item["output_index"]) == output_index
    )
    metadata = load_reach_metadata(Path(fixture["dataset"]))
    crop_config = crop_config_from_metadata(metadata)
    zarr_root = zarr.open_group(str(fixture["dataset"]), mode="r")
    context = _zarr_episode_context(zarr_root, int(review_episode["dataset_episode_index"]))
    nominal = _prequalified_dataset_nominal(
        zarr_root,
        context=context,
        episode_index=int(review_episode["dataset_episode_index"]),
    )
    candidate = _candidate_from_record(review_episode)
    constraints = u_shape_constraints(candidate, target="robot", name_prefix="u_shape_box_derived")

    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
    except Exception as exc:
        raise RuntimeError("ManiSkill and Gymnasium are required for planner validation") from exc
    register_pg3d_reach_envs()
    model_env = gym.make(str(metadata["env_id"]), **_env_kwargs(metadata))
    try:
        collision_model, world_from_base = _collision_model(model_env)
        initial_exact = _robot_clouds(
            collision_model,
            world_from_base,
            np.asarray(context["state"], dtype=np.float32),
        )[0]
        observed = np.asarray(context["point_cloud"], dtype=np.float32)[
            np.asarray(context["point_valid_mask"], dtype=bool)
            & np.asarray(context["robot_mask"], dtype=bool)
        ]
        initial_cloud = np.concatenate([initial_exact, observed], axis=0)
        initial_clearance = min(
            float(np.min(constraint.region.signed_distance(initial_cloud)))
            for constraint in constraints
        )
        geometry = UShapeGeometryCheck(
            accepted=initial_clearance >= args.minimum_clearance,
            reasons=() if initial_clearance >= args.minimum_clearance else ("initial_clearance",),
            initial_clearance=initial_clearance,
            goal_clearance=float("nan"),
            mouth_clearance=0.5 * candidate.mouth_width,
            goal_beyond_back=float("nan"),
            direct_back_clearance=float(review_episode["direct_path_back_clearance_m"]),
        )
        if initial_clearance < args.minimum_clearance:
            result = {
                "output_index": output_index,
                "dataset_episode_index": int(review_episode["dataset_episode_index"]),
                "status": "initial_clearance_below_gate",
                "initial_robot_clearance_m": initial_clearance,
                "minimum_clearance_m": args.minimum_clearance,
                "worker_wall_seconds": time.perf_counter() - started,
            }
            _write_json(args.output_dir / "results" / f"episode_{output_index:03d}.json", result)
            return 0
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            record, diagnostics = _validate_candidate_live(
                gym=gym,
                metadata=metadata,
                crop_config=crop_config,
                context=context,
                episode=source_episode,
                nominal=nominal,
                candidate=candidate,
                geometry=geometry,
                collision_model=collision_model,
                world_from_base=world_from_base,
                min_clearance=args.minimum_clearance,
                max_steps=args.max_steps,
                hold_steps=args.hold_steps,
                gripper_open=args.gripper_open,
                minimum_obstacle_points=0,
                planner_margins=(args.minimum_clearance, 0.0),
            )
    finally:
        model_env.close()

    result_path = args.output_dir / "results" / f"episode_{output_index:03d}.json"
    if record is None:
        result = {
            "output_index": output_index,
            "dataset_episode_index": int(review_episode["dataset_episode_index"]),
            "status": "no_valid_witness",
            "initial_robot_clearance_m": initial_clearance,
            "minimum_clearance_m": args.minimum_clearance,
            "worker_wall_seconds": time.perf_counter() - started,
            "diagnostics": diagnostics,
        }
        _write_json(result_path, result)
        return 0

    replay = record["replay"]
    validated = record["validated"]
    witness_path = args.output_dir / "witnesses" / f"episode_{output_index:03d}.npz"
    witness_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        witness_path,
        qpos=replay["qpos"],
        tcp_path=replay["tcp"],
        robot_points=replay["clouds"],
        clearance=replay["clearance"],
    )
    result = {
        "output_index": output_index,
        "dataset_episode_index": int(review_episode["dataset_episode_index"]),
        "status": "success",
        "geometry_modified": False,
        "witness_side": validated.witness_side,
        "witness_steps": validated.witness_steps,
        "witness_tcp_path_length_m": _path_length(replay["tcp"]),
        "initial_robot_clearance_m": initial_clearance,
        "witness_robot_clearance_m": float(np.min(replay["clearance"])),
        "stable_hold_steps": int(replay["stable_hold_steps"]),
        "obstacle_points_policy_input_reset": int(validated.obstacle_points),
        "physical_contacts": replay["physical_contacts"],
        "worker_wall_seconds": time.perf_counter() - started,
        "witness_file": str(witness_path),
        "rerun_file": str(args.output_dir / "rerun" / f"episode_{output_index:03d}.rrd"),
        "topdown_file": str(args.output_dir / "topdown" / f"episode_{output_index:03d}.png"),
        "diagnostics": diagnostics,
    }
    save_rerun_timeline(
        Path(result["rerun_file"]),
        replay["timeline"],
        constraints=constraints,
        recording_identity={
            "fixture_id": fixture["fixture_id"],
            "episode": output_index,
            "dataset_episode_index": int(review_episode["dataset_episode_index"]),
            "status": "mplib_validated_witness",
        },
        inspection={
            "nominal_tcp_path": nominal["tcp_path"],
            "witness_tcp_path": replay["tcp"],
            "witness_robot_points": replay["clouds"],
            "witness_robot_link_indices": collision_model.link_indices.detach().cpu().numpy(),
            "witness_clearance": replay["clearance"],
            "start_position": nominal["start"],
            "goal_position": nominal["goal"],
            "witness_side": validated.witness_side,
            "summary": result,
        },
    )
    _save_topdown(
        Path(result["topdown_file"]),
        candidate=candidate,
        nominal=nominal["tcp_path"],
        witness=replay["tcp"],
        start=nominal["start"],
        goal=nominal["goal"],
        title=f"MPLib witness — episode {output_index:03d}",
    )
    result["worker_wall_seconds"] = time.perf_counter() - started
    _write_json(result_path, result)
    return 0


def _candidate_from_record(record: dict[str, Any]) -> UShapeCandidate:
    return UShapeCandidate(
        root_center=np.asarray(record["root_center"], dtype=np.float32),
        yaw=float(record["yaw_rad"]),
        half_extents=np.asarray(record["envelope_half_extents_m"], dtype=np.float32),
        mouth_width=float(record["mouth_width_m"]),
        mouth_distance=float(record["mouth_distance_m"]),
        back_distance=float(record["back_distance_m"]),
        back_chunk_ratio=float(record["back_first_chunk_ratio"]),
        mouth_chunk_fraction=float(record["mouth_first_chunk_ratio"]),
        lateral_offset=float(record["lateral_offset_from_start_m"]),
        chunk_distance=float(record["first_chunk_distance_m"]),
        fallback=False,
    )


def _prepare_output(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {path}; pass --overwrite")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _parent_failure_result(
    episode: dict[str, Any],
    *,
    status: str,
    elapsed_seconds: float,
    message: str,
) -> dict[str, Any]:
    return {
        "output_index": int(episode["output_index"]),
        "dataset_episode_index": int(episode["dataset_episode_index"]),
        "status": status,
        "geometry_modified": False,
        "parent_wall_seconds": elapsed_seconds,
        "message": message,
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
