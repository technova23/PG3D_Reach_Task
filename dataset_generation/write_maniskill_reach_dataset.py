from __future__ import annotations

import argparse
import contextlib
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pg3d.envs.maniskill_adapter import adapt_observation, register_pg3d_reach_envs
from pg3d.envs.maniskill_adapter.dataset import (
    DEFAULT_WORKSPACE_BOUNDS,
    ActionMode,
    PointCloudCropConfig,
    ReachEpisodeData,
    git_commit_info,
    observation_to_dataset_row,
    write_reach_zarr,
)
from pg3d.envs.maniskill_adapter.reach_config import REACH_TASK_SPECS, reach_task_metadata
from pg3d.policies.dp3.goal_markers import goal_marker_points as _goal_marker_points
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
    to_numpy as _to_numpy,
)
from pg3d.utils.serialization import jsonable as _jsonable


@dataclass(frozen=True)
class PointCloudSaliencyConfig:
    goal_marker_points: int = 16
    goal_marker_radius: float = 0.045
    tcp_marker_points: int = 32
    tcp_marker_radius: float = 0.025

    def to_json(self) -> dict[str, Any]:
        return {
            "goal_marker_points": int(self.goal_marker_points),
            "goal_marker_radius": float(self.goal_marker_radius),
            "tcp_marker_points": int(self.tcp_marker_points),
            "tcp_marker_radius": float(self.tcp_marker_radius),
        }


@dataclass(frozen=True)
class TrajectoryFamilySpec:
    family_id: int
    name: str
    lateral_scale: float
    vertical_scale: float
    ratios: tuple[float, ...]
    min_curve_multiplier: float = 1.0
    lateral_jitter: float = 0.08
    vertical_jitter: float = 0.05


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        from mani_skill.examples.motionplanning.panda.motionplanner import (
            PandaArmMotionPlanningSolver,
        )
    except Exception as exc:
        print(
            f"Failed to import ManiSkill motion-planning stack: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print(
            "Install with: "
            "uv sync --extra cu129 --extra maniskill --group dev --group notebooks",
            file=sys.stderr,
        )
        return 2
    return run_generation(
        args,
        planner_cls=PandaArmMotionPlanningSolver,
        register_envs=register_pg3d_reach_envs,
        planner_name="PandaArmMotionPlanningSolver.move_to_pose_with_screw",
    )


def run_generation(
    args: argparse.Namespace,
    *,
    planner_cls: Any,
    register_envs: Any,
    planner_name: str,
) -> int:
    """Shared reach data-generation loop. Robot-specific bits (planner, env
    registration, planner label) are injected so multiple robots (Panda, xArm7)
    reuse the same sampling/replay/zarr pipeline."""
    try:
        import gymnasium as gym
        import mani_skill
        import mani_skill.envs  # noqa: F401
        import sapien
    except Exception as exc:
        print(
            f"Failed to import ManiSkill stack: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print(
            "Install with: "
            "uv sync --extra cu129 --extra maniskill --group dev --group notebooks",
            file=sys.stderr,
        )
        return 2

    register_envs()
    crop_config = PointCloudCropConfig(
        bounds=np.asarray(args.workspace_bounds),
        num_points=args.num_points,
        robot_point_fraction=args.robot_point_fraction,
    )
    env_kwargs = _env_kwargs(args)
    env: Any | None = None
    episodes: list[ReachEpisodeData] = []
    skipped: list[dict[str, Any]] = []
    family_specs = _trajectory_variant_specs(args.trajectory_variants_per_reset)
    print(
        "generation config: "
        f"num_demos={args.num_demos} max_attempts={args.max_attempts} "
        f"seed_start={args.seed_start} variants_per_reset={args.trajectory_variants_per_reset} "
        f"waypoint_attempts={args.waypoint_attempts}",
        flush=True,
    )
    print(
        "trajectory families: "
        + ", ".join(f"{spec.family_id}:{spec.name}" for spec in family_specs),
        flush=True,
    )
    if args.num_demos < args.trajectory_variants_per_reset:
        print(
            "progress note: --num-demos is smaller than --trajectory-variants-per-reset, "
            f"so only the first {args.num_demos} replayed episodes will be saved. "
            f"Use --num-demos {args.trajectory_variants_per_reset} to save all families "
            "for one start-goal pair.",
            flush=True,
        )
    try:
        env = gym.make(args.env_id, **env_kwargs)
        attempt = 0
        while len(episodes) < args.num_demos and attempt < args.max_attempts:
            seed = args.seed_start + attempt
            attempt += 1
            print(
                f"[seed {seed}] reset {attempt}/{args.max_attempts}; "
                f"collected={len(episodes)}/{args.num_demos}",
                flush=True,
            )
            new_episodes = _collect_multimodal_episodes(
                env=env,
                seed=seed,
                env_id=args.env_id,
                action_mode=args.action_mode,
                crop_config=crop_config,
                max_steps=args.max_steps_per_demo,
                hold_steps=args.hold_steps,
                settle_steps=args.settle_steps,
                gripper_open=args.gripper_open,
                sapien=sapien,
                planner_cls=planner_cls,
                variants_per_reset=args.trajectory_variants_per_reset,
                waypoint_attempts=args.waypoint_attempts,
                min_base_clearance=args.min_base_clearance,
                table_margin=args.table_margin,
                waypoint_xy_noise=args.waypoint_xy_noise,
                waypoint_z_noise=args.waypoint_z_noise,
                lateral_z_offset=args.lateral_z_offset,
                vertical_lateral_offset=args.vertical_lateral_offset,
                min_curve_offset=args.min_curve_offset,
                max_joint_step=args.max_joint_step,
                max_joint_accel=args.max_joint_accel,
                max_raw_plan_multiplier=args.max_raw_plan_multiplier,
                progress_interval=args.progress_interval,
                min_feasible_families=args.min_feasible_families,
                randomize_start=args.randomize_start,
                start_bounds=_start_workspace_bounds(
                    args.env_id,
                    args.start_bounds,
                    reach_workspace_bounds=args.reach_workspace_bounds,
                ),
                reach_workspace_bounds=args.reach_workspace_bounds,
                start_sample_attempts=args.start_sample_attempts,
                min_start_goal_distance=args.min_start_goal_distance,
                acceptance_success_distance=args.acceptance_success_distance,
                saliency_config=_point_cloud_saliency_config(args),
                require_complete_variant_set=not args.allow_partial_variant_sets,
                suppress_planner_output=not args.show_planner_output,
                viewer_step_delay=args.viewer_step_delay if args.viewer else 0.0,
            )
            if not new_episodes:
                print(f"[seed {seed}] skipped: no complete feasible variant set", flush=True)
                skipped.append({"seed": seed, "reason": "planner_failed_or_empty"})
                continue
            for episode in new_episodes:
                if len(episodes) >= args.num_demos:
                    break
                if not args.keep_failures and not bool(episode.metadata.get("success", False)):
                    skipped.append(
                        {
                            "seed": seed,
                            "reason": "unsuccessful_replay",
                            "trajectory_family": episode.metadata.get("trajectory_family"),
                            "final_distance": episode.metadata.get("final_distance"),
                        }
                    )
                    continue
                episodes.append(episode)
                print(
                    "demo "
                    f"{len(episodes)}/{args.num_demos}: seed={seed} "
                    f"variant={episode.metadata.get('trajectory_family')} "
                    f"steps={episode.state.shape[0]} "
                    f"hold={episode.metadata.get('hold_steps_recorded')} "
                    f"success={episode.metadata.get('success')} "
                    f"final_distance={episode.metadata.get('final_distance'):.4f}",
                    flush=True,
                )
    except Exception as exc:
        print(f"Failed to write reach dataset: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if env is not None:
            _hold_viewer(env, args.viewer_hold_seconds if args.viewer else 0.0)
            env.close()

    if not episodes:
        print("No usable reach demonstrations were collected.", file=sys.stderr)
        return 1

    repo_root = Path(__file__).resolve().parents[1]
    dataset_stats = _dataset_stats(episodes)
    metadata = {
        "env_id": args.env_id,
        "env_kwargs": env_kwargs,
        "num_requested_demos": args.num_demos,
        "num_collected_demos": len(episodes),
        "num_attempts": attempt,
        "skipped": skipped,
        "seed_start": args.seed_start,
        "action_mode": args.action_mode,
        "control_mode": args.control_mode,
        "hold_steps": args.hold_steps,
        "settle_steps": args.settle_steps,
        "start_sampling": {
            "randomize_start": args.randomize_start,
            "start_bounds": _start_workspace_bounds(
                args.env_id,
                args.start_bounds,
                reach_workspace_bounds=args.reach_workspace_bounds,
            ).tolist(),
            "reach_workspace_bounds": args.reach_workspace_bounds.tolist(),
            "start_sample_attempts": args.start_sample_attempts,
            "min_start_goal_distance": args.min_start_goal_distance,
            "min_base_clearance": args.min_base_clearance,
            "table_margin": args.table_margin,
            "note": (
                "Starts are sampled as Cartesian TCP poses, accepted only when the ManiSkill "
                "Panda motion planner can reach them from the reset configuration. The script "
                "also rejects starts and goals that violate the configured base clearance or "
                "XY table-margin inset."
            ),
        },
        "trajectory_generation": {
            "type": "multimodal_waypoint_planning",
            "variants_per_reset": args.trajectory_variants_per_reset,
            "waypoint_attempts": args.waypoint_attempts,
            "min_base_clearance": args.min_base_clearance,
            "table_margin": args.table_margin,
            "waypoint_xy_noise": args.waypoint_xy_noise,
            "waypoint_z_noise": args.waypoint_z_noise,
            "lateral_z_offset": args.lateral_z_offset,
            "vertical_lateral_offset": args.vertical_lateral_offset,
            "min_curve_offset": args.min_curve_offset,
            "max_joint_step": args.max_joint_step,
            "max_joint_accel": args.max_joint_accel,
            "max_raw_plan_multiplier": args.max_raw_plan_multiplier,
            "acceptance_success_distance": args.acceptance_success_distance,
            "allow_partial_variant_sets": args.allow_partial_variant_sets,
            "show_planner_output": args.show_planner_output,
            "planner": planner_name,
            "trajectory_families": [
                {
                    "trajectory_family_id": spec.family_id,
                    "trajectory_family_name": spec.name,
                    "lateral_scale": spec.lateral_scale,
                    "vertical_scale": spec.vertical_scale,
                    "ratios": list(spec.ratios),
                    "min_curve_offset_multiplier": spec.min_curve_multiplier,
                }
                for spec in _trajectory_variant_specs(args.trajectory_variants_per_reset)
            ],
            "note": (
                "Only trajectory generation is changed; env, observations, crop, "
                "and zarr writer are unchanged."
            ),
        },
        "task": reach_task_metadata(args.env_id),
        "crop": crop_config.to_json(),
        "point_cloud_saliency": _point_cloud_saliency_config(args).to_json(),
        "dataset_stats": dataset_stats,
        "camera": {
            "obs_mode": args.obs_mode,
            "shader": args.shader,
            "source": "ManiSkill default sensor config for PG3DReach",
        },
        "versions": {
            "mani_skill": getattr(mani_skill, "__version__", None),
            "sapien": getattr(sapien, "__version__", None),
        },
        "git": {
            "pg3d": git_commit_info(repo_root),
            "external_dp3": git_commit_info(repo_root / "external" / "dp3"),
        },
    }
    summary = write_reach_zarr(args.output, episodes, metadata=metadata, overwrite=args.overwrite)
    alias_arrays = _ensure_goal_observation_aliases(args.output)
    summary.get("arrays", {}).update(alias_arrays)
    family_arrays = _ensure_trajectory_family_arrays(args.output, episodes)
    summary.get("arrays", {}).update(family_arrays)
    print(f"saved dataset: {args.output}")
    print(f"summary: {summary}")
    print("dataset_stats: " + json_dumps(dataset_stats))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a structured multimodal pg3d ManiSkill reach dataset."
    )
    parser.add_argument("--env-id", default="PG3DReach-BalancedWorkspace-v0")
    parser.add_argument("--num-demos", type=int, default=4800)
    parser.add_argument("--max-attempts", type=int, default=500)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--obs-mode", default="pointcloud", choices=["pointcloud"])
    parser.add_argument("--action-mode", default="abs_joint", choices=["abs_joint", "delta_joint"])
    parser.add_argument("--control-mode", default="pd_joint_pos")
    parser.add_argument("--robot-uid", default="panda")
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument(
        "--robot-point-fraction",
        type=float,
        default=0.25,
        help=(
            "minimum fraction of saved point-cloud slots reserved for robot-mask points "
            "when enough robot points are available"
        ),
    )
    parser.add_argument(
        "--workspace-bounds",
        type=float,
        nargs=6,
        default=DEFAULT_WORKSPACE_BOUNDS.reshape(-1).tolist(),
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX", "Z_MIN", "Z_MAX"),
    )
    parser.add_argument(
        "--start-bounds",
        type=float,
        nargs=6,
        default=None,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX", "Z_MIN", "Z_MAX"),
        help=(
            "Cartesian TCP start sampling bounds. Defaults to --reach-workspace-bounds "
            "when unset."
        ),
    )
    parser.add_argument(
        "--reach-workspace-bounds",
        type=float,
        nargs=6,
        default=None,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX", "Z_MIN", "Z_MAX"),
        help=(
            "Wider table-safe Cartesian bounds used for goal sampling, start sampling, "
            "and waypoint detours. Defaults to a conservative Panda-reachable table box."
        ),
    )
    parser.add_argument(
        "--randomize-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="sample a reachable TCP start pose before generating variants",
    )
    parser.add_argument(
        "--start-sample-attempts",
        type=int,
        default=60,
        help="maximum reachable start pose samples to try per reset",
    )
    parser.add_argument(
        "--min-start-goal-distance",
        type=float,
        default=0.16,
        help="minimum Euclidean distance between sampled TCP start and goal in meters",
    )
    parser.add_argument("--sim-backend", default="auto")
    parser.add_argument("--render-backend", default="gpu")
    parser.add_argument("--shader", default="default")
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="open the ManiSkill human viewer and render frames during collection",
    )
    parser.add_argument(
        "--viewer-step-delay",
        type=float,
        default=0.0,
        help="seconds to sleep after each viewer frame; useful when watching collection live",
    )
    parser.add_argument(
        "--viewer-hold-seconds",
        type=float,
        default=0.0,
        help="seconds to keep the viewer open before closing the environment",
    )
    parser.add_argument("--max-steps-per-demo", type=int, default=100)
    parser.add_argument("--hold-steps", type=int, default=8)
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=30,
        help=(
            "extra final-target replay steps reserved inside --max-steps-per-demo "
            "before declaring a planned trajectory unsuccessful"
        ),
    )
    parser.add_argument("--gripper-open", type=float, default=0.04)
    parser.add_argument(
        "--trajectory-variants-per-reset",
        type=int,
        default=12,
        help=(
            "number of waypoint-conditioned trajectory variants to try for each identical "
            "environment reset/start/goal"
        ),
    )
    parser.add_argument(
        "--allow-partial-variant-sets",
        action="store_true",
        help=(
            "keep successful variants from a seed even if one requested trajectory family fails; "
            "by default, incomplete seed/start groups are skipped so datasets do not silently "
            "miss e.g. downward_arc for a seed"
        ),
    )
    parser.add_argument(
        "--show-planner-output",
        action="store_true",
        help=(
            "show ManiSkill planner stdout/stderr during expected retry failures; by default "
            "the writer suppresses repeated messages such as 'screw plan failed'"
        ),
    )
    parser.add_argument(
        "--waypoint-attempts",
        type=int,
        default=80,
        help="maximum waypoint samples to try per trajectory family",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=10,
        help="print waypoint-search progress every N candidate attempts per family",
    )
    parser.add_argument(
        "--min-feasible-families",
        type=int,
        default=4,
        help=(
            "minimum number of feasible trajectory families required to replay a start-goal "
            "pair; lower default keeps data generation moving while families are tuned"
        ),
    )
    parser.add_argument(
        "--acceptance-success-distance",
        type=float,
        default=0.030,
        help=(
            "writer-side success tolerance in meters; keeps near-threshold controller "
            "misses that are visually/replay close to the goal"
        ),
    )
    parser.add_argument(
        "--min-base-clearance",
        type=float,
        default=0.06,
        help=(
            "minimum horizontal XY distance from the robot base for sampled starts, goals, "
            "and waypoints, in meters"
        ),
    )
    parser.add_argument(
        "--table-margin",
        type=float,
        default=0.0,
        help=(
            "XY margin inset from the task/workspace bounds for sampled starts, goals, "
            "and waypoints, in meters"
        ),
    )
    parser.add_argument(
        "--waypoint-xy-noise",
        type=float,
        default=0.04,
        help="half-width of uniform XY perturbation added to sampled waypoints, in meters",
    )
    parser.add_argument(
        "--waypoint-z-noise",
        type=float,
        default=0.025,
        help="half-width of uniform Z perturbation added to sampled waypoints, in meters",
    )
    parser.add_argument(
        "--lateral-z-offset",
        type=float,
        default=0.15,
        help="maximum extra absolute Z offset for lateral curve waypoints, in meters",
    )
    parser.add_argument(
        "--vertical-lateral-offset",
        type=float,
        default=0.10,
        help="maximum lateral XY offset for upward/downward arc waypoints, in meters",
    )
    parser.add_argument(
        "--min-curve-offset",
        type=float,
        default=0.10,
        help="minimum maximum Cartesian deviation from the start-goal line for non-shallow families",
    )
    parser.add_argument(
        "--max-joint-step",
        type=float,
        default=2.50,
        help="reject planned waypoint trajectories with larger consecutive joint-space steps",
    )
    parser.add_argument(
        "--max-joint-accel",
        type=float,
        default=2.50,
        help="reject planned waypoint trajectories with larger second-difference joint-space jumps",
    )
    parser.add_argument(
        "--max-raw-plan-multiplier",
        type=float,
        default=4.0,
        help=(
            "reject raw planner paths longer than this multiple of the replay movement budget; "
            "very large raw paths usually fail after compression"
        ),
    )
    parser.add_argument(
        "--goal-marker-points",
        type=int,
        default=16,
        help="number of saved point-cloud slots reserved for structured goal marker points",
    )
    parser.add_argument(
        "--goal-marker-radius",
        type=float,
        default=0.045,
        help="radius of synthetic point-cloud goal marker samples in meters",
    )
    parser.add_argument(
        "--tcp-marker-points",
        type=int,
        default=32,
        help="number of saved point-cloud slots reserved for synthetic TCP/end-effector marker points",
    )
    parser.add_argument(
        "--tcp-marker-radius",
        type=float,
        default=0.025,
        help="radius of synthetic point-cloud TCP marker samples in meters",
    )
    parser.add_argument("--keep-failures", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/pg3d_reach_balanced.zarr"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.seed_start is None:
        args.seed_start = int(np.random.SeedSequence().entropy % (2**31 - 1))
    args.workspace_bounds = np.asarray(args.workspace_bounds, dtype=np.float32).reshape(3, 2)
    if args.start_bounds is not None:
        args.start_bounds = np.asarray(args.start_bounds, dtype=np.float32).reshape(3, 2)
    if args.reach_workspace_bounds is None:
        args.reach_workspace_bounds = _default_reach_workspace_bounds(args.workspace_bounds)
    else:
        args.reach_workspace_bounds = np.asarray(
            args.reach_workspace_bounds, dtype=np.float32
        ).reshape(3, 2)
    args.action_mode = _action_mode(args.action_mode)
    if args.hold_steps < 0:
        raise ValueError("--hold-steps must be non-negative")
    if args.settle_steps < 0:
        raise ValueError("--settle-steps must be non-negative")
    if args.max_steps_per_demo <= 0:
        raise ValueError("--max-steps-per-demo must be positive")
    if args.settle_steps >= args.max_steps_per_demo:
        raise ValueError("--settle-steps must be smaller than --max-steps-per-demo")
    if args.num_points <= 0:
        raise ValueError("--num-points must be positive")
    if not 0.0 <= args.robot_point_fraction <= 1.0:
        raise ValueError("--robot-point-fraction must be between 0 and 1")
    if args.trajectory_variants_per_reset <= 0:
        raise ValueError("--trajectory-variants-per-reset must be positive")
    if args.waypoint_attempts <= 0:
        raise ValueError("--waypoint-attempts must be positive")
    if args.progress_interval <= 0:
        raise ValueError("--progress-interval must be positive")
    if args.min_feasible_families <= 0:
        raise ValueError("--min-feasible-families must be positive")
    if args.min_feasible_families > args.trajectory_variants_per_reset:
        raise ValueError("--min-feasible-families cannot exceed --trajectory-variants-per-reset")
    if args.acceptance_success_distance <= 0:
        raise ValueError("--acceptance-success-distance must be positive")
    if args.min_base_clearance < 0:
        raise ValueError("--min-base-clearance must be non-negative")
    if args.table_margin < 0:
        raise ValueError("--table-margin must be non-negative")
    if args.waypoint_xy_noise < 0:
        raise ValueError("--waypoint-xy-noise must be non-negative")
    if args.waypoint_z_noise < 0:
        raise ValueError("--waypoint-z-noise must be non-negative")
    if args.lateral_z_offset < 0:
        raise ValueError("--lateral-z-offset must be non-negative")
    if args.vertical_lateral_offset < 0:
        raise ValueError("--vertical-lateral-offset must be non-negative")
    if args.min_curve_offset < 0:
        raise ValueError("--min-curve-offset must be non-negative")
    if args.max_joint_step <= 0:
        raise ValueError("--max-joint-step must be positive")
    if args.max_joint_accel <= 0:
        raise ValueError("--max-joint-accel must be positive")
    if args.max_raw_plan_multiplier < 1.0:
        raise ValueError("--max-raw-plan-multiplier must be at least 1.0")
    if args.goal_marker_points < 0:
        raise ValueError("--goal-marker-points must be non-negative")
    if args.goal_marker_radius < 0:
        raise ValueError("--goal-marker-radius must be non-negative")
    if args.tcp_marker_points < 0:
        raise ValueError("--tcp-marker-points must be non-negative")
    if args.tcp_marker_radius < 0:
        raise ValueError("--tcp-marker-radius must be non-negative")
    if args.start_sample_attempts <= 0:
        raise ValueError("--start-sample-attempts must be positive")
    if args.min_start_goal_distance < 0:
        raise ValueError("--min-start-goal-distance must be non-negative")
    if args.viewer_step_delay < 0:
        raise ValueError("--viewer-step-delay must be non-negative")
    if args.viewer_hold_seconds < 0:
        raise ValueError("--viewer-hold-seconds must be non-negative")
    return args


def _env_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    goal_center, goal_half_extents = _bounds_center_half_extents(args.reach_workspace_bounds)
    return {
        "obs_mode": args.obs_mode,
        "control_mode": args.control_mode,
        "render_mode": "human" if args.viewer else None,
        "robot_uids": args.robot_uid,
        "num_envs": 1,
        "max_episode_steps": int(args.max_steps_per_demo),
        "sim_backend": args.sim_backend,
        "render_backend": args.render_backend,
        "sensor_configs": {"shader_pack": args.shader},
        "goal_center": tuple(float(value) for value in goal_center),
        "goal_half_extents": tuple(float(value) for value in goal_half_extents),
        "goal_regions": (),
    }


def _point_cloud_saliency_config(args: argparse.Namespace) -> PointCloudSaliencyConfig:
    return PointCloudSaliencyConfig(
        goal_marker_points=args.goal_marker_points,
        goal_marker_radius=args.goal_marker_radius,
        tcp_marker_points=args.tcp_marker_points,
        tcp_marker_radius=args.tcp_marker_radius,
    )


def _collect_episode(
    *,
    env: Any,
    seed: int,
    env_id: str,
    action_mode: ActionMode,
    crop_config: PointCloudCropConfig,
    max_steps: int,
    hold_steps: int,
    settle_steps: int,
    acceptance_success_distance: float,
    gripper_open: float,
    sapien: Any,
    planner_cls: Any,
    saliency_config: PointCloudSaliencyConfig | None = None,
    viewer_step_delay: float = 0.0,
    suppress_planner_output: bool = True,
) -> ReachEpisodeData | None:
    obs, info = env.reset(seed=seed, options={"reconfigure": True})
    _render_viewer_frame(env, viewer_step_delay)
    unwrapped = env.unwrapped
    goal_pose = _goal_pose(unwrapped, sapien)
    planner = planner_cls(
        env,
        debug=False,
        vis=False,
        base_pose=unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
    )
    try:
        plan = _move_to_pose_with_screw(
            planner,
            goal_pose,
            suppress_output=suppress_planner_output,
        )
    finally:
        planner.close()
    if plan == -1 or "position" not in plan:
        return None

    return _replay_planned_positions_as_episode(
        env=env,
        env_id=env_id,
        action_mode=action_mode,
        crop_config=crop_config,
        max_steps=max_steps,
        hold_steps=hold_steps,
        settle_steps=settle_steps,
        acceptance_success_distance=0.025,
        gripper_open=gripper_open,
        obs=obs,
        info=info,
        positions=np.asarray(plan["position"], dtype=np.float32),
        saliency_config=saliency_config,
        viewer_step_delay=viewer_step_delay,
        metadata={
            "seed": seed,
            "planner_status": str(plan.get("status", "unknown")),
            "trajectory_family": "direct",
            "trajectory_type": -1,
            "trajectory_waypoints": [],
        },
    )


def _collect_multimodal_episodes(
    *,
    env: Any,
    seed: int,
    env_id: str,
    action_mode: ActionMode,
    crop_config: PointCloudCropConfig,
    max_steps: int,
    hold_steps: int,
    settle_steps: int,
    gripper_open: float,
    sapien: Any,
    planner_cls: Any,
    variants_per_reset: int,
    waypoint_attempts: int,
    min_base_clearance: float,
    table_margin: float,
    waypoint_xy_noise: float,
    waypoint_z_noise: float,
    lateral_z_offset: float,
    vertical_lateral_offset: float,
    min_curve_offset: float,
    max_joint_step: float,
    max_joint_accel: float,
    max_raw_plan_multiplier: float,
    progress_interval: int,
    min_feasible_families: int,
    randomize_start: bool,
    start_bounds: np.ndarray,
    reach_workspace_bounds: np.ndarray,
    start_sample_attempts: int,
    min_start_goal_distance: float,
    acceptance_success_distance: float,
    saliency_config: PointCloudSaliencyConfig,
    require_complete_variant_set: bool,
    suppress_planner_output: bool,
    viewer_step_delay: float = 0.0,
) -> list[ReachEpisodeData]:
    obs, info = env.reset(seed=seed, options={"reconfigure": True})
    _render_viewer_frame(env, viewer_step_delay)
    unwrapped = env.unwrapped
    goal_pose = _goal_pose(unwrapped, sapien)
    reset_tcp_pose = _tcp_pose(unwrapped)
    reset_qpos = _get_robot_qpos(env)
    rng = np.random.default_rng(seed)
    robot_base_position = _robot_base_position(unwrapped)
    waypoint_bounds = _inset_xy_bounds(
        _waypoint_workspace_bounds(
            env_id,
            crop_config,
            reach_workspace_bounds=reach_workspace_bounds,
        ),
        table_margin,
    )
    start_sampling_bounds = _inset_xy_bounds(start_bounds, table_margin)
    if waypoint_bounds is None or start_sampling_bounds is None:
        return []
    print(
        f"[seed {seed}] waypoint_bounds="
        f"{waypoint_bounds.astype(np.float32).tolist()} "
        f"start_bounds={start_sampling_bounds.astype(np.float32).tolist()}",
        flush=True,
    )
    goal_xyz = np.asarray(goal_pose.p, dtype=np.float64).reshape(-1, 3)[0]
    print(
        f"[seed {seed}] goal={goal_xyz.astype(np.float32).tolist()} "
        "checking workspace/base/table feasibility",
        flush=True,
    )
    if not _is_waypoint_valid(
        waypoint=goal_xyz,
        workspace_bounds=waypoint_bounds,
        robot_base_position=robot_base_position,
        min_base_clearance=min_base_clearance,
    ):
        print(f"[seed {seed}] rejected goal: outside feasible workspace filters", flush=True)
        return []

    planner = planner_cls(
        env,
        debug=False,
        vis=False,
        base_pose=unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
    )
    try:
        start_sample = _sample_reachable_start(
            env=env,
            planner=planner,
            sapien=sapien,
            rng=rng,
            reset_qpos=reset_qpos,
            reset_tcp_pose=reset_tcp_pose,
            goal_pose=goal_pose,
            start_bounds=start_sampling_bounds,
            randomize_start=randomize_start,
            max_attempts=start_sample_attempts,
            min_start_goal_distance=min_start_goal_distance,
            min_base_clearance=min_base_clearance,
            suppress_planner_output=suppress_planner_output,
        )
        if start_sample is None:
            print(f"[seed {seed}] rejected: no reachable start sample", flush=True)
            return []
        start_qpos, start_tcp_pose, start_metadata = start_sample
        goal_pose = _pose_with_orientation(
            sapien,
            position=goal_xyz,
            quat=start_tcp_pose[3:7],
        )
        print(
            f"[seed {seed}] start accepted after attempt={start_metadata.get('attempt')} "
            f"tcp={start_tcp_pose[:3].astype(np.float32).tolist()} "
            "goal_orientation=start_tcp_orientation",
            flush=True,
        )
        goal_reachability = _plan_to_pose(
            planner=planner,
            env=env,
            pose=goal_pose,
            start_qpos=start_qpos,
            suppress_planner_output=suppress_planner_output,
        )
        if goal_reachability is None:
            print(
                f"[seed {seed}] rejected: direct goal plan failed from sampled start",
                flush=True,
            )
            return []
        print(
            f"[seed {seed}] goal reachable from start: status={goal_reachability[1]}",
            flush=True,
        )
        start_metadata = {
            **start_metadata,
            "goal_reachable_from_start": True,
            "goal_reachability_status": goal_reachability[1],
        }
        _set_robot_qpos(env, start_qpos)
        _set_start_site_pose(env, start_tcp_pose[:3])
        variants = generate_multimodal_waypoints(
            current_tcp_pose=start_tcp_pose,
            goal_pose=goal_pose,
            workspace_bounds=waypoint_bounds,
            robot_base_position=robot_base_position,
            planner=planner,
            env=env,
            sapien=sapien,
            rng=rng,
            variants_per_reset=variants_per_reset,
            max_attempts=waypoint_attempts,
            min_base_clearance=min_base_clearance,
            waypoint_xy_noise=waypoint_xy_noise,
            waypoint_z_noise=waypoint_z_noise,
            lateral_z_offset=lateral_z_offset,
            vertical_lateral_offset=vertical_lateral_offset,
            min_curve_offset=min_curve_offset,
            max_joint_step=max_joint_step,
            max_joint_accel=max_joint_accel,
            max_raw_plan_multiplier=max_raw_plan_multiplier,
            progress_interval=progress_interval,
            max_replay_plan_steps=max(1, max_steps - settle_steps),
            seed=seed,
            start_qpos=start_qpos,
            suppress_planner_output=suppress_planner_output,
        )
    finally:
        planner.close()

    if require_complete_variant_set and len(variants) < min_feasible_families:
        found = sorted(int(variant["trajectory_type"]) for variant in variants)
        print(
            f"[seed {seed}] insufficient feasible families: found={found} "
            f"count={len(variants)}/{variants_per_reset} "
            f"required>={min_feasible_families}",
            flush=True,
        )
        return []

    episodes: list[ReachEpisodeData] = []
    for variant in variants:
        print(
            f"[seed {seed}] replay family {variant['trajectory_type']}:{variant['name']} "
            f"planned_steps={variant['positions'].shape[0]}",
            flush=True,
        )
        obs, info = env.reset(seed=seed, options={"reconfigure": False})
        _set_robot_qpos(env, start_qpos)
        _set_start_site_pose(env, start_tcp_pose[:3])
        obs, info = _refresh_obs_after_manual_qpos(
            env,
            info=info,
            gripper_open=gripper_open,
        )
        _render_viewer_frame(env, viewer_step_delay)
        episode = _replay_planned_positions_as_episode(
            env=env,
            env_id=env_id,
            action_mode=action_mode,
            crop_config=crop_config,
            max_steps=max_steps,
            hold_steps=hold_steps,
            settle_steps=settle_steps,
            acceptance_success_distance=acceptance_success_distance,
            gripper_open=gripper_open,
            obs=obs,
            info=info,
            positions=variant["positions"],
            saliency_config=saliency_config,
            viewer_step_delay=viewer_step_delay,
            metadata={
                "seed": seed,
                "planner_status": variant["planner_status"],
                "trajectory_family": variant["name"],
                "trajectory_family_name": variant["name"],
                "trajectory_family_id": variant["trajectory_type"],
                "trajectory_type": variant["trajectory_type"],
                "trajectory_waypoints": variant["waypoints"],
                "trajectory_waypoint_metadata": variant["waypoint_metadata"],
                "trajectory_quality": variant["quality"],
                "start_tcp_pose": start_tcp_pose.astype(np.float32).tolist(),
                "goal_pose": _pose_to_list(goal_pose),
                "start_sampling": start_metadata,
            },
        )
        if episode is not None:
            print(
                f"[seed {seed}] replay done {variant['trajectory_type']}:{variant['name']} "
                f"success={episode.metadata.get('success')} "
                f"final_distance={episode.metadata.get('final_distance'):.4f} "
                f"settle={episode.metadata.get('settle_steps_recorded', 0)}",
                flush=True,
            )
            episodes.append(episode)
    successful_episode_count = sum(
        bool(episode.metadata.get("success", False)) for episode in episodes
    )
    if require_complete_variant_set and successful_episode_count < min_feasible_families:
        print(
            f"[seed {seed}] rejected: replay produced {successful_episode_count}/"
            f"{len(episodes)} successful episodes; required>={min_feasible_families}",
            flush=True,
        )
        return []
    print(
        f"[seed {seed}] accepted {successful_episode_count}/{len(episodes)} "
        f"successful replayed families",
        flush=True,
    )
    return episodes


def _replay_planned_positions_as_episode(
    *,
    env: Any,
    env_id: str,
    action_mode: ActionMode,
    crop_config: PointCloudCropConfig,
    max_steps: int,
    hold_steps: int,
    settle_steps: int,
    acceptance_success_distance: float,
    gripper_open: float,
    obs: Any,
    info: Any,
    positions: np.ndarray,
    saliency_config: PointCloudSaliencyConfig | None,
    metadata: dict[str, Any],
    viewer_step_delay: float = 0.0,
) -> ReachEpisodeData | None:
    unwrapped = env.unwrapped
    rows: list[dict[str, np.ndarray]] = []
    successes: list[bool] = []
    distances: list[float] = []
    first_success_step: int | None = None
    pre_hold_final_distance: float | None = None
    hold_steps_recorded = 0
    settle_steps_recorded = 0
    planned_step_limit = max(1, max_steps - settle_steps)
    for planned_qpos in positions[:planned_step_limit]:
        sim_action = _format_sim_action(env, planned_qpos)
        row = _dataset_row_from_obs(
            obs=obs,
            info=info,
            env=env,
            env_id=env_id,
            sim_action=sim_action,
            action_mode=action_mode,
            crop_config=crop_config,
            saliency_config=saliency_config,
        )
        obs, _reward, terminated, truncated, info = env.step(sim_action)
        _render_viewer_frame(env, viewer_step_delay)
        success = _bool_info(info, "success")
        distance = _float_info(info, "tcp_to_goal_dist", default=_tcp_to_goal_distance(unwrapped))
        row["success"] = np.asarray(success, dtype=bool)
        rows.append(row)
        successes.append(success)
        distances.append(distance)
        if success:
            first_success_step = len(rows)
            pre_hold_final_distance = distance
            break
        if _bool_any(terminated) or _bool_any(truncated):
            break

    if not rows:
        return None

    final_planned_qpos = positions[-1]
    while (
        first_success_step is None
        and settle_steps_recorded < settle_steps
        and len(rows) < max_steps
    ):
        sim_action = _format_sim_action(env, final_planned_qpos)
        row = _dataset_row_from_obs(
            obs=obs,
            info=info,
            env=env,
            env_id=env_id,
            sim_action=sim_action,
            action_mode=action_mode,
            crop_config=crop_config,
            saliency_config=saliency_config,
        )
        obs, _reward, _terminated, truncated, info = env.step(sim_action)
        _render_viewer_frame(env, viewer_step_delay)
        success = _bool_info(info, "success")
        distance = _float_info(info, "tcp_to_goal_dist", default=_tcp_to_goal_distance(unwrapped))
        row["success"] = np.asarray(success, dtype=bool)
        rows.append(row)
        successes.append(success)
        distances.append(distance)
        settle_steps_recorded += 1
        if success:
            first_success_step = len(rows)
            pre_hold_final_distance = distance
            break
        if _bool_any(truncated):
            break

    while (
        first_success_step is not None
        and hold_steps_recorded < hold_steps
        and len(rows) < max_steps
    ):
        sim_action = _hold_sim_action(env, gripper_open=gripper_open)
        row = _dataset_row_from_obs(
            obs=obs,
            info=info,
            env=env,
            env_id=env_id,
            sim_action=sim_action,
            action_mode=action_mode,
            crop_config=crop_config,
            saliency_config=saliency_config,
        )
        obs, _reward, _terminated, truncated, info = env.step(sim_action)
        _render_viewer_frame(env, viewer_step_delay)
        success = _bool_info(info, "success")
        distance = _float_info(info, "tcp_to_goal_dist", default=_tcp_to_goal_distance(unwrapped))
        row["success"] = np.asarray(success, dtype=bool)
        rows.append(row)
        successes.append(success)
        distances.append(distance)
        hold_steps_recorded += 1
        if _bool_any(truncated):
            break

    final_distance = _float_info(
        info,
        "tcp_to_goal_dist",
        default=_tcp_to_goal_distance(unwrapped),
    )
    min_distance = float(np.min(distances)) if distances else final_distance
    accepted_success = (first_success_step is not None) or (min_distance <= acceptance_success_distance)
    if accepted_success and first_success_step is None and successes:
        closest_idx = int(np.argmin(np.asarray(distances, dtype=np.float32)))
        first_success_step = closest_idx + 1
        pre_hold_final_distance = float(distances[closest_idx])
        successes[closest_idx] = True
        rows[closest_idx]["success"] = np.asarray(True, dtype=bool)
    metadata = {
        **metadata,
        "length": len(rows),
        "first_success_step": first_success_step,
        "hold_steps_requested": hold_steps,
        "hold_steps_recorded": hold_steps_recorded,
        "settle_steps_requested": settle_steps,
        "settle_steps_recorded": settle_steps_recorded,
        "pre_hold_final_distance": pre_hold_final_distance,
        "final_distance": final_distance,
        "min_distance": min_distance,
        "env_success": first_success_step is not None and min_distance <= 0.025,
        "acceptance_success_distance": acceptance_success_distance,
        "success": accepted_success,
    }
    return ReachEpisodeData(
        state=np.stack([row["state"] for row in rows], axis=0),
        action=np.stack([row["action"] for row in rows], axis=0),
        sim_action=np.stack([row["sim_action"] for row in rows], axis=0),
        point_cloud=np.stack([row["point_cloud"] for row in rows], axis=0),
        robot_mask=np.stack([row["robot_mask"] for row in rows], axis=0),
        point_valid_mask=np.stack([row["point_valid_mask"] for row in rows], axis=0),
        target_position=np.stack([row["target_position"] for row in rows], axis=0),
        tcp_pose=np.stack([row["tcp_pose"] for row in rows], axis=0),
        success=np.asarray(successes, dtype=bool),
        metadata=metadata,
    )


def generate_multimodal_waypoints(
    *,
    current_tcp_pose: np.ndarray,
    goal_pose: Any,
    workspace_bounds: np.ndarray,
    robot_base_position: np.ndarray,
    planner: Any,
    env: Any,
    sapien: Any,
    rng: np.random.Generator,
    variants_per_reset: int,
    max_attempts: int,
    min_base_clearance: float,
    waypoint_xy_noise: float,
    waypoint_z_noise: float,
    lateral_z_offset: float,
    vertical_lateral_offset: float,
    min_curve_offset: float,
    max_joint_step: float,
    max_joint_accel: float,
    max_raw_plan_multiplier: float,
    progress_interval: int,
    max_replay_plan_steps: int,
    seed: int,
    start_qpos: np.ndarray,
    suppress_planner_output: bool = True,
) -> list[dict[str, Any]]:
    """Sample waypoint-conditioned variants while keeping the official planner in charge."""
    start = np.asarray(current_tcp_pose[:3], dtype=np.float64)
    goal = np.asarray(goal_pose.p, dtype=np.float64).reshape(-1, 3)[0]
    goal_quat = np.asarray(goal_pose.q, dtype=np.float64).reshape(4)
    delta = goal - start
    distance = float(np.linalg.norm(delta))
    if distance < 1e-6:
        return []

    specs = _trajectory_variant_specs(variants_per_reset)
    variants: list[dict[str, Any]] = []
    failed_family_count = 0
    max_failed_families = 5
    for spec in specs:
        selected_variant: dict[str, Any] | None = None
        geometry_candidates = 0
        planned_candidates = 0
        planner_failures = 0
        quality_candidates = 0
        resampled_long_candidates = 0
        raw_too_long_candidates = 0
        print(
            f"[seed {seed}] family {spec.family_id}:{spec.name} search started "
            f"({max_attempts} candidates)",
            flush=True,
        )
        for attempt_idx in range(1, max_attempts + 1):
            waypoints, waypoint_metadata = _sample_waypoint_set(
                start=start,
                goal=goal,
                spec=spec,
                workspace_bounds=workspace_bounds,
                robot_base_position=robot_base_position,
                min_base_clearance=min_base_clearance,
                xy_noise=waypoint_xy_noise,
                z_noise=waypoint_z_noise,
                lateral_z_offset=lateral_z_offset,
                vertical_lateral_offset=vertical_lateral_offset,
                rng=rng,
            )
            if waypoints is None:
                if attempt_idx % progress_interval == 0 or attempt_idx == max_attempts:
                    print(
                        f"[seed {seed}] family {spec.family_id}:{spec.name} "
                        f"attempt {attempt_idx}/{max_attempts}: "
                        f"geometry_ok={geometry_candidates} "
                        f"planner_ok={planned_candidates} planner_fail={planner_failures} quality_ok={quality_candidates} "
                        f"resampled_long={resampled_long_candidates} raw_too_long={raw_too_long_candidates}",
                        flush=True,
                    )
                continue
            geometry_candidates += 1

            waypoint_poses = [
                sapien.Pose(p=waypoint.astype(np.float32), q=goal_quat.astype(np.float32))
                for waypoint in waypoints
            ]
            plan_result = _plan_multisegment_trajectory(
                planner=planner,
                env=env,
                poses=[*waypoint_poses, goal_pose],
                start_qpos=start_qpos,
                suppress_planner_output=suppress_planner_output,
            )
            if plan_result is None:
                planner_failures += 1
                if attempt_idx % progress_interval == 0 or attempt_idx == max_attempts:
                    print(
                        f"[seed {seed}] family {spec.family_id}:{spec.name} "
                        f"attempt {attempt_idx}/{max_attempts}: "
                        f"geometry_ok={geometry_candidates} "
                        f"planner_ok={planned_candidates} planner_fail={planner_failures} quality_ok={quality_candidates} "
                        f"resampled_long={resampled_long_candidates} raw_too_long={raw_too_long_candidates}",
                        flush=True,
                    )
                continue
            planned_candidates += 1
            positions, planner_status = plan_result
            raw_plan_steps = int(positions.shape[0])
            max_raw_plan_steps = int(np.ceil(max_replay_plan_steps * max_raw_plan_multiplier))
            if raw_plan_steps > max_raw_plan_steps:
                raw_too_long_candidates += 1
                if attempt_idx % progress_interval == 0 or attempt_idx == max_attempts:
                    print(
                        f"[seed {seed}] family {spec.family_id}:{spec.name} "
                        f"attempt {attempt_idx}/{max_attempts}: "
                        f"geometry_ok={geometry_candidates} "
                        f"planner_ok={planned_candidates} planner_fail={planner_failures} quality_ok={quality_candidates} "
                        f"resampled_long={resampled_long_candidates} raw_too_long={raw_too_long_candidates} "
                        f"last_raw_steps={raw_plan_steps} max_raw={max_raw_plan_steps}",
                        flush=True,
                    )
                continue
            if raw_plan_steps > max_replay_plan_steps:
                resampled_long_candidates += 1
                positions = _resample_joint_positions(
                    positions,
                    max_steps=max_replay_plan_steps,
                )
            quality = _trajectory_quality(
                positions=positions,
                start=start,
                goal=goal,
                waypoints=waypoints,
                min_curve_offset=min_curve_offset * spec.min_curve_multiplier,
                max_joint_step=max_joint_step,
                max_joint_accel=max_joint_accel,
            )
            if quality is None:
                if attempt_idx % progress_interval == 0 or attempt_idx == max_attempts:
                    print(
                        f"[seed {seed}] family {spec.family_id}:{spec.name} "
                        f"attempt {attempt_idx}/{max_attempts}: "
                        f"geometry_ok={geometry_candidates} "
                        f"planner_ok={planned_candidates} planner_fail={planner_failures} quality_ok={quality_candidates} "
                        f"resampled_long={resampled_long_candidates} raw_too_long={raw_too_long_candidates}",
                        flush=True,
                    )
                continue
            quality_candidates += 1
            selected_variant = {
                "name": spec.name,
                "trajectory_type": spec.family_id,
                "positions": positions,
                "planner_status": planner_status,
                "raw_plan_steps": raw_plan_steps,
                "resampled_plan_steps": int(positions.shape[0]),
                "waypoints": [waypoint.astype(np.float32).tolist() for waypoint in waypoints],
                "waypoint_metadata": waypoint_metadata,
                "quality": quality,
            }
            print(
                f"[seed {seed}] family {spec.family_id}:{spec.name} first feasible "
                f"candidate at attempt {attempt_idx}/{max_attempts}: "
                f"steps={positions.shape[0]} raw_steps={raw_plan_steps} "
                f"deviation={quality['max_line_deviation']:.3f} "
                f"path_ratio={quality['path_direct_ratio']:.2f}",
                flush=True,
            )
            break
        if selected_variant is not None:
            variants.append(selected_variant)
            print(
                f"[seed {seed}] family {spec.family_id}:{spec.name} selected: "
                f"steps={selected_variant['positions'].shape[0]} "
                f"deviation={selected_variant['quality']['max_line_deviation']:.3f} "
                f"path_ratio={selected_variant['quality']['path_direct_ratio']:.2f}",
                flush=True,
            )
        else:
            failed_family_count += 1
            print(
                f"[seed {seed}] family {spec.family_id}:{spec.name} failed after "
                f"{max_attempts} candidates "
                f"(geometry_ok={geometry_candidates}, planner_ok={planned_candidates}, "
                f"quality_ok={quality_candidates}, resampled_long={resampled_long_candidates}, "
                f"raw_too_long={raw_too_long_candidates}; "
                f"failed_families={failed_family_count}/{max_failed_families})",
                flush=True,
            )
            if failed_family_count >= max_failed_families:
                remaining = len(specs) - (specs.index(spec) + 1)
                print(
                    f"[seed {seed}] stopping family search early: "
                    f"{failed_family_count} failed families; skipped_remaining={remaining}",
                    flush=True,
                )
                break
    _set_robot_qpos(env, start_qpos)
    return variants


def _resample_joint_positions(positions: np.ndarray, *, max_steps: int) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[0] <= max_steps:
        return positions.astype(np.float32, copy=True)
    if max_steps <= 1:
        return positions[-1:].astype(np.float32, copy=True)
    indices = np.linspace(0, positions.shape[0] - 1, max_steps)
    lower = np.floor(indices).astype(np.int64)
    upper = np.ceil(indices).astype(np.int64)
    alpha = (indices - lower).astype(np.float32).reshape(-1, 1)
    resampled = (1.0 - alpha) * positions[lower] + alpha * positions[upper]
    resampled[-1] = positions[-1]
    return resampled.astype(np.float32, copy=False)


def _sample_reachable_start(
    *,
    env: Any,
    planner: Any,
    sapien: Any,
    rng: np.random.Generator,
    reset_qpos: np.ndarray,
    reset_tcp_pose: np.ndarray,
    goal_pose: Any,
    start_bounds: np.ndarray,
    randomize_start: bool,
    max_attempts: int,
    min_start_goal_distance: float,
    min_base_clearance: float,
    suppress_planner_output: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    if not randomize_start:
        return (
            reset_qpos,
            reset_tcp_pose,
            {
                "randomized": False,
                "sampled_position": reset_tcp_pose[:3].astype(np.float32).tolist(),
                "actual_position": reset_tcp_pose[:3].astype(np.float32).tolist(),
                "attempt": 0,
            },
        )

    goal_xyz = np.asarray(goal_pose.p, dtype=np.float64).reshape(-1, 3)[0]
    reset_quat = np.asarray(reset_tcp_pose[3:7], dtype=np.float32)
    bounds = np.asarray(start_bounds, dtype=np.float64).reshape(3, 2)
    robot_base_position = _robot_base_position(env.unwrapped)
    reject_counts = {
        "sampled_geometry_fail": 0,
        "sampled_too_close": 0,
        "planner_fail": 0,
        "actual_too_close": 0,
        "actual_geometry_fail": 0,
    }
    for attempt in range(1, max_attempts + 1):
        sampled_xyz = rng.uniform(bounds[:, 0], bounds[:, 1]).astype(np.float64)
        if not _is_waypoint_valid(
            waypoint=sampled_xyz,
            workspace_bounds=bounds,
            robot_base_position=robot_base_position,
            min_base_clearance=min_base_clearance,
        ):
            reject_counts["sampled_geometry_fail"] += 1
            continue
        if float(np.linalg.norm(sampled_xyz - goal_xyz)) < min_start_goal_distance:
            reject_counts["sampled_too_close"] += 1
            continue
        sampled_pose = sapien.Pose(p=sampled_xyz.astype(np.float32), q=reset_quat)
        plan = _plan_to_pose(
            planner=planner,
            env=env,
            pose=sampled_pose,
            start_qpos=reset_qpos,
            suppress_planner_output=suppress_planner_output,
        )
        if plan is None:
            reject_counts["planner_fail"] += 1
            continue
        start_qpos, planner_status = plan
        _set_robot_qpos(env, start_qpos)
        actual_tcp_pose = _tcp_pose(env.unwrapped)
        _set_robot_qpos(env, reset_qpos)
        if float(np.linalg.norm(actual_tcp_pose[:3] - goal_xyz)) < min_start_goal_distance:
            reject_counts["actual_too_close"] += 1
            continue
        if not _is_waypoint_valid(
            waypoint=actual_tcp_pose[:3],
            workspace_bounds=bounds,
            robot_base_position=robot_base_position,
            min_base_clearance=min_base_clearance,
        ):
            reject_counts["actual_geometry_fail"] += 1
            continue
        return (
            start_qpos,
            actual_tcp_pose,
            {
                "randomized": True,
                "attempt": attempt,
                "sampled_position": sampled_xyz.astype(np.float32).tolist(),
                "actual_position": actual_tcp_pose[:3].astype(np.float32).tolist(),
                "planner_status": planner_status,
                "horizontal_base_clearance": float(
                    np.linalg.norm(actual_tcp_pose[:2] - robot_base_position[:2])
                ),
                "distance_to_goal": float(np.linalg.norm(actual_tcp_pose[:3] - goal_xyz)),
            },
        )
    print(
        "start sampling failed: "
        f"attempts={max_attempts} "
        f"sampled_geometry_fail={reject_counts['sampled_geometry_fail']} "
        f"sampled_too_close={reject_counts['sampled_too_close']} "
        f"planner_fail={reject_counts['planner_fail']} "
        f"actual_too_close={reject_counts['actual_too_close']} "
        f"actual_geometry_fail={reject_counts['actual_geometry_fail']} "
        f"bounds={bounds.astype(np.float32).tolist()} "
        f"min_start_goal_distance={min_start_goal_distance:.3f} "
        f"min_base_clearance={min_base_clearance:.3f}",
        flush=True,
    )
    _set_robot_qpos(env, reset_qpos)
    return None


def _trajectory_variant_specs(
    variants_per_reset: int,
) -> list[TrajectoryFamilySpec]:
    base_specs: list[TrajectoryFamilySpec] = [
        TrajectoryFamilySpec(0, "left_wide", -0.90, 0.00, (0.34, 0.68), 1.10),
        TrajectoryFamilySpec(1, "right_wide", 0.90, 0.00, (0.34, 0.68), 1.10),
        TrajectoryFamilySpec(2, "upper_left", -0.68, 0.90, (0.30, 0.62), 1.15),
        TrajectoryFamilySpec(3, "upper_right", 0.68, 0.90, (0.30, 0.62), 1.15),
        TrajectoryFamilySpec(4, "lower_left", -0.36, -0.34, (0.35, 0.70), 1.05),
        TrajectoryFamilySpec(5, "lower_right", 0.36, -0.34, (0.35, 0.70), 1.05),
        TrajectoryFamilySpec(6, "upper_arc", 0.00, 1.15, (0.50,), 0.90),
        TrajectoryFamilySpec(7, "lower_arc", 0.00, -0.48, (0.50,), 0.80),
        TrajectoryFamilySpec(8, "high_loop", -0.30, 0.95, (0.50,), 0.95),
        TrajectoryFamilySpec(9, "low_loop", 0.48, -0.52, (0.50,), 1.00),
        TrajectoryFamilySpec(
            10,
            "shallow_direct",
            0.16,
            0.08,
            (0.50,),
            min_curve_multiplier=0.0,
            lateral_jitter=0.025,
            vertical_jitter=0.015,
        ),
        TrajectoryFamilySpec(11, "extreme_detour", 0.75, 0.38, (0.30, 0.60), 1.00),
    ]
    if variants_per_reset <= len(base_specs):
        return base_specs[:variants_per_reset]
    specs = list(base_specs)
    for idx in range(len(base_specs), variants_per_reset):
        lateral_sign = -1.0 if idx % 2 == 0 else 1.0
        vertical_sign = -1.0 if (idx // 2) % 2 == 0 else 1.0
        specs.append(
            TrajectoryFamilySpec(
                idx,
                f"extreme_detour_{idx - len(base_specs) + 2}",
                lateral_sign * 1.25,
                vertical_sign * 0.85,
                (0.35, 0.68),
                1.35,
            )
        )
    return specs


def _has_complete_variant_set(variants: list[dict[str, Any]], *, variants_per_reset: int) -> bool:
    expected_types = {spec.family_id for spec in _trajectory_variant_specs(variants_per_reset)}
    actual_types = {int(variant["trajectory_type"]) for variant in variants}
    return expected_types.issubset(actual_types)


def _sample_waypoint_set(
    *,
    start: np.ndarray,
    goal: np.ndarray,
    spec: TrajectoryFamilySpec,
    workspace_bounds: np.ndarray,
    robot_base_position: np.ndarray,
    min_base_clearance: float,
    xy_noise: float,
    z_noise: float,
    lateral_z_offset: float,
    vertical_lateral_offset: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[dict[str, Any]]] | tuple[None, list[dict[str, Any]]]:
    delta = goal - start
    distance = float(np.linalg.norm(delta))
    if distance < 1e-6:
        return None, []

    bounds = np.asarray(workspace_bounds, dtype=np.float64).reshape(3, 2)
    base_xy = np.asarray(robot_base_position[:2], dtype=np.float64)
    lateral_axis, vertical_axis = _trajectory_basis(start, goal, rng)

    # Smaller offsets keep candidates in the valid workspace while still separating modes.
    lateral_mag = max(0.12, min(0.25, 0.60 * distance))
    vertical_mag = max(0.06, min(0.20, 0.50 * distance))
    if spec.name == "shallow_direct":
        lateral_mag = max(0.02, min(0.06, 0.15 * distance))
        vertical_mag = max(0.01, min(0.04, 0.10 * distance))

    # Family specs can intentionally be extreme; cap their effective scale here so they
    # stay sampleable in tight workspaces without erasing left/right/up/down identity.
    lateral_scale = float(np.clip(spec.lateral_scale, -1.0, 1.0))
    vertical_scale = float(np.clip(spec.vertical_scale, -0.85, 1.0))
    if spec.name in {"high_loop", "upper_arc"}:
        vertical_scale = min(vertical_scale, 0.90)
    elif spec.name in {"low_loop", "lower_arc"}:
        vertical_scale = max(vertical_scale, -0.70)

    def classify_waypoint(waypoint: np.ndarray) -> tuple[bool, list[str], float]:
        reasons: list[str] = []
        if not np.all(np.isfinite(waypoint)):
            reasons.append("other_validity_fail")
            return False, reasons, 0.0
        if not np.all((waypoint >= bounds[:, 0]) & (waypoint <= bounds[:, 1])):
            reasons.append("workspace_bounds_fail")
        clearance = float(np.linalg.norm(waypoint[:2] - base_xy))
        if clearance < min_base_clearance:
            reasons.append("base_clearance_fail")
        return not reasons, reasons, clearance

    def record_diagnostics(local_counts: dict[str, int]) -> None:
        diag = getattr(_sample_waypoint_set, "_diagnostics", None)
        if diag is None:
            diag = {}
            setattr(_sample_waypoint_set, "_diagnostics", diag)
        key = f"{spec.family_id}:{spec.name}"
        family_diag = diag.setdefault(
            key,
            {
                "candidate_rejects": 0,
                "workspace_bounds_fail": 0,
                "base_clearance_fail": 0,
                "other_validity_fail": 0,
            },
        )
        family_diag["candidate_rejects"] += 1
        for name, value in local_counts.items():
            family_diag[name] += int(value)
        rejects = int(family_diag["candidate_rejects"])
        if rejects <= 3 or rejects % 20 == 0:
            print(
                f"[waypoint diagnostics] family {key}: "
                f"candidate_rejects={rejects} "
                f"workspace_bounds_fail={family_diag['workspace_bounds_fail']} "
                f"base_clearance_fail={family_diag['base_clearance_fail']} "
                f"other_validity_fail={family_diag['other_validity_fail']}",
                flush=True,
            )

    waypoints: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    local_failure_counts = {
        "workspace_bounds_fail": 0,
        "base_clearance_fail": 0,
        "other_validity_fail": 0,
    }
    center = (len(spec.ratios) - 1) / 2.0
    per_waypoint_attempts = 4
    for idx, ratio in enumerate(spec.ratios):
        accepted: tuple[np.ndarray, dict[str, Any]] | None = None
        for local_attempt in range(1, per_waypoint_attempts + 1):
            ratio_jitter = float(rng.uniform(-0.035, 0.035))
            path_ratio = float(np.clip(ratio + ratio_jitter, 0.14, 0.86))
            envelope = 1.0 + 0.10 * (1.0 - abs(idx - center) / max(center, 1.0))
            lateral_noise = float(rng.uniform(-spec.lateral_jitter, spec.lateral_jitter))
            vertical_noise = float(rng.uniform(-spec.vertical_jitter, spec.vertical_jitter))
            lateral_offset = (lateral_scale + lateral_noise) * lateral_mag * envelope
            vertical_offset = (vertical_scale + vertical_noise) * vertical_mag * envelope
            vertical_offset = float(
                np.clip(
                    vertical_offset,
                    -max(0.08, vertical_lateral_offset + 0.5 * lateral_z_offset),
                    max(0.10, vertical_lateral_offset + 0.5 * lateral_z_offset, vertical_mag),
                )
            )
            perturbation = np.asarray(
                [
                    rng.uniform(-xy_noise, xy_noise),
                    rng.uniform(-xy_noise, xy_noise),
                    rng.uniform(-z_noise, z_noise),
                ],
                dtype=np.float64,
            )
            base_point = start + path_ratio * delta
            offset = lateral_offset * lateral_axis + vertical_offset * vertical_axis
            waypoint = base_point + offset + perturbation
            valid, reasons, clearance = classify_waypoint(waypoint)
            if not valid:
                for reason in reasons:
                    local_failure_counts[reason] += 1
                continue
            accepted = (
                waypoint,
                {
                    "family_id": spec.family_id,
                    "family_name": spec.name,
                    "path_ratio": path_ratio,
                    "lateral_offset": float(lateral_offset),
                    "vertical_offset": float(vertical_offset),
                    "offset": offset.astype(np.float32).tolist(),
                    "perturbation": perturbation.astype(np.float32).tolist(),
                    "xy_noise": float(xy_noise),
                    "z_noise": float(z_noise),
                    "local_attempt": local_attempt,
                    "horizontal_base_clearance": clearance,
                    "effective_lateral_scale": lateral_scale,
                    "effective_vertical_scale": vertical_scale,
                },
            )
            break
        if accepted is None:
            record_diagnostics(local_failure_counts)
            return None, []
        waypoint, waypoint_metadata = accepted
        waypoints.append(waypoint)
        metadata.append(waypoint_metadata)
    return np.stack(waypoints, axis=0), metadata


def _trajectory_basis(
    start: np.ndarray,
    goal: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    delta = np.asarray(goal - start, dtype=np.float64)
    horizontal = delta[:2]
    horizontal_norm = float(np.linalg.norm(horizontal))
    if horizontal_norm < 1e-6:
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        lateral_axis = np.asarray([np.cos(angle), np.sin(angle), 0.0], dtype=np.float64)
    else:
        lateral_axis = np.asarray([-horizontal[1], horizontal[0], 0.0], dtype=np.float64)
        lateral_axis /= np.linalg.norm(lateral_axis[:2])
    vertical_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    return lateral_axis, vertical_axis


def _trajectory_quality(
    *,
    positions: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    waypoints: np.ndarray,
    min_curve_offset: float,
    max_joint_step: float,
    max_joint_accel: float,
) -> dict[str, float] | None:
    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[0] < 2 or not np.all(np.isfinite(positions)):
        return None
    diffs = np.diff(positions, axis=0)
    joint_steps = np.linalg.norm(diffs, axis=1)
    if joint_steps.size == 0:
        return None
    max_step = float(np.max(joint_steps))
    if max_step > max_joint_step:
        return None
    if diffs.shape[0] > 1:
        accel = np.linalg.norm(np.diff(diffs, axis=0), axis=1)
        max_accel_seen = float(np.max(accel))
    else:
        max_accel_seen = 0.0
    if max_accel_seen > max_joint_accel:
        return None

    max_line_deviation = _max_line_deviation(
        np.concatenate([np.asarray(waypoints, dtype=np.float64), goal.reshape(1, 3)], axis=0),
        start=start,
        goal=goal,
    )
    if max_line_deviation < min_curve_offset:
        return None
    path_distance = float(
        np.linalg.norm(np.diff(np.vstack([start, waypoints, goal]), axis=0), axis=1).sum()
    )
    direct_distance = float(np.linalg.norm(goal - start))
    return {
        "max_joint_step": max_step,
        "max_joint_accel": max_accel_seen,
        "max_line_deviation": float(max_line_deviation),
        "cartesian_path_distance": path_distance,
        "direct_distance": direct_distance,
        "path_direct_ratio": float(path_distance / max(direct_distance, 1e-6)),
    }


def _max_line_deviation(points: np.ndarray, *, start: np.ndarray, goal: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    start = np.asarray(start, dtype=np.float64).reshape(3)
    goal = np.asarray(goal, dtype=np.float64).reshape(3)
    line = goal - start
    length = float(np.linalg.norm(line))
    if length < 1e-6:
        return 0.0
    unit = line / length
    rel = points - start
    projected = np.outer(rel @ unit, unit)
    deviations = np.linalg.norm(rel - projected, axis=1)
    return float(np.max(deviations)) if deviations.size else 0.0


def _inset_xy_bounds(bounds: np.ndarray, margin: float) -> np.ndarray | None:
    inset = np.asarray(bounds, dtype=np.float64).reshape(3, 2).copy()
    inset[:2, 0] += margin
    inset[:2, 1] -= margin
    if np.any(inset[:, 0] > inset[:, 1]):
        return None
    return inset


def _is_waypoint_valid(
    *,
    waypoint: np.ndarray,
    workspace_bounds: np.ndarray,
    robot_base_position: np.ndarray,
    min_base_clearance: float,
) -> bool:
    if not np.all(np.isfinite(waypoint)):
        return False
    bounds = np.asarray(workspace_bounds, dtype=np.float64).reshape(3, 2)
    if not np.all((waypoint >= bounds[:, 0]) & (waypoint <= bounds[:, 1])):
        return False
    horizontal_clearance = float(
        np.linalg.norm(waypoint[:2] - np.asarray(robot_base_position[:2], dtype=np.float64))
    )
    return horizontal_clearance >= min_base_clearance


def _plan_multisegment_trajectory(
    *,
    planner: Any,
    env: Any,
    poses: list[Any],
    start_qpos: np.ndarray,
    suppress_planner_output: bool = True,
) -> tuple[np.ndarray, str] | None:
    _set_robot_qpos(env, start_qpos)
    segments: list[np.ndarray] = []
    statuses: list[str] = []
    try:
        for pose in poses:
            plan = _move_to_pose_with_screw(
                planner,
                pose,
                suppress_output=suppress_planner_output,
            )
            if not _is_valid_plan(plan):
                return None
            positions = np.asarray(plan["position"], dtype=np.float32)
            if segments and np.allclose(segments[-1][-1], positions[0], atol=1e-5):
                positions = positions[1:]
            if positions.size == 0:
                return None
            segments.append(positions)
            statuses.append(str(plan.get("status", "unknown")))
            _set_robot_qpos(env, positions[-1])
    finally:
        _set_robot_qpos(env, start_qpos)
    if not segments:
        return None
    return np.concatenate(segments, axis=0).astype(np.float32), "+".join(statuses)


def _plan_to_pose(
    *,
    planner: Any,
    env: Any,
    pose: Any,
    start_qpos: np.ndarray,
    suppress_planner_output: bool = True,
) -> tuple[np.ndarray, str] | None:
    _set_robot_qpos(env, start_qpos)
    try:
        plan = _move_to_pose_with_screw(
            planner,
            pose,
            suppress_output=suppress_planner_output,
        )
        if not _is_valid_plan(plan):
            return None
        positions = np.asarray(plan["position"], dtype=np.float32)
        return positions[-1].astype(np.float32, copy=True), str(plan.get("status", "unknown"))
    finally:
        _set_robot_qpos(env, start_qpos)


def _is_valid_plan(plan: Any) -> bool:
    if plan == -1 or not isinstance(plan, dict) or "position" not in plan:
        return False
    positions = np.asarray(plan["position"], dtype=np.float32)
    if positions.ndim != 2 or positions.shape[0] == 0:
        return False
    if not np.all(np.isfinite(positions)):
        return False
    status = str(plan.get("status", "")).lower()
    return "fail" not in status and "error" not in status


def _move_to_pose_with_screw(
    planner: Any,
    pose: Any,
    *,
    suppress_output: bool = True,
) -> Any:
    if not suppress_output:
        return planner.move_to_pose_with_screw(pose, dry_run=True)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return planner.move_to_pose_with_screw(pose, dry_run=True)


def _dataset_row_from_obs(
    *,
    obs: Any,
    info: Any,
    env: Any,
    env_id: str,
    sim_action: np.ndarray,
    action_mode: ActionMode,
    crop_config: PointCloudCropConfig,
    saliency_config: PointCloudSaliencyConfig | None,
) -> dict[str, np.ndarray]:
    adapted = adapt_observation(obs, info=info, env=env, task_name=env_id)
    row = observation_to_dataset_row(
        adapted,
        sim_action=sim_action,
        action_mode=action_mode,
        crop_config=crop_config,
    )
    if saliency_config is not None:
        _inject_point_cloud_saliency(
            row,
            saliency_config=saliency_config,
            crop_bounds=crop_config.bounds,
        )
    return row


def _inject_point_cloud_saliency(
    row: dict[str, np.ndarray],
    *,
    saliency_config: PointCloudSaliencyConfig,
    crop_bounds: np.ndarray,
) -> None:
    points = row["point_cloud"]
    valid_mask = row["point_valid_mask"]
    robot_mask = row["robot_mask"]
    target_position = row["target_position"]
    tcp_pose = row["tcp_pose"]
    bounds = np.asarray(crop_bounds, dtype=np.float32).reshape(3, 2)
    reserved_goal_count = 0
    if saliency_config.goal_marker_points > 0 and np.all(np.isfinite(target_position)):
        goal_count = min(points.shape[0], int(saliency_config.goal_marker_points))
        goal_markers = _goal_marker_points(
            target_position,
            num_points=goal_count,
            radius=saliency_config.goal_marker_radius,
        ).reshape(goal_count, 3)
        goal_markers = np.clip(goal_markers, bounds[:, 0], bounds[:, 1])
        points[-goal_count:] = goal_markers
        valid_mask[-goal_count:] = True
        robot_mask[-goal_count:] = False
        reserved_goal_count = goal_count

    tcp_position = tcp_pose[:3]
    if saliency_config.tcp_marker_points <= 0 or not np.all(np.isfinite(tcp_position)):
        return
    tcp_markers = _marker_sphere_points(
        center=tcp_position,
        radius=saliency_config.tcp_marker_radius,
        count=saliency_config.tcp_marker_points,
    )
    if tcp_markers.size == 0:
        return
    tcp_markers = np.clip(tcp_markers.astype(np.float32, copy=False), bounds[:, 0], bounds[:, 1])
    replace_count = min(points.shape[0] - reserved_goal_count, tcp_markers.shape[0])
    if replace_count <= 0:
        return
    replace_indices = _saliency_replacement_indices(
        points=points[:-reserved_goal_count] if reserved_goal_count else points,
        valid_mask=valid_mask[:-reserved_goal_count] if reserved_goal_count else valid_mask,
        robot_mask=robot_mask[:-reserved_goal_count] if reserved_goal_count else robot_mask,
        target_position=target_position,
        tcp_position=tcp_position,
        count=replace_count,
    )
    points[replace_indices] = tcp_markers[:replace_count]
    valid_mask[replace_indices] = True
    robot_mask[replace_indices] = False


def _marker_sphere_points(*, center: np.ndarray, radius: float, count: int) -> np.ndarray:
    center = np.asarray(center, dtype=np.float32).reshape(3)
    if count <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    if radius <= 0.0:
        return np.repeat(center.reshape(1, 3), count, axis=0).astype(np.float32)
    indices = np.arange(count, dtype=np.float32)
    golden_angle = np.float32(np.pi * (3.0 - np.sqrt(5.0)))
    z = 1.0 - 2.0 * (indices + 0.5) / float(count)
    radial = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = indices * golden_angle
    shell = np.stack([radial * np.cos(theta), radial * np.sin(theta), z], axis=1)
    rings = 0.35 + 0.65 * (((indices.astype(np.int64) % 3) + 1) / 3.0)
    return (center.reshape(1, 3) + radius * shell * rings.reshape(-1, 1)).astype(np.float32)


def _saliency_replacement_indices(
    *,
    points: np.ndarray,
    valid_mask: np.ndarray,
    robot_mask: np.ndarray,
    target_position: np.ndarray,
    tcp_position: np.ndarray,
    count: int,
) -> np.ndarray:
    invalid = np.flatnonzero(~valid_mask)
    if invalid.size >= count:
        return invalid[:count]
    candidates = np.flatnonzero(valid_mask & ~robot_mask)
    if candidates.size:
        anchors = np.stack([target_position, tcp_position], axis=0).astype(np.float32)
        distances = np.min(
            np.linalg.norm(points[candidates, None, :] - anchors[None, :, :], axis=2),
            axis=1,
        )
        ordered_candidates = candidates[np.argsort(distances)[::-1]]
    else:
        ordered_candidates = np.flatnonzero(valid_mask)
    combined = np.concatenate([invalid, ordered_candidates], axis=0)
    if combined.size < count:
        combined = np.concatenate(
            [combined, np.arange(points.shape[0], dtype=np.int64)],
            axis=0,
        )
    _, first_indices = np.unique(combined, return_index=True)
    unique = combined[np.sort(first_indices)]
    return unique[:count].astype(np.int64, copy=False)


def _render_viewer_frame(env: Any, delay_seconds: float = 0.0) -> None:
    if not _is_human_render_env(env):
        return
    env.render()
    if delay_seconds > 0.0:
        import time

        time.sleep(delay_seconds)


def _hold_viewer(env: Any, seconds: float) -> None:
    if seconds <= 0.0 or not _is_human_render_env(env):
        return
    import time

    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        env.render()
        time.sleep(1.0 / 30.0)


def _is_human_render_env(env: Any) -> bool:
    return getattr(env, "render_mode", None) == "human" or (
        hasattr(env, "unwrapped") and getattr(env.unwrapped, "render_mode", None) == "human"
    )


def _refresh_obs_after_manual_qpos(
    env: Any,
    *,
    info: Any,
    gripper_open: float,
) -> tuple[Any, Any]:
    unwrapped = env.unwrapped
    if hasattr(unwrapped, "get_obs"):
        return unwrapped.get_obs(info), info
    if hasattr(env, "get_obs"):
        return env.get_obs(info), info
    action = _hold_sim_action(env, gripper_open=gripper_open)
    obs, _reward, _terminated, _truncated, info = env.step(action)
    return obs, info


def _set_start_site_pose(env: Any, position: np.ndarray) -> None:
    start_site = getattr(env.unwrapped, "start_site", None)
    if start_site is None:
        return
    from mani_skill.utils.structs.pose import Pose

    start_site.set_pose(Pose.create_from_pq(np.asarray(position, dtype=np.float32).reshape(1, 3)))


def _goal_pose(unwrapped_env: Any, sapien: Any) -> Any:
    goal_pos = _to_numpy(unwrapped_env.goal_site.pose.p).reshape(-1, 3)[0]
    tcp_pose = _to_numpy(unwrapped_env.agent.tcp.pose.raw_pose).reshape(-1, 7)[0]
    return sapien.Pose(p=goal_pos, q=tcp_pose[3:])


def _pose_with_orientation(sapien: Any, *, position: np.ndarray, quat: np.ndarray) -> Any:
    return sapien.Pose(
        p=np.asarray(position, dtype=np.float32).reshape(3),
        q=np.asarray(quat, dtype=np.float32).reshape(4),
    )


def _tcp_pose(unwrapped_env: Any) -> np.ndarray:
    tcp = getattr(unwrapped_env.agent, "tcp_pose", None)
    if tcp is not None and hasattr(tcp, "raw_pose"):
        return _to_numpy(tcp.raw_pose).reshape(-1, 7)[0].astype(np.float32)
    return _to_numpy(unwrapped_env.agent.tcp.pose.raw_pose).reshape(-1, 7)[0].astype(np.float32)


def _robot_base_position(unwrapped_env: Any) -> np.ndarray:
    pose = getattr(unwrapped_env.agent.robot, "pose", None)
    if pose is None or not hasattr(pose, "p"):
        return np.zeros(3, dtype=np.float32)
    return _to_numpy(pose.p).reshape(-1, 3)[0].astype(np.float32)


def _waypoint_workspace_bounds(
    env_id: str,
    crop_config: PointCloudCropConfig,
    *,
    reach_workspace_bounds: np.ndarray,
) -> np.ndarray:
    crop_bounds = np.asarray(crop_config.bounds, dtype=np.float32).reshape(3, 2)
    bounds = np.asarray(reach_workspace_bounds, dtype=np.float32).reshape(3, 2)
    return np.stack(
        [
            np.maximum(bounds[:, 0], crop_bounds[:, 0]),
            np.minimum(bounds[:, 1], crop_bounds[:, 1]),
        ],
        axis=1,
    ).astype(np.float32)


def _start_workspace_bounds(
    env_id: str,
    start_bounds: np.ndarray | None,
    *,
    reach_workspace_bounds: np.ndarray,
) -> np.ndarray:
    if start_bounds is not None:
        return np.asarray(start_bounds, dtype=np.float32).reshape(3, 2)
    return np.asarray(reach_workspace_bounds, dtype=np.float32).reshape(3, 2)


def _default_reach_workspace_bounds(workspace_bounds: np.ndarray) -> np.ndarray:
    crop_bounds = np.asarray(workspace_bounds, dtype=np.float32).reshape(3, 2)
    # Conservative table-safe / Panda-reachable box. The physical table is about
    # x=[-0.74, 0.47], y=[-1.21, 1.20], z_top=0.0, but we avoid extreme table
    # edges that often create IK or collision failures.
    bounds = np.asarray(
        [
            [-0.42, 0.42],
            [-0.45, 0.45],
            [0.20, 0.72],
        ],
        dtype=np.float32,
    )
    return np.stack(
        [
            np.maximum(bounds[:, 0], crop_bounds[:, 0]),
            np.minimum(bounds[:, 1], crop_bounds[:, 1]),
        ],
        axis=1,
    ).astype(np.float32)


def _bounds_center_half_extents(bounds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bounds = np.asarray(bounds, dtype=np.float32).reshape(3, 2)
    center = np.mean(bounds, axis=1).astype(np.float32)
    half_extents = ((bounds[:, 1] - bounds[:, 0]) * 0.5).astype(np.float32)
    return center, half_extents


def _pose_to_list(pose: Any) -> list[float]:
    return (
        np.concatenate(
            [
                np.asarray(pose.p, dtype=np.float32).reshape(-1)[:3],
                np.asarray(pose.q, dtype=np.float32).reshape(-1)[:4],
            ],
            axis=0,
        )
        .astype(float)
        .tolist()
    )


def _get_robot_qpos(env: Any) -> np.ndarray:
    robot = env.unwrapped.agent.robot
    if hasattr(robot, "get_qpos"):
        qpos = _to_numpy(robot.get_qpos())
    else:
        qpos = _to_numpy(robot.qpos)
    return qpos.reshape(-1).astype(np.float32, copy=True)


def _set_robot_qpos(env: Any, qpos: np.ndarray) -> None:
    robot = env.unwrapped.agent.robot
    qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
    if hasattr(robot, "get_qpos"):
        current = _to_numpy(robot.get_qpos()).astype(np.float32, copy=True)
    else:
        current = _to_numpy(robot.qpos).astype(np.float32, copy=True)
    current_shape = current.shape
    current = current.reshape(-1)
    if qpos.shape[0] > current.shape[0]:
        raise ValueError(
            f"planned qpos has {qpos.shape[0]} values, robot qpos has {current.shape[0]}"
        )
    next_qpos = current.copy()
    next_qpos[: qpos.shape[0]] = qpos
    next_qpos = next_qpos.reshape(current_shape)
    if hasattr(robot, "set_qpos"):
        robot.set_qpos(next_qpos)
    else:
        robot.qpos = next_qpos.reshape(-1)
    if hasattr(robot, "set_qvel"):
        robot.set_qvel(np.zeros_like(next_qpos, dtype=np.float32))


def _format_sim_action(env: Any, planned_qpos: np.ndarray) -> np.ndarray:
    action_dim = int(np.prod(env.action_space.shape))
    planned_qpos = np.asarray(planned_qpos, dtype=np.float32).reshape(-1)
    if planned_qpos.shape[0] == action_dim:
        return planned_qpos
    if planned_qpos.shape[0] >= 9 and action_dim == 8:
        return np.concatenate([planned_qpos[:7], [np.mean(planned_qpos[7:9])]]).astype(np.float32)
    if planned_qpos.shape[0] == 7 and action_dim == 8:
        return np.concatenate([planned_qpos, [0.04]]).astype(np.float32)
    raise ValueError(
        f"cannot convert planned qpos shape {planned_qpos.shape} to action_dim={action_dim}"
    )


def _hold_sim_action(env: Any, *, gripper_open: float) -> np.ndarray:
    """Return a simulator action that asks Panda to hold the current arm qpos."""
    qpos = _to_numpy(env.unwrapped.agent.robot.qpos).reshape(-1)
    action_dim = int(np.prod(env.action_space.shape))
    if qpos.shape[0] < 7:
        raise ValueError(f"robot qpos must have at least 7 values, got {qpos.shape}")
    if action_dim == 7:
        return qpos[:7].astype(np.float32, copy=True)
    if action_dim == 8:
        return np.concatenate([qpos[:7], [gripper_open]]).astype(np.float32)
    raise ValueError(f"unsupported action_dim={action_dim} for hold action")


def _dataset_stats(episodes: list[ReachEpisodeData]) -> dict[str, Any]:
    lengths = np.asarray([episode.state.shape[0] for episode in episodes], dtype=np.int64)
    final_distances = np.asarray(
        [episode.metadata.get("final_distance", np.nan) for episode in episodes],
        dtype=np.float32,
    )
    action_norms = np.concatenate(
        [np.linalg.norm(episode.action, axis=1).astype(np.float32) for episode in episodes],
        axis=0,
    )
    robot_counts = np.concatenate(
        [episode.robot_mask.sum(axis=1).astype(np.float32) for episode in episodes],
        axis=0,
    )
    valid_counts = np.concatenate(
        [episode.point_valid_mask.sum(axis=1).astype(np.float32) for episode in episodes],
        axis=0,
    )
    hold_requested = np.asarray(
        [episode.metadata.get("hold_steps_requested", 0) for episode in episodes],
        dtype=np.int64,
    )
    hold_recorded = np.asarray(
        [episode.metadata.get("hold_steps_recorded", 0) for episode in episodes],
        dtype=np.int64,
    )
    settle_requested = np.asarray(
        [episode.metadata.get("settle_steps_requested", 0) for episode in episodes],
        dtype=np.int64,
    )
    settle_recorded = np.asarray(
        [episode.metadata.get("settle_steps_recorded", 0) for episode in episodes],
        dtype=np.int64,
    )
    successes = np.asarray([bool(episode.metadata.get("success", False)) for episode in episodes])
    return {
        "num_episodes": int(len(episodes)),
        "num_steps": int(lengths.sum()) if lengths.size else 0,
        "success_rate": float(successes.mean()) if successes.size else 0.0,
        "episode_length": _summary_stats(lengths),
        "final_distance": _summary_stats(final_distances),
        "action_norm": _summary_stats(action_norms),
        "robot_mask_points": _summary_stats(robot_counts),
        "valid_points": _summary_stats(valid_counts),
        "hold_steps_requested": _summary_stats(hold_requested),
        "hold_steps_recorded": _summary_stats(hold_recorded),
        "settle_steps_requested": _summary_stats(settle_requested),
        "settle_steps_recorded": _summary_stats(settle_recorded),
        "settle_coverage": float(
            settle_recorded.sum() / max(int(settle_requested.sum()), 1)
        ),
        "hold_coverage": float(
            hold_recorded.sum() / max(int(hold_requested.sum()), 1)
        ),
    }


def _summary_stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"min": 0, "mean": 0.0, "max": 0}
    return {
        "min": float(np.min(finite)),
        "mean": float(np.mean(finite)),
        "max": float(np.max(finite)),
    }


def json_dumps(value: Any) -> str:
    import json

    return json.dumps(_jsonable(value), sort_keys=True)


def _ensure_goal_observation_aliases(dataset_path: Path) -> dict[str, dict[str, Any]]:
    """Add compatibility aliases requested by multimodal reach training configs."""
    import zarr

    root = zarr.open_group(str(dataset_path), mode="a")
    data = root["data"]
    goal_pos = np.asarray(data["target_position"][:], dtype=np.float32)
    tcp_pose = np.asarray(data["tcp_pose"][:], dtype=np.float32)
    eef_pos = tcp_pose[:, :3].astype(np.float32, copy=True)
    arrays = {
        "goal_pos": goal_pos,
        "goal_relative": (goal_pos - eef_pos).astype(np.float32, copy=False),
        "eef_pos": eef_pos,
    }
    summaries: dict[str, dict[str, Any]] = {}
    for key, value in arrays.items():
        if key in data:
            del data[key]
        chunks = (min(max(1, value.shape[0]), 1024),) + value.shape[1:]
        data.array(name=key, data=value, chunks=chunks)
        summaries[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
    return summaries


def _ensure_trajectory_family_arrays(
    dataset_path: Path,
    episodes: list[ReachEpisodeData],
) -> dict[str, dict[str, Any]]:
    """Add per-step family conditioning arrays without changing the shared writer."""
    import zarr

    family_ids_by_episode = np.asarray(
        [int(episode.metadata.get("trajectory_family_id", -1)) for episode in episodes],
        dtype=np.int64,
    )
    if family_ids_by_episode.size == 0:
        return {}
    max_family_id = int(np.max(family_ids_by_episode))
    num_families = max(max_family_id + 1, 1)
    family_ids = np.concatenate(
        [
            np.full((episode.state.shape[0], 1), family_id, dtype=np.int64)
            for episode, family_id in zip(episodes, family_ids_by_episode, strict=True)
        ],
        axis=0,
    )
    one_hot = np.zeros((family_ids.shape[0], num_families), dtype=np.float32)
    valid_rows = family_ids[:, 0] >= 0
    one_hot[np.flatnonzero(valid_rows), family_ids[valid_rows, 0]] = 1.0

    root = zarr.open_group(str(dataset_path), mode="a")
    data = root["data"]
    arrays = {
        "trajectory_family_id": family_ids,
        "trajectory_family_onehot": one_hot,
    }
    summaries: dict[str, dict[str, Any]] = {}
    for key, value in arrays.items():
        if key in data:
            del data[key]
        chunks = (min(max(1, value.shape[0]), 1024),) + value.shape[1:]
        data.array(name=key, data=value, chunks=chunks)
        summaries[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
    return summaries


def _tcp_to_goal_distance(unwrapped_env: Any) -> float:
    goal_pos = _to_numpy(unwrapped_env.goal_site.pose.p).reshape(-1, 3)[0]
    tcp_pos = _to_numpy(unwrapped_env.agent.tcp.pose.p).reshape(-1, 3)[0]
    return float(np.linalg.norm(goal_pos - tcp_pos))


def _action_mode(value: str) -> ActionMode:
    if value not in {"abs_joint", "delta_joint"}:
        raise ValueError(f"unsupported action mode {value!r}")
    return value  # type: ignore[return-value]


if __name__ == "__main__":
    raise SystemExit(main())
