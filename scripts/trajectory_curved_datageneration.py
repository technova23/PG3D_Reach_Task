"""Generate a reach dataset by smoothing the existing multimodal waypoint paths.

This is a *standalone* companion to ``scripts/write_maniskill_reach_dataset.py``.
It does **not** modify that file; it imports its tested helpers and reuses the
exact same environment setup, planner, start sampling, multimodal waypoint
generator, replay, and zarr writer.

Idea
----
The base data-gen already plots **random intermediate waypoints** per trajectory
family (``generate_multimodal_waypoints``) to create multimodality, then
screw-chains ``[start -> wp1 -> wp2 -> ... -> goal]``. That produces a *kinked*
piecewise-straight task-space path (straight into each waypoint, straight out).

Here we keep those very same multimodal waypoints, but instead of the kinked
polyline we fit a **smooth cubic spline through ``[start, *waypoints, goal]``**
and densely resample it. Every leg -- start->waypoint *and* waypoint->goal -- is
curved, and the spline passes through the original waypoints so the multimodal
diversity is preserved.

IK reachability
---------------
The densely-sampled spline points are screw-chained with
``_plan_multisegment_trajectory``, which IK-plans each short segment seeded from
the previous joint config. **If any sampled point on the curve is not
IK-reachable the whole curved variant is rejected** -- nothing off the reachable
manifold is ever written. A cheap workspace/base-clearance filter runs first.

The output zarr has the identical schema as the base generator.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline

# Reuse the tested helpers from the base generator without modifying it.
import scripts.write_maniskill_reach_dataset as wm
from pg3d.envs.maniskill_adapter import register_pg3d_reach_envs
from pg3d.envs.maniskill_adapter.dataset import (
    PointCloudCropConfig,
    ReachEpisodeData,
    git_commit_info,
    write_reach_zarr,
)
from pg3d.envs.maniskill_adapter.reach_config import reach_task_metadata


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.viz_trajectories:
        return _visualize_curved_families(args)
    try:
        import gymnasium as gym
        import mani_skill
        import mani_skill.envs  # noqa: F401
        import sapien
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

    register_pg3d_reach_envs()
    crop_config = PointCloudCropConfig(
        bounds=np.asarray(args.workspace_bounds),
        num_points=args.num_points,
        robot_point_fraction=args.robot_point_fraction,
    )
    env_kwargs = wm._env_kwargs(args)
    saliency_config = wm._point_cloud_saliency_config(args)
    start_bounds = wm._start_workspace_bounds(
        args.env_id,
        args.start_bounds,
        reach_workspace_bounds=args.reach_workspace_bounds,
    )
    family_specs = wm._trajectory_variant_specs(args.trajectory_variants_per_reset)

    env: Any | None = None
    episodes: list[ReachEpisodeData] = []
    skipped: list[dict[str, Any]] = []
    print(
        "curved generation config: "
        f"num_demos={args.num_demos} max_attempts={args.max_attempts} "
        f"seed_start={args.seed_start} variants_per_reset={args.trajectory_variants_per_reset} "
        f"waypoint_attempts={args.waypoint_attempts} "
        f"curve_samples_per_segment={args.curve_samples_per_segment}",
        flush=True,
    )
    print(
        "trajectory families: "
        + ", ".join(f"{spec.family_id}:{spec.name}" for spec in family_specs),
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
            new_episodes = _collect_curved_episodes(
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
                planner_cls=PandaArmMotionPlanningSolver,
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
                start_bounds=start_bounds,
                reach_workspace_bounds=args.reach_workspace_bounds,
                start_sample_attempts=args.start_sample_attempts,
                min_start_goal_distance=args.min_start_goal_distance,
                acceptance_success_distance=args.acceptance_success_distance,
                saliency_config=saliency_config,
                require_complete_variant_set=not args.allow_partial_variant_sets,
                suppress_planner_output=not args.show_planner_output,
                smooth_trajectory=args.smooth_trajectory,
                curved_paths=args.curved_paths,
                curvature_std=args.curvature_std,
                verbose_waypoints=args.verbose_waypoints,
                curve_samples_per_segment=args.curve_samples_per_segment,
                viewer_step_delay=args.viewer_step_delay if args.viewer else 0.0,
            )
            if not new_episodes:
                print(f"[seed {seed}] skipped: no curved variant produced", flush=True)
                skipped.append({"seed": seed, "reason": "no_curved_variant"})
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
                    f"demo {len(episodes)}/{args.num_demos}: seed={seed} "
                    f"variant={episode.metadata.get('trajectory_family')} "
                    f"steps={episode.state.shape[0]} "
                    f"success={episode.metadata.get('success')} "
                    f"final_distance={episode.metadata.get('final_distance'):.4f}",
                    flush=True,
                )
    except Exception as exc:
        print(f"Failed to write curved reach dataset: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if env is not None:
            wm._hold_viewer(env, args.viewer_hold_seconds if args.viewer else 0.0)
            env.close()

    if not episodes:
        print("No usable curved reach demonstrations were collected.", file=sys.stderr)
        return 1

    repo_root = Path(__file__).resolve().parents[1]
    dataset_stats = wm._dataset_stats(episodes)
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
            "start_bounds": start_bounds.tolist(),
            "reach_workspace_bounds": args.reach_workspace_bounds.tolist(),
            "start_sample_attempts": args.start_sample_attempts,
            "min_start_goal_distance": args.min_start_goal_distance,
            "min_base_clearance": args.min_base_clearance,
            "table_margin": args.table_margin,
        },
        "trajectory_generation": {
            "type": "multimodal_waypoint_spline_smoothing",
            "planner": "PandaArmMotionPlanningSolver.move_to_pose_with_screw",
            "variants_per_reset": args.trajectory_variants_per_reset,
            "waypoint_attempts": args.waypoint_attempts,
            "curve_samples_per_segment": args.curve_samples_per_segment,
            "smooth_trajectory": args.smooth_trajectory,
            "acceptance_success_distance": args.acceptance_success_distance,
            "note": (
                "Reuses generate_multimodal_waypoints to plot the same random multimodal "
                "waypoints, then fits a cubic spline through [start, *waypoints, goal] and "
                "densely resamples it so every leg (start->waypoint and waypoint->goal) is "
                "curved. Each spline sample is IK-checked by chaining "
                "move_to_pose_with_screw; any unreachable sample rejects the curved variant, "
                "so no out-of-reachability points are written. Family ids are not saved as "
                "zarr arrays."
            ),
        },
        "task": reach_task_metadata(args.env_id),
        "crop": crop_config.to_json(),
        "point_cloud_saliency": saliency_config.to_json(),
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
    wm._strip_trajectory_family_metadata(episodes)
    summary = write_reach_zarr(
        args.output, episodes, metadata=metadata, overwrite=args.overwrite, append=args.append
    )
    alias_arrays = wm._ensure_goal_observation_aliases(args.output)
    summary.get("arrays", {}).update(alias_arrays)
    print(f"saved dataset: {args.output}")
    print(f"summary: {summary}")
    print("dataset_stats: " + wm.json_dumps(dataset_stats))
    return 0


def _collect_curved_episodes(
    *,
    env: Any,
    seed: int,
    env_id: str,
    action_mode: Any,
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
    saliency_config: Any,
    require_complete_variant_set: bool,
    suppress_planner_output: bool,
    smooth_trajectory: bool,
    curved_paths: bool,
    curvature_std: float,
    verbose_waypoints: bool,
    curve_samples_per_segment: int,
    viewer_step_delay: float = 0.0,
) -> list[ReachEpisodeData]:
    """Mirror _collect_multimodal_episodes, but spline-smooth each variant's path.

    The setup (reset, bounds, start sampling, goal reachability, multimodal
    waypoint generation) is identical to the base generator. The only change is
    that each variant's kinked screw path is replaced by a cubic spline through
    ``[start, *waypoints, goal]``, densely resampled and re-planned.
    """
    obs, info = env.reset(seed=seed, options={"reconfigure": True})
    unwrapped = env.unwrapped
    goal_pose = wm._goal_pose(unwrapped, sapien)
    reset_tcp_pose = wm._tcp_pose(unwrapped)
    reset_qpos = wm._get_robot_qpos(env)
    rng = np.random.default_rng(seed)
    robot_base_position = wm._robot_base_position(unwrapped)

    waypoint_bounds = wm._inset_xy_bounds(
        wm._waypoint_workspace_bounds(
            env_id, crop_config, reach_workspace_bounds=reach_workspace_bounds
        ),
        table_margin,
    )
    start_sampling_bounds = wm._inset_xy_bounds(start_bounds, table_margin)
    if waypoint_bounds is None or start_sampling_bounds is None:
        return []

    goal_xyz = np.asarray(goal_pose.p, dtype=np.float64).reshape(-1, 3)[0]
    if not wm._is_waypoint_valid(
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
        start_sample = wm._sample_reachable_start(
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
        start_xyz = np.asarray(start_tcp_pose[:3], dtype=np.float64)
        goal_quat = np.asarray(start_tcp_pose[3:7], dtype=np.float32)
        goal_pose = wm._pose_with_orientation(sapien, position=goal_xyz, quat=goal_quat)
        print(
            f"[seed {seed}] start accepted attempt={start_metadata.get('attempt')} "
            f"tcp={start_xyz.astype(np.float32).tolist()} "
            f"goal={goal_xyz.astype(np.float32).tolist()}",
            flush=True,
        )
        goal_reachability = wm._plan_to_pose(
            planner=planner,
            env=env,
            pose=goal_pose,
            start_qpos=start_qpos,
            suppress_planner_output=suppress_planner_output,
        )
        if goal_reachability is None:
            print(f"[seed {seed}] rejected: direct goal plan failed from sampled start", flush=True)
            return []
        start_metadata = {
            **start_metadata,
            "goal_reachable_from_start": True,
            "goal_reachability_status": goal_reachability[1],
        }
        wm._set_robot_qpos(env, start_qpos)
        wm._set_start_site_pose(env, start_tcp_pose[:3])

        # Reuse the existing multimodal waypoint generator verbatim. Each returned
        # variant carries the random intermediate waypoints we will spline through.
        variants = wm.generate_multimodal_waypoints(
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
            smooth_trajectory=smooth_trajectory,
            curved_paths=curved_paths,
            curvature_std=curvature_std,
            verbose_waypoints=verbose_waypoints,
        )

        # Replace each variant's kinked screw path with a spline through its waypoints.
        curved_variants: list[dict[str, Any]] = []
        for variant in variants:
            waypoints = np.asarray(variant.get("waypoints", []), dtype=np.float64).reshape(-1, 3)
            curve_points = _spline_through_waypoints(
                start_xyz=start_xyz,
                waypoints=waypoints,
                goal_xyz=goal_xyz,
                samples_per_segment=curve_samples_per_segment,
                workspace_bounds=waypoint_bounds,
                robot_base_position=robot_base_position,
                min_base_clearance=min_base_clearance,
            )
            if curve_points is None:
                print(
                    f"[seed {seed}] family {variant['name']}: spline left feasible workspace; "
                    "skipping curved variant",
                    flush=True,
                )
                continue
            poses = [
                wm._pose_with_orientation(sapien, position=point, quat=goal_quat)
                for point in curve_points
            ]
            plan = wm._plan_multisegment_trajectory(
                planner=planner,
                env=env,
                poses=poses,
                start_qpos=start_qpos,
                suppress_planner_output=suppress_planner_output,
                smooth_trajectory=smooth_trajectory,
            )
            if plan is None:
                print(
                    f"[seed {seed}] family {variant['name']}: curved spline not fully "
                    "IK-reachable; skipping",
                    flush=True,
                )
                continue
            positions, planner_status = plan
            curved_variants.append(
                {
                    "name": variant["name"],
                    "trajectory_type": variant["trajectory_type"],
                    "positions": positions,
                    "planner_status": planner_status,
                    "waypoints": variant.get("waypoints", []),
                    "waypoint_metadata": variant.get("waypoint_metadata"),
                    "quality": variant.get("quality"),
                    "curve_samples": int(curve_points.shape[0]),
                }
            )
    finally:
        planner.close()

    if require_complete_variant_set and len(curved_variants) < min_feasible_families:
        print(
            f"[seed {seed}] insufficient feasible curved families: "
            f"count={len(curved_variants)}/{variants_per_reset} required>={min_feasible_families}",
            flush=True,
        )
        return []

    episodes: list[ReachEpisodeData] = []
    for variant in curved_variants:
        obs, info = env.reset(seed=seed, options={"reconfigure": False})
        wm._set_robot_qpos(env, start_qpos)
        wm._set_start_site_pose(env, start_tcp_pose[:3])
        obs, info = wm._refresh_obs_after_manual_qpos(env, info=info, gripper_open=gripper_open)
        wm._render_viewer_frame(env, viewer_step_delay)
        episode = wm._replay_planned_positions_as_episode(
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
                "curve_samples": variant["curve_samples"],
                "start_tcp_pose": np.asarray(start_tcp_pose, dtype=np.float32).tolist(),
                "goal_pose": wm._pose_to_list(goal_pose),
                "start_sampling": start_metadata,
            },
        )
        if episode is not None:
            episodes.append(episode)
            print(
                f"[seed {seed}] curved replay {variant['trajectory_type']}:{variant['name']} "
                f"success={episode.metadata.get('success')} "
                f"final_distance={episode.metadata.get('final_distance'):.4f}",
                flush=True,
            )

    successful = sum(bool(ep.metadata.get("success", False)) for ep in episodes)
    if require_complete_variant_set and successful < min_feasible_families:
        print(
            f"[seed {seed}] rejected: curved replay produced {successful}/{len(episodes)} "
            f"successful episodes; required>={min_feasible_families}",
            flush=True,
        )
        return []
    print(f"[seed {seed}] accepted {successful}/{len(episodes)} successful curved families", flush=True)
    return episodes


def _spline_through_waypoints(
    *,
    start_xyz: np.ndarray,
    waypoints: np.ndarray,
    goal_xyz: np.ndarray,
    samples_per_segment: int,
    workspace_bounds: np.ndarray,
    robot_base_position: np.ndarray,
    min_base_clearance: float,
) -> np.ndarray | None:
    """Fit a cubic spline through ``[start, *waypoints, goal]`` and resample it.

    The spline is parameterized by cumulative chord length so it passes through
    every original waypoint while smoothing the legs into and out of them. Dense
    samples between control points turn the kinked polyline into a continuous
    curve.

    Returns the sampled points *excluding* the start (first returned point is the
    first interior sample, last is exactly the goal), or ``None`` if any control
    or sampled point leaves the feasible workspace box or violates base
    clearance. ``None`` makes the caller drop this curved variant.
    """
    start = np.asarray(start_xyz, dtype=np.float64).reshape(3)
    goal = np.asarray(goal_xyz, dtype=np.float64).reshape(3)
    waypoints = np.asarray(waypoints, dtype=np.float64).reshape(-1, 3)
    control = np.vstack([start[None, :], waypoints, goal[None, :]])

    # Drop consecutive duplicate control points (zero-length legs break the spline).
    keep = [0]
    for idx in range(1, control.shape[0]):
        if np.linalg.norm(control[idx] - control[keep[-1]]) > 1e-6:
            keep.append(idx)
    control = control[keep]
    n_control = control.shape[0]
    if n_control < 2:
        return None

    if not _all_points_feasible(control, workspace_bounds, robot_base_position, min_base_clearance):
        return None

    # Cumulative chord-length parameterization in [0, 1].
    seg_len = np.linalg.norm(np.diff(control, axis=0), axis=1)
    t_control = np.concatenate([[0.0], np.cumsum(seg_len)])
    t_control = t_control / t_control[-1]

    spp = max(2, int(samples_per_segment))
    if n_control == 2:
        # No interior waypoint: straight leg, just sample densely between endpoints.
        sampled = (
            start[None, :]
            + np.linspace(0.0, 1.0, spp + 1)[:, None] * (goal - start)[None, :]
        )
    else:
        spline = CubicSpline(t_control, control, axis=0, bc_type="natural")
        n_samples = (n_control - 1) * spp + 1
        # Union uniform samples with the control parameters so the curve passes
        # exactly through every original multimodal waypoint, not merely near it.
        t_sample = np.union1d(np.linspace(0.0, 1.0, n_samples), t_control)
        sampled = np.asarray(spline(t_sample), dtype=np.float64)

    # Pin exact endpoints (guard against tiny spline overshoot at start/goal).
    sampled[0] = start
    sampled[-1] = goal

    if not _all_points_feasible(sampled, workspace_bounds, robot_base_position, min_base_clearance):
        return None

    # Drop the start; the planner chains screw moves *from* start_qpos.
    return sampled[1:].astype(np.float32)


def _all_points_feasible(
    points: np.ndarray,
    workspace_bounds: np.ndarray,
    robot_base_position: np.ndarray,
    min_base_clearance: float,
) -> bool:
    for point in np.asarray(points, dtype=np.float64):
        if not wm._is_waypoint_valid(
            waypoint=point,
            workspace_bounds=workspace_bounds,
            robot_base_position=robot_base_position,
            min_base_clearance=min_base_clearance,
        ):
            return False
    return True


def _visualize_curved_families(args: argparse.Namespace) -> int:
    """Plot the multimodal waypoints, the original kinked polyline, and the
    spline-smoothed curve for one start/goal pair -- no simulator required.

    Reuses the base generator's ``_sample_waypoint_set`` so the very same random
    multimodal waypoints are drawn, then overlays this script's cubic-spline
    smoothing (``_spline_through_waypoints``) on top so the difference between the
    straight kinked legs and the curved legs is visible per family.
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        print("matplotlib is required for --viz-trajectories. Install with: pip install matplotlib")
        return 1

    start = np.asarray(args.viz_start, dtype=np.float64)
    goal = np.asarray(args.viz_goal, dtype=np.float64)
    workspace_bounds = np.asarray(args.workspace_bounds, dtype=np.float64).reshape(3, 2)
    robot_base = np.zeros(3, dtype=np.float64)  # centred at origin for viz
    specs = wm._trajectory_variant_specs(args.trajectory_variants_per_reset)

    seed_cmaps = ["tab10", "Set2", "Dark2", "Paired"]
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*start, color="lime", s=120, zorder=5, label="start")
    ax.scatter(*goal, color="red", s=120, zorder=5, label="goal")
    ax.plot(
        [start[0], goal[0]], [start[1], goal[1]], [start[2], goal[2]],
        "k:", linewidth=1, alpha=0.35, label="direct",
    )

    for seed_idx in range(args.viz_seeds):
        rng = np.random.default_rng(seed_idx)
        cmap = plt.get_cmap(seed_cmaps[seed_idx % len(seed_cmaps)])
        colors = [cmap(i / max(len(specs) - 1, 1)) for i in range(len(specs))]
        for spec_idx, spec in enumerate(specs):
            color = colors[spec_idx]
            waypoints_arr, _ = wm._sample_waypoint_set(
                start=start,
                goal=goal,
                spec=spec,
                workspace_bounds=workspace_bounds,
                robot_base_position=robot_base,
                min_base_clearance=0.0,
                xy_noise=args.waypoint_xy_noise,
                z_noise=args.waypoint_z_noise,
                lateral_z_offset=args.lateral_z_offset,
                vertical_lateral_offset=args.vertical_lateral_offset,
                rng=rng,
                curved_paths=args.curved_paths,
                curvature_std=args.curvature_std,
                verbose_waypoints=False,
            )
            if waypoints_arr is None:
                continue
            # Original kinked polyline (start -> waypoints -> goal): faint dashed.
            kinked = np.vstack([start[None], waypoints_arr, goal[None]])
            ax.plot(
                kinked[:, 0], kinked[:, 1], kinked[:, 2],
                color=color, linewidth=1.0, alpha=0.30, linestyle="--",
            )
            # Spline-smoothed curve through the same waypoints: solid.
            curve = _spline_through_waypoints(
                start_xyz=start,
                waypoints=waypoints_arr,
                goal_xyz=goal,
                samples_per_segment=args.curve_samples_per_segment,
                workspace_bounds=workspace_bounds,
                robot_base_position=robot_base,
                min_base_clearance=0.0,
            )
            label = f"{spec.name}" if seed_idx == 0 else None
            if curve is not None:
                curve_full = np.vstack([start[None], curve])
                ax.plot(
                    curve_full[:, 0], curve_full[:, 1], curve_full[:, 2],
                    color=color, linewidth=2.0, alpha=0.9, label=label,
                )
            ax.scatter(
                waypoints_arr[:, 0], waypoints_arr[:, 1], waypoints_arr[:, 2],
                color=color, s=30, alpha=0.9,
            )
            if seed_idx == 0:
                wp = waypoints_arr[0]
                ax.text(wp[0], wp[1], wp[2], f" {spec.name}", fontsize=6, color=color)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(
        f"Curved (solid) vs kinked (dashed) trajectories\n"
        f"start={np.round(start, 2)} -> goal={np.round(goal, 2)} | "
        f"{len(specs)} families x {args.viz_seeds} seed(s) | "
        f"samples/segment={args.curve_samples_per_segment}"
    )
    ax.legend(loc="upper left", fontsize=7, ncol=2)

    if args.viz_save is not None:
        fig.savefig(args.viz_save, dpi=150, bbox_inches="tight")
        print(f"saved curved trajectory viz to {args.viz_save}")
    else:
        plt.tight_layout()
        plt.show()
    plt.close(fig)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Reuse the base generator's full argument set, plus spline controls."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--curve-samples-per-segment",
        type=int,
        default=6,
        help=(
            "number of waypoints sampled along the spline per leg between consecutive "
            "control points (start, each waypoint, goal); higher = smoother curve, "
            "more screw segments"
        ),
    )
    curve_args, remaining = pre.parse_known_args(argv)

    # Hand the rest to the base parser so every env/crop/start/waypoint/saliency
    # default and validation is shared verbatim with write_maniskill_reach_dataset.
    args = wm.parse_args(remaining)
    for key, value in vars(curve_args).items():
        setattr(args, key, value)

    if args.curve_samples_per_segment < 2:
        raise ValueError("--curve-samples-per-segment must be >= 2")
    return args


if __name__ == "__main__":
    raise SystemExit(main())
