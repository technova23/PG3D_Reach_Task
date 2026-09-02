#!/usr/bin/env python
"""Checkpoint-based pose-variety reachability test for xArm7.

Complements ``scripts/verify_pose_variety_reachability.py`` (pure ground-truth
mplib motion planning, no policy involved). This script instead loads a
trained DP3 checkpoint and tests whether the POLICY -- steered via reranking,
since the checkpoint has no direct conditioning input for a target
*orientation* (only goal position) -- can actually reach an arbitrary
start/goal pose pair.

No GraspGen. Design:

* One "config" = one fixed (start, goal) POSITION pair.
* For that config, run --episodes-per-config episodes that reuse the SAME
  two positions but each get their own INDEPENDENT, FPS-diversified random
  orientation at start/goal (mirrors --randomize-start-goal-orientation
  data generation's own pool-then-spread selection, so the orientations tested
  across episodes are spread apart rather than clustered).
* Each pose (position, orientation) is first IK-verified with mplib (ground
  truth, same as verify_pose_variety_reachability.py) before any policy
  rollout is attempted -- a trial whose start/goal isn't even IK-reachable is
  not a fair test of the checkpoint and is skipped. On top of that, the pair
  must be INTER-feasible: a collision-free plan has to actually connect the
  start configuration to the goal pose, not merely reach each independently.
* For every feasible pair: one episode. Reset at the start pose, then run
  rejection-free reranking (K candidate DP3 action chunks per replan, scored
  by imagined-rollout distance to a CartesianPoseConstraint, lowest cost
  executed) toward the GOAL pose. Reranking requires --geometry-mode exact
  (fast imagination freezes orientation at the window's starting value, which
  would silently make orientation-aware steering meaningless).

Usage:
    python scripts/eval_pose_variety_checkpoint_reachability.py \
        --checkpoint artifacts/xarm7_pose_variety_5000_checkpoints/final_step_00080000.pt \
        --dataset artifacts/pg3d_xarm7_pose_variety_reach_5000.zarr \
        --num-configs 4 --episodes-per-config 5 \
        --video-dir artifacts/pose_variety_checkpoint_eval \
        --summary-json artifacts/pose_variety_checkpoint_eval/summary.json

``--dataset`` is used ONLY to read metadata.json (env_id/env_kwargs/crop
config/action_mode) -- the exact observation shape the checkpoint was
trained on -- never to source episode content; every start/goal pose here
is generated fresh by this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _quat_angular_distance_rad(q1_wxyz: np.ndarray, q2_wxyz: np.ndarray) -> float:
    dot = float(np.clip(np.abs(np.dot(q1_wxyz, q2_wxyz)), 0.0, 1.0))
    return float(2.0 * np.arccos(dot))


# Rest-pose TCP orientation (wxyz): tool z-axis straight down. Used only as a fixed
# reference to report how far a sampled orientation tilts, in degrees -- a raw
# quaternion isn't human-readable on its own.
_DOWN_QUAT_WXYZ = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)


def _tilt_from_down_deg(quat_wxyz: np.ndarray) -> float:
    return float(np.degrees(_quat_angular_distance_rad(quat_wxyz, _DOWN_QUAT_WXYZ)))


def _orientation_variety_stats_deg(quats: list[np.ndarray]) -> dict[str, float | int | None]:
    """Pairwise angular spread (degrees) across a set of orientations.

    Quantifies whether the orientations actually tested for one label (start /
    goal) across a config's --episodes-per-config episodes are spread apart
    (FPS worked) or accidentally clustered -- independent of whether each one
    turned out to be IK-reachable or the checkpoint reached it.
    """
    n = len(quats)
    if n < 2:
        return {"count": n, "min_deg": None, "mean_deg": None, "max_deg": None}
    pairwise = [
        float(np.degrees(_quat_angular_distance_rad(quats[i], quats[j])))
        for i in range(n)
        for j in range(i + 1, n)
    ]
    return {
        "count": n,
        "min_deg": float(np.min(pairwise)),
        "mean_deg": float(np.mean(pairwise)),
        "max_deg": float(np.max(pairwise)),
    }


def _build_ik_env_and_planner(variant: str) -> tuple[Any, Any, Any, np.ndarray]:
    """Cheap obs_mode='none' env + mplib planner, used only for IK pre-verification."""
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    if variant == "gripper":
        from pg3d.envs.xarm_adapter import register_pg3d_xarm7_gripper_reach_envs
        from pg3d.envs.xarm_adapter.motionplanner import (
            XArm7GripperMotionPlanningSolver as Solver,
        )

        register_pg3d_xarm7_gripper_reach_envs()
        env_id, robot_uid = "PG3DReach-XArm7-Gripper-Workspace-v0", "xarm7_gripper"
    else:
        from pg3d.envs.xarm_adapter import register_pg3d_xarm7_reach_envs
        from pg3d.envs.xarm_adapter.motionplanner import (
            XArm7NoGripperMotionPlanningSolver as Solver,
        )

        register_pg3d_xarm7_reach_envs()
        env_id, robot_uid = "PG3DReach-XArm7-Workspace-v0", "xarm7_nogripper"

    env = gym.make(
        env_id, obs_mode="none", render_mode="rgb_array", robot_uids=robot_uid, num_envs=1
    )
    env.reset(seed=0)
    u = env.unwrapped
    solver = Solver(
        env,
        debug=False,
        vis=False,
        base_pose=u.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
    )
    n_ik = len(solver.planner.user_joint_names)
    rest_qpos = np.asarray(u.agent.robot.get_qpos()).reshape(-1)[:n_ik].astype(np.float32)
    # Return the solver itself as "planner" -- _resolve_reachable_orientation calls
    # planner.move_to_pose_with_screw(...), which only the ManiSkill solver wrapper
    # exposes (the raw mplib Planner, solver.planner, only has .IK).
    return env, solver, solver, rest_qpos


def _sample_workspace_pair(
    rng: np.random.Generator,
    *,
    bounds_world: np.ndarray,
    min_pairwise_distance: float,
    resample_attempts: int,
) -> list[np.ndarray]:
    """Sample a (start, goal) position pair at least min_pairwise_distance apart."""
    bounds = np.asarray(bounds_world, dtype=np.float32).reshape(3, 2)

    def sample_one() -> np.ndarray:
        return rng.uniform(bounds[:, 0], bounds[:, 1]).astype(np.float32)

    for _ in range(max(resample_attempts, 1)):
        points = [sample_one() for _ in range(2)]
        if float(np.linalg.norm(points[0] - points[1])) >= min_pairwise_distance:
            return points
    return points


def _set_site_pose(env: Any, site_name: str, position: np.ndarray) -> None:
    """Move one marker actor, using the ONLY pose-set form that works here.

    Mirrors dataset_generation.write_maniskill_reach_dataset._set_start_site_pose
    exactly: ManiSkill's BATCHED ``Pose.create_from_pq(<(1,3) array>)``, not a raw
    ``sapien.Pose(p=...)``. On this GPU-backed sim a raw sapien.Pose does not
    reliably land in the CUDA rigid-body buffer, so the actor silently keeps its
    old pose -- which previously left goal_site (and therefore the policy's goal
    conditioning, see _sync_pose_variety_markers) pinned to the env's own
    randomly sampled goal instead of ours.
    """
    site = getattr(env.unwrapped, site_name, None)
    if site is None:
        return
    from mani_skill.utils.structs.pose import Pose

    site.set_pose(Pose.create_from_pq(np.asarray(position, dtype=np.float32).reshape(1, 3)))


def _sync_pose_variety_markers(
    sim_env: Any,
    *,
    start_pos: np.ndarray,
    goal_pos: np.ndarray,
) -> None:
    """Point the env's start_site (red) / goal_site (green) at OUR actual pair.

    ``goal_site`` is NOT purely cosmetic: PG3DReachEnv._get_obs_extra exposes
    ``goal_site.pose.p`` as ``info["extra"]["goal_pos"]``, which
    pg3d/envs/maniskill_adapter/observation.py reads into
    ``sim_gt.target_position`` -- i.e. the actual goal-conditioning signal fed
    into the DP3 policy's observation (both ``obs["goal_xyz"]`` and, if the
    checkpoint uses goal-marker point-cloud tokens, points literally inserted
    into the point cloud at this position). Left alone it keeps the env's OWN
    randomly sampled goal, so the policy would chase a point unrelated to the
    one we're testing; CartesianPoseConstraint-based reranking cannot fix that,
    since it only reorders the K candidates the diffusion model already sampled
    around whatever goal_site says the goal is.

    Caller is responsible for refreshing sim_entry/obs_window (e.g. via
    _refresh_obs_after_manual_qpos) after calling this without stepping
    physics -- this only moves the SAPIEN actors, it does not re-derive the
    already-computed observation entries that fed the policy on prior steps.
    """
    _set_site_pose(sim_env, "start_site", start_pos)
    _set_site_pose(sim_env, "goal_site", goal_pos)
    update_render = getattr(sim_env.unwrapped.scene, "update_render", None)
    if callable(update_render):
        update_render()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    import torch

    from dataset_generation.write_maniskill_reach_dataset import (
        _farthest_point_select_orientations,
        _plan_multisegment_trajectory,
        _pose_with_orientation,
        _refresh_obs_after_manual_qpos,
        _resolve_reachable_orientation,
        _sample_broad_orientation_sapien,
        _set_robot_qpos,
        _tcp_pose,
    )
    from pg3d.constraints import CartesianPoseConstraint
    from pg3d.envs.maniskill_adapter import ManiSkillGhostPandaGeometryProvider
    from pg3d.envs.maniskill_adapter.dataset import load_reach_metadata
    from pg3d.envs.xarm_adapter.reach_config import XARM7_REACH_WORKSPACE_BOUNDS
    from pg3d.eval import TimingRecorder, scene_context_for_constraints
    from pg3d.policies.dp3.checkpoint import load_reach_policy_from_checkpoint
    from pg3d.utils.devices import select_device
    from pg3d.world_model import GeometricWorldModel
    from scripts.eval_constrained_reach import (
        DP3ChunkPolicyAdapter,
        _env_kwargs,
        _select_decision,
    )
    from scripts.rollout_dp3_reach_policy import (
        _action_mode,
        append_obs_window,
        crop_config_from_metadata,
        make_initial_obs_window,
        policy_action_to_sim_action,
        rollout_observation_entry,
        save_video,
    )
    from pg3d.utils.arrays import bool_any, frame_to_numpy

    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
        import sapien
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to import ManiSkill/SAPIEN: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    metadata = load_reach_metadata(args.dataset)
    device = select_device(args.device)
    policy = load_reach_policy_from_checkpoint(args.checkpoint, device=device)
    action_mode = _action_mode(str(metadata.get("action_mode", "abs_joint")))
    crop_config = crop_config_from_metadata(metadata)

    try:
        ik_env, ik_solver, planner, rest_qpos = _build_ik_env_and_planner(args.variant)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to build IK env/planner: {type(exc).__name__}: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1

    # metadata["env_kwargs"] carries whatever max_episode_steps the ORIGINAL dataset
    # was generated with. ManiSkill's TimeLimit wrapper truncates at that value, so if
    # it's BELOW this script's --max-steps budget the episode gets cut short and
    # misreported as "truncated early". Raise the env ceiling to at least our budget --
    # but never LOWER it, so a generous metadata limit isn't thrown away; the while
    # loop already caps the episode at args.max_steps either way.
    _meta_max_steps = metadata.get("env_kwargs", {}).get("max_episode_steps")
    _env_max_steps = max(int(args.max_steps), int(_meta_max_steps or 0))
    print(
        f"metadata env_kwargs max_episode_steps={_meta_max_steps!r}, "
        f"--max-steps={args.max_steps} -> env limit set to {_env_max_steps}"
    )
    sim_env = gym.make(
        str(metadata["env_id"]),
        **_env_kwargs(metadata, render_mode="rgb_array", max_episode_steps=_env_max_steps),
    )
    ghost_env = gym.make(str(metadata["env_id"]), **_env_kwargs(metadata, render_mode=None))

    adapter = DP3ChunkPolicyAdapter(
        policy, action_mode=action_mode, device=device, policy_batch_size=args.policy_batch_size
    )
    provider = ManiSkillGhostPandaGeometryProvider(
        ghost_env, task_name=str(metadata.get("env_id", "unknown")), crop_bounds=crop_config.bounds
    )
    world_model = GeometricWorldModel(provider)

    rng = np.random.default_rng(args.seed)
    bounds_world = np.asarray(XARM7_REACH_WORKSPACE_BOUNDS, dtype=np.float32).copy()
    # XARM7_REACH_WORKSPACE_BOUNDS' height (Z) row is [base_z + 0.05, base_z + 0.55] --
    # i.e. 5cm to 55cm above the table (see XARM7_REACH_BOX_BASE in reach_config.py).
    # Override just the Z row to this script's own --min-height/--max-height, scoped
    # to sampling here only -- NOT touching the shared constant, which other
    # scripts/training/dataset-gen still read unmodified.
    base_z_offset = float(bounds_world[2, 0]) - 0.05  # recover base_position[2]
    bounds_world[2, 0] = base_z_offset + args.min_height
    bounds_world[2, 1] = base_z_offset + args.max_height
    print(
        f"start/goal height range: {args.min_height:.2f}m to {args.max_height:.2f}m "
        f"above table (world Z {bounds_world[2, 0]:.3f} to {bounds_world[2, 1]:.3f})"
    )

    if args.video_dir is not None:
        args.video_dir.mkdir(parents=True, exist_ok=True)

    labels = ("start", "goal")
    all_rows: list[dict[str, Any]] = []
    counts = {
        "total_episodes": 0,
        "ik_failed": 0,
        "motion_plan_infeasible": 0,
        "goal_reached": 0,
    }

    try:
        for config_idx in range(args.num_configs):
            positions = _sample_workspace_pair(
                rng,
                bounds_world=bounds_world,
                min_pairwise_distance=args.min_pairwise_distance,
                resample_attempts=args.position_resample_attempts,
            )
            print(
                f"\n=== config {config_idx:02d} — start={positions[0].tolist()} "
                f"goal={positions[1].tolist()} ==="
            )

            pools: dict[str, list[np.ndarray]] = {}
            for label in labels:
                if label == "start" and not args.randomize_start_orientation:
                    # Keep start orientation fixed straight-down (the natural rest
                    # approach) so every episode has an easy, consistent starting
                    # condition -- only the GOAL orientation varies. This isolates
                    # what's being tested to "can the checkpoint steer to an arbitrary
                    # goal orientation", without also asking it to recover from an
                    # awkward/varied start.
                    pools[label] = [_DOWN_QUAT_WXYZ.copy() for _ in range(args.episodes_per_config)]
                    continue
                pool_size = max(args.episodes_per_config * 4, 12)
                candidates = [
                    _sample_broad_orientation_sapien(rng, cone_deg=args.orientation_cone_deg)
                    for _ in range(pool_size)
                ]
                pools[label] = _farthest_point_select_orientations(
                    candidates, k=args.episodes_per_config, rng=rng
                )

            print("  orientation variety sampled for this config (pairwise spread, degrees):")
            for label in labels:
                stats = _orientation_variety_stats_deg(pools[label])
                tilts = ", ".join(f"{_tilt_from_down_deg(q):.0f}" for q in pools[label])
                print(
                    f"    {label:8s}: min={stats['min_deg']:.1f} mean={stats['mean_deg']:.1f} "
                    f"max={stats['max_deg']:.1f}  (tilt-from-down per episode: [{tilts}] deg)"
                )

            config_counts = {
                "total": 0,
                "ik_failed": 0,
                "motion_plan_infeasible": 0,
                "goal_reached": 0,
            }
            for episode_idx in range(args.episodes_per_config):
                resolved: dict[str, tuple[np.ndarray, np.ndarray]] = {}
                failure_label: str | None = None
                for label, position in zip(labels, positions, strict=True):
                    result = _resolve_reachable_orientation(
                        planner=planner,
                        env=ik_env,
                        sapien=sapien,
                        position=position,
                        primary_quat=pools[label][episode_idx],
                        seed_qpos=rest_qpos,
                        rng=rng,
                        cone_deg=args.orientation_cone_deg,
                        extra_attempts=args.extra_orientation_attempts,
                        suppress_planner_output=True,
                    )
                    if result is None:
                        failure_label = label
                        break
                    resolved[label] = result

                counts["total_episodes"] += 1
                config_counts["total"] += 1
                row: dict[str, Any] = {
                    "config": config_idx,
                    "episode": episode_idx,
                    "start_pos": positions[0].tolist(),
                    "goal_pos": positions[1].tolist(),
                }
                if failure_label is not None:
                    counts["ik_failed"] += 1
                    config_counts["ik_failed"] += 1
                    row["outcome"] = f"{failure_label}_ik_failed"
                    print(f"  [ep {episode_idx}] SKIP — {failure_label} pose not IK-reachable")
                    all_rows.append(row)
                    continue

                start_qpos, start_quat = resolved["start"]
                goal_quat = resolved["goal"][1]
                row["start_quat"] = start_quat.tolist()
                row["goal_quat"] = goal_quat.tolist()

                # Each pose being independently IK-reachable (from rest_qpos, above) is
                # NOT enough -- two individually-reachable poses can still require
                # incompatible arm configurations, with no collision-free path between
                # them. Verify the pair is INTER-feasible: a plan that actually starts
                # from the resolved START configuration and reaches the goal pose
                # (mirrors verify_pose_variety_reachability.py's ground-truth check),
                # not each pose re-seeded from rest_qpos. Skip the checkpoint entirely
                # if this fails -- testing robustness on a pair no continuous trajectory
                # can even connect isn't a fair test of the checkpoint.
                inter_feasible_plan = _plan_multisegment_trajectory(
                    planner=planner,
                    env=ik_env,
                    poses=[_pose_with_orientation(sapien, position=positions[1], quat=goal_quat)],
                    start_qpos=start_qpos,
                    suppress_planner_output=True,
                    smooth_trajectory=True,
                )
                if inter_feasible_plan is None:
                    counts["motion_plan_infeasible"] += 1
                    config_counts["motion_plan_infeasible"] += 1
                    row["outcome"] = "motion_plan_infeasible"
                    print(
                        f"  [ep {episode_idx}] SKIP — start and goal each individually "
                        "IK-reachable, but no collision-free plan connects them "
                        "(not inter-feasible)"
                    )
                    all_rows.append(row)
                    continue
                print(
                    f"  [ep {episode_idx}] reachable and inter-feasible — proceeding "
                    f"regardless of what the checkpoint does with it. tilt-from-down: "
                    f"start={_tilt_from_down_deg(start_quat):.0f}deg "
                    f"goal={_tilt_from_down_deg(goal_quat):.0f}deg"
                )

                _reset_obs, reset_info = sim_env.reset(
                    seed=args.seed + config_idx * 1000 + episode_idx, options={"reconfigure": True}
                )
                _set_robot_qpos(sim_env, start_qpos)
                _sync_pose_variety_markers(
                    sim_env,
                    start_pos=positions[0],
                    goal_pos=positions[1],
                )
                sim_obs, sim_info = _refresh_obs_after_manual_qpos(
                    sim_env, info=reset_info, gripper_open=args.gripper_open
                )
                sim_entry = rollout_observation_entry(
                    sim_obs, sim_info, env=sim_env, crop_config=crop_config
                )
                obs_window = make_initial_obs_window(sim_entry, n_obs_steps=int(policy.n_obs_steps))

                # Verify the goal-conditioning the policy ACTUALLY sees matches the goal
                # we just pointed goal_site at. This is read back out of the observation
                # itself (via info["extra"]["goal_pos"]), so a mismatch here means the
                # marker pose-set silently didn't land -- exactly the failure that
                # previously left the policy chasing the env's own random goal.
                conditioned_on = np.asarray(sim_entry["target_position"], dtype=np.float64)
                if not np.allclose(conditioned_on, positions[1].astype(np.float64), atol=1e-3):
                    print(
                        f"  [ep {episode_idx}] WARNING — policy goal-conditioning is "
                        f"{conditioned_on.round(3).tolist()} but the goal target is "
                        f"{positions[1].astype(np.float64).round(3).tolist()}; "
                        "goal_site did not take the new pose"
                    )
                else:
                    print(
                        f"  [ep {episode_idx}] policy conditioned on "
                        f"goal_xyz={conditioned_on.round(3).tolist()}"
                    )

                target_pos = positions[1]
                target_quat = goal_quat
                # Single fixed target for the whole episode, so build the constraint and
                # scene context once instead of rebuilding identical objects per replan.
                constraint = CartesianPoseConstraint(
                    target_position=target_pos,
                    target_orientation=target_quat,
                    position_tolerance=args.position_tolerance,
                    rotation_tolerance=args.rotation_tolerance,
                    weight=1.0,
                    name="pose_variety_goal",
                )
                scene = scene_context_for_constraints(
                    target_position=target_pos,
                    constraints=[constraint],
                    metadata={"config": config_idx, "episode": episode_idx},
                )
                timer = TimingRecorder(enabled=False)
                frames = [frame_to_numpy(sim_env.render())]
                ema_sim_action: np.ndarray | None = None
                goal_reached_step: int | None = None
                reached_goal = False
                truncated_early = False
                best_pos_err = float("inf")
                best_rot_err = float("inf")
                best_pos_err_step = -1
                best_rot_err_step = -1
                total_steps = 0
                was_training = policy.training
                policy.eval()
                try:
                    while total_steps < args.max_steps:
                        decision = _select_decision(
                            method="reranking",
                            adapter=adapter,
                            world_model=world_model,
                            provider=provider,
                            current_entry=sim_entry,
                            obs_window=obs_window,
                            scene=scene,
                            constraints=[constraint],
                            crop_config=crop_config,
                            goal_thresh=args.position_tolerance,
                            planning_horizon_chunks=1,
                            geometry_mode="exact",
                            k_schedule=tuple(args.k_schedule),
                            match_current_robot_points=True,
                            rng=rng,
                            timer=timer,
                        )
                        steps_to_execute = min(
                            decision.selected_chunk.horizon,
                            int(policy.n_action_steps),
                            args.max_steps - total_steps,
                        )
                        reached_goal = False
                        truncated_early = False
                        for policy_action in decision.selected_chunk.actions[:steps_to_execute]:
                            sim_action = policy_action_to_sim_action(
                                policy_action,
                                np.asarray(sim_entry["agent_pos"], dtype=np.float32),
                                action_mode=action_mode,
                                sim_action_dim=int(np.prod(sim_env.action_space.shape)),
                                low=getattr(sim_env.action_space, "low", None),
                                high=getattr(sim_env.action_space, "high", None),
                                gripper_open=args.gripper_open,
                            )
                            # EMA-smooth the executed action to damp the vibratory,
                            # per-replan-chunk-boundary jitter seen in raw chunk execution
                            # (mirrors abhinav/main's --action-ema-alpha in
                            # rollout_dp3_reach_policy.py / train_dp3_reach.py's rollout
                            # eval). State persists across the whole episode.
                            if ema_sim_action is None or args.action_ema_alpha >= 1.0:
                                ema_sim_action = sim_action
                            else:
                                ema_sim_action = (
                                    args.action_ema_alpha * sim_action
                                    + (1.0 - args.action_ema_alpha) * ema_sim_action
                                )
                            sim_obs, _reward, _terminated, truncated, sim_info = sim_env.step(ema_sim_action)
                            total_steps += 1
                            sim_entry = rollout_observation_entry(
                                sim_obs, sim_info, env=sim_env, crop_config=crop_config
                            )
                            obs_window = append_obs_window(
                                obs_window, sim_entry, n_obs_steps=int(policy.n_obs_steps)
                            )
                            frames.append(frame_to_numpy(sim_env.render()))

                            tcp = _tcp_pose(sim_env.unwrapped)
                            pos_err = float(np.linalg.norm(tcp[:3].astype(np.float64) - target_pos))
                            rot_err = _quat_angular_distance_rad(
                                tcp[3:7].astype(np.float64), target_quat.astype(np.float64)
                            )
                            # Track the best each error got INDEPENDENTLY over the whole
                            # episode. If rotation gets close at some point but the final
                            # frame is far, the policy can reach the orientation and then
                            # drifts off it (a tuning/stopping problem). If it never gets
                            # close at all, the orientation is simply outside what this
                            # position-conditioned checkpoint can produce (a capability
                            # limit that reranking over K samples can't fix).
                            if pos_err < best_pos_err:
                                best_pos_err = pos_err
                                best_pos_err_step = total_steps
                            if rot_err < best_rot_err:
                                best_rot_err = rot_err
                                best_rot_err_step = total_steps
                            if pos_err <= args.position_tolerance and rot_err <= args.rotation_tolerance:
                                reached_goal = True
                                goal_reached_step = total_steps
                                break
                            # NOTE: `terminated` is intentionally NOT treated as a stop
                            # condition here. PG3DReachEnv.evaluate() derives it from
                            # POSITION ALONE (tcp_to_goal_dist <= goal_thresh=0.025m by
                            # default -- see reach_env.py), with no orientation check at
                            # all -- almost the same threshold as this script's own
                            # --position-tolerance (0.02m). Breaking on it was cutting
                            # episodes off the instant position got close, before
                            # rotation could converge, and logging them "failed" even
                            # when the arm was still actively closing the last bit of
                            # rotation error. This script's own position+rotation check
                            # above is the sole authority on "reached" -- gymnasium
                            # doesn't require reset() immediately after terminated=True,
                            # so continuing to step is safe. Only `truncated` (a real
                            # episode-length/physics cutoff) still stops the episode.
                            if bool_any(truncated):
                                truncated_early = True
                                break
                        if truncated_early:
                            break
                        if reached_goal:
                            print(
                                f"  [ep {episode_idx}] goal reached at step {total_steps} "
                                f"(pos_err={pos_err:.4f} rot_err={np.degrees(rot_err):.1f}deg)"
                            )
                            break
                    if not reached_goal:
                        reason = (
                            f"episode truncated at step {total_steps}"
                            if truncated_early
                            else f"step budget ({args.max_steps}) exhausted"
                        )
                        print(
                            f"  [ep {episode_idx}] goal NOT reached — {reason}. "
                            f"final pos_err={pos_err:.4f} rot_err={np.degrees(rot_err):.1f}deg | "
                            f"BEST pos_err={best_pos_err:.4f}@step{best_pos_err_step} "
                            f"rot_err={np.degrees(best_rot_err):.1f}deg@step{best_rot_err_step} "
                            f"(tolerances: pos<={args.position_tolerance} "
                            f"rot<={np.degrees(args.rotation_tolerance):.1f}deg)"
                        )
                finally:
                    if was_training:
                        policy.train()

                row["goal_reached_step"] = goal_reached_step
                row["total_steps"] = total_steps
                row["final_pos_err"] = float(pos_err)
                row["final_rot_err_deg"] = float(np.degrees(rot_err))
                row["best_pos_err"] = float(best_pos_err)
                row["best_pos_err_step"] = best_pos_err_step
                row["best_rot_err_deg"] = float(np.degrees(best_rot_err))
                row["best_rot_err_step"] = best_rot_err_step
                if goal_reached_step is not None:
                    counts["goal_reached"] += 1
                    config_counts["goal_reached"] += 1
                row["outcome"] = "goal_reached" if goal_reached_step is not None else "failed"

                # Video is saved unconditionally for every IK-reachable episode -- success
                # or failure -- since a robustness test needs to SEE the failures too, not
                # just the ones that happened to succeed.
                if args.video_dir is not None:
                    video_path = args.video_dir / f"config_{config_idx:02d}_episode_{episode_idx:02d}.mp4"
                    save_video(video_path, frames, fps=args.video_fps)
                    row["video"] = str(video_path)
                    print(f"  [ep {episode_idx}] outcome={row['outcome']}  video: {video_path}")

                all_rows.append(row)

            attempted = (
                config_counts["total"]
                - config_counts["ik_failed"]
                - config_counts["motion_plan_infeasible"]
            )
            print(
                f"  — config {config_idx:02d} success rate: "
                f"goal_reached={config_counts['goal_reached']}/{attempted} attempted "
                f"({100 * config_counts['goal_reached'] / max(attempted, 1):.1f}%), "
                f"ik_unreachable={config_counts['ik_failed']}/{config_counts['total']}, "
                f"not_inter_feasible={config_counts['motion_plan_infeasible']}/{config_counts['total']}"
            )
    finally:
        ik_solver.close()
        ik_env.close()
        sim_env.close()
        ghost_env.close()

    attempted_total = (
        counts["total_episodes"] - counts["ik_failed"] - counts["motion_plan_infeasible"]
    )
    print("\n── Summary (checkpoint robustness across all configs)")
    print(f"   total episodes         : {counts['total_episodes']}")
    print(f"   IK-unreachable (skipped): {counts['ik_failed']}")
    print(f"   not inter-feasible (skipped): {counts['motion_plan_infeasible']}")
    print(f"   attempted by checkpoint : {attempted_total}")
    print(
        f"   goal success rate       : {counts['goal_reached']}/{attempted_total} "
        f"({100 * counts['goal_reached'] / max(attempted_total, 1):.1f}% of attempted, "
        f"{100 * counts['goal_reached'] / max(counts['total_episodes'], 1):.1f}% of all)"
    )

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps({"counts": counts, "episodes": all_rows}, indent=2))
        print(f"\nwrote summary: {args.summary_json}")

    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Checkpoint-based pose-variety reachability test: DP3 + reranking steered "
            "from an IK-verified random start pose to a random goal pose (same 2 "
            "positions, diverse orientations across episodes-per-config)."
        )
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Used only for metadata.json (env_id/env_kwargs/crop/action_mode) -- "
        "the observation shape the checkpoint was trained on. No episode content is read.",
    )
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--variant", choices=["nogripper", "gripper"], default="gripper")
    p.add_argument("--num-configs", type=int, default=4, help="Distinct start/goal position pairs.")
    p.add_argument("--episodes-per-config", type=int, default=5, help="Orientation variants per config.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--orientation-cone-deg",
        type=float,
        default=60.0,
        help=(
            "Max tilt (degrees) of sampled start/goal orientations away from "
            "straight-down. Samples are drawn uniform-by-solid-angle within this cone "
            "(see _sample_broad_orientation_sapien), so a smaller value keeps the "
            "tested orientations closer to the training distribution's mode."
        ),
    )
    p.add_argument("--extra-orientation-attempts", type=int, default=3)
    p.add_argument(
        "--randomize-start-orientation",
        action="store_true",
        default=False,
        help=(
            "By default start orientation is fixed straight-down (the natural rest "
            "approach) and only the GOAL orientation is randomized within "
            "--orientation-cone-deg -- isolates testing to 'can the checkpoint steer "
            "to an arbitrary goal orientation' without also demanding recovery from "
            "an awkward start. Pass this to randomize start orientation too."
        ),
    )
    p.add_argument(
        "--min-height",
        type=float,
        default=0.10,
        help=(
            "Minimum start/goal height (m) above the table. Raised from the "
            "XARM7_REACH_WORKSPACE_BOUNDS default of 0.05 -- low + heavily tilted goal "
            "poses let the gripper BODY (fingers/wrist, not just the TCP point) dip "
            "into the table even when the TCP itself clears --min-height, since a "
            "tilted wrist extends below/around the TCP frame."
        ),
    )
    p.add_argument(
        "--max-height",
        type=float,
        default=0.25,
        help="Maximum start/goal height (m) above the table. XARM7_REACH_WORKSPACE_BOUNDS default is 0.55.",
    )
    p.add_argument(
        "--min-pairwise-distance",
        type=float,
        default=0.15,
        help="Minimum required separation (m) between the start and goal positions.",
    )
    p.add_argument("--position-resample-attempts", type=int, default=1)
    p.add_argument("--position-tolerance", type=float, default=0.02)
    p.add_argument("--rotation-tolerance", type=float, default=0.35, help="Radians.")
    p.add_argument(
        "--max-steps",
        type=int,
        default=120,
        help="Sim-step budget for the single start->goal reach before it's called failed.",
    )
    p.add_argument(
        "--action-ema-alpha",
        type=float,
        default=0.6,
        help=(
            "EMA smoothing factor applied to each executed sim action, persisted across "
            "the whole episode. 1.0 = no smoothing (raw, vibratory chunk-boundary jitter); "
            "lower = heavier smoothing. Matches abhinav/main's --action-ema-alpha."
        ),
    )
    p.add_argument("--k-schedule", type=int, nargs="+", default=[16, 32, 64])
    p.add_argument("--policy-batch-size", type=int, default=64)
    p.add_argument("--gripper-open", type=float, default=0.04)
    p.add_argument("--video-dir", type=Path, default=None)
    p.add_argument("--video-fps", type=int, default=10)
    p.add_argument("--summary-json", type=Path, default=None)
    args = p.parse_args(argv)
    if args.num_configs <= 0:
        raise ValueError("--num-configs must be positive")
    if args.episodes_per_config <= 0:
        raise ValueError("--episodes-per-config must be positive")
    if args.extra_orientation_attempts < 0:
        raise ValueError("--extra-orientation-attempts must be non-negative")
    if args.min_pairwise_distance < 0.0:
        raise ValueError("--min-pairwise-distance must be non-negative")
    if args.min_height < 0.0:
        raise ValueError("--min-height must be non-negative")
    if args.max_height <= args.min_height:
        raise ValueError("--max-height must be greater than --min-height")
    if args.position_resample_attempts <= 0:
        raise ValueError("--position-resample-attempts must be positive")
    if args.position_tolerance < 0.0 or args.rotation_tolerance < 0.0:
        raise ValueError("tolerances must be non-negative")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if not (0.0 < args.action_ema_alpha <= 1.0):
        raise ValueError("--action-ema-alpha must be in (0.0, 1.0]")
    if args.video_fps <= 0:
        raise ValueError("--video-fps must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(main())
