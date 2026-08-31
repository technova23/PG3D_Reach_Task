#!/usr/bin/env python
"""Checkpoint-based pose-variety reachability test for xArm7.

Complements ``scripts/verify_pose_variety_reachability.py`` (pure ground-truth
mplib motion planning, no policy involved). This script instead loads a
trained DP3 checkpoint and tests whether the POLICY -- steered via reranking,
since the checkpoint has no direct conditioning input for a target
*orientation* (only goal position) -- can actually reach an arbitrary
start/waypoint/goal pose triple.

No GraspGen. Design:

* One "config" = one fixed (start, waypoint, goal) POSITION triple.
* For that config, run --episodes-per-config episodes that reuse the SAME
  three positions but each get their own INDEPENDENT, FPS-diversified random
  orientation at start/waypoint/goal (mirrors --randomize-start-goal-orientation
  data generation's own pool-then-spread selection, so the orientations tested
  across episodes are spread apart rather than clustered).
* Each pose (position, orientation) is first IK-verified with mplib (ground
  truth, same as verify_pose_variety_reachability.py) before any policy
  rollout is attempted -- a trial whose start/waypoint/goal isn't even
  IK-reachable is not a fair test of the checkpoint and is skipped.
* For every IK-reachable triple: one continuous episode, two stages. Reset at
  the start pose, run rejection-free reranking (K candidate DP3 action chunks
  per replan, scored by imagined-rollout distance to a CartesianPoseConstraint,
  lowest cost executed) toward the WAYPOINT pose; once within tolerance (or a
  step budget expires), swap the steering target to the GOAL pose and continue
  in the SAME rollout/video. Reranking requires --geometry-mode exact (fast
  imagination freezes orientation at the window's starting value, which would
  silently make orientation-aware steering meaningless).

Usage:
    python scripts/eval_pose_variety_checkpoint_reachability.py \
        --checkpoint artifacts/xarm7_pose_variety_5000_checkpoints/final_step_00080000.pt \
        --dataset artifacts/pg3d_xarm7_pose_variety_reach_5000.zarr \
        --num-configs 4 --episodes-per-config 5 \
        --video-dir artifacts/pose_variety_checkpoint_eval \
        --summary-json artifacts/pose_variety_checkpoint_eval/summary.json

``--dataset`` is used ONLY to read metadata.json (env_id/env_kwargs/crop
config/action_mode) -- the exact observation shape the checkpoint was
trained on -- never to source episode content; every start/waypoint/goal
pose here is generated fresh by this script.
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
    waypoint / goal) across a config's --episodes-per-config episodes are spread
    apart (FPS worked) or accidentally clustered -- independent of whether each
    one turned out to be IK-reachable or the checkpoint reached it.
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


def _sample_workspace_triple(
    rng: np.random.Generator,
    *,
    bounds_world: np.ndarray,
    min_pairwise_distance: float,
    resample_attempts: int,
) -> list[np.ndarray]:
    bounds = np.asarray(bounds_world, dtype=np.float32).reshape(3, 2)

    def sample_one() -> np.ndarray:
        return rng.uniform(bounds[:, 0], bounds[:, 1]).astype(np.float32)

    for _ in range(max(resample_attempts, 1)):
        points = [sample_one() for _ in range(3)]
        ok = all(
            float(np.linalg.norm(points[i] - points[j])) >= min_pairwise_distance
            for i in range(3)
            for j in range(i + 1, 3)
        )
        if ok:
            return points
    return points


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    import torch

    from dataset_generation.write_maniskill_reach_dataset import (
        _farthest_point_select_orientations,
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

    sim_env = gym.make(str(metadata["env_id"]), **_env_kwargs(metadata, render_mode="rgb_array"))
    ghost_env = gym.make(str(metadata["env_id"]), **_env_kwargs(metadata, render_mode=None))

    adapter = DP3ChunkPolicyAdapter(
        policy, action_mode=action_mode, device=device, policy_batch_size=args.policy_batch_size
    )
    provider = ManiSkillGhostPandaGeometryProvider(
        ghost_env, task_name=str(metadata.get("env_id", "unknown")), crop_bounds=crop_config.bounds
    )
    world_model = GeometricWorldModel(provider)

    rng = np.random.default_rng(args.seed)
    bounds_world = np.asarray(XARM7_REACH_WORKSPACE_BOUNDS, dtype=np.float32)

    if args.video_dir is not None:
        args.video_dir.mkdir(parents=True, exist_ok=True)

    labels = ("start", "waypoint", "goal")
    all_rows: list[dict[str, Any]] = []
    counts = {
        "total_episodes": 0,
        "ik_failed": 0,
        "waypoint_reached": 0,
        "goal_reached": 0,
    }

    try:
        for config_idx in range(args.num_configs):
            positions = _sample_workspace_triple(
                rng,
                bounds_world=bounds_world,
                min_pairwise_distance=args.min_pairwise_distance,
                resample_attempts=args.position_resample_attempts,
            )
            print(
                f"\n=== config {config_idx:02d} — start={positions[0].tolist()} "
                f"waypoint={positions[1].tolist()} goal={positions[2].tolist()} ==="
            )

            pools: dict[str, list[np.ndarray]] = {}
            for label in labels:
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

            config_counts = {"total": 0, "ik_failed": 0, "waypoint_reached": 0, "goal_reached": 0}
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
                    "waypoint_pos": positions[1].tolist(),
                    "goal_pos": positions[2].tolist(),
                }
                if failure_label is not None:
                    counts["ik_failed"] += 1
                    config_counts["ik_failed"] += 1
                    row["outcome"] = f"{failure_label}_ik_failed"
                    print(f"  [ep {episode_idx}] SKIP — {failure_label} pose not IK-reachable")
                    all_rows.append(row)
                    continue

                start_qpos, start_quat = resolved["start"]
                waypoint_quat = resolved["waypoint"][1]
                goal_quat = resolved["goal"][1]
                row["start_quat"] = start_quat.tolist()
                row["waypoint_quat"] = waypoint_quat.tolist()
                row["goal_quat"] = goal_quat.tolist()
                print(
                    f"  [ep {episode_idx}] IK-reachable — proceeding regardless of what the "
                    f"checkpoint does with it. tilt-from-down: "
                    f"start={_tilt_from_down_deg(start_quat):.0f}deg "
                    f"waypoint={_tilt_from_down_deg(waypoint_quat):.0f}deg "
                    f"goal={_tilt_from_down_deg(goal_quat):.0f}deg"
                )

                stage_targets = [
                    ("waypoint", positions[1], waypoint_quat),
                    ("goal", positions[2], goal_quat),
                ]

                _reset_obs, reset_info = sim_env.reset(
                    seed=args.seed + config_idx * 1000 + episode_idx, options={"reconfigure": True}
                )
                _set_robot_qpos(sim_env, start_qpos)
                sim_obs, sim_info = _refresh_obs_after_manual_qpos(
                    sim_env, info=reset_info, gripper_open=args.gripper_open
                )
                sim_entry = rollout_observation_entry(
                    sim_obs, sim_info, env=sim_env, crop_config=crop_config
                )
                obs_window = make_initial_obs_window(sim_entry, n_obs_steps=int(policy.n_obs_steps))

                timer = TimingRecorder(enabled=False)
                frames = [sim_env.render()]
                stage_idx = 0
                stage_steps = 0
                stage_reached_step: dict[str, int | None] = {"waypoint": None, "goal": None}
                total_steps = 0
                was_training = policy.training
                policy.eval()
                try:
                    while total_steps < args.max_steps_per_stage * len(stage_targets):
                        target_label, target_pos, target_quat = stage_targets[stage_idx]
                        constraint = CartesianPoseConstraint(
                            target_position=target_pos,
                            target_orientation=target_quat,
                            position_tolerance=args.position_tolerance,
                            rotation_tolerance=args.rotation_tolerance,
                            weight=1.0,
                            name=f"pose_variety_{target_label}",
                        )
                        scene = scene_context_for_constraints(
                            target_position=target_pos,
                            constraints=[constraint],
                            metadata={"config": config_idx, "episode": episode_idx, "stage": target_label},
                        )
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
                            args.max_steps_per_stage * len(stage_targets) - total_steps,
                        )
                        reached_this_stage = False
                        terminated_or_truncated = False
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
                            sim_obs, _reward, terminated, truncated, sim_info = sim_env.step(sim_action)
                            total_steps += 1
                            stage_steps += 1
                            sim_entry = rollout_observation_entry(
                                sim_obs, sim_info, env=sim_env, crop_config=crop_config
                            )
                            obs_window = append_obs_window(
                                obs_window, sim_entry, n_obs_steps=int(policy.n_obs_steps)
                            )
                            frames.append(sim_env.render())

                            tcp = _tcp_pose(sim_env.unwrapped)
                            pos_err = float(np.linalg.norm(tcp[:3].astype(np.float64) - target_pos))
                            rot_err = _quat_angular_distance_rad(
                                tcp[3:7].astype(np.float64), target_quat.astype(np.float64)
                            )
                            if pos_err <= args.position_tolerance and rot_err <= args.rotation_tolerance:
                                reached_this_stage = True
                                stage_reached_step[target_label] = total_steps
                                break
                            if bool(np.any(terminated)) or bool(np.any(truncated)):
                                terminated_or_truncated = True
                                break
                        if terminated_or_truncated:
                            break
                        if reached_this_stage:
                            print(
                                f"  [ep {episode_idx}] {target_label} reached at step {total_steps} "
                                f"(pos_err={pos_err:.4f} rot_err={np.degrees(rot_err):.1f}deg)"
                            )
                            if stage_idx == len(stage_targets) - 1:
                                break
                            stage_idx += 1
                            stage_steps = 0
                        elif stage_steps >= args.max_steps_per_stage:
                            print(f"  [ep {episode_idx}] {target_label} FAILED — step budget exhausted")
                            break
                finally:
                    if was_training:
                        policy.train()

                row["waypoint_reached_step"] = stage_reached_step["waypoint"]
                row["goal_reached_step"] = stage_reached_step["goal"]
                row["total_steps"] = total_steps
                if stage_reached_step["waypoint"] is not None:
                    counts["waypoint_reached"] += 1
                    config_counts["waypoint_reached"] += 1
                if stage_reached_step["goal"] is not None:
                    counts["goal_reached"] += 1
                    config_counts["goal_reached"] += 1
                row["outcome"] = (
                    "goal_reached"
                    if stage_reached_step["goal"] is not None
                    else ("waypoint_reached_only" if stage_reached_step["waypoint"] is not None else "failed")
                )

                # Video is saved unconditionally for every IK-reachable episode -- success
                # or failure -- since a robustness test needs to SEE the failures too, not
                # just the ones that happened to succeed.
                if args.video_dir is not None:
                    video_path = args.video_dir / f"config_{config_idx:02d}_episode_{episode_idx:02d}.mp4"
                    save_video(video_path, [np.asarray(f) for f in frames], fps=args.video_fps)
                    row["video"] = str(video_path)
                    print(f"  [ep {episode_idx}] outcome={row['outcome']}  video: {video_path}")

                all_rows.append(row)

            attempted = config_counts["total"] - config_counts["ik_failed"]
            print(
                f"  — config {config_idx:02d} success rate: "
                f"goal_reached={config_counts['goal_reached']}/{attempted} attempted "
                f"({100 * config_counts['goal_reached'] / max(attempted, 1):.1f}%), "
                f"waypoint_reached={config_counts['waypoint_reached']}/{attempted} "
                f"({100 * config_counts['waypoint_reached'] / max(attempted, 1):.1f}%), "
                f"ik_unreachable={config_counts['ik_failed']}/{config_counts['total']}"
            )
    finally:
        ik_solver.close()
        ik_env.close()
        sim_env.close()
        ghost_env.close()

    attempted_total = counts["total_episodes"] - counts["ik_failed"]
    print("\n── Summary (checkpoint robustness across all configs)")
    print(f"   total episodes         : {counts['total_episodes']}")
    print(f"   IK-unreachable (skipped): {counts['ik_failed']}")
    print(f"   attempted by checkpoint : {attempted_total}")
    print(
        f"   waypoint success rate   : {counts['waypoint_reached']}/{attempted_total} "
        f"({100 * counts['waypoint_reached'] / max(attempted_total, 1):.1f}% of attempted, "
        f"{100 * counts['waypoint_reached'] / max(counts['total_episodes'], 1):.1f}% of all)"
    )
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
            "through IK-verified random start/waypoint/goal poses (same 3 positions, "
            "diverse orientations across episodes-per-config)."
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
    p.add_argument("--num-configs", type=int, default=4, help="Distinct start/waypoint/goal position triples.")
    p.add_argument("--episodes-per-config", type=int, default=5, help="Orientation variants per config.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--orientation-cone-deg", type=float, default=120.0)
    p.add_argument("--extra-orientation-attempts", type=int, default=3)
    p.add_argument("--min-pairwise-distance", type=float, default=0.15)
    p.add_argument("--position-resample-attempts", type=int, default=1)
    p.add_argument("--position-tolerance", type=float, default=0.02)
    p.add_argument("--rotation-tolerance", type=float, default=0.35, help="Radians.")
    p.add_argument("--max-steps-per-stage", type=int, default=60)
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
    if args.position_resample_attempts <= 0:
        raise ValueError("--position-resample-attempts must be positive")
    if args.position_tolerance < 0.0 or args.rotation_tolerance < 0.0:
        raise ValueError("tolerances must be non-negative")
    if args.max_steps_per_stage <= 0:
        raise ValueError("--max-steps-per-stage must be positive")
    if args.video_fps <= 0:
        raise ValueError("--video-fps must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(main())
