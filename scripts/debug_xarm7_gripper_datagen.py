#!/usr/bin/env python
"""Debug the xArm7 gripper data-gen for exactly 1 seed — saves one MP4 per family.

Runs the exact same sampling + planning pipeline as write_xarm7_gripper_reach_dataset.py
but with render_mode="rgb_array" and frame capture on every env.step, so you can
visually verify trajectory quality for all 12 families without writing a zarr.

Usage:
    # 1 seed, default seed (first one the real script would use)
    python scripts/debug_xarm7_gripper_datagen.py

    # Fixed seed
    python scripts/debug_xarm7_gripper_datagen.py --seed 42

    # Custom output dir
    python scripts/debug_xarm7_gripper_datagen.py --output-dir artifacts/datagen_debug
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np


# ── helpers ──────────────────────────────────────────────────────────────────

def _to_uint8(frame: Any) -> np.ndarray:
    import torch
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    arr = np.asarray(frame)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    return arr


def _render_frame(env: Any) -> np.ndarray | None:
    frame = env.render()
    if frame is None:
        return None
    return _to_uint8(frame)


def _save_mp4(frames: list[np.ndarray], path: Path, fps: int) -> None:
    import imageio.v2 as imageio
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), frames, fps=fps)
    print(f"  saved: {path}  ({len(frames)} frames)")


# ── replay with frame capture ─────────────────────────────────────────────────

_ACCEPTANCE_DIST = 0.025  # matches env goal threshold + data-gen acceptance (2.5 cm)


def _replay_and_record(
    *,
    env: Any,
    seed: int,
    start_qpos: np.ndarray,
    start_tcp_pose: np.ndarray,
    variant: dict[str, Any],
    max_steps: int,
    settle_steps: int,
    hold_steps: int,
    gripper_open: float,
    fps: int,
    output_dir: Path,
) -> bool:
    """Replay one family variant, capturing frames, return success flag.

    Success mirrors the real data-gen: env strict flag (tcp ≤ 2.5 cm) OR
    min_distance ≤ _ACCEPTANCE_DIST (3.0 cm) at any step.
    """
    from dataset_generation.write_maniskill_reach_dataset import (
        _format_sim_action,
        _hold_sim_action,
        _refresh_obs_after_manual_qpos,
        _set_robot_qpos,
        _set_start_site_pose,
    )
    from pg3d.utils.arrays import bool_any as _bool_any, bool_info as _bool_info, float_info as _float_info

    family_name = variant["name"]
    family_id   = variant["trajectory_type"]
    positions   = np.asarray(variant["positions"])

    obs, info = env.reset(seed=seed, options={"reconfigure": False})
    _set_robot_qpos(env, start_qpos)
    _set_start_site_pose(env, start_tcp_pose[:3])
    obs, info = _refresh_obs_after_manual_qpos(env, info=info, gripper_open=gripper_open)

    frames: list[np.ndarray] = []
    f = _render_frame(env)
    if f is not None:
        frames.append(f)

    env_success = False
    first_success_step = None
    hold_qpos = None  # qpos captured where success was first reached (hold locks to it)
    planned_step_limit = max(1, max_steps - settle_steps)
    settle_recorded = 0
    hold_recorded   = 0
    final_pos = positions[-1]
    distances: list[float] = []

    def _dist(info: Any) -> float:
        return float(_float_info(info, "tcp_to_goal_dist", default=float("inf")))

    def _cur_qpos() -> np.ndarray:
        return np.asarray(env.unwrapped.agent.robot.qpos).reshape(-1)

    # planned steps
    for pos in positions[:planned_step_limit]:
        sim_action = _format_sim_action(env, pos)
        obs, _r, terminated, truncated, info = env.step(sim_action)
        f = _render_frame(env)
        if f is not None:
            frames.append(f)
        distances.append(_dist(info))
        if _bool_info(info, "success"):
            env_success = True
            first_success_step = len(frames)
            hold_qpos = _cur_qpos()
            break
        if _bool_any(terminated) or _bool_any(truncated):
            break

    # settle steps (if not yet env-successful)
    while first_success_step is None and settle_recorded < settle_steps:
        sim_action = _format_sim_action(env, final_pos)
        obs, _r, _term, truncated, info = env.step(sim_action)
        f = _render_frame(env)
        if f is not None:
            frames.append(f)
        distances.append(_dist(info))
        settle_recorded += 1
        if _bool_info(info, "success"):
            env_success = True
            first_success_step = len(frames)
            hold_qpos = _cur_qpos()
            break
        if _bool_any(truncated):
            break

    # apply data-gen acceptance rule: near-miss within threshold still counts
    min_dist = float(min(distances)) if distances else float("inf")
    accepted = env_success or (min_dist <= _ACCEPTANCE_DIST)
    if accepted and first_success_step is None:
        first_success_step = int(np.argmin(np.asarray(distances, dtype=np.float32))) + 1

    # hold steps: lock to the pose where success was first reached (falls back to the
    # current qpos for near-miss acceptances that never tripped the strict flag).
    while first_success_step is not None and hold_recorded < hold_steps:
        sim_action = _hold_sim_action(env, gripper_open=gripper_open, qpos=hold_qpos)
        obs, _r, _term, truncated, info = env.step(sim_action)
        f = _render_frame(env)
        if f is not None:
            frames.append(f)
        hold_recorded += 1
        if _bool_any(truncated):
            break

    status = "ok" if accepted else "fail"
    print(f"min_dist={min_dist*100:.1f}cm  env_success={env_success}  accepted={accepted}", end="  ")
    out = output_dir / f"family{family_id:02d}_{family_name}_{status}.mp4"
    _save_mp4(frames, out, fps)
    return accepted


# ── main ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
        import sapien
    except Exception as exc:
        print(f"ManiSkill import failed: {exc}", file=sys.stderr)
        return 2

    from pg3d.envs.xarm_adapter import register_pg3d_xarm7_gripper_reach_envs
    from pg3d.envs.xarm_adapter.motionplanner import XArm7GripperMotionPlanningSolver
    from pg3d.envs.xarm_adapter.reach_config import (
        XARM7_REACH_WORKSPACE_BOUNDS,
        XARM7_WORKSPACE_BOUNDS,
    )
    from dataset_generation.write_maniskill_reach_dataset import (
        _bounds_center_half_extents,
        _goal_pose,
        _inset_xy_bounds,
        _plan_to_pose,
        _pose_with_orientation,
        _robot_base_position,
        _sample_reachable_start,
        _set_robot_qpos,
        _set_start_site_pose,
        _tcp_pose,
        _waypoint_workspace_bounds,
        generate_multimodal_waypoints,
        parse_args as _parse_args_gen,
    )

    register_pg3d_xarm7_gripper_reach_envs()

    # Build args object using the real parser with the same xArm7 defaults
    from dataset_generation.write_xarm7_gripper_reach_dataset import _XARM7_GRIPPER_DEFAULTS, _bounds_flag
    injected = (
        _XARM7_GRIPPER_DEFAULTS
        + _bounds_flag("--reach-workspace-bounds", XARM7_REACH_WORKSPACE_BOUNDS)
        + _bounds_flag("--workspace-bounds", XARM7_WORKSPACE_BOUNDS)
        + ["--num-demos", "12"]  # 1 seed = 12 families
    )
    if args.curved_paths:
        injected += ["--curved-paths", "--curvature-std", str(args.curvature_std)]
    gen_args = _parse_args_gen(injected)

    # Camera for a nice 3rd-person view (same as preview script)
    from mani_skill.utils import sapien_utils
    vis_pose = sapien_utils.look_at(eye=[0.7, 0.0, 0.45], target=[-0.315, 0.0, 0.22])
    p = np.asarray(vis_pose.p).reshape(-1)
    q = np.asarray(vis_pose.q).reshape(-1)
    vis_cam = {
        "pose": [float(p[0]), float(p[1]), float(p[2]),
                 float(q[0]), float(q[1]), float(q[2]), float(q[3])],
        "width": 640, "height": 480,
        "fov": float(np.deg2rad(60)), "near": 0.1, "far": 10.0,
    }

    goal_center, goal_half = _bounds_center_half_extents(XARM7_REACH_WORKSPACE_BOUNDS)
    env_kwargs = dict(
        obs_mode=gen_args.obs_mode,
        control_mode=gen_args.control_mode,
        render_mode="rgb_array",
        robot_uids="xarm7_gripper",
        num_envs=1,
        max_episode_steps=int(gen_args.max_steps_per_demo),
        sim_backend=gen_args.sim_backend,
        render_backend=gen_args.render_backend,
        sensor_configs={"shader_pack": gen_args.shader},
        goal_center=tuple(float(v) for v in goal_center),
        goal_half_extents=tuple(float(v) for v in goal_half),
        goal_regions=(),
        human_render_camera_configs={"render_camera": vis_cam},
    )

    seed = args.seed
    print(f"debug data-gen: seed={seed}  output={out_dir}")

    env: Any | None = None
    try:
        env = gym.make("PG3DReach-XArm7-Gripper-Workspace-v0", **env_kwargs)
        unwrapped = env.unwrapped

        # ── initial reset ────────────────────────────────────────────────────
        obs, info = env.reset(seed=seed, options={"reconfigure": True})
        goal_pose_  = _goal_pose(unwrapped, sapien)
        reset_tcp   = _tcp_pose(unwrapped)
        reset_qpos  = np.asarray(unwrapped.agent.robot.get_qpos()).reshape(-1).astype(np.float64)
        rng         = np.random.default_rng(seed)
        base_pos    = _robot_base_position(unwrapped)

        waypoint_bounds = _inset_xy_bounds(
            _waypoint_workspace_bounds(
                gen_args.env_id,
                __import__("pg3d.envs.maniskill_adapter.dataset",
                           fromlist=["PointCloudCropConfig"]).PointCloudCropConfig(
                    bounds=np.asarray(gen_args.workspace_bounds),
                    num_points=gen_args.num_points,
                    robot_point_fraction=gen_args.robot_point_fraction,
                ),
                reach_workspace_bounds=XARM7_REACH_WORKSPACE_BOUNDS,
            ),
            gen_args.table_margin,
        )
        start_bounds = _inset_xy_bounds(
            np.asarray(XARM7_REACH_WORKSPACE_BOUNDS, dtype=np.float64),
            gen_args.table_margin,
        )
        if waypoint_bounds is None or start_bounds is None:
            print("ERROR: waypoint/start bounds too tight after margin inset", file=sys.stderr)
            return 1

        goal_xyz = np.asarray(goal_pose_.p, dtype=np.float64).reshape(-1, 3)[0]
        print(f"goal xyz: {goal_xyz.astype(np.float32).tolist()}")

        # ── build planner ────────────────────────────────────────────────────
        planner = XArm7GripperMotionPlanningSolver(
            env, debug=False, vis=False,
            base_pose=unwrapped.agent.robot.pose,
            visualize_target_grasp_pose=False,
            print_env_info=False,
        )
        try:
            # ── sample reachable start ────────────────────────────────────────
            start_sample = _sample_reachable_start(
                env=env, planner=planner, sapien=sapien, rng=rng,
                reset_qpos=reset_qpos, reset_tcp_pose=reset_tcp,
                goal_pose=goal_pose_, start_bounds=start_bounds,
                randomize_start=gen_args.randomize_start,
                max_attempts=gen_args.start_sample_attempts,
                min_start_goal_distance=gen_args.min_start_goal_distance,
                min_base_clearance=gen_args.min_base_clearance,
                suppress_planner_output=True,
            )
            if start_sample is None:
                print("REJECTED: no reachable start sample found", file=sys.stderr)
                return 1
            start_qpos, start_tcp, start_meta = start_sample
            goal_pose_ = _pose_with_orientation(
                sapien, position=goal_xyz, quat=start_tcp[3:7])
            print(f"start tcp: {start_tcp[:3].astype(np.float32).tolist()}  "
                  f"attempt={start_meta.get('attempt')}")

            # Check goal reachable from start
            gr = _plan_to_pose(planner=planner, env=env, pose=goal_pose_,
                               start_qpos=start_qpos, suppress_planner_output=True)
            if gr is None:
                print("REJECTED: goal not reachable from start", file=sys.stderr)
                return 1
            print(f"goal reachable from start: status={gr[1]}")

            _set_robot_qpos(env, start_qpos)
            _set_start_site_pose(env, start_tcp[:3])

            # ── generate all 12 family variants ──────────────────────────────
            print("planning 12 families…")
            variants = generate_multimodal_waypoints(
                current_tcp_pose=start_tcp,
                goal_pose=goal_pose_,
                workspace_bounds=waypoint_bounds,
                robot_base_position=base_pos,
                planner=planner,
                env=env,
                sapien=sapien,
                rng=rng,
                variants_per_reset=gen_args.trajectory_variants_per_reset,
                max_attempts=gen_args.waypoint_attempts,
                min_base_clearance=gen_args.min_base_clearance,
                waypoint_xy_noise=gen_args.waypoint_xy_noise,
                waypoint_z_noise=gen_args.waypoint_z_noise,
                lateral_z_offset=gen_args.lateral_z_offset,
                vertical_lateral_offset=gen_args.vertical_lateral_offset,
                min_curve_offset=gen_args.min_curve_offset,
                max_joint_step=gen_args.max_joint_step,
                max_joint_accel=gen_args.max_joint_accel,
                max_raw_plan_multiplier=gen_args.max_raw_plan_multiplier,
                progress_interval=gen_args.progress_interval,
                max_replay_plan_steps=max(1, gen_args.max_steps_per_demo - gen_args.settle_steps),
                seed=seed,
                start_qpos=start_qpos,
                suppress_planner_output=True,
                smooth_trajectory=gen_args.smooth_trajectory,
                curved_paths=gen_args.curved_paths,
                curvature_std=gen_args.curvature_std,
            )
        finally:
            planner.close()

        print(f"\nplanned {len(variants)}/12 families; replaying + recording…\n")

        successes = []
        for variant in variants:
            fid   = variant["trajectory_type"]
            fname = variant["name"]
            nstep = np.asarray(variant["positions"]).shape[0]
            print(f"  family {fid:02d} {fname:20s}  planned_steps={nstep}", end="  ", flush=True)
            ok = _replay_and_record(
                env=env,
                seed=seed,
                start_qpos=start_qpos,
                start_tcp_pose=start_tcp,
                variant=variant,
                max_steps=gen_args.max_steps_per_demo,
                settle_steps=gen_args.settle_steps,
                hold_steps=gen_args.hold_steps,
                gripper_open=gen_args.gripper_open,
                fps=args.fps,
                output_dir=out_dir,
            )
            successes.append(ok)

        n_ok = sum(successes)
        print(f"\n{'='*60}")
        print(f"seed {seed}: {n_ok}/{len(variants)} families reached goal")
        print(f"videos: {out_dir}/")

    finally:
        if env is not None:
            env.close()

    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Debug xArm7 gripper data-gen: 1 seed → 12 family MP4s.")
    p.add_argument("--seed", type=int, default=0, help="Env seed (default: 0)")
    p.add_argument("--fps", type=int, default=10, help="Video FPS (default: 10)")
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/datagen_debug"),
                   help="Directory to write per-family MP4s")
    p.add_argument("--curved-paths", action="store_true", default=False,
                   help="Reproduce curved-path waypoint sampling (matches production data-gen).")
    p.add_argument("--curvature-std", type=float, default=0.10,
                   help="Gaussian curvature std when --curved-paths (default matches production: 0.10).")
    return p.parse_args(argv if argv is not None else sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
