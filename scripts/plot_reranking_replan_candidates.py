"""Visualize the 32-candidate bundle at every reranking replan step, top-down (XY).

Debug tool for the avoid_projection selection issue: reuses the real
eval_constrained_reach.py machinery (constraint placement, DP3ChunkPolicyAdapter,
GeometricWorldModel, `_select_multichunk`) so the candidate bundle and selection
shown here are byte-for-byte what a real `--constraint-type projection
--methods reranking` eval run would produce -- just rendered per replan step
instead of only reported in aggregate metrics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from pg3d.constraints import AvoidProjection, AvoidRegion, BoxRegion, SphereRegion
from pg3d.envs.maniskill_adapter import (
    ManiSkillGhostPandaGeometryProvider,
    register_pg3d_reach_envs,
)
from pg3d.envs.maniskill_adapter.dataset import load_reach_metadata
from pg3d.envs.xarm_adapter import register_pg3d_xarm7_gripper_reach_envs
from pg3d.eval import TimingRecorder, scene_context_for_constraints
from pg3d.policies.dp3.checkpoint import load_reach_policy_from_checkpoint
from pg3d.utils.arrays import bool_any as _bool_any
from pg3d.utils.arrays import bool_info as _bool_info
from pg3d.utils.devices import select_device
from pg3d.world_model import GeometricWorldModel
from scripts.eval_constrained_reach import (
    DP3ChunkPolicyAdapter,
    PendingObstacleSpawn,
    _action_mode,
    _constraints_for_episode,
    _env_kwargs,
    _env_task_name,
    _progress_fraction_along_path,
    _select_multichunk,
    parse_args as parse_eval_args,
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
    policy_action_to_sim_action,
    rollout_observation_entry,
    save_video,
    select_rollout_specs,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-model", choices=["ema", "raw"], default="ema")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--episode-indices", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--k", type=int, default=32, help="candidates sampled per replan")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help=(
            "Override the dataset's baked-in env_kwargs.max_episode_steps (env "
            "truncation point), same as eval_constrained_reach.py's flag of the "
            "same name. Without this, the env still truncates at whatever value "
            "is stored in the dataset's metadata.json, regardless of --max-steps."
        ),
    )
    parser.add_argument(
        "--constraint-type",
        choices=["region", "projection"],
        default="projection",
        help="Forwarded to eval_constrained_reach.py's --constraint-type.",
    )
    parser.add_argument(
        "--constraint-placement",
        choices=["direct_path", "candidate_midpath", "widest_trajectory", "obstacle_spawning"],
        default="candidate_midpath",
        help=(
            "Forwarded to eval_constrained_reach.py's --constraint-placement. "
            "obstacle_spawning only shows/uses its first (always-visible) obstacle here "
            "-- the second, dynamically-spawned one isn't wired into this script's replan "
            "loop or rendering yet."
        ),
    )
    parser.add_argument(
        "--projection-half-extents",
        type=float,
        nargs=2,
        default=[0.025, 0.025],
        metavar=("HX", "HY"),
    )
    parser.add_argument("--avoid-path-fractions", type=float, nargs="+", default=[0.5])
    parser.add_argument("--gripper-open", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reranking_candidates_debug"),
    )
    parser.add_argument("--video-fps", type=int, default=6)
    parser.add_argument(
        "--hold-frames",
        type=int,
        default=12,
        help="frames each replan step is held for (at --video-fps) so it's readable",
    )
    parser.add_argument(
        "--view-elev",
        type=float,
        default=25.0,
        help="3-D plot camera elevation angle in degrees (matplotlib Axes3D.view_init).",
    )
    parser.add_argument(
        "--view-azim",
        type=float,
        default=-60.0,
        help="3-D plot camera azimuth angle in degrees (matplotlib Axes3D.view_init).",
    )
    args = parser.parse_args(argv)
    if args.k <= 0:
        raise ValueError("--k must be positive")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.max_episode_steps is not None and args.max_episode_steps <= 0:
        raise ValueError("--max-episode-steps must be positive")
    if args.video_fps <= 0:
        raise ValueError("--video-fps must be positive")
    if args.hold_frames <= 0:
        raise ValueError("--hold-frames must be positive")
    return args


def _build_eval_args(args: argparse.Namespace, *, scratch_dir: Path) -> argparse.Namespace:
    """Build a full eval_constrained_reach argparse.Namespace via its own parser.

    This guarantees `_constraints_for_episode` sees exactly the attributes it
    expects, and that the constraint we visualize is identical to what a real
    `eval_constrained_reach.py --constraint-type projection` run would place.
    """
    argv = [
        "--checkpoint", str(args.checkpoint),
        "--checkpoint-model", args.checkpoint_model,
        "--dataset", str(args.dataset),
        "--output-dir", str(scratch_dir),
        "--device", args.device,
        "--source", "dataset",
        "--episodes", str(len(args.episode_indices)),
        "--episode-indices", *[str(i) for i in args.episode_indices],
        "--methods", "reranking",
        "--max-steps", str(args.max_steps),
        "--planning-horizon-chunks", "1",
        "--geometry-mode", "fast",
        "--k-schedule", str(args.k),
        "--constraint-placement", args.constraint_placement,
        "--constraint-type", args.constraint_type,
        "--projection-half-extents", str(args.projection_half_extents[0]), str(args.projection_half_extents[1]),
        "--avoid-path-fractions", *[str(f) for f in args.avoid_path_fractions],
        "--gripper-open", str(args.gripper_open),
        "--seed", str(args.seed),
    ]
    if args.max_episode_steps is not None:
        argv += ["--max-episode-steps", str(args.max_episode_steps)]
    return parse_eval_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
    except Exception as exc:
        print(f"Failed to import ManiSkill/Gymnasium: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    eval_args = _build_eval_args(args, scratch_dir=args.output_dir / "_eval_scratch")
    print(
        "resolved constraint config: "
        f"constraint_type={eval_args.constraint_type} "
        f"constraint_placement={eval_args.constraint_placement} "
        f"projection_half_extents={eval_args.projection_half_extents} "
        f"avoid_path_fractions={eval_args.avoid_path_fractions} "
        f"k_schedule={eval_args.k_schedule} "
        f"planning_horizon_chunks={eval_args.planning_horizon_chunks} "
        f"geometry_mode={eval_args.geometry_mode} "
        f"max_steps={eval_args.max_steps} "
        f"max_episode_steps={eval_args.max_episode_steps} "
        f"method={eval_args.methods}",
        flush=True,
    )

    register_pg3d_reach_envs()
    register_pg3d_xarm7_gripper_reach_envs()
    metadata = load_reach_metadata(args.dataset)
    device = select_device(args.device)
    policy = load_reach_policy_from_checkpoint(
        args.checkpoint, device=device, prefer_ema=args.checkpoint_model == "ema"
    )
    action_mode = _action_mode(str(metadata.get("action_mode", "abs_joint")))
    crop_config = crop_config_from_metadata(metadata)
    goal_thresh = float(dict(metadata.get("env_kwargs", {})).get("goal_thresh", 0.025))
    dataset_episode_seeds = [
        int(episode["seed"]) for episode in metadata.get("episodes", []) if "seed" in episode
    ]
    zarr_root = zarr.open_group(str(args.dataset), mode="r")
    specs = select_rollout_specs(
        source="dataset",
        dataset_episode_seeds=dataset_episode_seeds,
        episodes=len(args.episode_indices),
        episode_indices=args.episode_indices,
        seed_start=eval_args.seed_start,
    )

    timer = TimingRecorder(enabled=False)
    rng = np.random.default_rng(args.seed)
    sim_env = None
    ghost_env = None
    try:
        sim_env = gym.make(
            str(metadata["env_id"]),
            **_env_kwargs(metadata, render_mode=None, max_episode_steps=eval_args.max_episode_steps),
        )
        ghost_env = gym.make(
            str(metadata["env_id"]),
            **_env_kwargs(metadata, render_mode=None, max_episode_steps=eval_args.max_episode_steps),
        )
        adapter = DP3ChunkPolicyAdapter(
            policy, action_mode=action_mode, device=device, policy_batch_size=64, timer=timer
        )

        for spec in specs:
            zarr_context = _zarr_episode_context(zarr_root, spec.dataset_episode_index)
            constraints, pending_spawn = _constraints_for_episode(
                sim_env,
                spec=spec,
                policy=policy,
                adapter=adapter,
                action_mode=action_mode,
                crop_config=crop_config,
                goal_thresh=goal_thresh,
                args=eval_args,
                zarr_context=zarr_context,
            )
            if pending_spawn is not None:
                print(
                    f"episode {spec.output_index}: obstacle_spawning second obstacle pending "
                    f"(trigger_fraction={pending_spawn.trigger_fraction:.3f}, "
                    f"region={pending_spawn.constraint.to_json()}) -- will splice into the live "
                    "constraint list / replan scoring once the TCP crosses the first obstacle",
                    flush=True,
                )
            print(
                f"episode {spec.output_index} (dataset idx {spec.dataset_episode_index}, "
                f"seed {spec.seed}): constraints={[c.to_json() for c in constraints]}",
                flush=True,
            )
            records = _run_episode(
                sim_env=sim_env,
                ghost_env=ghost_env,
                policy=policy,
                adapter=adapter,
                spec=spec,
                constraints=constraints,
                pending_spawn=pending_spawn,
                action_mode=action_mode,
                crop_config=crop_config,
                goal_thresh=goal_thresh,
                zarr_context=zarr_context,
                k=args.k,
                max_steps=args.max_steps,
                gripper_open=args.gripper_open,
                rng=rng,
                timer=timer,
            )
            video_path = args.output_dir / f"episode_{spec.output_index:03d}.mp4"
            frames = _render_episode_frames(
                records,
                hold_frames=args.hold_frames,
                episode_index=spec.output_index,
                view_elev=args.view_elev,
                view_azim=args.view_azim,
            )
            save_video(video_path, frames, fps=args.video_fps)
            print(f"saved: {video_path} ({len(records)} replans, {len(frames)} frames)", flush=True)
    finally:
        if sim_env is not None:
            sim_env.close()
        if ghost_env is not None:
            ghost_env.close()
    return 0


def _run_episode(
    *,
    sim_env: Any,
    ghost_env: Any,
    policy: Any,
    adapter: DP3ChunkPolicyAdapter,
    spec: Any,
    constraints: list[AvoidRegion],
    action_mode: str,
    crop_config: Any,
    goal_thresh: float,
    zarr_context: dict[str, Any],
    k: int,
    max_steps: int,
    gripper_open: float,
    rng: np.random.Generator,
    timer: TimingRecorder,
    pending_spawn: PendingObstacleSpawn | None = None,
) -> list[dict[str, Any]]:
    """Roll out one episode with reranking, recording every replan's full candidate bundle."""
    # Defensive copies: this episode's initial constraints/pending_spawn come from the
    # caller's per-episode _constraints_for_episode() call, so appending to `constraints`
    # below (on obstacle_spawning's crossing trigger) must not leak into any other call.
    constraints = list(constraints)
    spawn_pending = pending_spawn
    sim_obs, sim_info = _reset_to_zarr_episode(sim_env, rollout_seed=spec.seed, zarr_context=zarr_context)
    sim_entry = rollout_observation_entry(sim_obs, sim_info, env=sim_env, crop_config=crop_config)
    sim_entry = _apply_zarr_initial_entry(sim_entry, zarr_context)
    obs_window = make_initial_obs_window(sim_entry, n_obs_steps=int(policy.n_obs_steps))
    target = np.asarray(sim_entry["target_position"], dtype=np.float32).reshape(3)
    scene = scene_context_for_constraints(
        target_position=target,
        constraints=constraints,
        metadata={"method": "reranking", "episode": spec.output_index, "seed": spec.seed},
    )

    provider = ManiSkillGhostPandaGeometryProvider(
        ghost_env, task_name=_env_task_name(sim_env), crop_bounds=crop_config.bounds
    )
    provider.reset(seed=spec.seed, options={"reconfigure": True})
    world_model = GeometricWorldModel(provider)

    executed_xyz = [np.asarray(sim_entry["tcp_pose"], dtype=np.float32).reshape(-1)[:3].copy()]
    records: list[dict[str, Any]] = []
    steps = 0
    replans = 0
    was_training = policy.training
    policy.eval()
    try:
        while steps < max_steps:
            result = _select_multichunk(
                method="reranking",
                adapter=adapter,
                world_model=world_model,
                provider=provider,
                current_entry=sim_entry,
                obs_window=obs_window,
                scene=scene,
                constraints=constraints,
                crop_config=crop_config,
                goal_thresh=goal_thresh,
                planning_horizon_chunks=1,
                geometry_mode="fast",
                k_schedule=(k,),
                rng=rng,
                timer=timer,
            )
            replans += 1
            candidate_xyz = [c.rollout.eef_path.copy() for c in result.candidates]
            feasible = [bool(c.feasible) for c in result.candidates]
            selected_local_idx = next(
                (i for i, c in enumerate(result.candidates) if c is result.selected), 0
            )
            feasible_count = sum(feasible)
            print(
                f"  episode={spec.output_index} replan={replans - 1} step={steps} "
                f"feasible={feasible_count}/{len(feasible)} "
                f"reason={result.selection_reason} "
                f"selected_feasible={feasible[selected_local_idx]}",
                flush=True,
            )
            records.append(
                {
                    "replan_index": replans - 1,
                    "step": steps,
                    "candidate_xyz": candidate_xyz,
                    "feasible": feasible,
                    "selected_local_idx": selected_local_idx,
                    "selection_reason": result.selection_reason,
                    "executed_xyz_before": np.stack(executed_xyz, axis=0),
                    "target_xyz": target.copy(),
                    # constraints actually used for this replan's cost/feasibility computation
                    # above -- for obstacle_spawning this grows mid-episode, so later records
                    # legitimately have more entries than earlier ones.
                    "constraints": list(constraints),
                }
            )

            steps_to_execute = min(
                result.action_chunk.horizon, int(policy.n_action_steps), max_steps - steps
            )
            stop = False
            for policy_action in result.action_chunk.actions[:steps_to_execute]:
                sim_action = policy_action_to_sim_action(
                    policy_action,
                    np.asarray(sim_entry["agent_pos"], dtype=np.float32),
                    action_mode=action_mode,
                    sim_action_dim=int(np.prod(sim_env.action_space.shape)),
                    low=getattr(sim_env.action_space, "low", None),
                    high=getattr(sim_env.action_space, "high", None),
                    gripper_open=gripper_open,
                )
                sim_obs, _reward, terminated, truncated, sim_info = sim_env.step(sim_action)
                steps += 1
                sim_entry = rollout_observation_entry(sim_obs, sim_info, env=sim_env, crop_config=crop_config)
                obs_window = append_obs_window(obs_window, sim_entry, n_obs_steps=int(policy.n_obs_steps))
                executed_xyz.append(
                    np.asarray(sim_entry["tcp_pose"], dtype=np.float32).reshape(-1)[:3].copy()
                )
                if spawn_pending is not None:
                    tcp_position = np.asarray(sim_entry["tcp_pose"], dtype=np.float32).reshape(-1)[:3]
                    progress = _progress_fraction_along_path(spawn_pending.reference_path, tcp_position)
                    if progress >= spawn_pending.trigger_fraction:
                        constraints.append(spawn_pending.constraint)
                        scene = scene_context_for_constraints(
                            target_position=target,
                            constraints=constraints,
                            metadata={"method": "reranking", "episode": spec.output_index, "seed": spec.seed},
                        )
                        print(
                            f"obstacle_spawning: spawned second obstacle episode={spec.output_index} "
                            f"step={steps} progress={progress:.3f} trigger={spawn_pending.trigger_fraction:.3f}",
                            flush=True,
                        )
                        spawn_pending = None
                success = _bool_info(sim_info, "success")
                if success or _bool_any(terminated) or _bool_any(truncated):
                    stop = True
                    break
            if stop:
                break
    finally:
        if was_training:
            policy.train()
    return records


_REGION_FACE_RGBA = (0.65, 0.05, 0.55, 0.22)
_REGION_EDGE_RGBA = (0.65, 0.05, 0.55, 0.55)
_PROJECTION_FACE_RGBA = (1.0, 0.25, 0.05, 0.18)
_PROJECTION_EDGE_RGBA = (1.0, 0.25, 0.05, 0.5)


def _sphere_mesh(center: np.ndarray, radius: float, *, resolution: int = 18) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, v = np.meshgrid(
        np.linspace(0.0, 2.0 * np.pi, resolution),
        np.linspace(0.0, np.pi, max(resolution // 2, 6)),
    )
    xs = center[0] + radius * np.cos(u) * np.sin(v)
    ys = center[1] + radius * np.sin(u) * np.sin(v)
    zs = center[2] + radius * np.cos(v)
    return xs, ys, zs


def _box_edges(center: np.ndarray, half: np.ndarray) -> list[np.ndarray]:
    """12 edges of an axis-aligned box, each as a [2,3] array of endpoints."""
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float32)
    corners = center.reshape(1, 3) + signs * half.reshape(1, 3)
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            # corners differing in exactly one sign flip (one coordinate) are edges
            if np.count_nonzero(signs[i] != signs[j]) == 1:
                edges.append(np.stack([corners[i], corners[j]], axis=0))
    return edges


def _draw_region_3d(ax: Any, constraint: Any) -> list[Any]:
    """Draw one constraint's keep-out region as a real 3-D shape on a 3-D Axes.

    AvoidRegion draws a true sphere/box at its actual height. AvoidProjection (a
    tabletop footprint with no z-extent of its own) is drawn as a translucent
    vertical slab spanning the axes' current z-limits, since it penalizes XY
    position "at any height".
    """
    handles: list[Any] = []
    if isinstance(constraint, AvoidRegion):
        label = f"avoid_region ({constraint.name})"
        region = constraint.region
        if isinstance(region, SphereRegion):
            center = np.asarray(region.center, dtype=np.float32)
            xs, ys, zs = _sphere_mesh(center, float(region.radius))
            surf = ax.plot_surface(
                xs, ys, zs, color=_REGION_FACE_RGBA, edgecolor=_REGION_EDGE_RGBA,
                linewidth=0.2, shade=True, label=label,
            )
            # matplotlib 3-D surface legend entries need a 2-D proxy facecolor array;
            # plot_surface's own Poly3DCollection isn't legend-safe without this fixup.
            surf._facecolors2d = surf._facecolor3d if hasattr(surf, "_facecolor3d") else surf._facecolors3d
            surf._edgecolors2d = surf._edgecolor3d if hasattr(surf, "_edgecolor3d") else surf._edgecolors3d
            handles.append(surf)
        elif isinstance(region, BoxRegion):
            center = np.asarray(region.center, dtype=np.float32)
            half = np.asarray(region.half_extents, dtype=np.float32)
            for i, edge in enumerate(_box_edges(center, half)):
                (line,) = ax.plot3D(
                    edge[:, 0], edge[:, 1], edge[:, 2], color=_REGION_EDGE_RGBA[:3],
                    linewidth=1.2, label=label if i == 0 else None,
                )
                handles.append(line)
        return handles
    if isinstance(constraint, AvoidProjection):
        label = f"avoid_projection ({constraint.name})"
        center = np.asarray(constraint.region.center, dtype=np.float32)
        half = np.asarray(constraint.region.half_extents, dtype=np.float32)
        zlo, zhi = ax.get_zlim()
        corners_xy = np.array(
            [
                [center[0] - half[0], center[1] - half[1]],
                [center[0] + half[0], center[1] - half[1]],
                [center[0] + half[0], center[1] + half[1]],
                [center[0] - half[0], center[1] + half[1]],
            ],
            dtype=np.float32,
        )
        first = True
        for corner in corners_xy:
            (line,) = ax.plot3D(
                [corner[0], corner[0]], [corner[1], corner[1]], [zlo, zhi],
                color=_PROJECTION_EDGE_RGBA[:3], linewidth=1.2, label=label if first else None,
            )
            handles.append(line)
            first = False
        for z in (zlo, zhi):
            ring = np.vstack([corners_xy, corners_xy[:1]])
            (line,) = ax.plot3D(ring[:, 0], ring[:, 1], np.full(ring.shape[0], z), color=_PROJECTION_EDGE_RGBA[:3], linewidth=1.2)
            handles.append(line)
        return handles
    return []


def _region_3d_bounds(constraint: Any) -> np.ndarray | None:
    """[2,3] XYZ points bounding a constraint's footprint, for axis-limit sizing."""
    if isinstance(constraint, AvoidRegion):
        region = constraint.region
        if isinstance(region, SphereRegion):
            center = np.asarray(region.center, dtype=np.float32)
            radius = float(region.radius)
            return np.stack([center - radius, center + radius], axis=0)
        if isinstance(region, BoxRegion):
            center = np.asarray(region.center, dtype=np.float32)
            half = np.asarray(region.half_extents, dtype=np.float32)
            return np.stack([center - half, center + half], axis=0)
    if isinstance(constraint, AvoidProjection):
        center = np.asarray(constraint.region.center, dtype=np.float32)
        half = np.asarray(constraint.region.half_extents, dtype=np.float32)
        # z left at 0 -- the projection has no inherent z-extent; the overall data
        # bounds (candidate/executed paths) supply the real z-range for the slab.
        lo = np.array([center[0] - half[0], center[1] - half[1], 0.0], dtype=np.float32)
        hi = np.array([center[0] + half[0], center[1] + half[1], 0.0], dtype=np.float32)
        return np.stack([lo, hi], axis=0)
    return None


def _render_episode_frames(
    records: list[dict[str, Any]],
    *,
    hold_frames: int,
    episode_index: int,
    view_elev: float = 25.0,
    view_azim: float = -60.0,
) -> list[np.ndarray]:
    import matplotlib.pyplot as plt

    if not records:
        raise RuntimeError(f"episode {episode_index}: no replan steps recorded")

    # Union of every constraint ever active across the episode (obstacle_spawning grows
    # this mid-episode; the last record has the superset since spawning is additive-only)
    # so axis limits fit every obstacle that will ever be drawn, even before it appears.
    final_constraints = records[-1]["constraints"]

    all_points = [records[-1]["target_xyz"].reshape(1, 3)]
    for record in records:
        all_points.extend(record["candidate_xyz"])
        all_points.append(record["executed_xyz_before"])
    for constraint in final_constraints:
        region_bounds = _region_3d_bounds(constraint)
        if region_bounds is not None:
            all_points.append(region_bounds)
    bounds = np.concatenate(all_points, axis=0)
    mins = np.min(bounds, axis=0)
    maxs = np.max(bounds, axis=0)
    mid = (mins + maxs) * 0.5
    # Single uniform span across x/y/z (not per-axis) so spheres render as true
    # spheres rather than squashed ellipsoids under an unequal 3-D aspect ratio.
    span = max(float(np.max(maxs - mins)) * 1.2, 0.12)

    frames: list[np.ndarray] = []
    for record in records:
        fig = plt.figure(figsize=(7.5, 7.5), dpi=130)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim(mid[0] - span * 0.5, mid[0] + span * 0.5)
        ax.set_ylim(mid[1] - span * 0.5, mid[1] + span * 0.5)
        ax.set_zlim(mid[2] - span * 0.5, mid[2] + span * 0.5)
        try:
            ax.set_box_aspect((1, 1, 1))
        except AttributeError:
            pass  # older matplotlib without set_box_aspect -- proportions may skew
        ax.view_init(elev=view_elev, azim=view_azim)

        # Only draws constraints active in *this* record (record["constraints"]) -- for
        # obstacle_spawning the second obstacle's shape only appears from the first replan
        # after the TCP crossed the first obstacle's trigger fraction, mirroring exactly
        # when it starts counting toward feasibility/cost in the real eval script.
        for constraint in record["constraints"]:
            _draw_region_3d(ax, constraint)
        for idx, path in enumerate(record["candidate_xyz"]):
            feasible = record["feasible"][idx]
            is_selected = idx == record["selected_local_idx"]
            if is_selected:
                continue
            color = "#2ca02c" if feasible else "#c9c9c9"
            ax.plot3D(
                path[:, 0], path[:, 1], path[:, 2],
                color=color, linewidth=1.0, alpha=0.7 if feasible else 0.45,
            )
        selected_path = record["candidate_xyz"][record["selected_local_idx"]]
        selected_feasible = record["feasible"][record["selected_local_idx"]]
        ax.plot3D(
            selected_path[:, 0],
            selected_path[:, 1],
            selected_path[:, 2],
            color="#1f5ecb" if selected_feasible else "#d62728",
            linewidth=3.2,
            alpha=0.95,
            label="selected",
        )
        ax.scatter3D(
            selected_path[-1:, 0], selected_path[-1:, 1], selected_path[-1:, 2],
            color="#1f5ecb", s=45,
        )

        executed = record["executed_xyz_before"]
        ax.plot3D(
            executed[:, 0], executed[:, 1], executed[:, 2],
            color="black", linewidth=2.0, alpha=0.9, label="executed so far",
        )
        ax.scatter3D(
            executed[:1, 0], executed[:1, 1], executed[:1, 2],
            color="gold", s=90, edgecolors="black", label="start",
        )
        target_xyz = record["target_xyz"]
        ax.scatter3D(
            [target_xyz[0]], [target_xyz[1]], [target_xyz[2]],
            color="limegreen", s=110, marker="*", edgecolors="black", label="goal",
        )

        feasible_count = sum(record["feasible"])
        ax.set_title(
            f"episode {episode_index}  replan {record['replan_index']}  step {record['step']}\n"
            f"feasible {feasible_count}/{len(record['feasible'])}  "
            f"selection={record['selection_reason']}  selected_feasible={selected_feasible}"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        handles, labels = ax.get_legend_handles_labels()
        seen = dict(zip(labels, handles))
        ax.legend(seen.values(), seen.keys(), loc="upper left", fontsize=7, framealpha=0.8)
        fig.tight_layout()
        fig.canvas.draw()
        image = _canvas_rgb_array(fig.canvas)
        plt.close(fig)
        frames.extend([image] * hold_frames)
    return frames


def _canvas_rgb_array(canvas: Any) -> np.ndarray:
    width, height = canvas.get_width_height()
    if hasattr(canvas, "buffer_rgba"):
        rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
        return rgba[:, :, :3].copy()
    if hasattr(canvas, "tostring_rgb"):
        return np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
    raise AttributeError("Matplotlib canvas cannot export RGB pixels")


if __name__ == "__main__":
    raise SystemExit(main())
