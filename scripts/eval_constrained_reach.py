from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import zarr

from pg3d.composition import (
    CandidateDiagnostics,
    ControllerInput,
    ControllerResult,
    RejectionController,
    RerankingController,
    ScoreWeights,
)
from pg3d.composition.scoring import (
    consensus_deviations,
    goal_distance,
    primary_constraint_penalty,
    trajectory_smoothness,
)
from pg3d.constraints import (
    AvoidProjection,
    AvoidRegion,
    BoxRegion,
    CartesianPoseConstraint,
    RectRegion2D,
    SphereRegion,
)
from pg3d.envs.maniskill_adapter import (
    ManiSkillGhostPandaGeometryProvider,
    register_pg3d_reach_envs,
)
from pg3d.envs.maniskill_adapter.dataset import (
    PointCloudCropConfig,
    load_reach_metadata,
)
from pg3d.envs.xarm_adapter import register_pg3d_xarm7_gripper_reach_envs
from pg3d.eval import (
    AvoidOverlayConfig,
    EpisodePath,
    TimingRecorder,
    candidate_feasibility_fraction,
    cartesian_pose_step_metrics,
    concatenate_rollouts,
    direct_path_avoid_region,
    episode_metric_row,
    load_episode_constraints,
    progress_series,
    save_episode_constraints,
    scene_context_for_constraints,
    select_artifact_episode_indices,
    should_emit_episode_artifact,
    summarize_metrics,
    validate_planning_horizons,
)
from pg3d.policies.dp3 import SimpleDP3
from pg3d.policies.dp3.checkpoint import (
    latest_reach_checkpoint,
    load_reach_policy_from_checkpoint,
)
from pg3d.policies.dp3.goal_markers import (
    DEFAULT_GOAL_MARKER_RADIUS,
    insert_goal_marker_points,
)
from pg3d.utils.arrays import bool_any as _bool_any
from pg3d.utils.arrays import bool_info as _bool_info
from pg3d.utils.arrays import frame_to_numpy as _frame_to_numpy
from pg3d.utils.devices import select_device
from pg3d.utils.serialization import jsonable as _jsonable
from pg3d.world_model import ActionChunk, GeometricWorldModel, ImaginedRollout
from pg3d.world_model.chunks import interpret_joint_chunk
from pg3d.world_model.compositor import compose_robot_cloud, static_scene_from_robot_mask
from scripts.compare_world_model_rollout import (
    entry_to_world_model_observation,
    world_model_entry_from_rollout_step,
)
from scripts.eval_reach_checkpoint_unique_seeds import (
    _apply_zarr_initial_entry,
    _reset_to_zarr_episode,
    _zarr_episode_context,
)
from scripts.rollout_dp3_reach_policy import (
    ActionMode,
    RolloutSpec,
    append_obs_window,
    crop_config_from_metadata,
    make_initial_obs_window,
    obs_window_to_torch,
    policy_action_to_sim_action,
    rollout_observation_entry,
    save_rerun_timeline,
    save_video,
    select_rollout_specs,
)

EvalMethod = Literal["base", "rejection", "reranking"]
GeometryMode = Literal["fast", "exact"]
Entry = dict[str, np.ndarray | bool | float]
# Z range used to extrude the height-agnostic avoid_projection footprint for the
# overlay video. Display-only; the constraint itself penalizes XY at any height.
_PROJECTION_OVERLAY_Z_RANGE = (0.0, 0.5)


@dataclass
class EvalDecisionSummary:
    """Compact per-replan diagnostic summary."""

    selected_chunk: ActionChunk
    result: ControllerResult | None
    candidate_feasible: int
    candidate_total: int
    selection_reason: str | None


class DP3ChunkPolicyAdapter:
    """Adapt `SimpleDP3.predict_action` to the P09 candidate-sampling protocol."""

    def __init__(
        self,
        policy: SimpleDP3,
        *,
        action_mode: ActionMode,
        device: torch.device,
        policy_batch_size: int = 64,
        timer: TimingRecorder | None = None,
        dt: float = 1.0,
    ) -> None:
        self.policy = policy
        self.action_mode = action_mode
        self.device = device
        self.policy_batch_size = int(policy_batch_size)
        self.timer = timer or TimingRecorder(enabled=False)
        self.dt = float(dt)

    def sample_action_chunks(
        self,
        policy_input: list[Entry],
        *,
        k: int,
        rng: np.random.Generator | None = None,
    ) -> list[ActionChunk]:
        """Sample `k` DP3 action chunks from one rolling observation window."""
        if k <= 0:
            raise ValueError("k must be positive")
        with self.timer.time("policy_sampling", windows=1, samples=k):
            batch = _repeat_obs_window_to_torch(
                policy_input,
                k=k,
                device=self.device,
                goal_marker_points=int(getattr(self.policy, "goal_marker_points", 0)),
                goal_marker_radius=float(
                    getattr(self.policy, "goal_marker_radius", DEFAULT_GOAL_MARKER_RADIUS)
                ),
            )
            actions = self._predict_actions(batch)
        return [
            ActionChunk(
                actions=actions[idx].astype(np.float32, copy=True),
                action_mode=self.action_mode,
                dt=self.dt,
                metadata={"candidate_index": idx},
            )
            for idx in range(actions.shape[0])
        ]

    def sample_action_chunks_for_windows(
        self,
        policy_inputs: list[list[Entry]],
        *,
        rng: np.random.Generator | None = None,
    ) -> list[ActionChunk]:
        """Sample one DP3 action chunk for each rolling observation window."""
        if not policy_inputs:
            return []
        del rng
        actions: list[np.ndarray] = []
        with self.timer.time("policy_sampling", windows=len(policy_inputs), samples=1):
            for start in range(0, len(policy_inputs), self.policy_batch_size):
                batch_windows = policy_inputs[start : start + self.policy_batch_size]
                batch = _obs_windows_to_torch(
                    batch_windows,
                    device=self.device,
                    goal_marker_points=int(getattr(self.policy, "goal_marker_points", 0)),
                    goal_marker_radius=float(
                        getattr(self.policy, "goal_marker_radius", DEFAULT_GOAL_MARKER_RADIUS)
                    ),
                )
                actions.append(self._predict_actions(batch))
        stacked = np.concatenate(actions, axis=0)
        return [
            ActionChunk(
                actions=stacked[idx].astype(np.float32, copy=True),
                action_mode=self.action_mode,
                dt=self.dt,
                metadata={"candidate_index": idx},
            )
            for idx in range(stacked.shape[0])
        ]

    def _predict_actions(self, batch: dict[str, torch.Tensor]) -> np.ndarray:
        with torch.inference_mode():
            output = self.policy.predict_action(batch)
            return output["action"].detach().cpu().numpy()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint, args.checkpoint_dir)
    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
    except Exception as exc:
        print(
            f"Failed to import ManiSkill/Gymnasium: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print(
            "Install with: "
            "uv sync --extra cu129 --extra maniskill --extra viz --group dev --group notebooks",
            file=sys.stderr,
        )
        return 2

    register_pg3d_reach_envs()
    register_pg3d_xarm7_gripper_reach_envs()
    metadata = load_reach_metadata(args.dataset)
    device = select_device(args.device)
    _seed_torch(args.seed)
    timer = TimingRecorder(
        enabled=args.profile,
        sync_fn=_cuda_sync_fn(device) if args.sync_cuda_timers else None,
    )
    policy = load_reach_policy_from_checkpoint(
        checkpoint_path,
        device=device,
        prefer_ema=args.checkpoint_model == "ema",
    )
    action_mode = _action_mode(str(metadata.get("action_mode", "abs_joint")))
    crop_config = crop_config_from_metadata(metadata)
    goal_thresh = (
        float(args.goal_thresh)
        if args.goal_thresh is not None
        else float(dict(metadata.get("env_kwargs", {})).get("goal_thresh", 0.025))
    )
    dataset_episode_seeds = [
        int(episode["seed"]) for episode in metadata.get("episodes", []) if "seed" in episode
    ]
    zarr_root = (
        zarr.open_group(str(args.dataset), mode="r") if args.source == "dataset" else None
    )
    episode_indices = _episode_indices_from_args(
        args,
        dataset_episode_seeds=dataset_episode_seeds,
    )
    specs = select_rollout_specs(
        source=args.source,
        dataset_episode_seeds=dataset_episode_seeds,
        episodes=args.episodes,
        episode_indices=episode_indices,
        seed_start=args.seed_start,
    )
    if not specs:
        raise RuntimeError("no constrained-reach episodes selected")
    if args.unique_dataset_seeds:
        print(
            "unique dataset seed selection: "
            f"selected={len(specs)} available_unique={len(set(dataset_episode_seeds))} "
            f"dataset_rows={len(dataset_episode_seeds)}",
            flush=True,
        )
    artifact_seed = args.artifact_selection_seed
    video_episode_indices = (
        set(spec.output_index for spec in specs)
        if args.video
        else set(
            select_artifact_episode_indices(
                [spec.output_index for spec in specs],
                selection=args.artifact_selection,
                count=args.artifact_episode_count,
                seed=artifact_seed,
                every_episodes=args.video_every_episodes,
            )
        )
    )
    rerun_episode_indices = set(
        select_artifact_episode_indices(
            [spec.output_index for spec in specs],
            selection=args.artifact_selection,
            count=args.artifact_episode_count,
            seed=artifact_seed,
            every_episodes=args.rerun_every_episodes,
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run = _init_wandb(args, metadata=metadata, checkpoint_path=checkpoint_path)
    sim_env: Any | None = None
    ghost_env: Any | None = None
    rows: list[dict[str, Any]] = []
    metrics_path = args.output_dir / "metrics.jsonl"
    decisions_path = args.output_dir / "decisions.jsonl"
    step_traces_path = args.output_dir / "step_traces.jsonl"
    timings_path = args.output_dir / "timings.jsonl"
    timing_written = 0
    rng = np.random.default_rng(args.seed)
    try:
        sim_env = gym.make(
            str(metadata["env_id"]),
            **_env_kwargs(metadata, render_mode="rgb_array" if args.video else None),
        )
        ghost_env = gym.make(str(metadata["env_id"]), **_env_kwargs(metadata, render_mode=None))
        adapter = DP3ChunkPolicyAdapter(
            policy,
            action_mode=action_mode,
            device=device,
            policy_batch_size=args.policy_batch_size,
            timer=timer,
        )
        with (
            metrics_path.open("w", encoding="utf-8") as metrics_file,
            decisions_path.open("w", encoding="utf-8") as decisions_file,
            step_traces_path.open("w", encoding="utf-8") as step_traces_file,
        ):
            for spec in specs:
                zarr_context = (
                    _zarr_episode_context(zarr_root, spec.dataset_episode_index)
                    if zarr_root is not None and spec.dataset_episode_index is not None
                    else None
                )
                constraints = _constraints_for_episode(
                    sim_env,
                    spec=spec,
                    policy=policy,
                    adapter=adapter,
                    action_mode=action_mode,
                    crop_config=crop_config,
                    goal_thresh=goal_thresh,
                    args=args,
                    zarr_context=zarr_context,
                )
                constraint_path = (
                    args.output_dir
                    / "constraints"
                    / f"episode_{spec.output_index:03d}.json"
                )
                with timer.time("json_write", artifact="constraint"):
                    save_episode_constraints(constraint_path, constraints)
                write_video = args.video and spec.output_index in video_episode_indices
                write_rerun = args.rerun and spec.output_index in rerun_episode_indices
                for method in args.methods:
                    row = run_eval_episode(
                        sim_env=sim_env,
                        ghost_env=ghost_env,
                        policy=policy,
                        adapter=adapter,
                        method=method,
                        spec=spec,
                        constraints=constraints,
                        action_mode=action_mode,
                        crop_config=crop_config,
                        goal_thresh=goal_thresh,
                        output_dir=args.output_dir,
                        max_steps=args.max_steps,
                        post_success_steps=args.post_success_steps,
                        planning_horizon_chunks=args.planning_horizon_chunks,
                        execution_horizon_chunks=args.execution_horizon_chunks,
                        geometry_mode=args.geometry_mode,
                        k_schedule=tuple(args.k_schedule),
                        gripper_open=args.gripper_open,
                        match_current_robot_points=args.match_current_robot_points,
                        video=write_video,
                        rerun=write_rerun,
                        video_fps=args.video_fps,
                        decisions_file=decisions_file,
                        step_traces_file=step_traces_file,
                        rng=rng,
                        timer=timer,
                        video_env_factory=_video_env_factory(
                            gym,
                            metadata=metadata,
                            enabled=write_video and args.constraint_overlay_video,
                        ),
                        constraint_overlay_alpha=args.constraint_overlay_alpha,
                        constraint_overlay_color=tuple(args.constraint_overlay_color),
                        robot_clearance_metric=args.robot_clearance_metric,
                        robot_clearance_stride=args.robot_clearance_stride,
                        zarr_context=zarr_context,
                    )
                    rows.append(row)
                    with timer.time("json_write", artifact="metrics"):
                        metrics_file.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
                        metrics_file.flush()
                    _log_wandb_episode(run, args=args, row=row, global_step=len(rows))
                    print(
                        f"method={method} episode={spec.output_index} seed={spec.seed} "
                        f"combined={row['combined_success']} reach={row['reach_success']} "
                        f"constraint={row['constraint_satisfied']} "
                        f"final={_format_optional(row['final_target_distance'])} "
                        f"clearance={_format_optional(row['min_clearance'])}"
                    )
                timing_written = _write_new_timing_events(
                    timer,
                    timings_path,
                    start_index=timing_written,
                )
                if should_emit_episode_artifact(spec.output_index, args.plot_every_episodes):
                    _maybe_emit_progress(
                        output_dir=args.output_dir,
                        rows=rows,
                        timer=timer,
                        episode_index=spec.output_index,
                        plots=args.plots or run is not None,
                        run=run,
                        args=args,
                    )
                if args.profile and should_emit_episode_artifact(
                    spec.output_index,
                    args.profile_every_episodes,
                ):
                    _print_timing_summary(timer)
    except Exception as exc:
        print(f"Failed constrained reach eval: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if sim_env is not None:
            sim_env.close()
        if ghost_env is not None:
            ghost_env.close()

    summary = {
        "checkpoint": str(checkpoint_path),
        "dataset": str(args.dataset),
        "source": args.source,
        "methods": list(args.methods),
        "env_id": metadata["env_id"],
        "env_kwargs": _env_kwargs(metadata, render_mode="rgb_array" if args.video else None),
        "planning_horizon_chunks": args.planning_horizon_chunks,
        "execution_horizon_chunks": args.execution_horizon_chunks,
        "geometry_mode": args.geometry_mode,
        "k_schedule": list(args.k_schedule),
        "constraint_source": _constraint_source_summary(args),
        "artifact_selection": _artifact_selection_summary(
            specs,
            video_episode_indices=video_episode_indices,
            rerun_episode_indices=rerun_episode_indices,
            args=args,
        ),
        "constraint_overlay_video": bool(args.constraint_overlay_video),
        "constraint_overlay_alpha": float(args.constraint_overlay_alpha),
        "constraint_overlay_color": list(args.constraint_overlay_color),
        "metrics_jsonl": str(metrics_path),
        "decisions_jsonl": str(decisions_path),
        "step_traces_jsonl": str(step_traces_path),
        "timing": timer.summary(),
        "episodes": rows,
        "by_method": summarize_metrics(rows),
        "code_only_baseline_note": (
            "Code-only waypoint planning is a strong reach baseline and is intentionally "
            "not implemented in this P10 scaffold; do not over-claim reach-only results."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if args.plots:
        _maybe_emit_progress(
            output_dir=args.output_dir,
            rows=rows,
            timer=timer,
            episode_index=max((int(row["episode"]) for row in rows), default=0),
            plots=True,
            run=None,
            args=args,
            final=True,
        )
        summary_plot_paths = _write_summary_plots(args.output_dir, rows=rows)
        if run is not None and summary_plot_paths:
            try:
                import wandb

                metrics = {
                    f"summary_plot/{path.stem}": wandb.Image(str(path))
                    for path in summary_plot_paths
                }
                run.log(metrics)
            except Exception as exc:
                if not args.wandb_required:
                    print(
                        f"warning: W&B summary plot logging failed: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
    if run is not None:
        _log_wandb_summary(run, args=args, rows=rows, summary=summary)

    failures = sum(0 if row["combined_success"] else 1 for row in rows)
    return 0 if args.allow_failure or failures == 0 else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate base DP3, rejection, and reranking on constrained reach."
    )
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument("--checkpoint", type=Path, default=None)
    checkpoint_group.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-model", choices=["ema", "raw"], default="ema")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--source", choices=["dataset", "fresh"], default="fresh")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--episode-indices", type=int, nargs="+", default=None)
    parser.add_argument(
        "--episode-indices-file",
        type=Path,
        default=None,
        help="Text file with one dataset episode index per line.",
    )
    parser.add_argument(
        "--unique-dataset-seeds",
        action="store_true",
        help=(
            "With --source dataset, evaluate only the first dataset row for each unique "
            "episode seed, capped by --episodes. This avoids repeated env resets when "
            "the dataset contains multiple rows with the same seed."
        ),
    )
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["base", "rejection", "reranking"],
        default=["base", "rejection", "reranking"],
    )
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--post-success-steps", type=int, default=16)
    parser.add_argument("--planning-horizon-chunks", type=int, default=1)
    parser.add_argument("--execution-horizon-chunks", type=int, default=1)
    parser.add_argument("--geometry-mode", choices=["fast", "exact"], default="fast")
    parser.add_argument("--k-schedule", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--policy-batch-size", type=int, default=64)
    parser.add_argument("--goal-thresh", type=float, default=None)
    parser.add_argument(
        "--constraint-placement",
        choices=["direct_path", "candidate_midpath", "widest_trajectory"],
        default="direct_path",
        help=(
            "Where to place generated avoid regions. direct_path uses the midpoint of "
            "start TCP and goal; candidate_midpath first rolls out base-policy candidates "
            "and places the region in the middle of their natural path bundle; "
            "widest_trajectory rolls out candidates, selects the single path that bows out "
            "the most from the straight start-goal line, and plants one region per "
            "--avoid-path-fractions value along that widest path."
        ),
    )
    parser.add_argument("--constraint-placement-candidates", type=int, default=10)
    parser.add_argument(
        "--constraint-placement-steps",
        type=int,
        default=None,
        help="Max sim steps for candidate_midpath placement; defaults to --max-steps.",
    )
    parser.add_argument(
        "--constraint-placement-path-fraction",
        type=float,
        default=0.5,
        help="Arc-length fraction sampled from each candidate path before aggregating.",
    )
    parser.add_argument(
        "--constraint-placement-success-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For candidate_midpath, aggregate successful candidate paths when any are "
            "available; otherwise fall back to all sampled paths."
        ),
    )
    parser.add_argument(
        "--constraint-type",
        choices=["region", "projection"],
        default="region",
        help=(
            "Constraint family to place. 'region' (default) is a 3-D keep-out volume "
            "(sphere/box) penalizing the EEF/robot for entering it. 'projection' is the "
            "no-overflight analog: a 2-D tabletop rectangle that penalizes the XY "
            "projection of the EEF/robot for passing over it, regardless of height z. "
            "Projection placement reuses the candidate_midpath logic; pass "
            "--constraint-placement candidate_midpath."
        ),
    )
    parser.add_argument(
        "--projection-half-extents",
        type=float,
        nargs=2,
        default=[0.025, 0.025],
        metavar=("MIN_HX", "MIN_HY"),
        help=(
            "Minimum per-axis XY half-extents (meters) for --constraint-type projection. "
            "The rectangle is sized per axis from the sampled candidate paths' XY spread "
            "(37.5th percentile of deviations from the placed center, mirroring how the "
            "sphere radius is computed), then floored at these values. The rectangle is "
            "centered at the placed XY location and extends through all z."
        ),
    )
    parser.add_argument("--avoid-radius", type=float, default=0.08)
    parser.add_argument("--avoid-min-radius", type=float, default=0.025)
    parser.add_argument("--avoid-margin", type=float, default=0.0)
    parser.add_argument("--avoid-weight", type=float, default=1.0)
    parser.add_argument(
        "--avoid-shape",
        choices=["sphere", "box", "cuboid"],
        default="sphere",
        help=(
            "Shape of the placed avoid region. Applies to all placement modes. The region "
            "is centered exactly where the sphere would be; 'box' and 'cuboid' are "
            "synonyms for an axis-aligned rectangular keep-out sized by "
            "--avoid-box-half-extents (or the effective sphere radius if those are omitted)."
        ),
    )
    parser.add_argument(
        "--avoid-box-half-extents",
        type=float,
        nargs=3,
        default=None,
        metavar=("HX", "HY", "HZ"),
        help=(
            "Half-extents (meters) for --avoid-shape box. If omitted, the box uses isotropic "
            "half-extents equal to the effective avoid radius."
        ),
    )
    parser.add_argument(
        "--avoid-path-fractions",
        type=float,
        nargs="+",
        default=[0.5],
        help=(
            "Fraction(s) along the direct start-to-goal line at which to place avoid-region "
            "sphere(s) (direct_path mode only). Each value in [0, 1] places one sphere. "
            "E.g. --avoid-path-fractions 0.4 0.8 places two spheres at 40%% and 80%% of the path."
        ),
    )
    parser.add_argument(
        "--constraint-target",
        choices=["eef", "robot"],
        default="eef",
        help=(
            "Body checked against avoid regions during planner guidance (rejection/"
            "reranking). 'eef' (default) penalizes only the end-effector path; 'robot' "
            "penalizes the whole arm and base using the imagined robot point clouds. "
            "'robot' requires --geometry-mode exact for guidance methods (fast mode imagines "
            "no robot cloud). The whole-robot evaluation metric (--robot-clearance-metric) is "
            "independent of this flag and defaults to off."
        ),
    )
    parser.add_argument(
        "--robot-clearance-metric",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Evaluate constraint_satisfied against the whole robot (URDF/mesh point cloud "
            "sampled across the executed trajectory) rather than only the TCP. Off by default "
            "since guidance (rejection/reranking) only ever avoids with the TCP/EEF path under "
            "the default --geometry-mode fast (which imagines no robot point cloud), so grading "
            "against the whole robot body checks a signal guidance was never steering against. "
            "The whole-robot result is still reported under constraint_satisfied / min_clearance "
            "when enabled; the TCP-only result is always reported under constraint_satisfied_tcp "
            "/ min_clearance_tcp."
        ),
    )
    parser.add_argument(
        "--robot-clearance-stride",
        type=int,
        default=4,
        help="Subsample stride over executed timesteps when sampling the whole-robot "
        "clearance cloud (1 = every step).",
    )
    parser.add_argument(
        "--robot-clearance-placement",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Adjust each placed avoid region (shrink/translate) so it does not intersect the "
            "robot's links or base at the episode start configuration, keeping the plotted "
            "keep-out volume in free space."
        ),
    )
    parser.add_argument(
        "--robot-clearance-placement-margin",
        type=float,
        default=0.02,
        help="Minimum required clearance (meters) between a placed avoid region and the "
        "start-configuration robot point cloud.",
    )
    parser.add_argument(
        "--constraints-dir",
        type=Path,
        default=None,
        help="Directory containing precomputed constraints/episode_XXX.json files.",
    )
    parser.add_argument("--gripper-open", type=float, default=0.04)
    parser.add_argument(
        "--match-current-robot-points",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cap ghost robot clouds to the current cropped robot-mask count.",
    )
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-every-episodes", type=int, default=10)
    parser.add_argument(
        "--constraint-overlay-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render avoid-region overlays in constrained-eval MP4s using a separate env.",
    )
    parser.add_argument("--constraint-overlay-alpha", type=float, default=0.25)
    parser.add_argument(
        "--constraint-overlay-color",
        type=float,
        nargs=3,
        default=[1.0, 0.25, 0.05],
        metavar=("R", "G", "B"),
    )
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--rerun-every-episodes", type=int, default=10)
    parser.add_argument(
        "--artifact-selection",
        choices=["periodic", "random", "all"],
        default="periodic",
    )
    parser.add_argument("--artifact-episode-count", type=int, default=5)
    parser.add_argument("--artifact-selection-seed", type=int, default=None)
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--plot-every-episodes", type=int, default=10)
    parser.add_argument(
        "--plot-candidate-paths",
        action="store_true",
        help=(
            "When using --constraint-placement candidate_midpath, save a matplotlib figure "
            "showing the sampled candidate TCP paths, the constraint sphere center, and its "
            "radius to <output-dir>/candidate_path_plots/episode_XXX.png."
        ),
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-every-episodes", type=int, default=10)
    parser.add_argument("--sync-cuda-timers", action="store_true")
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument(
        "--wandb-mode",
        choices=["disabled", "offline", "online"],
        default=os.environ.get("WANDB_MODE", "disabled"),
    )
    parser.add_argument("--wandb-project", default="pg3d")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-required", action="store_true")
    parser.add_argument("--allow-failure", action="store_true")
    args = parser.parse_args(argv)
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.episode_indices is not None and args.episode_indices_file is not None:
        raise ValueError("--episode-indices and --episode-indices-file are mutually exclusive")
    if args.episode_indices_file is not None and args.source != "dataset":
        raise ValueError("--episode-indices-file requires --source dataset")
    if args.unique_dataset_seeds and args.source != "dataset":
        raise ValueError("--unique-dataset-seeds requires --source dataset")
    if args.unique_dataset_seeds and (
        args.episode_indices is not None or args.episode_indices_file is not None
    ):
        raise ValueError(
            "--unique-dataset-seeds cannot be combined with --episode-indices or "
            "--episode-indices-file"
        )
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.post_success_steps < 0:
        raise ValueError("--post-success-steps must be non-negative")
    validate_planning_horizons(
        planning_horizon_chunks=args.planning_horizon_chunks,
        execution_horizon_chunks=args.execution_horizon_chunks,
    )
    if not args.k_schedule or any(k <= 0 for k in args.k_schedule):
        raise ValueError("--k-schedule values must be positive")
    if args.policy_batch_size <= 0:
        raise ValueError("--policy-batch-size must be positive")
    if args.avoid_radius <= 0.0 or args.avoid_min_radius <= 0.0:
        raise ValueError("avoid radii must be positive")
    if any(h <= 0.0 for h in args.projection_half_extents):
        raise ValueError("--projection-half-extents components must be positive")
    if args.constraint_type == "projection" and args.constraint_placement != "candidate_midpath":
        raise ValueError(
            "--constraint-type projection requires --constraint-placement candidate_midpath"
        )
    if args.avoid_box_half_extents is not None and any(
        h <= 0.0 for h in args.avoid_box_half_extents
    ):
        raise ValueError("--avoid-box-half-extents components must be positive")
    if not args.avoid_path_fractions:
        raise ValueError("--avoid-path-fractions must have at least one value")
    if not all(0.0 <= f <= 1.0 for f in args.avoid_path_fractions):
        raise ValueError("--avoid-path-fractions values must be in [0, 1]")
    if args.constraint_placement_candidates <= 0:
        raise ValueError("--constraint-placement-candidates must be positive")
    if args.constraint_placement_steps is not None and args.constraint_placement_steps <= 0:
        raise ValueError("--constraint-placement-steps must be positive when set")
    if not 0.0 <= args.constraint_placement_path_fraction <= 1.0:
        raise ValueError("--constraint-placement-path-fraction must be in [0, 1]")
    for name in [
        "video_every_episodes",
        "rerun_every_episodes",
        "plot_every_episodes",
        "profile_every_episodes",
    ]:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.video_fps <= 0:
        raise ValueError("--video-fps must be positive")
    if not 0.0 <= args.constraint_overlay_alpha <= 1.0:
        raise ValueError("--constraint-overlay-alpha must be in [0, 1]")
    if any(value < 0.0 or value > 1.0 for value in args.constraint_overlay_color):
        raise ValueError("--constraint-overlay-color components must be in [0, 1]")
    if args.artifact_episode_count <= 0:
        raise ValueError("--artifact-episode-count must be positive")
    if args.robot_clearance_stride <= 0:
        raise ValueError("--robot-clearance-stride must be positive")
    if args.robot_clearance_placement_margin < 0.0:
        raise ValueError("--robot-clearance-placement-margin must be non-negative")
    guidance_methods = {"rejection", "reranking"}
    if (
        args.constraint_target == "robot"
        and args.geometry_mode == "fast"
        and any(method in guidance_methods for method in args.methods)
    ):
        raise ValueError(
            "--constraint-target robot requires --geometry-mode exact for guidance methods "
            "(rejection/reranking): fast geometry imagines no robot point cloud, so whole-robot "
            "guidance would be a silent no-op. Re-run with --geometry-mode exact, or use "
            "--constraint-target eef, or restrict --methods to base."
        )
    if args.artifact_selection_seed is None:
        args.artifact_selection_seed = args.seed
    return args


def resolve_checkpoint_path(checkpoint: Path | None, checkpoint_dir: Path | None) -> Path:
    """Resolve an explicit checkpoint or the latest step-named checkpoint in a directory."""
    if checkpoint is not None:
        return checkpoint
    if checkpoint_dir is None:
        raise ValueError("checkpoint or checkpoint_dir is required")
    return latest_reach_checkpoint(checkpoint_dir)


def run_eval_episode(
    *,
    sim_env: Any,
    ghost_env: Any,
    policy: SimpleDP3,
    adapter: DP3ChunkPolicyAdapter,
    method: EvalMethod,
    spec: RolloutSpec,
    constraints: list[AvoidRegion],
    action_mode: ActionMode,
    crop_config: PointCloudCropConfig,
    goal_thresh: float,
    output_dir: Path,
    max_steps: int,
    post_success_steps: int,
    planning_horizon_chunks: int,
    execution_horizon_chunks: int,
    geometry_mode: GeometryMode,
    k_schedule: tuple[int, ...],
    gripper_open: float,
    match_current_robot_points: bool,
    video: bool,
    rerun: bool,
    video_fps: int,
    decisions_file: Any,
    step_traces_file: Any,
    rng: np.random.Generator,
    timer: TimingRecorder,
    video_env_factory: Callable[[], Any] | None = None,
    constraint_overlay_alpha: float = 0.25,
    constraint_overlay_color: tuple[float, float, float] = (1.0, 0.25, 0.05),
    robot_clearance_metric: bool = False,
    robot_clearance_stride: int = 4,
    zarr_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if zarr_context is not None:
        sim_obs, sim_info = _reset_to_zarr_episode(
            sim_env, rollout_seed=spec.seed, zarr_context=zarr_context
        )
    else:
        sim_obs, sim_info = sim_env.reset(seed=spec.seed, options={"reconfigure": True})
    video_env: Any | None = None
    with timer.time("observation_adapt_crop", source="reset"):
        sim_entry = rollout_observation_entry(
            sim_obs,
            sim_info,
            env=sim_env,
            crop_config=crop_config,
        )
    if zarr_context is not None:
        sim_entry = _apply_zarr_initial_entry(sim_entry, zarr_context)
    obs_window = make_initial_obs_window(sim_entry, n_obs_steps=int(policy.n_obs_steps))
    target = np.asarray(sim_entry["target_position"], dtype=np.float32).reshape(3)
    scene = scene_context_for_constraints(
        target_position=target,
        constraints=constraints,
        metadata={"method": method, "episode": spec.output_index, "seed": spec.seed},
    )
    path = EpisodePath()
    _append_path(path, sim_entry)
    timeline = [sim_entry.copy()]
    _write_step_trace(
        step_traces_file,
        method=method,
        spec=spec,
        step=0,
        replan_index=None,
        selected_chunk_step=None,
        decision=None,
        entry=sim_entry,
        constraints=constraints,
        policy_action=None,
        sim_action=None,
    )
    frames = []
    if video:
        video_env = _maybe_create_overlay_video_env(
            video_env_factory=video_env_factory,
            spec=spec,
            constraints=constraints,
            color=constraint_overlay_color,
            alpha=constraint_overlay_alpha,
        )
        with timer.time("video_frame_render", method=method):
            frames.append(_frame_to_numpy(_render_video_frame(sim_env, video_env)))
    provider: ManiSkillGhostPandaGeometryProvider | None = None
    world_model: GeometricWorldModel | None = None
    if method != "base" or robot_clearance_metric:
        provider = ManiSkillGhostPandaGeometryProvider(
            ghost_env,
            task_name=_env_task_name(sim_env),
            crop_bounds=crop_config.bounds,
        )
        provider.reset(seed=spec.seed, options={"reconfigure": True})
        if method != "base":
            world_model = GeometricWorldModel(provider)

    steps = 0
    replans = 0
    first_success_step: int | None = None
    observed_post_success_steps = 0
    candidate_feasible = 0
    candidate_total = 0
    fallback_count = 0
    terminated_or_truncated = False
    was_training = policy.training
    policy.eval()
    try:
        while steps < max_steps:
            if first_success_step is not None and observed_post_success_steps >= post_success_steps:
                break
            decision = _select_decision(
                method=method,
                adapter=adapter,
                world_model=world_model,
                provider=provider,
                current_entry=sim_entry,
                obs_window=obs_window,
                scene=scene,
                constraints=constraints,
                crop_config=crop_config,
                goal_thresh=goal_thresh,
                planning_horizon_chunks=planning_horizon_chunks,
                geometry_mode=geometry_mode,
                k_schedule=k_schedule,
                match_current_robot_points=match_current_robot_points,
                rng=rng,
                timer=timer,
            )
            replans += 1
            if decision.result is not None:
                candidate_feasible += decision.candidate_feasible
                candidate_total += decision.candidate_total
                if decision.selection_reason == "least_bad_fallback":
                    fallback_count += 1
            _write_decision(
                decisions_file,
                method=method,
                spec=spec,
                replan_index=replans - 1,
                step=steps,
                decision=decision,
            )
            steps_to_execute = min(
                decision.selected_chunk.horizon,
                int(policy.n_action_steps) * execution_horizon_chunks,
                max_steps - steps,
            )
            if replans == 1:
                print(
                    "action chunk diagnostic: "
                    f"method={method} episode={spec.output_index} "
                    f"predicted_shape={decision.selected_chunk.actions.shape} "
                    f"chunk_horizon={decision.selected_chunk.horizon} "
                    f"policy_n_action_steps={int(policy.n_action_steps)} "
                    f"execution_horizon_chunks={execution_horizon_chunks} "
                    f"steps_to_execute={steps_to_execute}",
                    flush=True,
                )
            for selected_chunk_step, policy_action in enumerate(
                decision.selected_chunk.actions[:steps_to_execute]
            ):
                sim_action = policy_action_to_sim_action(
                    policy_action,
                    np.asarray(sim_entry["agent_pos"], dtype=np.float32),
                    action_mode=action_mode,
                    sim_action_dim=int(np.prod(sim_env.action_space.shape)),
                    low=getattr(sim_env.action_space, "low", None),
                    high=getattr(sim_env.action_space, "high", None),
                    gripper_open=gripper_open,
                )
                with timer.time("sim_step", method=method):
                    sim_obs, _reward, terminated, truncated, sim_info = sim_env.step(sim_action)
                steps += 1
                with timer.time("observation_adapt_crop", source="step"):
                    sim_entry = rollout_observation_entry(
                        sim_obs,
                        sim_info,
                        env=sim_env,
                        crop_config=crop_config,
                    )
                obs_window = append_obs_window(
                    obs_window,
                    sim_entry,
                    n_obs_steps=int(policy.n_obs_steps),
                )
                _append_path(path, sim_entry)
                timeline.append(sim_entry.copy())
                _write_step_trace(
                    step_traces_file,
                    method=method,
                    spec=spec,
                    step=steps,
                    replan_index=replans - 1,
                    selected_chunk_step=selected_chunk_step,
                    decision=decision,
                    entry=sim_entry,
                    constraints=constraints,
                    policy_action=policy_action,
                    sim_action=sim_action,
                )
                if video:
                    if video_env is not None:
                        try:
                            video_env.step(sim_action)
                        except Exception as exc:
                            print(
                                "warning: constraint overlay video step failed, "
                                f"falling back to plain render: {type(exc).__name__}: {exc}",
                                file=sys.stderr,
                            )
                            _close_env(video_env)
                            video_env = None
                    with timer.time("video_frame_render", method=method):
                        frames.append(_frame_to_numpy(_render_video_frame(sim_env, video_env)))
                success = _bool_info(sim_info, "success")
                if success and first_success_step is None:
                    first_success_step = steps
                elif first_success_step is not None:
                    observed_post_success_steps += 1
                terminated_or_truncated = _bool_any(terminated) or _bool_any(truncated)
                if terminated_or_truncated:
                    break
                if (
                    first_success_step is not None
                    and observed_post_success_steps >= post_success_steps
                ):
                    break
            if terminated_or_truncated:
                break
    finally:
        if was_training:
            policy.train()
        if video_env is not None:
            _close_env(video_env)

    video_path = None
    if video:
        video_path = output_dir / "videos" / method / f"episode_{spec.output_index:03d}.mp4"
        with timer.time("video_write", method=method):
            save_video(video_path, frames, fps=video_fps)
    rerun_path = None
    if rerun:
        rerun_path = output_dir / "rerun" / method / f"episode_{spec.output_index:03d}.rrd"
        with timer.time("rerun_write", method=method):
            save_rerun_timeline(rerun_path, timeline, constraints=constraints)
    robot_clearance_points: np.ndarray | None = None
    if robot_clearance_metric and constraints and provider is not None:
        try:
            with timer.time("robot_clearance_points", method=method):
                robot_clearance_points = _whole_robot_clearance_points(
                    path, provider, stride=robot_clearance_stride
                )
        except Exception as exc:
            print(
                "warning: whole-robot clearance metric failed, falling back to TCP-only: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            robot_clearance_points = None
    return episode_metric_row(
        method=method,
        episode=spec.output_index,
        seed=spec.seed,
        path=path,
        constraints=constraints,
        robot_clearance_points=robot_clearance_points,
        reach_success=first_success_step is not None,
        first_success_step=first_success_step,
        steps=steps,
        replans=replans,
        candidate_feasibility_fraction=candidate_feasibility_fraction(
            candidate_feasible,
            candidate_total,
        ),
        fallback_count=fallback_count,
        video=str(video_path) if video_path is not None else None,
        rerun=str(rerun_path) if rerun_path is not None else None,
    )


def _select_decision(
    *,
    method: EvalMethod,
    adapter: DP3ChunkPolicyAdapter,
    world_model: GeometricWorldModel | None,
    provider: ManiSkillGhostPandaGeometryProvider | None,
    current_entry: Entry,
    obs_window: list[Entry],
    scene: Any,
    constraints: list[AvoidRegion],
    crop_config: PointCloudCropConfig,
    goal_thresh: float,
    planning_horizon_chunks: int,
    geometry_mode: GeometryMode,
    k_schedule: tuple[int, ...],
    match_current_robot_points: bool,
    rng: np.random.Generator,
    timer: TimingRecorder,
) -> EvalDecisionSummary:
    if method == "base":
        chunk = adapter.sample_action_chunks(obs_window, k=1, rng=rng)[0]
        return EvalDecisionSummary(
            selected_chunk=chunk,
            result=None,
            candidate_feasible=0,
            candidate_total=0,
            selection_reason=None,
        )
    if world_model is None or provider is None:
        raise RuntimeError("controller methods require a world model and ghost provider")
    if match_current_robot_points:
        provider.set_robot_point_budget_from_mask(
            np.asarray(current_entry["robot_mask"], dtype=bool),
            point_valid_mask=np.asarray(current_entry["point_valid_mask"], dtype=bool),
        )
    controller_input = ControllerInput(
        observation=entry_to_world_model_observation(current_entry),
        scene=scene,
        policy_input=obs_window,
    )
    if geometry_mode == "exact" and planning_horizon_chunks == 1:
        controller_cls = RejectionController if method == "rejection" else RerankingController
        with timer.time("candidate_scoring", method=method, geometry_mode=geometry_mode):
            result = controller_cls(
                policy=adapter,
                world_model=world_model,
                constraints=constraints,
                k_schedule=k_schedule,
            ).select(controller_input, rng=rng)
    else:
        result = _select_multichunk(
            method=method,
            adapter=adapter,
            world_model=world_model,
            provider=provider,
            current_entry=current_entry,
            obs_window=obs_window,
            scene=scene,
            constraints=constraints,
            crop_config=crop_config,
            goal_thresh=goal_thresh,
            planning_horizon_chunks=planning_horizon_chunks,
            geometry_mode=geometry_mode,
            k_schedule=k_schedule,
            rng=rng,
            timer=timer,
        )
    feasible = sum(1 for candidate in result.candidates if candidate.feasible)
    return EvalDecisionSummary(
        selected_chunk=result.action_chunk,
        result=result,
        candidate_feasible=feasible,
        candidate_total=len(result.candidates),
        selection_reason=result.selection_reason,
    )


def _select_multichunk(
    *,
    method: EvalMethod,
    adapter: DP3ChunkPolicyAdapter,
    world_model: GeometricWorldModel,
    provider: ManiSkillGhostPandaGeometryProvider,
    current_entry: Entry,
    obs_window: list[Entry],
    scene: Any,
    constraints: list[AvoidRegion],
    crop_config: PointCloudCropConfig,
    goal_thresh: float,
    planning_horizon_chunks: int,
    geometry_mode: GeometryMode,
    k_schedule: tuple[int, ...],
    rng: np.random.Generator,
    timer: TimingRecorder,
) -> ControllerResult:
    candidates: list[CandidateDiagnostics] = []
    attempted: list[int] = []
    for k in k_schedule:
        attempted.append(k)
        with timer.time(
            "candidate_scoring",
            method=method,
            geometry_mode=geometry_mode,
            attempted_k=k,
        ):
            batch = _build_multichunk_candidates(
                adapter=adapter,
                world_model=world_model,
                provider=provider,
                current_entry=current_entry,
                obs_window=obs_window,
                scene=scene,
                constraints=constraints,
                crop_config=crop_config,
                goal_thresh=goal_thresh,
                planning_horizon_chunks=planning_horizon_chunks,
                geometry_mode=geometry_mode,
                attempted_k=k,
                start_index=len(candidates),
                rng=rng,
                timer=timer,
            )
        candidates.extend(batch)
        feasible = [candidate for candidate in candidates if candidate.feasible]
        if feasible:
            if method == "rejection":
                selected = feasible[0]
                return _controller_result(selected, candidates, attempted, "first_feasible")
            selected = min(feasible, key=lambda candidate: candidate.total_score)
            return _controller_result(selected, candidates, attempted, "best_feasible")
    if not candidates:
        raise RuntimeError("policy returned no candidate action chunks")
    selected = min(candidates, key=lambda candidate: candidate.total_score)
    return _controller_result(selected, candidates, attempted, "least_bad_fallback")


def _build_multichunk_candidates(
    *,
    adapter: DP3ChunkPolicyAdapter,
    world_model: GeometricWorldModel,
    provider: ManiSkillGhostPandaGeometryProvider,
    current_entry: Entry,
    obs_window: list[Entry],
    scene: Any,
    constraints: list[AvoidRegion],
    crop_config: PointCloudCropConfig,
    goal_thresh: float,
    planning_horizon_chunks: int,
    geometry_mode: GeometryMode,
    attempted_k: int,
    start_index: int,
    rng: np.random.Generator,
    timer: TimingRecorder,
) -> list[CandidateDiagnostics]:
    first_chunks = adapter.sample_action_chunks(obs_window, k=attempted_k, rng=rng)
    branch_entries = [_copy_entry(current_entry) for _ in first_chunks]
    branch_windows = [_copy_window(obs_window) for _ in first_chunks]
    branch_rollout_lists: list[list[ImaginedRollout]] = [[] for _ in first_chunks]
    next_chunks = list(first_chunks)
    for chunk_idx in range(planning_horizon_chunks):
        if chunk_idx > 0:
            next_chunks = adapter.sample_action_chunks_for_windows(
                branch_windows,
                rng=rng,
            )
        for branch_idx, next_chunk in enumerate(next_chunks):
            if geometry_mode == "exact":
                rollout = world_model.imagine(
                    entry_to_world_model_observation(branch_entries[branch_idx]),
                    next_chunk,
                    metadata={"branch": branch_idx, "chunk_index": chunk_idx},
                )
                branch_rollout_lists[branch_idx].append(rollout)
                for step_idx in range(rollout.action_chunk.horizon):
                    branch_entries[branch_idx] = world_model_entry_from_rollout_step(
                        rollout,
                        step_idx,
                        previous_entry=branch_entries[branch_idx],
                        crop_config=crop_config,
                        goal_thresh=goal_thresh,
                    )
                    branch_windows[branch_idx] = append_obs_window(
                        branch_windows[branch_idx],
                        branch_entries[branch_idx],
                        n_obs_steps=int(adapter.policy.n_obs_steps),
                    )
            else:
                rollout = _fast_imagine_rollout(
                    provider=provider,
                    observation=entry_to_world_model_observation(branch_entries[branch_idx]),
                    action_chunk=next_chunk,
                    metadata={"branch": branch_idx, "chunk_index": chunk_idx},
                    timer=timer,
                )
                branch_rollout_lists[branch_idx].append(rollout)
                if chunk_idx < planning_horizon_chunks - 1:
                    feedback_start = max(
                        0,
                        rollout.action_chunk.horizon - int(adapter.policy.n_obs_steps),
                    )
                    for step_idx in range(feedback_start, rollout.action_chunk.horizon):
                        branch_entries[branch_idx] = _render_feedback_entry(
                            provider=provider,
                            rollout=rollout,
                            step_index=step_idx,
                            previous_entry=branch_entries[branch_idx],
                            crop_config=crop_config,
                            goal_thresh=goal_thresh,
                            timer=timer,
                        )
                        branch_windows[branch_idx] = append_obs_window(
                            branch_windows[branch_idx],
                            branch_entries[branch_idx],
                            n_obs_steps=int(adapter.policy.n_obs_steps),
                        )

    branch_rollouts = [
        concatenate_rollouts(
            rollouts,
            metadata={"candidate_index": start_index + branch_idx},
        )
        for branch_idx, rollouts in enumerate(branch_rollout_lists)
    ]

    chunks = [rollout.action_chunk for rollout in branch_rollouts]
    consensus = consensus_deviations(chunks)
    return [
        _candidate_diagnostics(
            index=start_index + idx,
            attempted_k=attempted_k,
            action_chunk=rollout.action_chunk,
            rollout=rollout,
            scene=scene,
            constraints=constraints,
            consensus_deviation=consensus[idx],
        )
        for idx, rollout in enumerate(branch_rollouts)
    ]


def _candidate_diagnostics(
    *,
    index: int,
    attempted_k: int,
    action_chunk: ActionChunk,
    rollout: ImaginedRollout,
    scene: Any,
    constraints: list[AvoidRegion],
    consensus_deviation: float,
) -> CandidateDiagnostics:
    constraint_costs: dict[str, float] = {}
    constraint_satisfied: dict[str, bool] = {}
    for constraint_idx, constraint in enumerate(constraints):
        label = f"{constraint_idx}:{constraint.name}"
        costs = constraint.cost(rollout, scene)
        for key, value in costs.items():
            constraint_costs[_unique_cost_key(constraint_costs, key)] = float(value)
        constraint_satisfied[label] = bool(constraint.satisfied(rollout, scene))
    feasible = all(constraint_satisfied.values()) if constraint_satisfied else True
    distance = goal_distance(rollout, scene.target_position)
    smoothness = trajectory_smoothness(rollout, order=2)
    penalty = primary_constraint_penalty(constraint_costs)
    weights = ScoreWeights()
    total_score = (
        weights.constraint * penalty
        + weights.goal_distance * (0.0 if distance is None else distance)
        + weights.smoothness * smoothness
        + weights.consensus * consensus_deviation
    )
    return CandidateDiagnostics(
        index=index,
        attempted_k=attempted_k,
        action_chunk=action_chunk,
        rollout=rollout,
        constraint_costs=constraint_costs,
        constraint_satisfied=constraint_satisfied,
        feasible=feasible,
        goal_distance=distance,
        constraint_penalty=penalty,
        smoothness=smoothness,
        consensus_deviation=consensus_deviation,
        policy_surrogate=None,
        total_score=float(total_score),
    )


def _fast_imagine_rollout(
    *,
    provider: ManiSkillGhostPandaGeometryProvider,
    observation: Any,
    action_chunk: ActionChunk,
    metadata: dict[str, Any],
    timer: TimingRecorder,
) -> ImaginedRollout:
    """Imagine q/EEF trajectories without rendering robot point clouds for every step."""
    q = interpret_joint_chunk(action_chunk, observation.robot_state.joint_positions)
    eef_positions: list[np.ndarray] = []
    for q_step in q:
        with timer.time("ghost_eef_lookup", geometry_mode="fast"):
            eef_positions.append(provider.end_effector_position_only(q_step))
    horizon = action_chunk.horizon
    return ImaginedRollout(
        q=q,
        eef_path=np.stack(eef_positions, axis=0).astype(np.float32, copy=False),
        robot_point_clouds=[np.zeros((0, 3), dtype=np.float32) for _ in range(horizon)],
        scene_point_clouds=[np.zeros((0, 3), dtype=np.float32) for _ in range(horizon)],
        robot_masks=[np.zeros((0,), dtype=bool) for _ in range(horizon)],
        action_chunk=action_chunk,
        metadata={**metadata, "geometry_mode": "fast"},
        eef_orientations=(
            np.repeat(
                np.asarray(observation.robot_state.tcp_pose[3:7], dtype=np.float32).reshape(1, 4),
                repeats=horizon,
                axis=0,
            )
            if observation.robot_state.tcp_pose is not None
            and observation.robot_state.tcp_pose.shape[0] >= 7
            else None
        ),
    )


def _render_feedback_entry(
    *,
    provider: ManiSkillGhostPandaGeometryProvider,
    rollout: ImaginedRollout,
    step_index: int,
    previous_entry: Entry,
    crop_config: PointCloudCropConfig,
    goal_thresh: float,
    timer: TimingRecorder,
) -> Entry:
    """Render one imagined q state into a policy-shaped observation entry."""
    q = rollout.q[step_index]
    with timer.time("ghost_pointcloud_render", geometry_mode="fast"):
        robot_points = provider.robot_point_cloud(q)
    static_scene = static_scene_from_robot_mask(
        entry_to_world_model_observation(previous_entry).point_cloud,
        entry_to_world_model_observation(previous_entry).robot_mask,
    )
    scene, robot_mask = compose_robot_cloud(static_scene, robot_points)
    one_step_rollout = ImaginedRollout(
        q=q.reshape(1, -1),
        eef_path=rollout.eef_path[step_index].reshape(1, 3),
        robot_point_clouds=[robot_points],
        scene_point_clouds=[scene],
        robot_masks=[robot_mask],
        action_chunk=ActionChunk(
            actions=rollout.action_chunk.actions[step_index].reshape(1, -1),
            action_mode=rollout.action_chunk.action_mode,
            dt=rollout.action_chunk.dt,
            metadata=rollout.action_chunk.metadata,
        ),
        metadata=rollout.metadata,
        eef_orientations=(
            rollout.eef_orientations[step_index].reshape(1, -1)
            if rollout.eef_orientations is not None
            else None
        ),
    )
    return world_model_entry_from_rollout_step(
        one_step_rollout,
        0,
        previous_entry=previous_entry,
        crop_config=crop_config,
        goal_thresh=goal_thresh,
    )


def _controller_result(
    selected: CandidateDiagnostics,
    candidates: list[CandidateDiagnostics],
    attempted: list[int],
    reason: str,
) -> ControllerResult:
    selected.selection_reason = reason
    return ControllerResult(
        selected=selected,
        candidates=candidates,
        attempted_k_values=list(attempted),
        selection_reason=reason,
    )


def _action_chunk_trace(action_chunk: ActionChunk) -> dict[str, Any]:
    return {
        "actions": action_chunk.actions,
        "action_mode": action_chunk.action_mode,
        "dt": float(action_chunk.dt),
        "horizon": int(action_chunk.horizon),
        "metadata": action_chunk.metadata,
    }


def _rollout_trace(rollout: ImaginedRollout) -> dict[str, Any]:
    return {
        "q": rollout.q,
        "eef_path": rollout.eef_path,
        "eef_orientations": rollout.eef_orientations,
        "metadata": rollout.metadata,
        "action_chunk_metadata": rollout.action_chunk.metadata,
    }


def _candidate_summary(candidate: CandidateDiagnostics) -> dict[str, Any]:
    return {
        "index": candidate.index,
        "attempted_k": candidate.attempted_k,
        "feasible": candidate.feasible,
        "total_score": candidate.total_score,
        "goal_distance": candidate.goal_distance,
        "constraint_penalty": candidate.constraint_penalty,
        "smoothness": candidate.smoothness,
        "consensus_deviation": candidate.consensus_deviation,
        "policy_surrogate": candidate.policy_surrogate,
        "directional": candidate.directional,
        "constraint_costs": candidate.constraint_costs,
        "constraint_satisfied": candidate.constraint_satisfied,
        "action_chunk_metadata": candidate.action_chunk.metadata,
        "rollout_metadata": candidate.rollout.metadata,
    }


def _selected_branch_trace(decision: EvalDecisionSummary | None) -> dict[str, Any]:
    if decision is None:
        return {}
    row: dict[str, Any] = {
        "selected_action_chunk": _action_chunk_trace(decision.selected_chunk),
        "selection_reason": decision.selection_reason,
        "candidate_feasible": decision.candidate_feasible,
        "candidate_total": decision.candidate_total,
    }
    if decision.result is not None:
        selected = decision.result.selected
        row.update(
            {
                "attempted_k_values": decision.result.attempted_k_values,
                "selected_index": selected.index,
                "selected_score": selected.total_score,
                "selected_feasible": selected.feasible,
                "selected_goal_distance": selected.goal_distance,
                "selected_constraint_penalty": selected.constraint_penalty,
                "selected_constraint_costs": selected.constraint_costs,
                "selected_action_chunk": _action_chunk_trace(selected.action_chunk),
                "selected_rollout": _rollout_trace(selected.rollout),
            }
        )
    return row


def _write_step_trace(
    step_traces_file: Any,
    *,
    method: EvalMethod,
    spec: RolloutSpec,
    step: int,
    replan_index: int | None,
    selected_chunk_step: int | None,
    decision: EvalDecisionSummary | None,
    entry: Entry,
    constraints: list[Any],
    policy_action: np.ndarray | None,
    sim_action: np.ndarray | None,
) -> None:
    branch = _selected_branch_trace(decision)
    row = {
        "method": method,
        "episode": spec.output_index,
        "seed": spec.seed,
        "source": spec.source,
        "dataset_episode_index": spec.dataset_episode_index,
        "step": int(step),
        "frame_index": int(step),
        "replan_index": replan_index,
        "selected_chunk_step": selected_chunk_step,
        "policy_action": policy_action,
        "sim_action": sim_action,
        "tcp_pose": entry["tcp_pose"],
        "agent_pos": entry["agent_pos"],
        "target_position": entry["target_position"],
        "target_distance": float(np.asarray(entry["final_distance"]).reshape(-1)[0]),
        "success": bool(entry["success"]),
        "cartesian_pose_metrics": cartesian_pose_step_metrics(
            tcp_pose=entry["tcp_pose"],
            constraints=constraints,
            step=step,
        ),
    }
    row.update(
        {
            "selected_index": branch.get("selected_index"),
            "selection_reason": branch.get("selection_reason"),
            "candidate_feasible": branch.get("candidate_feasible"),
            "candidate_total": branch.get("candidate_total"),
            "selected_score": branch.get("selected_score"),
            "selected_constraint_costs": branch.get("selected_constraint_costs"),
        }
    )
    step_traces_file.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
    step_traces_file.flush()


def _write_decision(
    decisions_file: Any,
    *,
    method: EvalMethod,
    spec: RolloutSpec,
    replan_index: int,
    step: int,
    decision: EvalDecisionSummary,
) -> None:
    result = decision.result
    row = {
        "method": method,
        "episode": spec.output_index,
        "seed": spec.seed,
        "source": spec.source,
        "dataset_episode_index": spec.dataset_episode_index,
        "replan_index": replan_index,
        "step": step,
        "selection_reason": decision.selection_reason,
        "candidate_feasible": decision.candidate_feasible,
        "candidate_total": decision.candidate_total,
        "selected_action_chunk": _action_chunk_trace(decision.selected_chunk),
    }
    if result is not None:
        scores = [candidate.total_score for candidate in result.candidates]
        row.update(
            {
                "attempted_k_values": result.attempted_k_values,
                "selected_index": result.selected.index,
                "selected_score": result.selected.total_score,
                "selected_feasible": result.selected.feasible,
                "selected_goal_distance": result.selected.goal_distance,
                "selected_constraint_penalty": result.selected.constraint_penalty,
                "selected_smoothness": result.selected.smoothness,
                "selected_constraint_costs": result.selected.constraint_costs,
                "selected_action_chunk": _action_chunk_trace(result.selected.action_chunk),
                "selected_rollout": _rollout_trace(result.selected.rollout),
                "candidate_summaries": [
                    _candidate_summary(candidate) for candidate in result.candidates
                ],
                "score_min": min(scores) if scores else None,
                "score_mean": float(np.mean(scores)) if scores else None,
            }
        )
    decisions_file.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
    decisions_file.flush()


def _entry_robot_points(entry: Entry) -> np.ndarray:
    """Return the start-configuration robot point cloud from an observation entry.

    Uses the segmented robot mask (and validity mask when present) so the keep-out
    placement check sees the actual robot links and base at episode start.
    """
    point_cloud = np.asarray(entry["point_cloud"], dtype=np.float32).reshape(-1, 3)
    robot_mask = np.asarray(entry["robot_mask"], dtype=bool).reshape(-1)
    mask = robot_mask
    valid = entry.get("point_valid_mask")
    if valid is not None:
        mask = mask & np.asarray(valid, dtype=bool).reshape(-1)
    if mask.shape[0] != point_cloud.shape[0]:
        return np.zeros((0, 3), dtype=np.float32)
    return point_cloud[mask].astype(np.float32, copy=False)


def _clear_region_from_robot(
    region: SphereRegion | BoxRegion,
    robot_points: np.ndarray,
    args: argparse.Namespace,
    *,
    clearance: float,
    name: str,
    max_iter: int = 64,
) -> SphereRegion | BoxRegion:
    """Shrink/translate an avoid region so it does not intersect the robot point cloud.

    A sphere is first shrunk toward --avoid-min-radius; if its center is still too close
    to the robot it is translated directly away from the nearest robot point. A box is
    translated away from the nearest point. Guarantees ``min signed distance >= clearance``
    when achievable within ``max_iter`` iterations.
    """
    pts = np.asarray(robot_points, dtype=np.float32).reshape(-1, 3)
    if pts.size == 0:
        return region
    min_radius = float(args.avoid_min_radius)
    if isinstance(region, SphereRegion):
        center = region.center.astype(np.float32).copy()
        radius = float(region.radius)
        for _ in range(max_iter):
            dist = np.linalg.norm(pts - center.reshape(1, 3), axis=1)
            min_d = float(np.min(dist))
            if min_d - radius >= clearance:
                return SphereRegion(center=center, radius=radius)
            target_r = min_d - clearance
            if target_r >= min_radius:
                return SphereRegion(center=center, radius=float(target_r))
            nearest = pts[int(np.argmin(dist))]
            direction = center - nearest
            norm = float(np.linalg.norm(direction))
            if norm < 1e-6:
                direction = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                norm = 1.0
            step = (min_radius + clearance - min_d) + 1e-3
            center = (center + direction / norm * step).astype(np.float32)
            radius = min_radius
        print(
            f"warning: could not fully clear avoid region '{name}' from robot "
            f"(min clearance still < {clearance:.3f} m after {max_iter} iters)",
            file=sys.stderr,
        )
        return SphereRegion(center=center, radius=max(min_radius, radius))
    # BoxRegion: translate away from nearest robot point until clear.
    center = region.center.astype(np.float32).copy()
    half_extents = region.half_extents.astype(np.float32)
    for _ in range(max_iter):
        candidate = BoxRegion(center=center, half_extents=half_extents)
        signed = candidate.signed_distance(pts)
        min_sd = float(np.min(signed))
        if min_sd >= clearance:
            return candidate
        nearest = pts[int(np.argmin(signed))]
        direction = center - nearest
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            direction = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            norm = 1.0
        step = (clearance - min_sd) + 1e-3
        center = (center + direction / norm * step).astype(np.float32)
    print(
        f"warning: could not fully clear avoid box '{name}' from robot "
        f"(min clearance still < {clearance:.3f} m after {max_iter} iters)",
        file=sys.stderr,
    )
    return BoxRegion(center=center, half_extents=half_extents)


def _finalize_constraints(
    constraints: list[AvoidRegion],
    *,
    robot_points: np.ndarray | None,
    args: argparse.Namespace,
) -> list[AvoidRegion]:
    """Apply the guidance target and robot-clearance-aware placement to placed regions."""
    enable_clearance = (
        bool(args.robot_clearance_placement)
        and robot_points is not None
        and len(robot_points) > 0
    )
    finalized: list[AvoidRegion] = []
    for constraint in constraints:
        region = constraint.region
        if enable_clearance and isinstance(region, (SphereRegion, BoxRegion)):
            region = _clear_region_from_robot(
                region,
                robot_points,
                args,
                clearance=float(args.robot_clearance_placement_margin),
                name=constraint.name,
            )
        finalized.append(replace(constraint, region=region, target=args.constraint_target))
    return finalized


def _episode_start_robot_points(
    env: Any,
    *,
    spec: RolloutSpec,
    crop_config: PointCloudCropConfig,
    zarr_context: dict[str, Any] | None,
) -> np.ndarray:
    """Reset to the episode start and return the robot point cloud for placement clearance."""
    if zarr_context is not None:
        obs, info = _reset_to_zarr_episode(env, rollout_seed=spec.seed, zarr_context=zarr_context)
    else:
        obs, info = env.reset(seed=spec.seed, options={"reconfigure": True})
    entry = rollout_observation_entry(obs, info, env=env, crop_config=crop_config)
    if zarr_context is not None:
        entry = _apply_zarr_initial_entry(entry, zarr_context)
    return _entry_robot_points(entry)


def _constraints_for_episode(
    env: Any,
    *,
    spec: RolloutSpec,
    policy: SimpleDP3,
    adapter: DP3ChunkPolicyAdapter,
    action_mode: ActionMode,
    crop_config: PointCloudCropConfig,
    goal_thresh: float,
    args: argparse.Namespace,
    zarr_context: dict[str, Any] | None = None,
) -> list[AvoidRegion]:
    if args.constraints_dir is not None:
        return load_episode_constraints(_precomputed_constraint_path(args.constraints_dir, spec))
    if args.constraint_placement == "candidate_midpath":
        constraints = _candidate_midpath_constraints(
            env,
            spec=spec,
            policy=policy,
            adapter=adapter,
            action_mode=action_mode,
            crop_config=crop_config,
            goal_thresh=goal_thresh,
            args=args,
            zarr_context=zarr_context,
            output_dir=getattr(args, "output_dir", None),
        )
    elif args.constraint_placement == "widest_trajectory":
        constraints = _widest_trajectory_constraints(
            env,
            spec=spec,
            policy=policy,
            adapter=adapter,
            action_mode=action_mode,
            crop_config=crop_config,
            goal_thresh=goal_thresh,
            args=args,
            zarr_context=zarr_context,
            output_dir=getattr(args, "output_dir", None),
        )
    else:
        constraints = _episode_constraints(
            env, spec=spec, crop_config=crop_config, args=args, zarr_context=zarr_context
        )
    robot_points: np.ndarray | None = None
    if args.robot_clearance_placement:
        robot_points = _episode_start_robot_points(
            env, spec=spec, crop_config=crop_config, zarr_context=zarr_context
        )
    return _finalize_constraints(constraints, robot_points=robot_points, args=args)


def _precomputed_constraint_path(constraints_dir: Path, spec: RolloutSpec) -> Path:
    path = constraints_dir / f"episode_{spec.output_index:03d}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"missing precomputed constraint file for output episode {spec.output_index}: {path}"
        )
    return path


def _episode_constraints(
    env: Any,
    *,
    spec: RolloutSpec,
    crop_config: PointCloudCropConfig,
    args: argparse.Namespace,
    zarr_context: dict[str, Any] | None = None,
) -> list[AvoidRegion]:
    if zarr_context is not None:
        obs, info = _reset_to_zarr_episode(env, rollout_seed=spec.seed, zarr_context=zarr_context)
    else:
        obs, info = env.reset(seed=spec.seed, options={"reconfigure": True})
    entry = rollout_observation_entry(obs, info, env=env, crop_config=crop_config)
    if zarr_context is not None:
        entry = _apply_zarr_initial_entry(entry, zarr_context)
    start_tcp = np.asarray(entry["tcp_pose"], dtype=np.float32).reshape(-1)[:3]
    target = np.asarray(entry["target_position"], dtype=np.float32).reshape(3)
    fractions = list(args.avoid_path_fractions)
    multi = len(fractions) > 1
    constraints = []
    for i, frac in enumerate(fractions):
        name = f"direct_path_avoid_region_{i}" if multi else "direct_path_avoid_region"
        constraints.append(
            direct_path_avoid_region(
                start_tcp=start_tcp,
                target_position=target,
                config=AvoidOverlayConfig(
                    radius=args.avoid_radius,
                    min_radius=args.avoid_min_radius,
                    margin=args.avoid_margin,
                    weight=args.avoid_weight,
                    path_fraction=frac,
                    name=name,
                    shape=args.avoid_shape,
                    box_half_extents=(
                        tuple(args.avoid_box_half_extents)
                        if args.avoid_box_half_extents is not None
                        else None
                    ),
                ),
            )
        )
    return constraints


def _avoid_region_of_shape(
    center: np.ndarray,
    radius: float,
    args: argparse.Namespace,
) -> SphereRegion | BoxRegion:
    """Build an avoid-region primitive of the configured shape at the given center.

    The center matches exactly where a sphere would be placed; only the shape differs.
    A box uses --avoid-box-half-extents when provided, otherwise isotropic half-extents
    equal to the effective sphere radius so it occupies a comparable volume.
    """
    center = np.asarray(center, dtype=np.float32).reshape(3)
    if getattr(args, "avoid_shape", "sphere") in ("box", "cuboid"):
        if args.avoid_box_half_extents is not None:
            half_extents = np.asarray(args.avoid_box_half_extents, dtype=np.float32)
        else:
            half_extents = np.full(3, float(radius), dtype=np.float32)
        return BoxRegion(center=center, half_extents=half_extents)
    return SphereRegion(center=center, radius=float(radius))


def _build_placed_constraint(
    *,
    center: np.ndarray,
    radius: float,
    name: str,
    args: argparse.Namespace,
    projection_half_extents: np.ndarray | None = None,
) -> AvoidRegion | AvoidProjection:
    """Build the placed keep-out constraint of the configured family at ``center``.

    'region' returns a 3-D AvoidRegion (sphere/box) sized by ``radius``/shape args;
    'projection' returns an AvoidProjection over a 2-D XY rectangle (no-overflight)
    using the XY of ``center``. The rectangle half-extents come from
    ``projection_half_extents`` when provided (sized from the candidate bundle,
    analogous to ``radius``); otherwise they fall back to the
    --projection-half-extents floor. ``radius`` is ignored for projection.
    """
    if args.constraint_type == "projection":
        half_extents = (
            np.asarray(projection_half_extents, dtype=np.float32)
            if projection_half_extents is not None
            else np.asarray(args.projection_half_extents, dtype=np.float32)
        )
        return AvoidProjection(
            region=RectRegion2D(
                center=np.asarray(center, dtype=np.float32).reshape(3)[:2],
                half_extents=half_extents,
            ),
            margin=float(args.avoid_margin),
            weight=float(args.avoid_weight),
            name=name,
        )
    return AvoidRegion(
        region=_avoid_region_of_shape(center, radius, args),
        margin=float(args.avoid_margin),
        weight=float(args.avoid_weight),
        name=name,
    )


def _placed_constraint_name(base: str, index: int, *, multi: bool, args: argparse.Namespace) -> str:
    """Return the canonical placed-constraint name, reflecting the constraint family."""
    suffix = "avoid_projection" if args.constraint_type == "projection" else "avoid_region"
    stem = f"{base}_{suffix}"
    return f"{stem}_{index}" if multi else stem


def _collect_candidate_paths(
    env: Any,
    *,
    spec: RolloutSpec,
    policy: SimpleDP3,
    adapter: DP3ChunkPolicyAdapter,
    action_mode: ActionMode,
    crop_config: PointCloudCropConfig,
    goal_thresh: float,
    args: argparse.Namespace,
    zarr_context: dict[str, Any] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Roll out base-policy candidate TCP paths for constraint placement.

    Returns (all_paths, successful_paths). Shared by candidate_midpath and
    widest_trajectory placement so both see the same candidate bundle.
    """
    max_steps = int(args.constraint_placement_steps or args.max_steps)
    candidate_count = int(args.constraint_placement_candidates)
    paths: list[np.ndarray] = []
    successful_paths: list[np.ndarray] = []
    was_training = policy.training
    policy.eval()
    try:
        for _ in range(candidate_count):
            path, success = _rollout_base_candidate_path(
                env,
                spec=spec,
                adapter=adapter,
                action_mode=action_mode,
                crop_config=crop_config,
                max_steps=max_steps,
                goal_thresh=goal_thresh,
                gripper_open=float(args.gripper_open),
                zarr_context=zarr_context,
            )
            paths.append(path)
            if success:
                successful_paths.append(path)
    finally:
        if was_training:
            policy.train()
    return paths, successful_paths


def _path_lateral_width(path: np.ndarray) -> float:
    """Return a path's peak lateral deviation from its straight start-goal chord.

    This is the largest perpendicular distance of any path point from the line that
    joins the path's first and last TCP positions -- i.e. how far the trajectory
    "bows out". Degenerate paths (no motion) report their max spread from the start.
    """
    points = np.asarray(path, dtype=np.float32).reshape(-1, 3)
    if points.shape[0] < 2:
        return 0.0
    start = points[0]
    chord = points[-1] - start
    chord_length = float(np.linalg.norm(chord))
    rel = points - start.reshape(1, 3)
    if chord_length <= 1e-8:
        return float(np.max(np.linalg.norm(rel, axis=1)))
    unit = chord / chord_length
    projection = rel @ unit
    perpendicular = rel - np.outer(projection, unit)
    return float(np.max(np.linalg.norm(perpendicular, axis=1)))


def _widest_trajectory_constraints(
    env: Any,
    *,
    spec: RolloutSpec,
    policy: SimpleDP3,
    adapter: DP3ChunkPolicyAdapter,
    action_mode: ActionMode,
    crop_config: PointCloudCropConfig,
    goal_thresh: float,
    args: argparse.Namespace,
    zarr_context: dict[str, Any] | None = None,
    output_dir: Path | None = None,
) -> list[AvoidRegion]:
    """Place avoid regions along the widest sampled candidate trajectory.

    Rolls out base-policy candidates, selects the single path that bows out the most
    from its straight start-goal chord, then plants one region per
    --avoid-path-fractions value at those arc-length fractions along that path.
    """
    fractions = list(args.avoid_path_fractions)
    multi = len(fractions) > 1
    paths, successful_paths = _collect_candidate_paths(
        env,
        spec=spec,
        policy=policy,
        adapter=adapter,
        action_mode=action_mode,
        crop_config=crop_config,
        goal_thresh=goal_thresh,
        args=args,
        zarr_context=zarr_context,
    )
    selected_paths = (
        successful_paths if args.constraint_placement_success_only and successful_paths else paths
    )
    if not selected_paths:
        raise RuntimeError("widest_trajectory constraint placement produced no candidate paths")

    widths = [_path_lateral_width(path) for path in selected_paths]
    widest_index = int(np.argmax(widths))
    widest_path = selected_paths[widest_index]
    widest_width = float(widths[widest_index])

    constraints: list[AvoidRegion] = []
    for i, frac in enumerate(fractions):
        center = _point_at_arc_fraction(widest_path, fraction=frac).astype(np.float32)
        radius = max(float(args.avoid_min_radius), float(args.avoid_radius))
        name = (
            f"widest_trajectory_avoid_region_{i}" if multi else "widest_trajectory_avoid_region"
        )
        print(
            f"widest-trajectory constraint [{i}]: "
            f"episode={spec.output_index} frac={frac:.2f} "
            f"sampled={len(paths)} successful={len(successful_paths)} "
            f"used={len(selected_paths)} widest_idx={widest_index} "
            f"width={widest_width:.4f} center={center.tolist()} "
            f"shape={args.avoid_shape} radius={radius:.4f}",
            flush=True,
        )
        if getattr(args, "plot_candidate_paths", False) and output_dir is not None:
            _plot_candidate_paths(
                output_dir=output_dir,
                episode_index=spec.output_index,
                paths=paths,
                successful_paths=successful_paths,
                selected_paths=[widest_path],
                center=center,
                radius=radius,
                path_fraction=frac,
                constraint_index=i if multi else None,
                projection_half_extents=None,
            )
        constraints.append(
            AvoidRegion(
                region=_avoid_region_of_shape(center, radius, args),
                margin=float(args.avoid_margin),
                weight=float(args.avoid_weight),
                name=name,
            )
        )
    return constraints


def _candidate_midpath_constraints(
    env: Any,
    *,
    spec: RolloutSpec,
    policy: SimpleDP3,
    adapter: DP3ChunkPolicyAdapter,
    action_mode: ActionMode,
    crop_config: PointCloudCropConfig,
    goal_thresh: float,
    args: argparse.Namespace,
    zarr_context: dict[str, Any] | None = None,
    output_dir: Path | None = None,
) -> list[AvoidRegion]:
    fractions = list(args.avoid_path_fractions)
    multi = len(fractions) > 1
    paths, successful_paths = _collect_candidate_paths(
        env,
        spec=spec,
        policy=policy,
        adapter=adapter,
        action_mode=action_mode,
        crop_config=crop_config,
        goal_thresh=goal_thresh,
        args=args,
        zarr_context=zarr_context,
    )
    selected_paths = (
        successful_paths if args.constraint_placement_success_only and successful_paths else paths
    )
    if not selected_paths:
        raise RuntimeError("candidate_midpath constraint placement produced no candidate paths")

    constraints: list[AvoidRegion] = []
    is_projection = args.constraint_type == "projection"
    # For a height-agnostic projection only the XY footprint matters, so place the
    # center by XY (tabletop) arc length -- a vertical lift must not shift where the
    # rectangle lands. Region (sphere/box) placement stays on the 3-D arc length.
    arc_at = _point_at_arc_fraction_xy if is_projection else _point_at_arc_fraction
    for i, frac in enumerate(fractions):
        centers = np.stack(
            [arc_at(path, fraction=frac) for path in selected_paths],
            axis=0,
        )
        center = np.median(centers, axis=0).astype(np.float32)
        radius = _effective_avoid_radius(
            center=center,
            paths=selected_paths,
            requested_radius=float(args.avoid_radius),
            min_radius=float(args.avoid_min_radius),
        )
        name = _placed_constraint_name("candidate_midpath", i, multi=multi, args=args)
        projection_half_extents: np.ndarray | None = None
        if is_projection:
            projection_half_extents = _effective_projection_half_extents(
                center_xy=center[:2],
                paths=selected_paths,
                fraction=frac,
                min_half_extents=np.asarray(args.projection_half_extents, dtype=np.float32),
            )
            shape_desc = (
                f"shape=rect2d half_extents={projection_half_extents.tolist()} "
                f"min_half_extents={list(args.projection_half_extents)}"
            )
        else:
            shape_desc = f"shape={args.avoid_shape} radius={radius:.4f}"
        print(
            f"candidate-midpath constraint [{i}]: "
            f"episode={spec.output_index} frac={frac:.2f} "
            f"type={args.constraint_type} "
            f"sampled={len(paths)} successful={len(successful_paths)} "
            f"used={len(selected_paths)} center={center.tolist()} "
            f"{shape_desc}",
            flush=True,
        )
        if getattr(args, "plot_candidate_paths", False) and output_dir is not None:
            _plot_candidate_paths(
                output_dir=output_dir,
                episode_index=spec.output_index,
                paths=paths,
                successful_paths=successful_paths,
                selected_paths=selected_paths,
                center=center,
                radius=radius,
                path_fraction=frac,
                constraint_index=i if multi else None,
                projection_half_extents=projection_half_extents,
            )
        constraints.append(
            _build_placed_constraint(
                center=center,
                radius=radius,
                name=name,
                args=args,
                projection_half_extents=projection_half_extents,
            )
        )
    return constraints


def _rollout_base_candidate_path(
    env: Any,
    *,
    spec: RolloutSpec,
    adapter: DP3ChunkPolicyAdapter,
    action_mode: ActionMode,
    crop_config: PointCloudCropConfig,
    max_steps: int,
    goal_thresh: float,
    gripper_open: float,
    zarr_context: dict[str, Any] | None = None,
) -> tuple[np.ndarray, bool]:
    if zarr_context is not None:
        obs, info = _reset_to_zarr_episode(env, rollout_seed=spec.seed, zarr_context=zarr_context)
    else:
        obs, info = env.reset(seed=spec.seed, options={"reconfigure": True})
    entry = rollout_observation_entry(obs, info, env=env, crop_config=crop_config)
    if zarr_context is not None:
        entry = _apply_zarr_initial_entry(entry, zarr_context)
    obs_window = make_initial_obs_window(entry, n_obs_steps=int(adapter.policy.n_obs_steps))
    tcp_path = [np.asarray(entry["tcp_pose"], dtype=np.float32).reshape(-1)[:3]]
    success = _bool_info(info, "success")
    steps = 0
    while steps < max_steps and not success:
        chunk = adapter.sample_action_chunks(obs_window, k=1)[0]
        steps_to_execute = min(int(adapter.policy.n_action_steps), chunk.horizon, max_steps - steps)
        for policy_action in chunk.actions[:steps_to_execute]:
            sim_action = policy_action_to_sim_action(
                policy_action,
                np.asarray(entry["agent_pos"], dtype=np.float32),
                action_mode=action_mode,
                sim_action_dim=int(np.prod(env.action_space.shape)),
                low=getattr(env.action_space, "low", None),
                high=getattr(env.action_space, "high", None),
                gripper_open=gripper_open,
            )
            obs, _reward, terminated, truncated, info = env.step(sim_action)
            steps += 1
            entry = rollout_observation_entry(obs, info, env=env, crop_config=crop_config)
            obs_window = append_obs_window(
                obs_window,
                entry,
                n_obs_steps=int(adapter.policy.n_obs_steps),
            )
            tcp = np.asarray(entry["tcp_pose"], dtype=np.float32).reshape(-1)[:3]
            tcp_path.append(tcp)
            distance = float(np.asarray(entry["final_distance"], dtype=np.float32).reshape(-1)[0])
            success = _bool_info(info, "success") or distance <= goal_thresh
            if success or _bool_any(terminated) or _bool_any(truncated) or steps >= max_steps:
                break
    return np.asarray(tcp_path, dtype=np.float32), bool(success)


def _point_at_arc_fraction(path: np.ndarray, *, fraction: float) -> np.ndarray:
    points = np.asarray(path, dtype=np.float32).reshape(-1, 3)
    if points.shape[0] == 0:
        raise ValueError("path must contain at least one point")
    if points.shape[0] == 1:
        return points[0].astype(np.float32, copy=True)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total = float(np.sum(segment_lengths))
    if total <= 1e-8:
        return points[0].astype(np.float32, copy=True)
    target = float(np.clip(fraction, 0.0, 1.0)) * total
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    idx = int(np.searchsorted(cumulative, target, side="right") - 1)
    idx = min(max(idx, 0), points.shape[0] - 2)
    segment = float(segment_lengths[idx])
    alpha = 0.0 if segment <= 1e-8 else (target - float(cumulative[idx])) / segment
    return ((1.0 - alpha) * points[idx] + alpha * points[idx + 1]).astype(np.float32)


def _effective_avoid_radius(
    *,
    center: np.ndarray,
    paths: list[np.ndarray],
    requested_radius: float,
    min_radius: float,
) -> float:
    if requested_radius <= 0.0 or min_radius <= 0.0:
        raise ValueError("avoid radii must be positive")
    points = np.concatenate([np.asarray(path, dtype=np.float32).reshape(-1, 3) for path in paths])
    if points.size == 0:
        return float(max(min_radius, requested_radius))
    distances = np.linalg.norm(points - center.reshape(1, 3), axis=1)
    spread_radius = float(np.percentile(distances, 37.5)) if distances.size else requested_radius
    return float(max(min_radius, spread_radius))


def _effective_projection_half_extents(
    *,
    center_xy: np.ndarray,
    paths: list[np.ndarray],
    fraction: float,
    min_half_extents: np.ndarray,
    window: float = 0.15,
    percentile: float = 37.5,
) -> np.ndarray:
    """Return the projection rectangle half-extents (= ``min_half_extents``).

    The spread-based inflation was removed because it sized the rectangle to
    match the path-bundle width at the chokepoint, which — for a height-agnostic
    projection constraint — left no lateral escape route and caused every candidate
    to fail. The caller supplies the desired obstacle footprint via
    ``--projection-half-extents``; that value is used directly.
    """
    min_half = np.asarray(min_half_extents, dtype=np.float32).reshape(2)
    if np.any(min_half <= 0.0):
        raise ValueError("projection min half-extents must be positive")
    return min_half.copy()


def _point_at_arc_fraction_xy(path: np.ndarray, *, fraction: float) -> np.ndarray:
    """Like ``_point_at_arc_fraction`` but parameterized by XY (tabletop) arc length.

    The along-path position is chosen by cumulative XY distance so a vertical lift
    does not move where the projection footprint lands. The returned value is the
    full interpolated 3-D point (its z is ignored by the rectangle).
    """
    points = np.asarray(path, dtype=np.float32).reshape(-1, 3)
    if points.shape[0] == 0:
        raise ValueError("path must contain at least one point")
    if points.shape[0] == 1:
        return points[0].astype(np.float32, copy=True)
    segment_lengths = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    total = float(np.sum(segment_lengths))
    if total <= 1e-8:
        return points[0].astype(np.float32, copy=True)
    target = float(np.clip(fraction, 0.0, 1.0)) * total
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    idx = int(np.searchsorted(cumulative, target, side="right") - 1)
    idx = min(max(idx, 0), points.shape[0] - 2)
    segment = float(segment_lengths[idx])
    alpha = 0.0 if segment <= 1e-8 else (target - float(cumulative[idx])) / segment
    return ((1.0 - alpha) * points[idx] + alpha * points[idx + 1]).astype(np.float32)


def _local_path_points_xy(
    path: np.ndarray,
    *,
    fraction: float,
    window: float,
) -> np.ndarray:
    """Return the XY points of ``path`` within ``+/- window`` (XY arc fraction) of ``fraction``.

    Always returns at least one point -- the interpolated XY point at ``fraction`` --
    so callers can concatenate across candidate paths without special-casing empties.
    """
    points = np.asarray(path, dtype=np.float32).reshape(-1, 3)
    if points.shape[0] == 0:
        raise ValueError("path must contain at least one point")
    if points.shape[0] == 1:
        return points[:, :2].astype(np.float32, copy=True)
    segment_lengths = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    total = float(np.sum(segment_lengths))
    if total <= 1e-8:
        return points[:1, :2].astype(np.float32, copy=True)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)]) / total
    mask = (cumulative >= float(fraction) - float(window)) & (
        cumulative <= float(fraction) + float(window)
    )
    if not np.any(mask):
        return _point_at_arc_fraction_xy(path, fraction=fraction)[:2].reshape(1, 2)
    return points[mask, :2].astype(np.float32, copy=False)


def _plot_candidate_paths(
    *,
    output_dir: Path,
    episode_index: int,
    paths: list[np.ndarray],
    successful_paths: list[np.ndarray],
    selected_paths: list[np.ndarray],
    center: np.ndarray,
    radius: float,
    path_fraction: float,
    constraint_index: int | None = None,
    projection_half_extents: np.ndarray | None = None,
) -> None:
    try:
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception as exc:
        print(
            f"warning: matplotlib unavailable, skipping candidate path plot: {exc}",
            file=sys.stderr,
        )
        return

    is_projection = projection_half_extents is not None
    successful_set = {id(p) for p in successful_paths}
    selected_set = {id(p) for p in selected_paths}

    fig = plt.figure(figsize=(13, 5))

    # --- 3-D view ---
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    for path in paths:
        pts = np.asarray(path, dtype=np.float32).reshape(-1, 3)
        is_success = id(path) in successful_set
        is_selected = id(path) in selected_set
        color = "tab:green" if is_selected else ("tab:orange" if is_success else "tab:gray")
        alpha = 0.85 if is_selected else 0.4
        lw = 1.5 if is_selected else 0.8
        ax3d.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, alpha=alpha, linewidth=lw)
        ax3d.scatter(pts[0, 0], pts[0, 1], pts[0, 2], color="tab:blue", s=18, zorder=5)
        ax3d.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], color="tab:red", s=18, zorder=5)

    if is_projection:
        # Draw projection rectangle extruded over the visible z range.
        hx, hy = float(projection_half_extents[0]), float(projection_half_extents[1])
        cx, cy = float(center[0]), float(center[1])
        z_lo, z_hi = _PROJECTION_OVERLAY_Z_RANGE
        xs = [cx - hx, cx + hx, cx + hx, cx - hx, cx - hx]
        ys = [cy - hy, cy - hy, cy + hy, cy + hy, cy - hy]
        for z in (z_lo, z_hi):
            ax3d.plot(xs, ys, [z] * 5, color="crimson", alpha=0.5, linewidth=1.2)
        for xc, yc in zip(xs[:-1], ys[:-1], strict=False):
            ax3d.plot(
                [xc, xc],
                [yc, yc],
                [z_lo, z_hi],
                color="crimson",
                alpha=0.3,
                linewidth=0.8,
            )
        ax3d.scatter(cx, cy, 0.5 * (z_lo + z_hi), color="crimson", s=60, zorder=10)
        constraint_legend_label = f"projection {2*hx:.3f}×{2*hy:.3f}m"
    else:
        # Draw sphere wireframe.
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        sx = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        sy = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        sz = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
        ax3d.plot_wireframe(sx, sy, sz, color="crimson", alpha=0.25, linewidth=0.6)
        ax3d.scatter(*center, color="crimson", s=60, zorder=10)
        constraint_legend_label = f"sphere r={radius:.3f}m"

    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.set_title(f"Episode {episode_index} — 3-D candidate paths")
    legend_handles = [
        Line2D([0], [0], color="tab:green", lw=1.5, label=f"selected ({len(selected_paths)})"),
        Line2D(
            [0],
            [0],
            color="tab:gray",
            lw=0.8,
            alpha=0.5,
            label=f"failed ({len(paths) - len(successful_paths)})",
        ),
        Line2D([0], [0], color="tab:blue", marker="o", lw=0, markersize=5, label="start"),
        Line2D([0], [0], color="tab:red", marker="o", lw=0, markersize=5, label="end"),
        Line2D([0], [0], color="crimson", lw=1, label=constraint_legend_label),
    ]
    ax3d.legend(handles=legend_handles, fontsize=7, loc="upper left")

    # --- XY top-down view ---
    arc_fn = _point_at_arc_fraction_xy if is_projection else _point_at_arc_fraction
    ax2d = fig.add_subplot(1, 2, 2)
    for path in paths:
        pts = np.asarray(path, dtype=np.float32).reshape(-1, 3)
        is_selected = id(path) in selected_set
        color = "tab:green" if is_selected else "tab:gray"
        alpha = 0.85 if is_selected else 0.35
        lw = 1.5 if is_selected else 0.8
        ax2d.plot(pts[:, 0], pts[:, 1], color=color, alpha=alpha, linewidth=lw)
        ax2d.scatter(pts[0, 0], pts[0, 1], color="tab:blue", s=18, zorder=5)
        ax2d.scatter(pts[-1, 0], pts[-1, 1], color="tab:red", s=18, zorder=5)
        # mark the arc-fraction point used for center computation
        try:
            arc_pt = arc_fn(pts, fraction=path_fraction)
            ax2d.scatter(arc_pt[0], arc_pt[1], color="gold", s=22, zorder=6, marker="x")
        except Exception:
            pass

    if is_projection:
        hx, hy = float(projection_half_extents[0]), float(projection_half_extents[1])
        rect = mpatches.Rectangle(
            (float(center[0]) - hx, float(center[1]) - hy),
            2 * hx, 2 * hy,
            linewidth=1.5, edgecolor="crimson", facecolor="crimson", alpha=0.15,
            label=f"projection {2*hx:.3f}×{2*hy:.3f}m",
        )
        ax2d.add_patch(rect)
        ax2d.scatter(center[0], center[1], color="crimson", s=60, zorder=10)
    else:
        circle = plt.Circle(
            (center[0], center[1]), radius, color="crimson", fill=False, linewidth=1.5,
            label=f"sphere r={radius:.3f}m",
        )
        ax2d.add_patch(circle)
        ax2d.scatter(center[0], center[1], color="crimson", s=60, zorder=10)

    ax2d.set_aspect("equal")
    ax2d.set_xlabel("x")
    ax2d.set_ylabel("y")
    ax2d.set_title(f"Episode {episode_index} — XY top-down")
    ax2d.grid(True, alpha=0.25)
    ax2d.legend(fontsize=7)

    n_total = len(paths)
    n_success = len(successful_paths)
    n_selected = len(selected_paths)
    constraint_tag = f" | constraint {constraint_index}" if constraint_index is not None else ""
    fig.suptitle(
        f"Candidate midpath — episode {episode_index}{constraint_tag} | "
        f"sampled={n_total} success={n_success} selected={n_selected} | "
        f"arc_frac={path_fraction:.2f} {constraint_legend_label}",
        fontsize=9,
    )
    fig.tight_layout()

    plots_dir = output_dir / "candidate_path_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if constraint_index is None else f"_c{constraint_index}"
    out_path = plots_dir / f"episode_{episode_index:03d}{suffix}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"candidate path plot saved: {out_path}", flush=True)


def _episode_indices_from_args(
    args: argparse.Namespace,
    *,
    dataset_episode_seeds: list[int],
) -> list[int] | None:
    if args.unique_dataset_seeds:
        return _unique_seed_episode_indices(
            dataset_episode_seeds,
            max_count=int(args.episodes),
        )
    if args.episode_indices_file is None:
        return args.episode_indices
    return _read_episode_indices_file(args.episode_indices_file)


def _unique_seed_episode_indices(
    dataset_episode_seeds: list[int],
    *,
    max_count: int,
) -> list[int]:
    if max_count <= 0:
        raise ValueError("max_count must be positive")
    seen: set[int] = set()
    indices: list[int] = []
    for dataset_idx, seed in enumerate(dataset_episode_seeds):
        if seed in seen:
            continue
        seen.add(seed)
        indices.append(dataset_idx)
        if len(indices) >= max_count:
            break
    if not indices:
        raise ValueError("dataset metadata did not contain any episode seeds")
    return indices


def _read_episode_indices_file(path: Path) -> list[int]:
    indices: list[int] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            value = int(line)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number} is not an integer episode index") from exc
        if value < 0:
            raise ValueError(f"{path}:{line_number} episode index must be non-negative")
        indices.append(value)
    if not indices:
        raise ValueError(f"{path} did not contain any episode indices")
    return indices


def _constraint_source_summary(args: argparse.Namespace) -> dict[str, Any]:
    if args.constraints_dir is not None:
        return {
            "type": "precomputed",
            "constraints_dir": str(args.constraints_dir),
            "episode_indices_file": (
                str(args.episode_indices_file) if args.episode_indices_file is not None else None
            ),
        }
    return {
        "type": str(args.constraint_placement),
        "constraint_type": str(args.constraint_type),
        "projection_half_extents": [float(h) for h in args.projection_half_extents],
        "constraint_placement_candidates": int(args.constraint_placement_candidates),
        "constraint_placement_steps": (
            None
            if args.constraint_placement_steps is None
            else int(args.constraint_placement_steps)
        ),
        "avoid_path_fractions": [float(f) for f in args.avoid_path_fractions],
        "constraint_placement_success_only": bool(args.constraint_placement_success_only),
        "avoid_radius": float(args.avoid_radius),
        "avoid_min_radius": float(args.avoid_min_radius),
        "avoid_margin": float(args.avoid_margin),
        "avoid_weight": float(args.avoid_weight),
        "avoid_shape": str(args.avoid_shape),
        "avoid_box_half_extents": (
            [float(h) for h in args.avoid_box_half_extents]
            if args.avoid_box_half_extents is not None
            else None
        ),
    }


def _repeat_obs_window_to_torch(
    window: list[Entry],
    *,
    k: int,
    device: torch.device,
    goal_marker_points: int = 0,
    goal_marker_radius: float = DEFAULT_GOAL_MARKER_RADIUS,
) -> dict[str, torch.Tensor]:
    batch = obs_window_to_torch(
        window,
        device=device,
        goal_marker_points=goal_marker_points,
        goal_marker_radius=goal_marker_radius,
    )
    return {
        key: value.repeat((k, *([1] * (value.ndim - 1))))
        for key, value in batch.items()
    }


def _obs_windows_to_torch(
    windows: list[list[Entry]],
    *,
    device: torch.device,
    goal_marker_points: int = 0,
    goal_marker_radius: float = DEFAULT_GOAL_MARKER_RADIUS,
) -> dict[str, torch.Tensor]:
    if not windows:
        raise ValueError("windows must not be empty")
    point_cloud = np.stack(
        [np.stack([entry["point_cloud"] for entry in window], axis=0) for window in windows],
        axis=0,
    )
    if goal_marker_points:
        target_position = np.stack(
            [
                np.stack([entry["target_position"] for entry in window], axis=0)
                for window in windows
            ],
            axis=0,
        )
        point_cloud = insert_goal_marker_points(
            point_cloud,
            target_position,
            num_points=goal_marker_points,
            radius=goal_marker_radius,
        )
    agent_pos = np.stack(
        [np.stack([entry["agent_pos"] for entry in window], axis=0) for window in windows],
        axis=0,
    )
    goal_xyz = np.stack(
        [
            np.stack([entry["target_position"] for entry in window], axis=0)
            for window in windows
        ],
        axis=0,
    )
    ee_position = np.stack(
        [
            np.stack(
                [
                    np.asarray(entry["tcp_pose"], dtype=np.float32).reshape(-1)[:3]
                    for entry in window
                ],
                axis=0,
            )
            for window in windows
        ],
        axis=0,
    )
    return {
        "point_cloud": torch.from_numpy(point_cloud.astype(np.float32)).to(device),
        "agent_pos": torch.from_numpy(agent_pos.astype(np.float32)).to(device),
        "goal_xyz": torch.from_numpy(goal_xyz.astype(np.float32)).to(device),
        "ee_position": torch.from_numpy(ee_position.astype(np.float32)).to(device),
    }


def _append_path(path: EpisodePath, entry: Entry) -> None:
    path.append_pose(
        tcp_pose=entry["tcp_pose"],
        q=np.asarray(entry["agent_pos"], dtype=np.float32),
        target_distance=float(np.asarray(entry["final_distance"], dtype=np.float32).reshape(-1)[0]),
    )


def _whole_robot_clearance_points(
    path: EpisodePath,
    provider: ManiSkillGhostPandaGeometryProvider,
    *,
    stride: int = 4,
) -> np.ndarray:
    """Sample the URDF/mesh robot point cloud across the executed trajectory.

    Sets the ghost env to each stored qpos (subsampled by ``stride``) and collects the
    mesh-derived robot points, so constraint clearance reflects the whole arm and base
    rather than only the TCP. Returns a [M, 3] cloud (empty when no q is available).
    """
    q_array = path.q_array
    if q_array.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    stride = max(1, int(stride))
    indices = list(range(0, q_array.shape[0], stride))
    if indices[-1] != q_array.shape[0] - 1:
        indices.append(q_array.shape[0] - 1)
    clouds: list[np.ndarray] = []
    for idx in indices:
        points = np.asarray(provider.robot_point_cloud(q_array[idx]), dtype=np.float32)
        if points.size:
            clouds.append(points.reshape(-1, 3))
    if not clouds:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(clouds, axis=0).astype(np.float32, copy=False)


def _env_kwargs(metadata: dict[str, Any], *, render_mode: str | None) -> dict[str, Any]:
    env_kwargs = dict(metadata["env_kwargs"])
    env_kwargs["obs_mode"] = "pointcloud"
    env_kwargs["num_envs"] = 1
    if render_mode is None:
        env_kwargs.pop("render_mode", None)
    else:
        env_kwargs["render_mode"] = render_mode
    return env_kwargs


def _video_env_factory(
    gym: Any,
    *,
    metadata: dict[str, Any],
    enabled: bool,
) -> Callable[[], Any] | None:
    if not enabled:
        return None
    env_kwargs = _env_kwargs(metadata, render_mode="rgb_array")

    def factory() -> Any:
        return gym.make(str(metadata["env_id"]), **env_kwargs)

    return factory


def _maybe_create_overlay_video_env(
    *,
    video_env_factory: Callable[[], Any] | None,
    spec: RolloutSpec,
    constraints: list[AvoidRegion],
    color: tuple[float, float, float],
    alpha: float,
) -> Any | None:
    if video_env_factory is None:
        return None
    video_env = None
    try:
        video_env = video_env_factory()
        video_env.reset(seed=spec.seed, options={"reconfigure": True})
        _add_constraint_overlay_actors(
            video_env,
            constraints=constraints,
            color=color,
            alpha=alpha,
        )
        return video_env
    except Exception as exc:
        print(
            "warning: constraint overlay video setup failed, falling back to plain render: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        if video_env is not None:
            _close_env(video_env)
        return None


def _add_constraint_overlay_actors(
    env: Any,
    *,
    constraints: list[AvoidRegion | CartesianPoseConstraint],
    color: tuple[float, float, float],
    alpha: float,
) -> None:
    """Add visual-only keep-out actors to a render-only ManiSkill env."""
    import sapien
    from mani_skill.utils.building import actors

    unwrapped = getattr(env, "unwrapped", env)
    scene = unwrapped.scene
    rgba = [float(color[0]), float(color[1]), float(color[2]), float(alpha)]
    for constraint_idx, constraint in enumerate(constraints):
        if isinstance(constraint, CartesianPoseConstraint):
            _add_cartesian_pose_overlay_actors(
                scene,
                constraint=constraint,
                actors=actors,
                sapien=sapien,
                alpha=alpha,
            )
            continue
        region = constraint.region
        name = f"pg3d_avoid_region_overlay_{constraint_idx}"
        if isinstance(region, SphereRegion):
            actors.build_sphere(
                scene,
                radius=float(region.radius),
                color=rgba,
                name=name,
                body_type="kinematic",
                add_collision=False,
                initial_pose=sapien.Pose(p=region.center.tolist()),
            )
        elif isinstance(region, BoxRegion):
            actors.build_box(
                scene,
                half_sizes=region.half_extents.tolist(),
                color=rgba,
                name=name,
                body_type="kinematic",
                add_collision=False,
                initial_pose=sapien.Pose(p=region.center.tolist()),
            )
        elif isinstance(region, RectRegion2D):
            # Render the height-agnostic XY footprint as an extruded box for visibility.
            z_lo, z_hi = _PROJECTION_OVERLAY_Z_RANGE
            actors.build_box(
                scene,
                half_sizes=[
                    float(region.half_extents[0]),
                    float(region.half_extents[1]),
                    0.5 * (z_hi - z_lo),
                ],
                color=rgba,
                name=name,
                body_type="kinematic",
                add_collision=False,
                initial_pose=sapien.Pose(
                    p=[float(region.center[0]), float(region.center[1]), 0.5 * (z_lo + z_hi)]
                ),
            )
    update_render = getattr(scene, "update_render", None)
    if callable(update_render):
        update_render()


def _add_cartesian_pose_overlay_actors(
    scene: Any,
    *,
    constraint: CartesianPoseConstraint,
    actors: Any,
    sapien: Any,
    alpha: float,
) -> None:
    position = np.asarray(constraint.target_position, dtype=np.float32).reshape(3)
    orientation = np.asarray(constraint.target_orientation, dtype=np.float32).reshape(-1)
    rotation = _rotation_matrix_from_pose_orientation(orientation)
    rgba_origin = [1.0, 1.0, 1.0, float(alpha)]
    actors.build_sphere(
        scene,
        radius=0.01,
        color=rgba_origin,
        name=f"{constraint.name}_target_position",
        body_type="kinematic",
        add_collision=False,
        initial_pose=sapien.Pose(p=position.tolist()),
    )
    axis_specs = [
        ("x", (1.0, 0.3, 0.3, alpha)),
        ("y", (0.3, 1.0, 0.3, alpha)),
        ("z", (0.3, 0.5, 1.0, alpha)),
    ]
    for axis_idx, (label, rgba) in enumerate(axis_specs):
        axis = rotation[:, axis_idx]
        length = 0.04
        center = position + axis * (0.5 * length)
        quat = _quat_from_two_vectors(np.array([0.0, 0.0, 1.0], dtype=np.float32), axis)
        actors.build_box(
            scene,
            half_sizes=[0.003, 0.003, 0.5 * length],
            color=list(map(float, rgba)),
            name=f"{constraint.name}_{label}_axis",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=center.tolist(), q=quat.tolist()),
        )


def _rotation_matrix_from_pose_orientation(orientation: np.ndarray) -> np.ndarray:
    q = np.asarray(orientation, dtype=np.float32).reshape(-1)
    if q.shape == (4,):
        return _quat_to_matrix(q)
    if q.shape == (9,):
        return q.reshape(3, 3)
    raise ValueError(f"unsupported target_orientation shape: {q.shape}")


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float32).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("quaternion must be finite and non-zero")
    w, x, y, z = q / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _quat_from_two_vectors(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    a = np.asarray(u, dtype=np.float32)
    b = np.asarray(v, dtype=np.float32)
    a /= max(float(np.linalg.norm(a)), 1e-8)
    b /= max(float(np.linalg.norm(b)), 1e-8)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot < -0.999999:
        axis = np.cross(a, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        if float(np.linalg.norm(axis)) < 1e-6:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        axis /= max(float(np.linalg.norm(axis)), 1e-8)
        return np.asarray([0.0, axis[0], axis[1], axis[2]], dtype=np.float32)
    cross = np.cross(a, b)
    q = np.asarray([1.0 + dot, cross[0], cross[1], cross[2]], dtype=np.float32)
    q /= max(float(np.linalg.norm(q)), 1e-8)
    return q


def _render_video_frame(sim_env: Any, video_env: Any | None) -> Any:
    return video_env.render() if video_env is not None else sim_env.render()


def _close_env(env: Any) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()


def _copy_entry(entry: Entry) -> Entry:
    return {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in entry.items()
    }


def _copy_window(window: list[Entry]) -> list[Entry]:
    return [_copy_entry(entry) for entry in window]


def _action_mode(value: str) -> ActionMode:
    if value not in {"abs_joint", "delta_joint"}:
        raise ValueError(f"unsupported action_mode {value!r}")
    return value  # type: ignore[return-value]


def _env_task_name(env: Any) -> str:
    unwrapped = getattr(env, "unwrapped", env)
    spec = getattr(unwrapped, "spec", None)
    return str(getattr(spec, "id", "unknown"))


def _artifact_selection_summary(
    specs: list[RolloutSpec],
    *,
    video_episode_indices: set[int],
    rerun_episode_indices: set[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    spec_by_output = {spec.output_index: spec for spec in specs}
    return {
        "selection": args.artifact_selection,
        "episode_count": args.artifact_episode_count,
        "seed": args.artifact_selection_seed,
        "video": _selected_spec_summary(spec_by_output, video_episode_indices),
        "rerun": _selected_spec_summary(spec_by_output, rerun_episode_indices),
    }


def _selected_spec_summary(
    spec_by_output: dict[int, RolloutSpec],
    selected_output_indices: set[int],
) -> list[dict[str, int | str | None]]:
    rows: list[dict[str, int | str | None]] = []
    for output_index in sorted(selected_output_indices):
        spec = spec_by_output[output_index]
        rows.append(
            {
                "output_index": spec.output_index,
                "seed": spec.seed,
                "source": spec.source,
                "dataset_episode_index": spec.dataset_episode_index,
            }
        )
    return rows


def _unique_cost_key(costs: dict[str, float], key: str) -> str:
    if key not in costs:
        return key
    suffix = 1
    while f"{key}#{suffix}" in costs:
        suffix += 1
    return f"{key}#{suffix}"


def _init_wandb(
    args: argparse.Namespace,
    *,
    metadata: dict[str, Any],
    checkpoint_path: Path,
) -> Any | None:
    if args.wandb_mode == "disabled":
        return None
    try:
        import wandb

        return wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            mode=args.wandb_mode,
            config={
                "dataset": str(args.dataset),
                "checkpoint": str(checkpoint_path),
                "env_id": metadata.get("env_id"),
                "methods": list(args.methods),
                "planning_horizon_chunks": args.planning_horizon_chunks,
                "execution_horizon_chunks": args.execution_horizon_chunks,
                "k_schedule": list(args.k_schedule),
                "constraint_source": _constraint_source_summary(args),
                "artifact_selection": args.artifact_selection,
                "artifact_episode_count": args.artifact_episode_count,
                "artifact_selection_seed": args.artifact_selection_seed,
                "constraint_overlay_video": bool(args.constraint_overlay_video),
                "constraint_overlay_alpha": float(args.constraint_overlay_alpha),
                "constraint_overlay_color": list(args.constraint_overlay_color),
                "command": "scripts/eval_constrained_reach.py",
            },
        )
    except Exception as exc:
        if args.wandb_required:
            raise
        print(
            f"warning: W&B init failed, continuing without W&B: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return None


def _log_wandb_summary(
    run: Any,
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    try:
        import wandb

        metrics: dict[str, Any] = {}
        for method, method_summary in summary["by_method"].items():
            for key, value in method_summary.items():
                if isinstance(value, (int, float)) and value is not None:
                    metrics[f"eval/{method}/{key}"] = value
        columns = sorted({key for row in rows for key in row.keys()})
        table = wandb.Table(columns=columns)
        for row in rows:
            table.add_data(*[_jsonable(row.get(column)) for column in columns])
        metrics["eval/episodes"] = table
        if args.video:
            for row in rows:
                video = row.get("video")
                if video and Path(str(video)).exists():
                    metrics[f"eval_video/{row['method']}/episode_{int(row['episode']):03d}"] = (
                        wandb.Video(str(video), fps=args.video_fps, format="mp4")
                    )
        run.log(metrics)
    except Exception as exc:
        if args.wandb_required:
            raise
        print(
            f"warning: W&B summary logging failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )


def _log_wandb_episode(
    run: Any | None,
    *,
    args: argparse.Namespace,
    row: dict[str, Any],
    global_step: int,
) -> None:
    if run is None:
        return
    try:
        metrics = {
            f"episode/{row['method']}/reach_success": float(row["reach_success"]),
            f"episode/{row['method']}/constraint_satisfied": float(
                row["constraint_satisfied"]
            ),
            f"episode/{row['method']}/combined_success": float(row["combined_success"]),
            f"episode/{row['method']}/final_target_distance": row["final_target_distance"],
            f"episode/{row['method']}/min_clearance": row["min_clearance"],
            f"episode/{row['method']}/candidate_feasibility_fraction": row[
                "candidate_feasibility_fraction"
            ],
            f"episode/{row['method']}/fallback_count": row["fallback_count"],
            "episode/index": row["episode"],
        }
        metrics = {key: value for key, value in metrics.items() if value is not None}
        video = row.get("video")
        if video and Path(str(video)).exists():
            import wandb

            metrics[f"episode_video/{row['method']}/episode_{int(row['episode']):03d}"] = (
                wandb.Video(str(video), fps=args.video_fps, format="mp4")
            )
        with _null_timer():
            run.log(metrics, step=global_step)
    except Exception as exc:
        if args.wandb_required:
            raise
        print(
            f"warning: W&B episode logging failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )


def _maybe_emit_progress(
    *,
    output_dir: Path,
    rows: list[dict[str, Any]],
    timer: TimingRecorder,
    episode_index: int,
    plots: bool,
    run: Any | None,
    args: argparse.Namespace,
    final: bool = False,
) -> None:
    if not rows:
        return
    by_method = summarize_metrics(rows)
    plot_paths: list[Path] = []
    if plots:
        with timer.time("plot_write", final=final):
            plot_paths = _write_progress_plots(
                output_dir,
                rows=rows,
                timing=timer.summary(),
                episode_index=episode_index,
                final=final,
            )
    if run is None:
        return
    try:
        metrics: dict[str, Any] = {}
        for method, method_summary in by_method.items():
            for key, value in method_summary.items():
                if isinstance(value, (int, float)) and value is not None:
                    metrics[f"progress/{method}/{key}"] = value
        metrics["progress/episode"] = episode_index
        if plot_paths:
            import wandb

            for path in plot_paths:
                metrics[f"progress_plot/{path.stem}"] = wandb.Image(str(path))
        with timer.time("wandb_log", kind="progress"):
            run.log(metrics)
    except Exception as exc:
        if args.wandb_required:
            raise
        print(
            f"warning: W&B progress logging failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )


def _write_progress_plots(
    output_dir: Path,
    *,
    rows: list[dict[str, Any]],
    timing: dict[str, dict[str, float]],
    episode_index: int,
    final: bool,
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(
            f"warning: matplotlib unavailable for plots: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return []

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    series = progress_series(rows)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for method, method_series in series.items():
        x = np.arange(1, len(method_series["episode"]) + 1)
        axes[0, 0].plot(x, method_series["combined_success_rate"], label=method)
        axes[0, 1].plot(x, method_series["final_target_distance"], label=method)
        axes[1, 0].plot(x, method_series["min_clearance"], label=method)
        axes[1, 1].plot(x, method_series["candidate_feasibility_fraction"], label=method)
    axes[0, 0].set_ylim(0.0, 1.0)
    axes[0, 0].set_title("Cumulative combined success")
    axes[0, 1].set_title("Final target distance")
    axes[1, 0].set_title("Minimum clearance")
    axes[1, 1].set_title("Candidate feasibility fraction")
    for ax in axes.flat:
        ax.set_xlabel("Completed episode rows per method")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    fig.tight_layout()
    suffix = "final" if final else f"episode_{episode_index:04d}"
    progress_path = plots_dir / f"progress_{suffix}.png"
    latest_progress = plots_dir / "latest_progress.png"
    fig.savefig(progress_path)
    fig.savefig(latest_progress)
    plt.close(fig)
    paths = [progress_path, latest_progress]

    if timing:
        names = list(timing.keys())
        totals = [timing[name]["total"] for name in names]
        order = np.argsort(totals)[-10:]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh([names[idx] for idx in order], [totals[idx] for idx in order])
        ax.set_xlabel("Total seconds")
        ax.set_title("Timing breakdown")
        fig.tight_layout()
        timing_path = plots_dir / f"timing_{suffix}.png"
        latest_timing = plots_dir / "latest_timing.png"
        fig.savefig(timing_path)
        fig.savefig(latest_timing)
        plt.close(fig)
        paths.extend([timing_path, latest_timing])
    return paths


def _write_new_timing_events(
    timer: TimingRecorder,
    path: Path,
    *,
    start_index: int,
) -> int:
    if not timer.enabled:
        return start_index
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        for idx, event in enumerate(timer.events[start_index:], start=start_index):
            file.write(json.dumps({"index": idx, **event.to_json()}, sort_keys=True) + "\n")
    return len(timer.events)


def _write_summary_plots(
    output_dir: Path,
    *,
    rows: list[dict[str, Any]],
) -> list[Path]:
    """Generate final summary plots showing success rates and counts by method."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(
            f"warning: matplotlib unavailable for summary plots: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return []

    if not rows:
        return []

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Group rows by method
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        method = row.get("method")
        if method not in by_method:
            by_method[method] = []
        by_method[method].append(row)

    methods = sorted(by_method.keys())
    if not methods:
        return []

    # Compute statistics per method
    stats = {}
    for method in methods:
        method_rows = by_method[method]
        reach_success = sum(1 for r in method_rows if r.get("reach_success"))
        constraint_satisfied = sum(1 for r in method_rows if r.get("constraint_satisfied"))
        combined_success = sum(1 for r in method_rows if r.get("combined_success"))
        total = len(method_rows)
        stats[method] = {
            "reach_success": reach_success,
            "constraint_satisfied": constraint_satisfied,
            "combined_success": combined_success,
            "total": total,
        }

    paths: list[Path] = []

    # Plot 1: Success counts by method
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(methods))
    width = 0.25
    reach_counts = [stats[m]["reach_success"] for m in methods]
    constraint_counts = [stats[m]["constraint_satisfied"] for m in methods]
    combined_counts = [stats[m]["combined_success"] for m in methods]
    ax.bar(x - width, reach_counts, width, label="Reach Success", color="skyblue")
    ax.bar(x, constraint_counts, width, label="Constraint Satisfied", color="lightcoral")
    ax.bar(x + width, combined_counts, width, label="Combined Success", color="lightgreen")
    ax.set_xlabel("Method")
    ax.set_ylabel("Count")
    ax.set_title("Success Counts by Method")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    summary_counts_path = plots_dir / "summary_counts.png"
    fig.savefig(summary_counts_path)
    plt.close(fig)
    paths.append(summary_counts_path)

    # Plot 2: Success rates by method
    fig, ax = plt.subplots(figsize=(10, 5))
    reach_rates = [stats[m]["reach_success"] / stats[m]["total"] for m in methods]
    constraint_rates = [stats[m]["constraint_satisfied"] / stats[m]["total"] for m in methods]
    combined_rates = [stats[m]["combined_success"] / stats[m]["total"] for m in methods]
    ax.bar(x - width, reach_rates, width, label="Reach Success", color="skyblue")
    ax.bar(x, constraint_rates, width, label="Constraint Satisfied", color="lightcoral")
    ax.bar(x + width, combined_rates, width, label="Combined Success", color="lightgreen")
    ax.set_xlabel("Method")
    ax.set_ylabel("Success Rate")
    ax.set_title("Success Rates by Method")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim([0.0, 1.05])
    ax.legend()
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    summary_rates_path = plots_dir / "summary_rates.png"
    fig.savefig(summary_rates_path)
    plt.close(fig)
    paths.append(summary_rates_path)

    return paths


def _print_timing_summary(timer: TimingRecorder) -> None:
    summary = timer.summary()
    if not summary:
        return
    top = sorted(summary.items(), key=lambda item: item[1]["total"], reverse=True)[:6]
    text = ", ".join(
        f"{name}={values['total']:.2f}s/{int(values['count'])}x" for name, values in top
    )
    print(f"timing: {text}")


def _cuda_sync_fn(device: torch.device) -> Any | None:
    if device.type != "cuda":
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.synchronize


def _seed_torch(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


class _null_timer:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_args: Any) -> bool:
        return False


def _format_optional(value: Any) -> str:
    if value is None:
        return "nan"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(numeric):
        return "nan"
    return f"{numeric:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
