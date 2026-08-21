from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import zarr

from pg3d.composition import (
    CandidateDiagnostics,
    ControllerInput,
    ControllerResult,
    ConvexScoreWeights,
    GuidedScoreConfig,
    GuidedScoreMode,
    RejectionController,
    RerankingController,
    ScoreWeights,
)
from pg3d.composition.scoring import (
    consensus_deviations,
    directional_preference,
    goal_distance,
    primary_constraint_penalty,
    trajectory_smoothness,
)
from pg3d.constraints import (
    DEFAULT_AVOID_MARGIN_M,
    AvoidProjection,
    AvoidRegion,
    BoxRegion,
    CylinderRegion,
    RectRegion2D,
    SphereRegion,
)
from pg3d.constraints.torch_geometry import AvoidanceEnergyMode, avoidance_energy
from pg3d.envs.maniskill_adapter import (
    ManiSkillGhostPandaGeometryProvider,
    register_pg3d_reach_envs,
)
from pg3d.envs.maniskill_adapter.dataset import (
    PointCloudCropConfig,
    git_commit_info,
    load_reach_metadata,
)
from pg3d.envs.maniskill_adapter.panda_collision import (
    load_panda_collision_point_template,
)
from pg3d.envs.obstacles import (
    CABINET_COMPONENTS,
    U_SHAPE_REFERENCE_HALF_EXTENTS,
    scaled_cabinet_components,
    transform_box_component,
    u_shape_components,
)
from pg3d.envs.xarm_adapter import register_pg3d_xarm7_gripper_reach_envs
from pg3d.eval import (
    AvoidOverlayConfig,
    EpisodePath,
    GuidedProposalTraceWriter,
    TimingRecorder,
    action_sha256,
    candidate_feasibility_fraction,
    concatenate_rollouts,
    constraint_fingerprint,
    direct_path_avoid_region,
    episode_metric_row,
    load_episode_constraints,
    load_guided_fixture_manifest,
    min_constraint_clearance,
    paired_method_comparisons,
    progress_series,
    save_episode_constraints,
    scene_context_for_constraints,
    select_artifact_episode_indices,
    should_emit_episode_artifact,
    summarize_metrics,
    validate_paired_episode_rows,
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
from pg3d.world_model.panda_collision import DifferentiablePandaCollisionPoints
from pg3d.world_model.panda_fk import panda_end_effector_position
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
    _rerun35_exporter_python,
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

EvalMethod = Literal[
    "base", "beam", "rejection", "reranking", "itps", "itps_reranking", "itps_beam"
]
_HISTORICAL_SCORE_CONFIG = GuidedScoreConfig(mode="avoidance_only")
GeometryMode = Literal["fast", "exact"]
Entry = dict[str, np.ndarray | bool | float]
_CARTON_HALF_EXTENTS = (0.055, 0.08, 0.16)
_CYLINDER_DIMENSIONS = (0.055, 0.055, 0.12)
_CABINET_ENVELOPE_HALF_EXTENTS = (0.08, 0.085, 0.20)
_U_SHAPE_ENVELOPE_HALF_EXTENTS = U_SHAPE_REFERENCE_HALF_EXTENTS


@dataclass(frozen=True)
class ITPSGuidanceConfig:
    guide_ratio: float = 60.0
    mcmc_steps: int = 4
    energy: AvoidanceEnergyMode = "smooth"
    barrier_temperature: float = 0.01
    robot_points: int = 1024
    robot_sample_seed: int = 0

    def to_json(self) -> dict[str, float | int | str]:
        return {
            "scheduler": "ddim",
            "eta": 0.0,
            "guide_ratio": self.guide_ratio,
            "mcmc_steps": self.mcmc_steps,
            "energy": self.energy,
            "barrier_temperature": self.barrier_temperature,
            "robot_points": self.robot_points,
            "robot_sample_seed": self.robot_sample_seed,
        }


@dataclass
class ComputeOperationCounts:
    """Measured action-selection work for one evaluation episode."""

    denoiser_forward_calls: int = 0
    denoiser_evaluations: int = 0
    differentiable_fk_calls: int = 0
    differentiable_fk_pose_evaluations: int = 0
    differentiable_robot_point_calls: int = 0
    differentiable_robot_point_evaluations: int = 0
    end_effector_position_queries: int = 0
    end_effector_position_only_queries: int = 0
    eef_geometry_queries: int = 0
    robot_point_cloud_queries: int = 0
    robot_point_cloud_renders: int = 0
    peak_gpu_memory_bytes: int | None = None
    peak_gpu_memory_delta_bytes: int | None = None
    _denoiser_hook: Any | None = field(default=None, init=False, repr=False)

    def start_denoiser_tracking(self, model: torch.nn.Module) -> None:
        """Attach a hook to the actual denoiser module used by DP3."""
        if self._denoiser_hook is not None:
            raise RuntimeError("denoiser tracking is already active")
        self._denoiser_hook = model.register_forward_hook(self._record_denoiser_forward)

    def stop_denoiser_tracking(self) -> None:
        """Remove the denoiser hook without leaking it into later episodes."""
        if self._denoiser_hook is not None:
            self._denoiser_hook.remove()
            self._denoiser_hook = None

    def record_provider_delta(
        self,
        before: dict[str, int],
        after: dict[str, int],
    ) -> None:
        """Accumulate geometry-provider work performed during action selection."""
        for key in (
            "end_effector_position_queries",
            "end_effector_position_only_queries",
            "eef_geometry_queries",
            "robot_point_cloud_queries",
            "robot_point_cloud_renders",
        ):
            delta = int(after.get(key, 0)) - int(before.get(key, 0))
            if delta < 0:
                raise ValueError(f"provider counter {key!r} decreased")
            setattr(self, key, int(getattr(self, key)) + delta)

    def record_differentiable_fk(self, trajectory: torch.Tensor) -> None:
        """Count one vectorized FK call and the poses evaluated within it."""
        if trajectory.ndim < 2:
            raise ValueError("ITPS trajectory must have batch and horizon dimensions")
        self.differentiable_fk_calls += 1
        self.differentiable_fk_pose_evaluations += int(trajectory.shape[0] * trajectory.shape[1])

    def record_differentiable_robot_points(self, points: torch.Tensor) -> None:
        """Count one vectorized collision-point transform and all generated XYZ points."""
        if points.ndim != 4 or points.shape[-1] != 3:
            raise ValueError("ITPS robot points must have shape [B, H, N, 3]")
        self.differentiable_robot_point_calls += 1
        self.differentiable_robot_point_evaluations += int(np.prod(points.shape[:-1]))

    def begin_action_selection(self, device: torch.device) -> int | None:
        """Reset PyTorch CUDA peak stats and return the current allocation."""
        if device.type != "cuda" or not torch.cuda.is_available():
            return None
        torch.cuda.synchronize(device)
        baseline = int(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
        return baseline

    def end_action_selection(
        self,
        device: torch.device,
        baseline_bytes: int | None,
    ) -> None:
        """Retain the largest absolute and incremental CUDA allocation peaks."""
        if baseline_bytes is None:
            return
        torch.cuda.synchronize(device)
        peak = int(torch.cuda.max_memory_allocated(device))
        delta = max(0, peak - int(baseline_bytes))
        self.peak_gpu_memory_bytes = max(self.peak_gpu_memory_bytes or 0, peak)
        self.peak_gpu_memory_delta_bytes = max(
            self.peak_gpu_memory_delta_bytes or 0,
            delta,
        )

    def to_metric_row(self, *, replans: int) -> dict[str, int | float | None]:
        """Return schema-stable episode and per-replan operation metrics."""
        geometry_evaluations = (
            self.eef_geometry_queries
            + self.robot_point_cloud_renders
            + self.differentiable_fk_pose_evaluations
        )
        return {
            "denoiser_forward_calls": int(self.denoiser_forward_calls),
            "denoiser_evaluations": int(self.denoiser_evaluations),
            "denoiser_forward_calls_per_replan": _per_replan(self.denoiser_forward_calls, replans),
            "denoiser_evaluations_per_replan": _per_replan(self.denoiser_evaluations, replans),
            "differentiable_fk_calls": int(self.differentiable_fk_calls),
            "differentiable_fk_pose_evaluations": int(self.differentiable_fk_pose_evaluations),
            "differentiable_robot_point_calls": int(self.differentiable_robot_point_calls),
            "differentiable_robot_point_evaluations": int(
                self.differentiable_robot_point_evaluations
            ),
            "end_effector_position_queries": int(self.end_effector_position_queries),
            "end_effector_position_only_queries": int(self.end_effector_position_only_queries),
            "eef_geometry_queries": int(self.eef_geometry_queries),
            "robot_point_cloud_queries": int(self.robot_point_cloud_queries),
            "robot_point_cloud_renders": int(self.robot_point_cloud_renders),
            "geometry_evaluations": int(geometry_evaluations),
            "geometry_evaluations_per_replan": _per_replan(geometry_evaluations, replans),
            "peak_gpu_memory_bytes": self.peak_gpu_memory_bytes,
            "peak_gpu_memory_delta_bytes": self.peak_gpu_memory_delta_bytes,
        }

    def _record_denoiser_forward(
        self,
        _module: torch.nn.Module,
        _inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        batch_size = _first_tensor_batch_size(output)
        if batch_size is None:
            raise RuntimeError("denoiser output does not contain a batched tensor")
        self.denoiser_forward_calls += 1
        self.denoiser_evaluations += batch_size


def _first_tensor_batch_size(value: Any) -> int | None:
    """Return the leading dimension of the first tensor in a nested output."""
    if isinstance(value, torch.Tensor):
        return int(value.shape[0]) if value.ndim > 0 else None
    if isinstance(value, dict):
        for item in value.values():
            batch_size = _first_tensor_batch_size(item)
            if batch_size is not None:
                return batch_size
    elif isinstance(value, (list, tuple)):
        for item in value:
            batch_size = _first_tensor_batch_size(item)
            if batch_size is not None:
                return batch_size
    return None


def _per_replan(value: int, replans: int) -> float | None:
    return float(value) / float(replans) if replans > 0 else None


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
    beam_trace: BeamSearchTrace | None = None


def _guided_seed(root_seed: int, *identity: object) -> int:
    """Derive a stable Torch seed without depending on container or method order."""
    payload = ":".join(["pg3d-itps-guided", str(root_seed), *(str(x) for x in identity)])
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little") % (2**31 - 1)


def _beam_ancestry(node_id: str) -> list[str]:
    """Return deterministic root-to-node identifiers for proposal telemetry."""
    if node_id == "root":
        return ["root"]
    parts = node_id.split("/")
    return ["/".join(parts[:index]) for index in range(1, len(parts) + 1)]


@dataclass
class BeamNode:
    """One retained prefix in the long-horizon beam frontier."""

    node_id: str
    parent_id: str | None
    depth: int
    current_entry: Entry
    obs_window: list[Entry]
    action_chunks: list[ActionChunk]
    rollouts: list[ImaginedRollout]
    candidate: CandidateDiagnostics | None = None

    @property
    def feasible(self) -> bool:
        return _beam_candidate(self).feasible

    @property
    def constraint_penalty(self) -> float:
        return _beam_candidate(self).constraint_penalty

    @property
    def full_prefix_score(self) -> float:
        return _beam_candidate(self).total_score

    def to_rollout(self) -> ImaginedRollout:
        if not self.rollouts:
            raise RuntimeError("beam node has no imagined rollouts")
        return concatenate_rollouts(
            self.rollouts,
            metadata={
                "beam_depth": int(self.depth),
                "beam_node_id": self.node_id,
                "beam_parent_id": self.parent_id,
            },
        )


@dataclass(frozen=True)
class BeamRetainedNodeTrace:
    node_id: str
    parent_id: str | None
    candidate_index: int
    feasible: bool
    score: float
    constraint_penalty: float
    eef_path: np.ndarray
    min_clearance: float | None = None
    violation_max: float = 0.0
    violation_integral: float = 0.0
    seed_metadata: dict[str, Any] = field(default_factory=dict)
    normalized_score_terms: dict[str, float] = field(default_factory=dict)
    applied_score_weights: dict[str, float] = field(default_factory=dict)
    goal_distance: float | None = None
    smoothness: float = 0.0
    fallback_key: tuple[Any, ...] | None = None
    retained: bool = True
    active_width: int = 0
    observation_hash: str | None = None
    action_hash: str | None = None
    compute_counters: dict[str, int] = field(default_factory=dict)
    executed_lineage: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "candidate_index": int(self.candidate_index),
            "feasible": bool(self.feasible),
            "score": float(self.score),
            "constraint_penalty": float(self.constraint_penalty),
            "min_clearance": self.min_clearance,
            "violation_max": float(self.violation_max),
            "violation_integral": float(self.violation_integral),
            "seed_metadata": self.seed_metadata,
            "eef_path": self.eef_path,
            "normalized_score_terms": self.normalized_score_terms,
            "applied_score_weights": self.applied_score_weights,
            "goal_distance": self.goal_distance,
            "smoothness": float(self.smoothness),
            "fallback_key": self.fallback_key,
            "retained": bool(self.retained),
            "active_width": int(self.active_width),
            "observation_hash": self.observation_hash,
            "action_hash": self.action_hash,
            "compute_counters": self.compute_counters,
            "executed_lineage": bool(self.executed_lineage),
        }


@dataclass(frozen=True)
class BeamDepthTrace:
    depth: int
    expanded: int
    feasible: int
    retained: tuple[BeamRetainedNodeTrace, ...]
    nodes: tuple[BeamRetainedNodeTrace, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "depth": int(self.depth),
            "expanded": int(self.expanded),
            "feasible": int(self.feasible),
            "retained_count": len(self.retained),
            "retained": [node.to_json() for node in self.retained],
            "nodes": [node.to_json() for node in self.nodes],
        }


@dataclass(frozen=True)
class BeamSearchTrace:
    width: int
    branch_factor: int
    depths: tuple[BeamDepthTrace, ...]
    selected_node_id: str
    selected_lineage: tuple[str, ...]
    selected_score: float
    selected_constraint_penalty: float
    selected_feasible: bool

    @property
    def expanded_total(self) -> int:
        return sum(depth.expanded for depth in self.depths)

    @property
    def retained_total(self) -> int:
        return sum(len(depth.retained) for depth in self.depths)

    def to_json(self) -> dict[str, Any]:
        return {
            "width": int(self.width),
            "branch_factor": int(self.branch_factor),
            "expanded_total": int(self.expanded_total),
            "retained_total": int(self.retained_total),
            "depths": [depth.to_json() for depth in self.depths],
            "selected": {
                "node_id": self.selected_node_id,
                "lineage": list(self.selected_lineage),
                "score": float(self.selected_score),
                "constraint_penalty": float(self.selected_constraint_penalty),
                "feasible": bool(self.selected_feasible),
            },
        }


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
    itps_config = ITPSGuidanceConfig(
        guide_ratio=float(args.itps_guide_ratio),
        mcmc_steps=int(args.itps_mcmc_steps),
        energy=args.itps_energy,
        barrier_temperature=float(args.itps_barrier_temperature),
        robot_points=int(args.itps_robot_points),
        robot_sample_seed=int(args.itps_robot_sample_seed),
    )
    checkpoint_path = resolve_checkpoint_path(args.checkpoint, args.checkpoint_dir)
    checkpoint_sha256 = _artifact_file_record(checkpoint_path)["sha256"]
    score_config = _score_config_from_args(args)
    fixture_manifest = (
        load_guided_fixture_manifest(
            args.fixture_manifest,
            repo_root=Path(__file__).resolve().parents[1],
        )
        if args.fixture_manifest is not None
        else None
    )
    if fixture_manifest is not None:
        if fixture_manifest.dataset != args.dataset.resolve():
            raise ValueError("--dataset does not match the fixture manifest")
        if fixture_manifest.checkpoint != checkpoint_path.resolve():
            raise ValueError("--checkpoint does not match the fixture manifest")
    args._guided_fixture_manifest = fixture_manifest
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
    policy.set_ddim_eta(args.ddim_eta)
    action_mode = _action_mode(str(metadata.get("action_mode", "abs_joint")))
    crop_config = crop_config_from_metadata(metadata)
    if args.embody_obstacle:
        crop_config = replace(
            crop_config,
            obstacle_point_quota=args.obstacle_point_quota,
        )
    goal_thresh = (
        float(args.goal_thresh)
        if args.goal_thresh is not None
        else float(dict(metadata.get("env_kwargs", {})).get("goal_thresh", 0.025))
    )
    dataset_episode_seeds = [
        int(episode["seed"]) for episode in metadata.get("episodes", []) if "seed" in episode
    ]
    zarr_root = zarr.open_group(str(args.dataset), mode="r") if args.source == "dataset" else None
    if fixture_manifest is not None:
        specs = [
            RolloutSpec(
                output_index=episode.output_index,
                seed=episode.simulator_seed,
                source="dataset",
                dataset_episode_index=episode.dataset_episode_index,
            )
            for episode in fixture_manifest.episodes
        ]
        for episode in fixture_manifest.episodes:
            if dataset_episode_seeds[episode.dataset_episode_index] != episode.simulator_seed:
                raise ValueError(
                    "fixture simulator seed does not match dataset metadata for "
                    f"episode {episode.output_index}"
                )
    else:
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
    _resolve_grounded_embodied_obstacle_height(
        args,
        specs=specs,
        zarr_root=zarr_root,
    )
    if args.unique_dataset_seeds:
        print(
            "unique dataset seed selection: "
            f"selected={len(specs)} available_unique={len(set(dataset_episode_seeds))} "
            f"dataset_rows={len(dataset_episode_seeds)}",
            flush=True,
        )
    artifact_seed = args.artifact_selection_seed
    artifact_output_indices = (
        list(range(int(args.target_valid_episodes)))
        if args.target_valid_episodes is not None
        else [spec.output_index for spec in specs]
    )
    video_episode_indices = (
        set(
            select_artifact_episode_indices(
                artifact_output_indices,
                selection=args.artifact_selection,
                count=args.artifact_episode_count,
                seed=artifact_seed,
                every_episodes=args.video_every_episodes,
            )
        )
        if args.video
        else set()
    )
    rerun_episode_indices = (
        set(
            select_artifact_episode_indices(
                artifact_output_indices,
                selection=args.artifact_selection,
                count=args.artifact_episode_count,
                seed=artifact_seed,
                every_episodes=args.rerun_every_episodes,
            )
        )
        if args.rerun
        else set()
    )
    missing_rerun = video_episode_indices - rerun_episode_indices
    if missing_rerun:
        raise ValueError(
            "every selected video episode must also have Rerun output; missing "
            f"episode indices {sorted(missing_rerun)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_manifest_path = args.output_dir / "artifact_manifest.json"
    git_info = git_commit_info(Path(__file__).resolve().parents[1])
    run = _init_wandb(args, metadata=metadata, checkpoint_path=checkpoint_path)
    sim_env: Any | None = None
    ghost_env: Any | None = None
    rows: list[dict[str, Any]] = []
    accepted_specs: list[RolloutSpec] = []
    placement_attempts: list[dict[str, Any]] = []
    placement_exclusions: list[dict[str, Any]] = []
    metrics_path = args.output_dir / "metrics.jsonl"
    decisions_path = args.output_dir / "decisions.jsonl"
    placement_exclusions_path = args.output_dir / "placement_exclusions.jsonl"
    timings_path = args.output_dir / "timings.jsonl"
    timing_written = 0
    run_id = str(args.output_dir.resolve())
    try:
        sim_env = gym.make(
            str(metadata["env_id"]),
            **_env_kwargs(
                metadata,
                render_mode="rgb_array" if args.video else None,
                obstacle_half_extents=_embodied_obstacle_half_extents(args),
                obstacle_family=args.obstacle_family,
                max_episode_steps=args.max_steps,
            ),
        )
        ghost_env = gym.make(str(metadata["env_id"]), **_env_kwargs(metadata, render_mode=None))
        adapter = DP3ChunkPolicyAdapter(
            policy,
            action_mode=action_mode,
            device=device,
            policy_batch_size=args.policy_batch_size,
            timer=timer,
        )
        itps_collision_model = _make_itps_collision_model(
            sim_env,
            policy=policy,
            constraint_target=args.constraint_target,
            methods=args.methods,
            config=itps_config,
            gripper_open=float(args.gripper_open),
        )
        with (
            metrics_path.open("w", encoding="utf-8") as metrics_file,
            decisions_path.open("w", encoding="utf-8") as decisions_file,
            placement_exclusions_path.open("w", encoding="utf-8") as exclusions_file,
        ):
            for pool_spec in specs:
                if (
                    args.target_valid_episodes is not None
                    and len(accepted_specs) >= args.target_valid_episodes
                ):
                    break
                zarr_context = (
                    _zarr_episode_context(zarr_root, pool_spec.dataset_episode_index)
                    if zarr_root is not None and pool_spec.dataset_episode_index is not None
                    else None
                )
                placement_policy_seed: int | None = None
                if args.target_valid_episodes is not None:
                    placement_policy_seed = _episode_policy_seed(
                        args.seed,
                        pool_spec.output_index,
                    )
                    _seed_torch(placement_policy_seed)
                constraints = _constraints_for_episode(
                    sim_env,
                    spec=pool_spec,
                    policy=policy,
                    adapter=adapter,
                    action_mode=action_mode,
                    crop_config=crop_config,
                    goal_thresh=goal_thresh,
                    args=args,
                    zarr_context=zarr_context,
                )
                spec = pool_spec
                placement_record: dict[str, Any] | None = None
                if args.target_valid_episodes is not None:
                    spec, placement_record = _clearance_safe_candidate_spec(
                        pool_spec,
                        constraints,
                        zarr_context=zarr_context,
                        minimum_clearance=float(args.robot_clearance_placement_margin),
                        accepted_output_index=len(accepted_specs),
                    )
                    placement_record["placement_policy_seed"] = placement_policy_seed
                    placement_attempts.append(placement_record)
                    if spec is None:
                        placement_exclusions.append(placement_record)
                        exclusions_file.write(
                            json.dumps(_jsonable(placement_record), sort_keys=True) + "\n"
                        )
                        exclusions_file.flush()
                        print(
                            "excluded candidate-midpath placement: "
                            f"pool={pool_spec.output_index} "
                            f"dataset={pool_spec.dataset_episode_index} seed={pool_spec.seed} "
                            f"clearance={placement_record['initial_robot_clearance']:.6f} "
                            f"required={args.robot_clearance_placement_margin:.6f}",
                            flush=True,
                        )
                        continue
                accepted_specs.append(spec)
                _validate_precomputed_initial_clearance(
                    constraints,
                    zarr_context=zarr_context,
                    minimum_clearance=args.precomputed_initial_clearance_margin,
                )
                constraint_path = (
                    args.output_dir / "constraints" / f"episode_{spec.output_index:03d}.json"
                )
                with timer.time("json_write", artifact="constraint"):
                    save_episode_constraints(constraint_path, constraints)
                constraint_id = constraint_fingerprint(constraints)
                fixture_episode = (
                    fixture_manifest.episode(spec.output_index)
                    if fixture_manifest is not None
                    else None
                )
                policy_seed = (
                    _lineage_policy_seed(fixture_episode.policy_seed, args.seed)
                    if fixture_episode is not None
                    else _episode_policy_seed(args.seed, spec.output_index)
                )
                source_pool_index = (
                    fixture_episode.source_output_index
                    if fixture_episode is not None
                    else int(pool_spec.output_index)
                )
                write_video = args.video and spec.output_index in video_episode_indices
                write_rerun = args.rerun and spec.output_index in rerun_episode_indices
                for method in args.methods:
                    _seed_torch(policy_seed)
                    method_rng = np.random.default_rng(policy_seed)
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
                        beam_width=args.beam_width,
                        beam_branch_factor=args.beam_branch_factor,
                        guided_candidates=args.guided_candidates,
                        gripper_open=args.gripper_open,
                        match_current_robot_points=args.match_current_robot_points,
                        video=write_video,
                        rerun=write_rerun,
                        video_fps=args.video_fps,
                        decisions_file=decisions_file,
                        rng=method_rng,
                        timer=timer,
                        video_env_factory=_video_env_factory(
                            gym,
                            metadata=metadata,
                            enabled=(
                                write_video
                                and args.constraint_overlay_video
                                and not args.embody_obstacle
                            ),
                        ),
                        constraint_overlay_alpha=args.constraint_overlay_alpha,
                        constraint_overlay_color=tuple(args.constraint_overlay_color),
                        robot_clearance_metric=args.robot_clearance_metric,
                        robot_clearance_stride=args.robot_clearance_stride,
                        zarr_context=zarr_context,
                        embody_obstacle=args.embody_obstacle,
                        obstacle_family=args.obstacle_family,
                        terminate_on_obstacle_contact=args.terminate_on_obstacle_contact,
                        geometric_contact_threshold=args.geometric_contact_threshold,
                        directional_sign=_steer_sign(args.steer),
                        directional_weight=args.steer_weight,
                        itps_config=itps_config,
                        itps_collision_model=itps_collision_model,
                        constraint_target=args.constraint_target,
                        score_config=score_config,
                        artifact_identity={
                            "run_id": run_id,
                            "git_commit": git_info["commit"],
                            "git_revision": git_info["commit"],
                            "checkpoint_id": str(checkpoint_path),
                            "checkpoint_sha256": checkpoint_sha256,
                            "checkpoint_model": args.checkpoint_model,
                            "dataset": str(args.dataset),
                            "source": spec.source,
                            "source_pool_index": source_pool_index,
                            "dataset_episode_index": spec.dataset_episode_index,
                            "simulator_seed": int(spec.seed),
                            "policy_seed": policy_seed,
                            "constraint_id": constraint_id,
                        },
                    )
                    row.update(
                        {
                            "run_id": run_id,
                            "checkpoint_id": str(checkpoint_path),
                            "source": spec.source,
                            "source_pool_index": source_pool_index,
                            "dataset_episode_index": spec.dataset_episode_index,
                            "simulator_seed": int(spec.seed),
                            "policy_seed": policy_seed,
                            "constraint_id": constraint_id,
                            "initial_robot_clearance": (
                                placement_record["initial_robot_clearance"]
                                if placement_record is not None
                                else None
                            ),
                            "placement_policy_seed": placement_policy_seed,
                            "constraint_path": str(constraint_path),
                            "git_commit": git_info["commit"],
                            "method_config": _method_config(
                                args,
                                method=method,
                                itps_config=itps_config,
                                score_config=score_config,
                            ),
                            "artifact_manifest_path": (
                                str(artifact_manifest_path)
                                if row.get("video") is not None or row.get("rerun") is not None
                                else None
                            ),
                        }
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

    target_valid_shortfall = (
        args.target_valid_episodes is not None and len(accepted_specs) < args.target_valid_episodes
    )
    if args.target_valid_episodes is not None:
        episode_indices_path = args.output_dir / "episode_indices.txt"
        episode_indices_path.write_text(
            "".join(
                f"{spec.dataset_episode_index}\n"
                for spec in accepted_specs
                if spec.dataset_episode_index is not None
            ),
            encoding="utf-8",
        )
    validate_paired_episode_rows(rows, methods=list(args.methods))
    artifact_manifest = _write_artifact_manifest(
        artifact_manifest_path,
        rows=rows,
        run_id=run_id,
        checkpoint_path=checkpoint_path,
        dataset_path=args.dataset,
        git_info=git_info,
    )
    summary = {
        "run_id": run_id,
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
        "beam": {
            "width": int(args.beam_width),
            "branch_factor": int(args.beam_branch_factor),
            "expansion_formula": "B + (D - 1) * W * B",
        },
        "ddim_eta": float(args.ddim_eta),
        "itps": itps_config.to_json(),
        "score": score_config.to_json(),
        "fixture_manifest": (fixture_manifest.to_json() if fixture_manifest is not None else None),
        "constraint_source": _constraint_source_summary(args),
        "placement_selection": {
            "enabled": args.target_valid_episodes is not None,
            "target_valid_episodes": args.target_valid_episodes,
            "candidate_pool_size": len(specs),
            "attempted_placements": len(placement_attempts),
            "accepted_placements": len(accepted_specs),
            "minimum_initial_robot_clearance": (
                float(args.robot_clearance_placement_margin)
                if args.target_valid_episodes is not None
                else None
            ),
            "pool_exhausted": bool(target_valid_shortfall),
            "attempts": placement_attempts,
            "exclusions": placement_exclusions,
            "exclusions_path": str(placement_exclusions_path),
        },
        "artifact_selection": _artifact_selection_summary(
            accepted_specs,
            video_episode_indices=video_episode_indices,
            rerun_episode_indices=rerun_episode_indices,
            args=args,
        ),
        "artifact_manifest": str(artifact_manifest_path),
        "artifact_manifest_entries": len(artifact_manifest["artifacts"]),
        "constraint_overlay_video": bool(args.constraint_overlay_video),
        "constraint_overlay_alpha": float(args.constraint_overlay_alpha),
        "constraint_overlay_color": list(args.constraint_overlay_color),
        "contact_termination": {
            "enabled": bool(args.terminate_on_obstacle_contact),
            "sources": ["physx", "whole_robot_signed_clearance"],
            "geometric_contact_threshold_m": float(args.geometric_contact_threshold),
            "keeps_first_contact_frame": True,
        },
        "timing": timer.summary(),
        "episodes": rows,
        "by_method": summarize_metrics(rows),
        "paired_comparisons": paired_method_comparisons(
            rows,
            methods=list(args.methods),
            bootstrap_samples=args.paired_bootstrap_samples,
            bootstrap_seed=args.paired_bootstrap_seed,
        ),
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

    if target_valid_shortfall:
        print(
            "Failed constrained reach eval: candidate pool exhausted after "
            f"{len(placement_attempts)} attempts with {len(accepted_specs)}/"
            f"{args.target_valid_episodes} valid placements",
            file=sys.stderr,
        )
        return 1
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
    parser.add_argument(
        "--fixture-manifest",
        type=Path,
        default=None,
        help="Versioned guided-search fixture with exact episode, seed, and constraint mapping.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--source", choices=["dataset", "fresh"], default="fresh")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument(
        "--target-valid-episodes",
        type=int,
        default=None,
        help=(
            "For clearance-safe candidate-midpath pilots, scan the selected episode pool "
            "in order until this many finalized grounded placements satisfy "
            "--robot-clearance-placement-margin. Unsafe placements are excluded before "
            "constraints or methods are written and accepted outputs are remapped from zero."
        ),
    )
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
        choices=["base", "beam", "rejection", "reranking", "itps", "itps_reranking", "itps_beam"],
        default=["base", "rejection", "reranking"],
    )
    parser.add_argument("--itps-guide-ratio", type=float, default=60.0)
    parser.add_argument("--itps-mcmc-steps", type=int, default=4)
    parser.add_argument(
        "--itps-energy",
        choices=["smooth", "hinge"],
        default="smooth",
    )
    parser.add_argument("--itps-barrier-temperature", type=float, default=0.01)
    parser.add_argument(
        "--itps-robot-points",
        type=int,
        default=1024,
        help="Collision-surface point count per pose for whole-body ITPS guidance.",
    )
    parser.add_argument(
        "--itps-robot-sample-seed",
        type=int,
        default=0,
        help="Deterministic Panda collision-surface sampling seed.",
    )
    parser.add_argument(
        "--guided-candidates",
        type=int,
        default=10,
        help="Independent ITPS proposals per itps_reranking decision (default: 10).",
    )
    parser.add_argument(
        "--score-mode",
        choices=["avoidance_only", "fixed_task", "mass_mean", "mass_lcb", "adaptive_mass"],
        default="avoidance_only",
        help="Feasible-prefix score; hard feasibility always remains outside this score.",
    )
    parser.add_argument(
        "--score-config",
        type=Path,
        default=None,
        help="JSON containing physical score normalizers and default convex weights.",
    )
    parser.add_argument(
        "--score-weights",
        type=float,
        nargs=4,
        default=None,
        metavar=("GOAL", "CLEARANCE", "SMOOTHNESS", "MASS"),
    )
    parser.add_argument("--verification-buffer", type=float, default=0.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=150,
        help=(
            "Maximum task steps before timeout (default: 150). A successful rollout "
            "continues only for the configured post-success hold."
        ),
    )
    parser.add_argument("--post-success-steps", type=int, default=16)
    parser.add_argument("--planning-horizon-chunks", type=int, default=1)
    parser.add_argument("--execution-horizon-chunks", type=int, default=1)
    parser.add_argument("--geometry-mode", choices=["fast", "exact"], default="fast")
    parser.add_argument("--k-schedule", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument(
        "--beam-width",
        type=int,
        default=8,
        help="Maximum retained beam prefixes after each planning depth (default: 8).",
    )
    parser.add_argument(
        "--beam-branch-factor",
        type=int,
        default=32,
        help="DP3 continuations sampled per retained beam prefix (default: 32).",
    )
    parser.add_argument("--policy-batch-size", type=int, default=64)
    parser.add_argument(
        "--ddim-eta",
        type=float,
        default=0.0,
        help=(
            "DDIM reverse-process stochasticity for base/rejection/reranking samples. "
            "Use 1.0 for the U-scene diversity run. ITPS keeps its isolated eta=0 path."
        ),
    )
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
    parser.add_argument(
        "--avoid-margin",
        type=float,
        default=DEFAULT_AVOID_MARGIN_M,
        help=(
            "Required signed clearance used by avoidance cost and hard satisfaction "
            f"(default: {DEFAULT_AVOID_MARGIN_M:.2f} m). This also overrides the margin "
            "in precomputed avoid constraints."
        ),
    )
    parser.add_argument("--avoid-weight", type=float, default=1.0)
    parser.add_argument(
        "--avoid-clearance-scale",
        type=float,
        default=0.05,
        help=(
            "Soft-clearance decay scale in meters. Feasible candidates retain a positive "
            "avoidance cost that decreases with minimum obstacle clearance, allowing "
            "reranking to prefer more clearance. Set to 0 for the historical hinge-only cost."
        ),
    )
    parser.add_argument(
        "--avoid-shape",
        choices=["sphere", "box", "cuboid", "cylinder"],
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
        "--steer",
        choices=["none", "left", "right"],
        default="none",
        help=(
            "Bias candidate selection toward paths that bow to one side of the "
            "start-to-goal sightline (TCP-only, ground-plane XY). 'left'/'right' are "
            "relative to facing from the path start toward the goal at each replan "
            "step, not fixed world axes. Adds a soft term to total_score (weight "
            "--steer-weight) that only breaks ties among already-feasible candidates; "
            "it never overrides constraint satisfaction. Default 'none' disables the "
            "term entirely -- no directional cost is computed when evaluating plain "
            "avoid-region/projection constraints."
        ),
    )
    parser.add_argument(
        "--steer-weight",
        type=float,
        default=1.0,
        help="Weight for --steer's directional cost term (meters, same scale as "
        "goal_distance). Ignored when --steer none.",
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
        default=1,
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
    parser.add_argument(
        "--precomputed-initial-clearance-margin",
        type=float,
        default=None,
        help=(
            "Fail before method execution when a precomputed constraint is closer "
            "than this margin to the stored initial robot point cloud."
        ),
    )
    parser.add_argument(
        "--no-constraints",
        action="store_true",
        help=(
            "Disable constraint generation/loading for the E1 nominal checkpoint gate. "
            "All methods receive an empty constraint program."
        ),
    )
    parser.add_argument(
        "--embody-obstacle",
        action="store_true",
        help=(
            "Create the single box avoid region as a collidable actor in the control "
            "environment so ordinary camera point clouds, rather than synthetic point "
            "injection, expose it to the policy."
        ),
    )
    parser.add_argument(
        "--obstacle-point-quota",
        type=int,
        default=32,
        help=(
            "Minimum camera-visible obstacle points reserved during fixed-count "
            "downsampling in --embody-obstacle mode (default: 32)."
        ),
    )
    parser.add_argument(
        "--obstacle-yaw-deg",
        type=float,
        default=0.0,
        help=(
            "World-frame yaw of an embodied box and its synchronized BoxRegion, "
            "in degrees (default: 0)."
        ),
    )
    parser.add_argument(
        "--obstacle-family",
        choices=["box", "carton", "cylinder", "cabinet", "u_shape"],
        default="box",
        help=(
            "Named embodied-obstacle family. Carton defaults to half-extents "
            f"{_CARTON_HALF_EXTENTS}; cylinder uses radius/half-length encoded as "
            f"{_CYLINDER_DIMENSIONS}; cabinet uses a composite open structure; "
            f"u_shape defaults to envelope half-extents {_U_SHAPE_ENVELOPE_HALF_EXTENTS}. "
            "Explicit --avoid-box-half-extents overrides box/carton/u_shape only."
        ),
    )
    parser.add_argument(
        "--ground-embodied-obstacle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep embodied box/carton/cylinder/cabinet/u_shape actors supported by a plane.",
    )
    parser.add_argument(
        "--obstacle-support-plane-z",
        type=float,
        default=0.0,
        help="World Z of the support plane used for grounded embodied obstacles.",
    )
    parser.add_argument(
        "--obstacle-path-height-margin",
        type=float,
        default=0.02,
        help="Extra obstacle height above the selected direct-path point.",
    )
    parser.add_argument(
        "--obstacle-top-z",
        type=float,
        default=None,
        help=(
            "Explicit world-Z top for a grounded obstacle. If omitted for dataset "
            "direct-path placement, height is resolved from the selected path point."
        ),
    )
    parser.add_argument(
        "--terminate-on-obstacle-contact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "End an embodied-obstacle episode on the first PhysX contact or non-positive "
            "whole-robot geometric clearance (default: enabled)."
        ),
    )
    parser.add_argument(
        "--geometric-contact-threshold",
        type=float,
        default=None,
        help=(
            "Signed whole-robot clearance at or below which obstacle contact terminates "
            "the episode. Defaults to --avoid-margin so planning and execution use the "
            "same safety boundary."
        ),
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
    parser.add_argument("--paired-bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--paired-bootstrap-seed", type=int, default=0)
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
    if args.target_valid_episodes is not None:
        if args.target_valid_episodes <= 0:
            raise ValueError("--target-valid-episodes must be positive")
        if args.artifact_selection != "all":
            raise ValueError("--target-valid-episodes requires --artifact-selection all")
        if args.source != "dataset":
            raise ValueError("--target-valid-episodes requires --source dataset")
        if args.constraint_placement != "candidate_midpath":
            raise ValueError(
                "--target-valid-episodes requires --constraint-placement candidate_midpath"
            )
        if args.constraints_dir is not None or args.no_constraints:
            raise ValueError("--target-valid-episodes requires generated constraints")
        if not args.embody_obstacle or not args.ground_embodied_obstacle:
            raise ValueError("--target-valid-episodes requires a grounded embodied obstacle")
        if not args.robot_clearance_placement:
            raise ValueError("--target-valid-episodes requires --robot-clearance-placement")
    if args.paired_bootstrap_samples <= 0:
        raise ValueError("--paired-bootstrap-samples must be positive")
    if args.no_constraints and args.constraints_dir is not None:
        raise ValueError("--no-constraints and --constraints-dir are mutually exclusive")
    if args.fixture_manifest is not None:
        if args.source != "dataset":
            raise ValueError("--fixture-manifest requires --source dataset")
        if (
            any(
                value is not None
                for value in (args.episode_indices, args.episode_indices_file, args.constraints_dir)
            )
            or args.unique_dataset_seeds
            or args.no_constraints
        ):
            raise ValueError(
                "--fixture-manifest is mutually exclusive with episode selectors, "
                "--constraints-dir, --unique-dataset-seeds, and --no-constraints"
            )
    if not np.isfinite(args.verification_buffer) or args.verification_buffer < 0.0:
        raise ValueError("--verification-buffer must be finite and non-negative")
    if args.score_weights is not None:
        ConvexScoreWeights(*map(float, args.score_weights))
    if args.embody_obstacle and args.obstacle_family == "cylinder":
        args.avoid_shape = "cylinder"
        if args.avoid_box_half_extents is None:
            args.avoid_box_half_extents = list(_CYLINDER_DIMENSIONS)
    if args.embody_obstacle and args.obstacle_family == "cabinet":
        args.avoid_shape = "box"
        args.avoid_box_half_extents = list(_CABINET_ENVELOPE_HALF_EXTENTS)
    if args.embody_obstacle and args.obstacle_family == "u_shape":
        args.avoid_shape = "box"
        if args.avoid_box_half_extents is None:
            args.avoid_box_half_extents = list(_U_SHAPE_ENVELOPE_HALF_EXTENTS)
    if args.embody_obstacle and args.obstacle_family == "carton":
        args.avoid_shape = "box"
        if args.avoid_box_half_extents is None:
            args.avoid_box_half_extents = list(_CARTON_HALF_EXTENTS)
    if args.embody_obstacle and args.avoid_shape not in ("box", "cuboid", "cylinder"):
        raise ValueError("--embody-obstacle currently requires a box or cylinder avoid shape")
    if (
        args.embody_obstacle
        and args.obstacle_family != "cylinder"
        and args.avoid_shape == "cylinder"
    ):
        raise ValueError("--avoid-shape cylinder requires --obstacle-family cylinder")
    if args.embody_obstacle and len(args.avoid_path_fractions) != 1:
        raise ValueError("--embody-obstacle currently supports exactly one avoid region")
    if args.video and not args.rerun:
        raise ValueError("--video requires --rerun so every MP4 has a point-cloud timeline")
    if args.obstacle_point_quota < 0:
        raise ValueError("--obstacle-point-quota must be non-negative")
    if not np.isfinite(args.obstacle_yaw_deg):
        raise ValueError("--obstacle-yaw-deg must be finite")
    if not np.isfinite(args.obstacle_support_plane_z):
        raise ValueError("--obstacle-support-plane-z must be finite")
    if args.obstacle_path_height_margin < 0.0:
        raise ValueError("--obstacle-path-height-margin must be non-negative")
    if args.geometric_contact_threshold is None:
        args.geometric_contact_threshold = float(args.avoid_margin)
    if not np.isfinite(args.geometric_contact_threshold):
        raise ValueError("--geometric-contact-threshold must be finite")
    if args.obstacle_top_z is not None and (
        not np.isfinite(args.obstacle_top_z) or args.obstacle_top_z <= args.obstacle_support_plane_z
    ):
        raise ValueError("--obstacle-top-z must be finite and above the support plane")
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
    if args.beam_width <= 0:
        raise ValueError("--beam-width must be positive")
    if args.beam_branch_factor <= 0:
        raise ValueError("--beam-branch-factor must be positive")
    if args.policy_batch_size <= 0:
        raise ValueError("--policy-batch-size must be positive")
    if not np.isfinite(args.ddim_eta) or not 0.0 <= args.ddim_eta <= 1.0:
        raise ValueError("--ddim-eta must be finite and in [0, 1]")
    if args.itps_guide_ratio < 0.0:
        raise ValueError("--itps-guide-ratio must be non-negative")
    if args.itps_mcmc_steps <= 0:
        raise ValueError("--itps-mcmc-steps must be positive")
    if args.guided_candidates <= 0:
        raise ValueError("--guided-candidates must be positive")
    if args.itps_barrier_temperature <= 0.0:
        raise ValueError("--itps-barrier-temperature must be positive")
    if args.itps_robot_points < 320:
        raise ValueError("--itps-robot-points must be at least 320")
    if args.itps_robot_sample_seed < 0:
        raise ValueError("--itps-robot-sample-seed must be non-negative")
    if args.avoid_radius <= 0.0 or args.avoid_min_radius <= 0.0:
        raise ValueError("avoid radii must be positive")
    if not np.isfinite(args.avoid_margin) or args.avoid_margin < 0.0:
        raise ValueError("--avoid-margin must be finite and non-negative")
    if not np.isfinite(args.avoid_clearance_scale) or args.avoid_clearance_scale < 0.0:
        raise ValueError("--avoid-clearance-scale must be finite and non-negative")
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
    if args.precomputed_initial_clearance_margin is not None and (
        not np.isfinite(args.precomputed_initial_clearance_margin)
        or args.precomputed_initial_clearance_margin < 0.0
    ):
        raise ValueError("--precomputed-initial-clearance-margin must be finite and non-negative")
    if (
        args.constraint_target == "robot"
        and any(method in {"itps", "itps_reranking", "itps_beam"} for method in args.methods)
        and (not np.isfinite(args.gripper_open) or not 0.0 <= args.gripper_open <= 0.04)
    ):
        raise ValueError("whole-body ITPS requires --gripper-open within [0, 0.04]")
    guidance_methods = {"beam", "rejection", "reranking", "itps_reranking", "itps_beam"}
    if (
        args.constraint_target == "robot"
        and args.geometry_mode == "fast"
        and any(method in guidance_methods for method in args.methods)
    ):
        raise ValueError(
            "--constraint-target robot requires --geometry-mode exact for guidance methods "
            "(beam/rejection/reranking): fast geometry imagines no robot point cloud, so "
            "whole-robot "
            "guidance would be a silent no-op. Re-run with --geometry-mode exact, or use "
            "--constraint-target eef, or restrict --methods to base."
        )
    if args.artifact_selection_seed is None:
        args.artifact_selection_seed = args.seed
    return args


def _steer_sign(steer: str) -> int:
    """Map --steer's CLI choice to directional_preference's sign convention."""
    return {"none": 0, "left": 1, "right": -1}[steer]


def _score_config_from_args(args: argparse.Namespace) -> GuidedScoreConfig:
    weights = (
        ConvexScoreWeights(*map(float, args.score_weights))
        if args.score_weights is not None
        else None
    )
    mode: GuidedScoreMode = args.score_mode
    if args.score_config is not None:
        return GuidedScoreConfig.from_path(
            args.score_config,
            mode=mode,
            weights=weights,
            verification_buffer_m=float(args.verification_buffer),
        )
    return GuidedScoreConfig(
        mode=mode,
        verification_buffer_m=float(args.verification_buffer),
        weights=weights or ConvexScoreWeights(),
    )


def _guided_dependency_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0], "torch": torch.__version__}
    for distribution in ("numpy", "diffusers", "zarr", "mani-skill"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def _lineage_policy_seed(base_policy_seed: int, lineage: int) -> int:
    """Keep lineage zero identical to the fixture and derive later lineages stably."""
    if lineage < 0:
        raise ValueError("lineage seed must be non-negative")
    if lineage == 0:
        return int(base_policy_seed)
    return _guided_seed(int(base_policy_seed), "diffusion_lineage", int(lineage))


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
    beam_width: int = 8,
    beam_branch_factor: int = 32,
    guided_candidates: int = 10,
    gripper_open: float,
    match_current_robot_points: bool,
    video: bool,
    rerun: bool,
    video_fps: int,
    decisions_file: Any,
    rng: np.random.Generator,
    timer: TimingRecorder,
    itps_config: ITPSGuidanceConfig,
    itps_collision_model: DifferentiablePandaCollisionPoints | None,
    constraint_target: Literal["eef", "robot"],
    score_config: GuidedScoreConfig = _HISTORICAL_SCORE_CONFIG,
    video_env_factory: Callable[[], Any] | None = None,
    constraint_overlay_alpha: float = 0.25,
    constraint_overlay_color: tuple[float, float, float] = (1.0, 0.25, 0.05),
    robot_clearance_metric: bool = False,
    robot_clearance_stride: int = 4,
    zarr_context: dict[str, Any] | None = None,
    directional_sign: int = 0,
    directional_weight: float = 1.0,
    embody_obstacle: bool = False,
    obstacle_family: str = "box",
    terminate_on_obstacle_contact: bool = True,
    geometric_contact_threshold: float = DEFAULT_AVOID_MARGIN_M,
    artifact_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if embody_obstacle:
        _validate_embodied_obstacle_geometry(sim_env, constraints)
    obstacle_options = _embodied_obstacle_reset_options(constraints) if embody_obstacle else None
    if zarr_context is not None:
        sim_obs, sim_info = _reset_to_zarr_episode(
            sim_env,
            rollout_seed=spec.seed,
            zarr_context=zarr_context,
            reset_options=obstacle_options,
        )
    else:
        reset_options = {"reconfigure": True}
        if obstacle_options is not None:
            reset_options.update(obstacle_options)
        sim_obs, sim_info = sim_env.reset(seed=spec.seed, options=reset_options)
    video_env: Any | None = None
    with timer.time("observation_adapt_crop", source="reset"):
        sim_entry = rollout_observation_entry(
            sim_obs,
            sim_info,
            env=sim_env,
            crop_config=crop_config,
        )
    if zarr_context is not None and not embody_obstacle:
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
    rerun_replans: list[dict[str, Any]] = []
    frames = []
    raw_action_log = (
        output_dir / "debug" / method / f"episode_{spec.output_index:03d}_actions.jsonl"
    )
    raw_action_log.parent.mkdir(parents=True, exist_ok=True)
    raw_action_log.write_text("", encoding="utf-8")
    verification_log = (
        output_dir / "verification" / method / f"episode_{spec.output_index:03d}.jsonl"
    )
    verification_log.parent.mkdir(parents=True, exist_ok=True)
    verification_log.write_text("", encoding="utf-8")
    verification_rows: list[dict[str, Any]] = []
    proposal_writer = (
        GuidedProposalTraceWriter(
            output_dir / "proposals" / method / f"episode_{spec.output_index:03d}",
            run_metadata={
                **(artifact_identity or {}),
                "dependency_versions": _guided_dependency_versions(),
                "constraints": [constraint.to_json() for constraint in constraints],
            },
        )
        if method in {"itps", "itps_reranking", "itps_beam"}
        else None
    )
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
    if (
        method != "base"
        or robot_clearance_metric
        or rerun
        or (embody_obstacle and terminate_on_obstacle_contact)
    ):
        provider = ManiSkillGhostPandaGeometryProvider(
            ghost_env,
            task_name=_env_task_name(sim_env),
            crop_bounds=crop_config.bounds,
        )
        provider.reset(seed=spec.seed, options={"reconfigure": True})
        if method != "base":
            world_model = GeometricWorldModel(provider)
    contact_provider: ManiSkillGhostPandaGeometryProvider | None = None
    online_robot_clouds: list[np.ndarray] = []
    if embody_obstacle and terminate_on_obstacle_contact and constraints:
        contact_provider = ManiSkillGhostPandaGeometryProvider(
            ghost_env,
            task_name=_env_task_name(sim_env),
            crop_bounds=crop_config.bounds,
        )
        contact_provider.reset(seed=spec.seed, options={"reconfigure": True})
        online_robot_clouds.append(
            np.asarray(
                contact_provider.robot_point_cloud(
                    np.asarray(sim_entry["agent_pos"], dtype=np.float32)
                ),
                dtype=np.float32,
            )
        )

    steps = 0
    replans = 0
    first_goal_entry_step: int | None = None
    stable_entry_step: int | None = None
    current_hold_count = 0
    maximum_hold_count = 0
    candidate_feasible = 0
    candidate_total = 0
    fallback_count = 0
    beam_expanded_nodes = 0
    beam_feasible_nodes = 0
    beam_retained_nodes = 0
    action_selection_times: list[float] = []
    executed_action_targets: list[np.ndarray] = []
    replan_start_action_indices: list[int] = []
    compute_counts = ComputeOperationCounts()
    terminated_or_truncated = False
    physical_collision = False
    physical_collision_step: int | None = None
    physical_collision_pairs: list[list[str]] = []
    geometric_collision = False
    geometric_collision_step: int | None = None
    geometric_collision_clearance: float | None = None
    was_training = policy.training
    policy.eval()
    compute_counts.start_denoiser_tracking(policy.model)
    try:
        while True:
            if first_goal_entry_step is None and steps >= max_steps:
                break
            if stable_entry_step is not None:
                break
            step_limit = _episode_step_limit(
                max_task_steps=max_steps,
                post_success_steps=post_success_steps,
                first_success_step=first_goal_entry_step,
            )
            provider_counts_before = provider.counter_snapshot() if provider is not None else {}
            memory_baseline = compute_counts.begin_action_selection(policy.device)
            try:
                with timer.time(
                    "action_selection",
                    method=method,
                    episode=spec.output_index,
                    replan=replans,
                ):
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
                        beam_width=beam_width,
                        beam_branch_factor=beam_branch_factor,
                        guided_candidates=guided_candidates,
                        match_current_robot_points=match_current_robot_points,
                        rng=rng,
                        timer=timer,
                        compute_counts=compute_counts,
                        directional_sign=directional_sign,
                        directional_weight=directional_weight,
                        itps_config=itps_config,
                        itps_collision_model=itps_collision_model,
                        constraint_target=constraint_target,
                        score_config=score_config,
                        proposal_writer=proposal_writer,
                        replan=replans,
                    )
            finally:
                compute_counts.end_action_selection(policy.device, memory_baseline)
                if provider is not None:
                    compute_counts.record_provider_delta(
                        provider_counts_before,
                        provider.counter_snapshot(),
                    )
            if timer.enabled:
                action_selection_times.append(timer.events[-1].seconds)
            replans += 1
            if decision.result is not None:
                candidate_feasible += decision.candidate_feasible
                candidate_total += decision.candidate_total
                if decision.selection_reason in {
                    "least_bad_fallback",
                    "beam_least_bad_fallback",
                }:
                    fallback_count += 1
            if decision.beam_trace is not None:
                beam_expanded_nodes += decision.beam_trace.expanded_total
                beam_feasible_nodes += sum(depth.feasible for depth in decision.beam_trace.depths)
                beam_retained_nodes += decision.beam_trace.retained_total
            _write_decision(
                decisions_file,
                method=method,
                spec=spec,
                replan_index=replans - 1,
                step=steps,
                decision=decision,
            )
            if rerun and provider is not None:
                rerun_replans.append(
                    _rerun_replan_record(
                        decision,
                        provider=provider,
                        current_entry=sim_entry,
                        constraints=constraints,
                        constraint_target=constraint_target,
                        collision_model=itps_collision_model,
                        step=steps,
                        replan_index=replans - 1,
                        timer=timer,
                    )
                )
            steps_to_execute = min(
                decision.selected_chunk.horizon,
                int(policy.n_action_steps) * execution_horizon_chunks,
                step_limit - steps,
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
            if executed_action_targets and steps_to_execute > 0:
                replan_start_action_indices.append(len(executed_action_targets))
            raw_chunk = np.asarray(decision.selected_chunk.actions, dtype=np.float32)
            selected_validation_rollout = (
                decision.result.selected.rollout
                if decision.result is not None
                else (
                    world_model.imagine(
                        entry_to_world_model_observation(sim_entry),
                        decision.selected_chunk,
                        metadata={"purpose": "imagined_execution_validation"},
                    )
                    if world_model is not None
                    else None
                )
            )
            executed_actions: list[np.ndarray] = []
            executed_agent_positions: list[np.ndarray] = []
            action_tcp_poses: list[np.ndarray] = [
                np.asarray(sim_entry["tcp_pose"], dtype=np.float32).copy()
            ]
            for policy_action in decision.selected_chunk.actions[:steps_to_execute]:
                sim_action = policy_action_to_sim_action(
                    policy_action,
                    np.asarray(sim_entry["agent_pos"], dtype=np.float32),
                    action_mode=action_mode,
                    sim_action_dim=int(np.prod(sim_env.action_space.shape)),
                    low=getattr(sim_env.action_space, "low", None),
                    high=getattr(sim_env.action_space, "high", None),
                    gripper_open=gripper_open,
                )
                executed_actions.append(np.asarray(sim_action, dtype=np.float32).copy())
                executed_action_targets.append(
                    np.asarray(sim_action, dtype=np.float32).reshape(-1)[:7].copy()
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
                action_tcp_poses.append(np.asarray(sim_entry["tcp_pose"], dtype=np.float32).copy())
                executed_agent_positions.append(
                    np.asarray(sim_entry["agent_pos"], dtype=np.float32).copy()
                )
                obs_window = append_obs_window(
                    obs_window,
                    sim_entry,
                    n_obs_steps=int(policy.n_obs_steps),
                )
                _append_path(path, sim_entry)
                timeline.append(sim_entry.copy())
                current_robot_cloud: np.ndarray | None = None
                if contact_provider is not None:
                    with timer.time("online_robot_contact_points", method=method):
                        current_robot_cloud = np.asarray(
                            contact_provider.robot_point_cloud(
                                np.asarray(sim_entry["agent_pos"], dtype=np.float32)
                            ),
                            dtype=np.float32,
                        )
                    online_robot_clouds.append(current_robot_cloud)
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
                tcp_to_goal_distance = float(
                    np.linalg.norm(
                        np.asarray(sim_entry["tcp_pose"], dtype=np.float32).reshape(-1)[:3] - target
                    )
                )
                (
                    first_goal_entry_step,
                    stable_entry_step,
                    current_hold_count,
                    maximum_hold_count,
                ) = _update_goal_hold(
                    distance=tcp_to_goal_distance,
                    goal_threshold=goal_thresh,
                    step=steps,
                    hold_steps=post_success_steps,
                    first_entry_step=first_goal_entry_step,
                    stable_entry_step=stable_entry_step,
                    current_hold_count=current_hold_count,
                    maximum_hold_count=maximum_hold_count,
                )
                contact_pairs = (
                    _robot_obstacle_contact_pairs(sim_env)
                    if embody_obstacle and terminate_on_obstacle_contact
                    else []
                )
                if contact_pairs and physical_collision_step is None:
                    physical_collision = True
                    physical_collision_step = steps
                    physical_collision_pairs = contact_pairs
                if (
                    current_robot_cloud is not None
                    and not geometric_collision
                    and len(current_robot_cloud)
                ):
                    clearance = _minimum_raw_obstacle_clearance(
                        current_robot_cloud,
                        constraints,
                    )
                    if clearance <= float(geometric_contact_threshold):
                        geometric_collision = True
                        geometric_collision_step = steps
                        geometric_collision_clearance = clearance
                obstacle_contact = physical_collision or geometric_collision
                terminated_or_truncated = obstacle_contact or _episode_should_stop(
                    terminated=terminated,
                    truncated=truncated,
                    success=success,
                )
                if terminated_or_truncated:
                    break
                if stable_entry_step is not None:
                    break
            with raw_action_log.open("a", encoding="utf-8") as action_file:
                action_file.write(
                    json.dumps(
                        _jsonable(
                            {
                                "episode": spec.output_index,
                                "seed": spec.seed,
                                "method": method,
                                "replan_index": replans - 1,
                                "step": steps,
                                "selected_chunk_shape": list(raw_chunk.shape),
                                "selected_chunk": raw_chunk,
                                "executed_steps": int(steps_to_execute),
                                "executed_actions": executed_actions,
                                "tcp_poses": action_tcp_poses,
                                "selected_chunk_metadata": decision.selected_chunk.metadata,
                            }
                        ),
                        sort_keys=True,
                    )
                    + "\n"
                )
            if selected_validation_rollout is not None and provider is not None:
                verification_row = _imagined_execution_validation(
                    rollout=selected_validation_rollout,
                    executed_agent_positions=executed_agent_positions,
                    executed_tcp_poses=action_tcp_poses[1:],
                    provider=provider,
                    constraints=constraints,
                    constraint_target=constraint_target,
                    replan=replans - 1,
                    episode=spec.output_index,
                    method=method,
                )
                verification_rows.append(verification_row)
                with verification_log.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(_jsonable(verification_row), sort_keys=True) + "\n")
            if terminated_or_truncated:
                break
    finally:
        compute_counts.stop_denoiser_tracking()
        if was_training:
            policy.train()
        if video_env is not None:
            _close_env(video_env)

    video_path = (
        output_dir / "videos" / method / f"episode_{spec.output_index:03d}.mp4" if video else None
    )
    rerun_path = (
        output_dir / "rerun" / method / f"episode_{spec.output_index:03d}.rrd" if rerun else None
    )
    control_dt = _env_control_dt(sim_env)
    robot_clearance_point_clouds: list[np.ndarray] | None = None
    if robot_clearance_metric and constraints and provider is not None:
        if online_robot_clouds:
            robot_clearance_point_clouds = _subsample_robot_clouds(
                online_robot_clouds,
                stride=robot_clearance_stride,
            )
        else:
            try:
                with timer.time("robot_clearance_points", method=method):
                    robot_clearance_point_clouds = _whole_robot_clearance_point_clouds(
                        path, provider, stride=robot_clearance_stride
                    )
            except Exception as exc:
                print(
                    "warning: whole-robot clearance metric failed, falling back to TCP-only: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                robot_clearance_point_clouds = None
    row = episode_metric_row(
        method=method,
        episode=spec.output_index,
        seed=spec.seed,
        path=path,
        constraints=constraints,
        robot_clearance_point_clouds=robot_clearance_point_clouds,
        robot_clearance_dt=control_dt * robot_clearance_stride,
        reach_success=first_goal_entry_step is not None,
        first_success_step=first_goal_entry_step,
        steps=steps,
        replans=replans,
        candidate_feasibility_fraction=candidate_feasibility_fraction(
            candidate_feasible,
            candidate_total,
        ),
        fallback_count=fallback_count,
        video=str(video_path) if video_path is not None else None,
        rerun=str(rerun_path) if rerun_path is not None else None,
        goal_threshold=goal_thresh,
        hold_steps=post_success_steps,
        control_dt=control_dt,
        action_selection_times=action_selection_times,
        executed_actions=executed_action_targets,
        replan_start_action_indices=replan_start_action_indices,
    )
    row.update(compute_counts.to_metric_row(replans=replans))
    row.update(
        _beam_episode_metric_row(
            method=method,
            replans=replans,
            expanded=beam_expanded_nodes,
            feasible=beam_feasible_nodes,
            retained=beam_retained_nodes,
        )
    )
    row.update(
        {
            "physical_collision": physical_collision,
            "physical_collision_step": physical_collision_step,
            "physical_collision_pairs": physical_collision_pairs,
            "geometric_collision": geometric_collision,
            "geometric_collision_step": geometric_collision_step,
            "geometric_collision_clearance": geometric_collision_clearance,
            "geometric_contact_threshold": float(geometric_contact_threshold),
            "terminate_on_obstacle_contact": bool(terminate_on_obstacle_contact),
            "current_hold_count": int(current_hold_count),
            "maximum_hold_count": int(maximum_hold_count),
            "first_goal_entry_step": first_goal_entry_step,
            "stable_entry_step": stable_entry_step,
            "obstacle_contact": physical_collision or geometric_collision,
            "obstacle_contact_step": (
                min(
                    step
                    for step in (physical_collision_step, geometric_collision_step)
                    if step is not None
                )
                if physical_collision_step is not None or geometric_collision_step is not None
                else None
            ),
            "obstacle_contact_source": _obstacle_contact_source(
                physical_collision=physical_collision,
                geometric_collision=geometric_collision,
            ),
            "termination_reason": _termination_reason(
                physical_collision=physical_collision,
                geometric_collision=geometric_collision,
                terminated_or_truncated=terminated_or_truncated,
                stable_entry_step=stable_entry_step,
                steps=steps,
                max_steps=max_steps,
            ),
        }
    )
    if physical_collision or geometric_collision:
        row["constraint_satisfied"] = False
        row["combined_success"] = False
        row["stable_combined_success"] = False
    if rerun_path is not None:
        row.update(
            {
                "rerun_writer_version": "0.35.0",
                "policy_pointcloud_bundle": str(rerun_path.with_suffix(".policy_input.npz")),
                "policy_pointcloud_metadata": str(rerun_path.with_suffix(".policy_input.json")),
            }
        )
    if embody_obstacle:
        goal_marker_points = int(getattr(policy, "goal_marker_points", 0))
        obstacle_reset = _embodied_obstacle_reset_options(constraints)
        raw_counts = [int(entry.get("obstacle_points_raw", 0)) for entry in timeline]
        cropped_counts = [int(entry.get("obstacle_points_cropped", 0)) for entry in timeline]
        policy_counts = [
            _policy_obstacle_point_count(entry, goal_marker_points=goal_marker_points)
            for entry in timeline
        ]
        row.update(
            {
                "obstacle_id": f"{obstacle_family}:episode_{spec.output_index:03d}",
                "obstacle_family": obstacle_family,
                "obstacle_pose": {
                    "center": obstacle_reset["pg3d_obstacle_center"],
                    "yaw": obstacle_reset["pg3d_obstacle_yaw"],
                    "bottom_z": _constraint_bottom_z(constraints),
                    "top_z": _constraint_top_z(constraints),
                },
                "obstacle_collision_geometry": [
                    constraint.region.to_json() for constraint in constraints
                ],
                "obstacle_points_raw": min(raw_counts, default=0),
                "obstacle_points_cropped": min(cropped_counts, default=0),
                "obstacle_points_policy_input": min(policy_counts, default=0),
            }
        )
    embedded_identity = _episode_artifact_identity(
        row=row,
        base_identity=artifact_identity,
    )
    if video_path is not None:
        annotated_frames = _annotate_episode_video_frames(
            frames,
            identity=embedded_identity,
        )
        with timer.time("video_write", method=method):
            save_video(video_path, annotated_frames, fps=video_fps)
        row["video_labels_embedded"] = True
    if rerun_path is not None:
        with timer.time("rerun_write", method=method):
            save_rerun_timeline(
                rerun_path,
                timeline,
                constraints=constraints,
                replans=rerun_replans,
                goal_marker_points=int(getattr(policy, "goal_marker_points", 0)),
                goal_marker_radius=float(
                    getattr(policy, "goal_marker_radius", DEFAULT_GOAL_MARKER_RADIUS)
                ),
                recording_identity=embedded_identity,
            )
        row["rerun_identity_embedded"] = True
    if video_path is not None or rerun_path is not None:
        row["embedded_artifact_identity"] = embedded_identity
    row["proposal_trace"] = (
        str(proposal_writer.records_path) if proposal_writer is not None else None
    )
    row.update(_verification_episode_metrics(verification_rows))
    row["imagined_execution_trace"] = str(verification_log)
    return row


def _episode_artifact_identity(
    *,
    row: dict[str, Any],
    base_identity: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the exact identity/outcome payload embedded in MP4 and RRD artifacts."""
    identity = dict(_jsonable(base_identity or {}))
    identity.update(
        {
            "method": str(row["method"]),
            "episode": int(row["episode"]),
            "simulator_seed": int(identity.get("simulator_seed", row.get("seed", 0))),
            "dataset_episode_index": identity.get("dataset_episode_index"),
            "obstacle_id": row.get("obstacle_id"),
            "obstacle_family": row.get("obstacle_family"),
            "reach_success": bool(row["reach_success"]),
            "stable_goal_reached": bool(row["stable_goal_reached"]),
            "constraint_satisfied": bool(row["constraint_satisfied"]),
            "combined_success": bool(row["combined_success"]),
            "stable_combined_success": bool(row["stable_combined_success"]),
            "physical_collision": bool(row.get("physical_collision", False)),
            "geometric_collision": bool(row.get("geometric_collision", False)),
            "obstacle_contact": bool(row.get("obstacle_contact", False)),
            "termination_reason": row.get("termination_reason"),
            "min_clearance_m": row.get("min_clearance"),
            "steps": int(row["steps"]),
        }
    )
    return identity


def _annotate_episode_video_frames(
    frames: list[np.ndarray],
    *,
    identity: dict[str, Any],
) -> list[np.ndarray]:
    """Burn method, paired identity, obstacle, and final outcome into every frame."""
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    method = str(identity.get("method", "unknown")).upper()
    episode = int(identity.get("episode", 0))
    simulator_seed = int(identity.get("simulator_seed", 0))
    obstacle = str(identity.get("obstacle_family") or "none")
    reach = "YES" if identity.get("reach_success") else "NO"
    stable = "YES" if identity.get("stable_combined_success") else "NO"
    safe = "YES" if identity.get("constraint_satisfied") else "NO"
    collision = "YES" if identity.get("obstacle_contact") else "NO"
    clearance = identity.get("min_clearance_m")
    clearance_text = (
        "n/a"
        if clearance is None or not np.isfinite(float(clearance))
        else f"{float(clearance):+.3f} m"
    )
    lines = [
        f"{method} | EP {episode:03d} | SIM SEED {simulator_seed}",
        f"OBSTACLE {obstacle} | GOAL {reach} | STABLE+SAFE {stable}",
        f"SAFE {safe} | CLEARANCE {clearance_text} | CONTACT {collision}",
    ]
    annotated: list[np.ndarray] = []
    for frame in frames:
        array = np.asarray(frame)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ValueError(
                "video annotation requires uint8 RGB/RGBA frames, "
                f"got dtype={array.dtype} shape={array.shape}"
            )
        image = Image.fromarray(array).convert("RGB")
        line_height = 19
        # Keep the total 512+64 height codec-aligned so FFmpeg never rescales the
        # untouched simulator image.
        panel_height = 64
        canvas = Image.new("RGB", (image.width, image.height + panel_height), (0, 0, 0))
        canvas.paste(image, (0, panel_height))
        draw = ImageDraw.Draw(canvas)
        for line_index, line in enumerate(lines):
            draw.text(
                (8, 3 + line_index * line_height),
                line,
                fill=(255, 255, 255, 255),
                font=font,
            )
        annotated.append(np.asarray(canvas))
    return annotated


def _env_control_dt(env: Any) -> float:
    """Return the simulator control timestep used for physical-time metrics."""
    unwrapped = getattr(env, "unwrapped", env)
    for owner in (unwrapped, env):
        value = getattr(owner, "control_timestep", None)
        if value is not None:
            dt = float(value)
            if np.isfinite(dt) and dt > 0.0:
                return dt
    control_freq = getattr(unwrapped, "control_freq", None)
    if control_freq is not None:
        frequency = float(control_freq)
        if np.isfinite(frequency) and frequency > 0.0:
            return 1.0 / frequency
    raise RuntimeError("ManiSkill environment does not expose a valid control timestep")


def _episode_should_stop(*, terminated: Any, truncated: Any, success: bool) -> bool:
    """Ignore success termination while collecting the required post-success hold."""
    if _bool_any(truncated):
        return True
    return bool(_bool_any(terminated) and not success)


def _minimum_raw_obstacle_clearance(
    robot_points: np.ndarray,
    constraints: list[AvoidRegion],
) -> float:
    """Return raw robot-to-obstacle distance without subtracting safety margins."""
    points = np.asarray(robot_points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("robot_points must have non-empty shape [N, 3]")
    if not constraints:
        raise ValueError("raw obstacle clearance requires at least one constraint")
    return min(
        float(np.min(constraint.region.signed_distance(points))) for constraint in constraints
    )


def _imagined_execution_validation(
    *,
    rollout: ImaginedRollout,
    executed_agent_positions: list[np.ndarray],
    executed_tcp_poses: list[np.ndarray],
    provider: ManiSkillGhostPandaGeometryProvider,
    constraints: list[AvoidRegion],
    constraint_target: Literal["eef", "robot"],
    replan: int,
    episode: int,
    method: str,
) -> dict[str, Any]:
    count = min(8, len(executed_agent_positions), rollout.q.shape[0], rollout.eef_path.shape[0])
    predicted_q = np.asarray(rollout.q[:count, :7], dtype=np.float32)
    executed_q = (
        np.stack(
            [
                np.asarray(value, dtype=np.float32).reshape(-1)[:7]
                for value in executed_agent_positions[:count]
            ],
            axis=0,
        )
        if count
        else np.empty((0, 7), dtype=np.float32)
    )
    predicted_tcp = np.asarray(rollout.eef_path[:count], dtype=np.float32)
    executed_tcp = (
        np.stack(
            [
                np.asarray(value, dtype=np.float32).reshape(-1)[:3]
                for value in executed_tcp_poses[:count]
            ],
            axis=0,
        )
        if count
        else np.empty((0, 3), dtype=np.float32)
    )
    joint_errors = np.linalg.norm(predicted_q - executed_q, axis=1) if count else np.asarray([])
    tcp_errors = np.linalg.norm(predicted_tcp - executed_tcp, axis=1) if count else np.asarray([])

    predicted_clearance: list[float] = []
    executed_clearance: list[float] = []
    if constraints:
        for step_index in range(count):
            predicted_points = (
                np.asarray(rollout.eef_path[step_index], dtype=np.float32).reshape(1, 3)
                if constraint_target == "eef"
                else np.asarray(rollout.robot_point_clouds[step_index], dtype=np.float32)
            )
            executed_points = (
                executed_tcp[step_index].reshape(1, 3)
                if constraint_target == "eef"
                else np.asarray(
                    provider.robot_point_cloud(executed_q[step_index]), dtype=np.float32
                )
            )
            predicted_clearance.append(
                _minimum_raw_obstacle_clearance(predicted_points, constraints)
            )
            executed_clearance.append(_minimum_raw_obstacle_clearance(executed_points, constraints))
    predicted_clearance_array = np.asarray(predicted_clearance, dtype=np.float64)
    executed_clearance_array = np.asarray(executed_clearance, dtype=np.float64)
    clearance_errors = predicted_clearance_array - executed_clearance_array
    optimistic_errors = np.maximum(clearance_errors, 0.0)
    predicted_feasible = predicted_clearance_array >= DEFAULT_AVOID_MARGIN_M
    executed_feasible = executed_clearance_array >= DEFAULT_AVOID_MARGIN_M
    return {
        "schema_version": "pg3d.imagined_execution_validation.v1",
        "method": method,
        "episode": int(episode),
        "replan": int(replan),
        "compared_actions": int(count),
        "joint_error_rad": joint_errors.tolist(),
        "tcp_error_m": tcp_errors.tolist(),
        "predicted_clearance_m": predicted_clearance_array.tolist(),
        "executed_clearance_m": executed_clearance_array.tolist(),
        "clearance_error_m": clearance_errors.tolist(),
        "optimistic_clearance_error_m": optimistic_errors.tolist(),
        "predicted_feasible": predicted_feasible.tolist(),
        "executed_feasible": executed_feasible.tolist(),
        "feasibility_confusion": {
            "true_safe": int(np.sum(predicted_feasible & executed_feasible)),
            "false_safe": int(np.sum(predicted_feasible & ~executed_feasible)),
            "false_unsafe": int(np.sum(~predicted_feasible & executed_feasible)),
            "true_unsafe": int(np.sum(~predicted_feasible & ~executed_feasible)),
        },
    }


def _verification_episode_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    optimistic = [
        float(value) for row in rows for value in row.get("optimistic_clearance_error_m", [])
    ]
    joint = [float(value) for row in rows for value in row.get("joint_error_rad", [])]
    tcp = [float(value) for row in rows for value in row.get("tcp_error_m", [])]
    return {
        "imagined_execution_compared_actions": sum(int(row["compared_actions"]) for row in rows),
        "imagined_joint_error_mean_rad": float(np.mean(joint)) if joint else None,
        "imagined_joint_error_max_rad": float(np.max(joint)) if joint else None,
        "imagined_tcp_error_mean_m": float(np.mean(tcp)) if tcp else None,
        "imagined_tcp_error_max_m": float(np.max(tcp)) if tcp else None,
        "optimistic_clearance_error_p95_m": (
            float(np.quantile(optimistic, 0.95)) if optimistic else None
        ),
    }


def _contact_body_name(body: Any) -> str:
    name = getattr(body, "name", None)
    if name is None and hasattr(body, "get_name"):
        name = body.get_name()
    return str(name or "")


def _robot_obstacle_contact_pairs(env: Any) -> list[list[str]]:
    """Return PhysX contact pairs between robot links and embodied obstacles."""
    unwrapped = getattr(env, "unwrapped", env)
    scene = getattr(unwrapped, "scene", None)
    agent = getattr(unwrapped, "agent", None)
    robot = getattr(agent, "robot", None)
    links = getattr(robot, "links", None)
    if links is None and robot is not None and hasattr(robot, "get_links"):
        links = robot.get_links()
    robot_names = {_contact_body_name(link) for link in (links or [])}
    robot_names.discard("")
    if scene is None or not hasattr(scene, "get_contacts") or not robot_names:
        return []
    pairs: list[list[str]] = []
    for contact in scene.get_contacts():
        names = {_contact_body_name(body) for body in contact.bodies}
        names.discard("")
        obstacle_names = {name for name in names if "pg3d_obstacle" in name}
        contacted_robot = names & robot_names
        for robot_name in sorted(contacted_robot):
            for obstacle_name in sorted(obstacle_names):
                pairs.append([robot_name, obstacle_name])
    return pairs


def _termination_reason(
    *,
    physical_collision: bool,
    terminated_or_truncated: bool,
    stable_entry_step: int | None,
    steps: int,
    max_steps: int,
    geometric_collision: bool = False,
) -> str:
    if physical_collision:
        return "physical_obstacle_collision"
    if geometric_collision:
        return "geometric_obstacle_collision"
    if stable_entry_step is not None:
        return "stable_success"
    if terminated_or_truncated:
        return "simulator_termination"
    if steps >= max_steps:
        return "task_horizon"
    return "stopped"


def _obstacle_contact_source(
    *,
    physical_collision: bool,
    geometric_collision: bool,
) -> str | None:
    if physical_collision and geometric_collision:
        return "physx+geometry"
    if physical_collision:
        return "physx"
    if geometric_collision:
        return "geometry"
    return None


def _episode_step_limit(
    *,
    max_task_steps: int,
    post_success_steps: int,
    first_success_step: int | None,
) -> int:
    """Keep the nominal task horizon fixed while allowing a complete hold window."""
    if max_task_steps <= 0 or post_success_steps < 0:
        raise ValueError("invalid task or post-success step limit")
    return max_task_steps if first_success_step is None else max_task_steps + post_success_steps


def _update_goal_hold(
    *,
    distance: float,
    goal_threshold: float,
    step: int,
    hold_steps: int,
    first_entry_step: int | None,
    stable_entry_step: int | None,
    current_hold_count: int,
    maximum_hold_count: int,
) -> tuple[int | None, int | None, int, int]:
    """Update consecutive distance-threshold hold telemetry for one executed state."""
    if not np.isfinite(distance) or not np.isfinite(goal_threshold) or goal_threshold < 0.0:
        raise ValueError("distance and goal_threshold must be finite and threshold non-negative")
    if step < 0 or hold_steps < 0 or current_hold_count < 0 or maximum_hold_count < 0:
        raise ValueError("hold steps and counts must be non-negative")
    if distance > goal_threshold:
        return first_entry_step, stable_entry_step, 0, maximum_hold_count
    if first_entry_step is None:
        first_entry_step = step
    current_hold_count += 1
    maximum_hold_count = max(maximum_hold_count, current_hold_count)
    if stable_entry_step is None and current_hold_count >= hold_steps + 1:
        stable_entry_step = step - hold_steps
    return first_entry_step, stable_entry_step, current_hold_count, maximum_hold_count


def _beam_episode_metric_row(
    *, method: EvalMethod, replans: int, expanded: int, feasible: int, retained: int
) -> dict[str, int | float | None]:
    """Expose episode beam work for both unguided and ITPS-guided beam methods."""
    if method not in {"beam", "itps_beam"}:
        return {}
    return {
        "beam_expanded_nodes": int(expanded),
        "beam_feasible_nodes": int(feasible),
        "beam_retained_nodes": int(retained),
        "beam_expanded_nodes_per_replan": _per_replan(expanded, replans),
        "beam_feasible_nodes_per_replan": _per_replan(feasible, replans),
        "beam_retained_nodes_per_replan": _per_replan(retained, replans),
        "beam_feasible_fraction": (float(feasible) / float(expanded) if expanded > 0 else None),
    }


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
    beam_width: int = 8,
    beam_branch_factor: int = 32,
    guided_candidates: int = 10,
    match_current_robot_points: bool,
    rng: np.random.Generator,
    timer: TimingRecorder,
    compute_counts: ComputeOperationCounts,
    itps_config: ITPSGuidanceConfig,
    itps_collision_model: DifferentiablePandaCollisionPoints | None,
    constraint_target: Literal["eef", "robot"],
    score_config: GuidedScoreConfig = _HISTORICAL_SCORE_CONFIG,
    directional_sign: int = 0,
    directional_weight: float = 1.0,
    proposal_writer: GuidedProposalTraceWriter | None = None,
    replan: int = 0,
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
    if method == "itps":
        if provider is None:
            raise RuntimeError("ITPS guidance requires a live Panda geometry provider")
        return EvalDecisionSummary(
            selected_chunk=_select_itps_chunk(
                policy=adapter.policy,
                provider=provider,
                obs_window=obs_window,
                constraints=constraints,
                rng=rng,
                config=itps_config,
                compute_counts=compute_counts,
                collision_model=itps_collision_model,
                constraint_target=constraint_target,
                proposal_writer=proposal_writer,
                proposal_context={
                    "proposal_id": f"r{replan}/itps",
                    "purpose": "standalone_itps",
                    "replan": replan,
                    "depth": 1,
                    "parent_id": "root",
                    "ancestry": ["root"],
                    "branch_index": 0,
                },
            ),
            result=None,
            candidate_feasible=0,
            candidate_total=0,
            selection_reason="itps",
        )
    if method in {"beam", "itps_beam"}:
        if world_model is None or provider is None:
            raise RuntimeError("beam search requires a world model and ghost provider")
        if match_current_robot_points:
            provider.set_robot_point_budget_from_mask(
                np.asarray(current_entry["robot_mask"], dtype=bool),
                point_valid_mask=np.asarray(current_entry["point_valid_mask"], dtype=bool),
            )
        with timer.time("candidate_scoring", method=method, geometry_mode=geometry_mode):
            result, beam_trace = _select_beam_search(
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
                beam_width=beam_width,
                branch_factor=beam_branch_factor,
                rng=rng,
                timer=timer,
                directional_sign=directional_sign,
                directional_weight=directional_weight,
                guided=method == "itps_beam",
                policy=adapter.policy,
                itps_config=itps_config,
                compute_counts=compute_counts,
                collision_model=itps_collision_model,
                constraint_target=constraint_target,
                guided_root_seed=(int(rng.integers(0, 2**31 - 1)) if method == "itps_beam" else 0),
                score_config=score_config,
                proposal_writer=proposal_writer,
                replan=replan,
            )
        feasible = sum(1 for candidate in result.candidates if candidate.feasible)
        return EvalDecisionSummary(
            selected_chunk=result.action_chunk,
            result=result,
            candidate_feasible=feasible,
            candidate_total=len(result.candidates),
            selection_reason=result.selection_reason,
            beam_trace=beam_trace,
        )
    if method == "itps_reranking":
        if world_model is None or provider is None:
            raise RuntimeError("ITPS reranking requires an exact world model and ghost provider")
        root_seed = int(rng.integers(0, 2**31 - 1))
        chunks = sample_itps_candidates(
            policy=adapter.policy,
            provider=provider,
            observation_windows=[obs_window] * guided_candidates,
            constraints=constraints,
            seeds=[_guided_seed(root_seed, 1, "root", i) for i in range(guided_candidates)],
            config=itps_config,
            compute_counts=compute_counts,
            collision_model=itps_collision_model,
            constraint_target=constraint_target,
            proposal_writer=proposal_writer,
            proposal_contexts=[
                {
                    "proposal_id": f"r{replan}/h1/b{index}",
                    "purpose": "search_expansion",
                    "replan": replan,
                    "depth": 1,
                    "parent_id": "root",
                    "ancestry": ["root"],
                    "branch_index": index,
                }
                for index in range(guided_candidates)
            ],
        )
        consensus = consensus_deviations(chunks)
        candidates = []
        for index, (chunk, deviation) in enumerate(zip(chunks, consensus, strict=True)):
            rollout = world_model.imagine(entry_to_world_model_observation(current_entry), chunk)
            candidates.append(
                _candidate_diagnostics(
                    index=index,
                    attempted_k=guided_candidates,
                    action_chunk=chunk,
                    rollout=rollout,
                    scene=scene,
                    constraints=constraints,
                    consensus_deviation=deviation,
                    directional_sign=directional_sign,
                    directional_weight=directional_weight,
                    score_config=score_config,
                )
            )
        result = _select_guided_candidates(candidates, attempted_k=guided_candidates)
        if proposal_writer is not None:
            for candidate in candidates:
                _record_guided_candidate(
                    proposal_writer,
                    candidate,
                    retained=candidate.index == result.selected.index,
                    executed_lineage=candidate.index == result.selected.index,
                )
        return EvalDecisionSummary(
            selected_chunk=result.action_chunk,
            result=result,
            candidate_feasible=sum(candidate.feasible for candidate in candidates),
            candidate_total=len(candidates),
            selection_reason=result.selection_reason,
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
    if (
        geometry_mode == "exact"
        and planning_horizon_chunks == 1
        and directional_sign == 0
        and score_config.mode == "avoidance_only"
    ):
        controller_cls = RejectionController if method == "rejection" else RerankingController
        with timer.time("candidate_scoring", method=method, geometry_mode=geometry_mode):
            result = controller_cls(
                policy=adapter,
                world_model=world_model,
                constraints=constraints,
                k_schedule=k_schedule,
                score_weights=ScoreWeights(),
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
            directional_sign=directional_sign,
            directional_weight=directional_weight,
            score_config=score_config,
        )
    feasible = sum(1 for candidate in result.candidates if candidate.feasible)
    return EvalDecisionSummary(
        selected_chunk=result.action_chunk,
        result=result,
        candidate_feasible=feasible,
        candidate_total=len(result.candidates),
        selection_reason=result.selection_reason,
    )


def _select_beam_search(
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
    beam_width: int,
    branch_factor: int,
    rng: np.random.Generator,
    timer: TimingRecorder,
    score_config: GuidedScoreConfig = _HISTORICAL_SCORE_CONFIG,
    directional_sign: int = 0,
    directional_weight: float = 1.0,
    guided: bool = False,
    policy: SimpleDP3 | None = None,
    itps_config: ITPSGuidanceConfig | None = None,
    compute_counts: ComputeOperationCounts | None = None,
    collision_model: DifferentiablePandaCollisionPoints | None = None,
    constraint_target: Literal["eef", "robot"] = "eef",
    guided_root_seed: int = 0,
    proposal_writer: GuidedProposalTraceWriter | None = None,
    replan: int = 0,
) -> tuple[ControllerResult, BeamSearchTrace]:
    """Expand and feasible-first prune a DP3 continuation beam."""
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    if branch_factor <= 0:
        raise ValueError("branch_factor must be positive")
    if planning_horizon_chunks <= 0:
        raise ValueError("planning_horizon_chunks must be positive")

    frontier = [
        BeamNode(
            node_id="root",
            parent_id=None,
            depth=0,
            current_entry=_copy_entry(current_entry),
            obs_window=_copy_window(obs_window),
            action_chunks=[],
            rollouts=[],
        )
    ]
    depth_traces: list[BeamDepthTrace] = []
    expanded_depths: list[tuple[list[BeamNode], set[str]]] = []
    candidate_index = 0
    for depth in range(1, planning_horizon_chunks + 1):
        expanded, candidate_index = _expand_beam_frontier(
            frontier=frontier,
            depth=depth,
            next_candidate_index=candidate_index,
            adapter=adapter,
            world_model=world_model,
            provider=provider,
            scene=scene,
            constraints=constraints,
            crop_config=crop_config,
            goal_thresh=goal_thresh,
            geometry_mode=geometry_mode,
            branch_factor=branch_factor,
            rng=rng,
            timer=timer,
            directional_sign=directional_sign,
            directional_weight=directional_weight,
            guided=guided,
            policy=policy,
            itps_config=itps_config,
            compute_counts=compute_counts,
            collision_model=collision_model,
            constraint_target=constraint_target,
            guided_root_seed=guided_root_seed,
            score_config=score_config,
            proposal_writer=proposal_writer,
            replan=replan,
        )
        if not expanded:
            raise RuntimeError(f"beam search produced no nodes at depth {depth}")
        frontier = _prune_beam_nodes(expanded, beam_width=beam_width)
        retained_ids = {node.node_id for node in frontier}
        expanded_depths.append((expanded, retained_ids))
        compute_snapshot = _compute_count_snapshot(compute_counts)
        depth_traces.append(
            BeamDepthTrace(
                depth=depth,
                expanded=len(expanded),
                feasible=sum(
                    1 for node in expanded if node.candidate is not None and node.candidate.feasible
                ),
                retained=tuple(
                    _beam_node_trace(
                        node,
                        retained=True,
                        active_width=len(frontier),
                        compute_counters=compute_snapshot,
                    )
                    for node in frontier
                ),
                nodes=tuple(
                    _beam_node_trace(
                        node,
                        retained=node.node_id in retained_ids,
                        active_width=len(frontier),
                        compute_counters=compute_snapshot,
                    )
                    for node in expanded
                ),
            )
        )

    final_candidates = [node.candidate for node in frontier if node.candidate is not None]
    if not final_candidates:
        raise RuntimeError("beam search produced no final candidates")
    feasible = [candidate for candidate in final_candidates if candidate.feasible]
    if feasible:
        selected = min(feasible, key=lambda candidate: candidate.total_score)
        reason = "beam_best_feasible"
    else:
        selected = _beam_candidate(
            min(
                frontier,
                key=lambda node: (
                    _beam_candidate(node).violation_max,
                    _beam_candidate(node).violation_integral,
                    float("inf")
                    if _beam_candidate(node).goal_distance is None
                    else _beam_candidate(node).goal_distance,
                    node.node_id,
                ),
            )
        )
        reason = "beam_least_bad_fallback"
    selected_node = next(
        node
        for node in frontier
        if node.candidate is not None and node.candidate.index == selected.index
    )
    parent_by_id = {
        retained.node_id: retained.parent_id
        for depth_trace in depth_traces
        for retained in depth_trace.retained
    }
    lineage = [selected_node.node_id]
    while parent_by_id.get(lineage[-1]) not in {None, "root"}:
        lineage.append(str(parent_by_id[lineage[-1]]))
    lineage.reverse()
    executed_ids = set(lineage)
    depth_traces = [
        replace(
            depth_trace,
            retained=tuple(
                replace(node, executed_lineage=node.node_id in executed_ids)
                for node in depth_trace.retained
            ),
            nodes=tuple(
                replace(node, executed_lineage=node.node_id in executed_ids)
                for node in depth_trace.nodes
            ),
        )
        for depth_trace in depth_traces
    ]
    if proposal_writer is not None and guided:
        for expanded_nodes, retained_ids in expanded_depths:
            for node in expanded_nodes:
                _record_guided_candidate(
                    proposal_writer,
                    _beam_candidate(node),
                    action_chunk=node.action_chunks[-1],
                    retained=node.node_id in retained_ids,
                    executed_lineage=node.node_id in executed_ids,
                )
    return (
        _controller_result(
            selected,
            final_candidates,
            [branch_factor] * planning_horizon_chunks,
            reason,
        ),
        BeamSearchTrace(
            width=beam_width,
            branch_factor=branch_factor,
            depths=tuple(depth_traces),
            selected_node_id=selected_node.node_id,
            selected_lineage=tuple(lineage),
            selected_score=selected.total_score,
            selected_constraint_penalty=selected.constraint_penalty,
            selected_feasible=selected.feasible,
        ),
    )


def _expand_beam_frontier(
    *,
    frontier: list[BeamNode],
    depth: int,
    next_candidate_index: int,
    adapter: DP3ChunkPolicyAdapter,
    world_model: GeometricWorldModel,
    provider: ManiSkillGhostPandaGeometryProvider,
    scene: Any,
    constraints: list[AvoidRegion],
    crop_config: PointCloudCropConfig,
    goal_thresh: float,
    geometry_mode: GeometryMode,
    branch_factor: int,
    rng: np.random.Generator,
    timer: TimingRecorder,
    score_config: GuidedScoreConfig = _HISTORICAL_SCORE_CONFIG,
    directional_sign: int,
    directional_weight: float,
    guided: bool = False,
    policy: SimpleDP3 | None = None,
    itps_config: ITPSGuidanceConfig | None = None,
    compute_counts: ComputeOperationCounts | None = None,
    collision_model: DifferentiablePandaCollisionPoints | None = None,
    constraint_target: Literal["eef", "robot"] = "eef",
    guided_root_seed: int = 0,
    proposal_writer: GuidedProposalTraceWriter | None = None,
    replan: int = 0,
) -> tuple[list[BeamNode], int]:
    """Batch policy sampling across all parents, then imagine each child."""
    parent_indices = [
        parent_index for parent_index in range(len(frontier)) for _ in range(branch_factor)
    ]
    policy_windows = [frontier[parent_index].obs_window for parent_index in parent_indices]
    if guided:
        if policy is None or itps_config is None or compute_counts is None:
            raise RuntimeError("guided beam requires policy, ITPS config, and compute counters")
        seeds = [
            _guided_seed(
                guided_root_seed,
                depth,
                frontier[parent_index].node_id,
                local_index % branch_factor,
            )
            for local_index, parent_index in enumerate(parent_indices)
        ]
        chunks = sample_itps_candidates(
            policy=policy,
            provider=provider,
            observation_windows=policy_windows,
            constraints=constraints,
            seeds=seeds,
            config=itps_config,
            compute_counts=compute_counts,
            collision_model=collision_model,
            constraint_target=constraint_target,
            proposal_writer=proposal_writer,
            proposal_contexts=[
                {
                    "proposal_id": (
                        f"r{replan}/d{depth}/{frontier[parent_index].node_id}/"
                        f"b{local_index % branch_factor}"
                    ),
                    "purpose": "search_expansion",
                    "replan": replan,
                    "depth": depth,
                    "parent_id": frontier[parent_index].node_id,
                    "ancestry": _beam_ancestry(frontier[parent_index].node_id),
                    "branch_index": local_index % branch_factor,
                }
                for local_index, parent_index in enumerate(parent_indices)
            ],
        )
    else:
        chunks = adapter.sample_action_chunks_for_windows(policy_windows, rng=rng)
    if len(chunks) != len(parent_indices):
        raise RuntimeError("beam policy sampling returned an unexpected candidate count")

    expanded: list[BeamNode] = []
    for local_index, (parent_index, chunk) in enumerate(zip(parent_indices, chunks, strict=True)):
        parent = frontier[parent_index]
        rollout, next_entry, next_window = _imagine_beam_child(
            parent=parent,
            chunk=chunk,
            depth=depth,
            branch_index=local_index % branch_factor,
            world_model=world_model,
            provider=provider,
            crop_config=crop_config,
            goal_thresh=goal_thresh,
            geometry_mode=geometry_mode,
            timer=timer,
        )
        node = BeamNode(
            node_id=f"{parent.node_id}/b{local_index % branch_factor}",
            parent_id=parent.node_id,
            depth=depth,
            current_entry=next_entry,
            obs_window=next_window,
            action_chunks=[*parent.action_chunks, chunk],
            rollouts=[*parent.rollouts, rollout],
        )
        expanded.append(node)
    for parent_index in range(len(frontier)):
        start = parent_index * branch_factor
        siblings = expanded[start : start + branch_factor]
        consensus = consensus_deviations([node.action_chunks[-1] for node in siblings])
        for sibling_offset, (node, consensus_deviation) in enumerate(
            zip(siblings, consensus, strict=True)
        ):
            prefix_rollout = node.to_rollout()
            node.candidate = _candidate_diagnostics(
                index=next_candidate_index + start + sibling_offset,
                attempted_k=branch_factor,
                action_chunk=prefix_rollout.action_chunk,
                rollout=prefix_rollout,
                scene=scene,
                constraints=constraints,
                consensus_deviation=consensus_deviation,
                directional_sign=directional_sign,
                directional_weight=directional_weight,
                score_config=score_config,
            )
    return expanded, next_candidate_index + len(expanded)


def _imagine_beam_child(
    *,
    parent: BeamNode,
    chunk: ActionChunk,
    depth: int,
    branch_index: int,
    world_model: GeometricWorldModel,
    provider: ManiSkillGhostPandaGeometryProvider,
    crop_config: PointCloudCropConfig,
    goal_thresh: float,
    geometry_mode: GeometryMode,
    timer: TimingRecorder,
) -> tuple[ImaginedRollout, Entry, list[Entry]]:
    metadata = {
        "beam_depth": int(depth),
        "beam_branch": int(branch_index),
        "beam_parent_id": parent.node_id,
    }
    if geometry_mode == "exact":
        rollout = world_model.imagine(
            entry_to_world_model_observation(parent.current_entry),
            chunk,
            metadata=metadata,
        )
        next_entry = parent.current_entry
        next_window = _copy_window(parent.obs_window)
        for step_index in range(rollout.action_chunk.horizon):
            next_entry = world_model_entry_from_rollout_step(
                rollout,
                step_index,
                previous_entry=next_entry,
                crop_config=crop_config,
                goal_thresh=goal_thresh,
            )
            next_window = append_obs_window(
                next_window,
                next_entry,
                n_obs_steps=len(parent.obs_window),
            )
        return rollout, next_entry, next_window

    rollout = _fast_imagine_rollout(
        provider=provider,
        observation=entry_to_world_model_observation(parent.current_entry),
        action_chunk=chunk,
        metadata=metadata,
        timer=timer,
    )
    next_entry = parent.current_entry
    next_window = _copy_window(parent.obs_window)
    feedback_start = max(0, rollout.action_chunk.horizon - len(parent.obs_window))
    for step_index in range(feedback_start, rollout.action_chunk.horizon):
        next_entry = _render_feedback_entry(
            provider=provider,
            rollout=rollout,
            step_index=step_index,
            previous_entry=next_entry,
            crop_config=crop_config,
            goal_thresh=goal_thresh,
            timer=timer,
        )
        next_window = append_obs_window(
            next_window,
            next_entry,
            n_obs_steps=len(parent.obs_window),
        )
    return rollout, next_entry, next_window


def _prune_beam_nodes(nodes: list[BeamNode], *, beam_width: int) -> list[BeamNode]:
    """Retain feasible prefixes first, then least-violating infeasible prefixes."""
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    scored = [(node, node.candidate) for node in nodes if node.candidate is not None]
    feasible = [node for node, candidate in scored if candidate.feasible]
    feasible.sort(key=lambda node: (_beam_candidate(node).total_score, node.node_id))
    infeasible = [node for node, candidate in scored if not candidate.feasible]
    infeasible.sort(
        key=lambda node: (
            _beam_candidate(node).violation_max,
            _beam_candidate(node).violation_integral,
            float("inf")
            if _beam_candidate(node).goal_distance is None
            else _beam_candidate(node).goal_distance,
            node.node_id,
        )
    )
    return [*feasible, *infeasible][:beam_width]


def _beam_candidate(node: BeamNode) -> CandidateDiagnostics:
    if node.candidate is None:
        raise RuntimeError("beam node has no candidate diagnostics")
    return node.candidate


def _beam_node_trace(
    node: BeamNode,
    *,
    retained: bool = True,
    active_width: int = 0,
    compute_counters: dict[str, int] | None = None,
) -> BeamRetainedNodeTrace:
    candidate = node.candidate
    if candidate is None:
        raise RuntimeError("retained beam node has no candidate diagnostics")
    return BeamRetainedNodeTrace(
        node_id=node.node_id,
        parent_id=node.parent_id,
        candidate_index=candidate.index,
        feasible=candidate.feasible,
        score=candidate.total_score,
        constraint_penalty=candidate.constraint_penalty,
        eef_path=np.asarray(candidate.rollout.eef_path, dtype=np.float32).copy(),
        min_clearance=candidate.min_clearance,
        violation_max=candidate.violation_max,
        violation_integral=candidate.violation_integral,
        seed_metadata={
            key: value
            for key, value in node.action_chunks[-1].metadata.items()
            if key in {"guidance_seed", "candidate_index", "noise_lineage"}
        },
        normalized_score_terms=dict(candidate.normalized_score_terms),
        applied_score_weights=dict(candidate.applied_score_weights),
        goal_distance=candidate.goal_distance,
        smoothness=candidate.smoothness,
        fallback_key=(
            candidate.violation_max,
            candidate.violation_integral,
            float("inf") if candidate.goal_distance is None else candidate.goal_distance,
            node.node_id,
        ),
        retained=retained,
        active_width=active_width,
        observation_hash=node.action_chunks[-1].metadata.get("conditioning_hash"),
        action_hash=action_sha256(node.action_chunks[-1]),
        compute_counters=dict(compute_counters or {}),
    )


def _compute_count_snapshot(counts: ComputeOperationCounts | None) -> dict[str, int]:
    if counts is None:
        return {}
    return {
        "denoiser_forward_calls": int(counts.denoiser_forward_calls),
        "denoiser_evaluations": int(counts.denoiser_evaluations),
        "differentiable_fk_calls": int(counts.differentiable_fk_calls),
        "differentiable_fk_pose_evaluations": int(counts.differentiable_fk_pose_evaluations),
        "robot_point_cloud_renders": int(counts.robot_point_cloud_renders),
    }


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
    score_config: GuidedScoreConfig = _HISTORICAL_SCORE_CONFIG,
    directional_sign: int = 0,
    directional_weight: float = 1.0,
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
                directional_sign=directional_sign,
                directional_weight=directional_weight,
                score_config=score_config,
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
    selected = min(candidates, key=_infeasible_candidate_key)
    return _controller_result(selected, candidates, attempted, "least_bad_fallback")


def _select_itps_chunk(
    *,
    policy: SimpleDP3,
    provider: ManiSkillGhostPandaGeometryProvider,
    obs_window: list[Entry],
    constraints: list[AvoidRegion | AvoidProjection],
    rng: np.random.Generator,
    config: ITPSGuidanceConfig,
    compute_counts: ComputeOperationCounts,
    collision_model: DifferentiablePandaCollisionPoints | None,
    constraint_target: Literal["eef", "robot"],
    proposal_writer: GuidedProposalTraceWriter | None = None,
    proposal_context: dict[str, Any] | None = None,
) -> ActionChunk:
    seed = int(rng.integers(0, 2**31 - 1))
    chunk = sample_itps_candidates(
        policy=policy,
        provider=provider,
        observation_windows=[obs_window],
        constraints=constraints,
        seeds=[seed],
        config=config,
        compute_counts=compute_counts,
        collision_model=collision_model,
        constraint_target=constraint_target,
        proposal_writer=proposal_writer,
        proposal_contexts=[proposal_context or {}],
    )[0]
    if proposal_writer is not None:
        _record_guided_action_without_score(proposal_writer, chunk)
    return chunk


def sample_itps_candidates(
    *,
    policy: SimpleDP3,
    provider: ManiSkillGhostPandaGeometryProvider,
    observation_windows: list[list[Entry]],
    constraints: list[AvoidRegion | AvoidProjection],
    seeds: list[int],
    config: ITPSGuidanceConfig,
    compute_counts: ComputeOperationCounts,
    collision_model: DifferentiablePandaCollisionPoints | None,
    constraint_target: Literal["eef", "robot"],
    proposal_writer: GuidedProposalTraceWriter | None = None,
    proposal_contexts: list[dict[str, Any]] | None = None,
) -> list[ActionChunk]:
    """Generate independent faithful ITPS proposals for real or imagined windows."""
    if len(observation_windows) != len(seeds):
        raise ValueError("observation_windows and seeds must have equal length")
    contexts = proposal_contexts or [{} for _ in seeds]
    if len(contexts) != len(seeds):
        raise ValueError("proposal_contexts and seeds must have equal length")
    device = policy.device
    world_from_base = torch.as_tensor(
        provider.world_from_robot_base(),
        device=device,
        dtype=policy.dtype,
    )

    def guidance_fn(traj: torch.Tensor) -> torch.Tensor:
        compute_counts.record_differentiable_fk(traj)
        if constraint_target == "robot":
            if collision_model is None:
                raise RuntimeError("whole-body ITPS requires Panda collision geometry")
            guidance_points = _itps_robot_points(
                policy,
                traj,
                world_from_base,
                collision_model,
            )
            compute_counts.record_differentiable_robot_points(guidance_points)
        else:
            guidance_points = _itps_eef_path(policy, traj, world_from_base)
        return avoidance_energy(
            guidance_points,
            constraints,
            target=constraint_target,
            mode=config.energy,
            temperature=config.barrier_temperature,
        )

    geometry_metadata: dict[str, Any] = {}
    if constraint_target == "robot" and collision_model is not None:
        geometry_metadata = {
            "collision_geometry_source": "maniskill_panda_urdf",
            "collision_link_allocation": collision_model.allocation(),
            "gripper_open": collision_model.gripper_open,
            "excluded_collision_links": ["panda_link0"],
        }
    chunks: list[ActionChunk] = []
    for candidate_index, (obs_window, seed, proposal_context) in enumerate(
        zip(observation_windows, seeds, contexts, strict=True)
    ):
        noise_lineage = policy.make_itps_noise_lineage(
            candidate_seed=int(seed),
            mcmc_steps=config.mcmc_steps,
            guided=True,
            root_identity="guided_search",
        )
        conditioning_hash = None
        guidance_geometry_hash = None
        guidance_geometry_bundle = None
        if proposal_writer is not None:
            conditioning_hash, _ = proposal_writer.store_conditioning(obs_window)
            geometry_arrays = {
                "world_from_base": world_from_base.detach().cpu().numpy(),
            }
            if collision_model is not None:
                geometry_arrays.update(
                    {
                        "local_points": collision_model.local_points.detach().cpu().numpy(),
                        "link_indices": collision_model.link_indices.detach().cpu().numpy(),
                        "link_counts": np.asarray(collision_model.link_counts, dtype=np.int64),
                        "sample_seed": np.asarray(collision_model.sample_seed, dtype=np.int64),
                        "gripper_open": np.asarray(collision_model.gripper_open, dtype=np.float64),
                    }
                )
            guidance_geometry_hash, geometry_path = proposal_writer.store_guidance_geometry(
                geometry_arrays
            )
            guidance_geometry_bundle = str(geometry_path)
        obs_batch = _repeat_obs_window_to_torch(
            obs_window,
            k=1,
            device=device,
            goal_marker_points=int(getattr(policy, "goal_marker_points", 0)),
            goal_marker_radius=float(
                getattr(policy, "goal_marker_radius", DEFAULT_GOAL_MARKER_RADIUS)
            ),
        )
        output = policy.predict_action_itps(
            obs_batch,
            noise_lineage=noise_lineage,
            guidance_fn=guidance_fn,
            guide_ratio=config.guide_ratio,
            mcmc_steps=config.mcmc_steps,
        )
        action = output["action"][0]
        chunks.append(
            ActionChunk(
                actions=action.detach().cpu().numpy().astype(np.float32, copy=True),
                action_mode=_action_mode("abs_joint"),
                dt=1.0,
                metadata={
                    "method": "itps",
                    "candidate_index": candidate_index,
                    "guidance_seed": int(seed),
                    "noise_lineage": noise_lineage.to_json(),
                    "conditioning_hash": conditioning_hash,
                    "proposal_context": dict(proposal_context),
                    "guidance_geometry_hash": guidance_geometry_hash,
                    "guidance_geometry_bundle": guidance_geometry_bundle,
                    "guidance_target": constraint_target,
                    **config.to_json(),
                    **geometry_metadata,
                },
            )
        )
    return chunks


def _record_guided_action_without_score(
    writer: GuidedProposalTraceWriter,
    action_chunk: ActionChunk,
) -> None:
    _write_guided_proposal_record(
        writer,
        action_chunk=action_chunk,
        score_data=None,
        retained=True,
        executed_lineage=True,
    )


def _record_guided_candidate(
    writer: GuidedProposalTraceWriter,
    candidate: CandidateDiagnostics,
    *,
    action_chunk: ActionChunk | None = None,
    retained: bool,
    executed_lineage: bool,
) -> None:
    _write_guided_proposal_record(
        writer,
        action_chunk=action_chunk or candidate.action_chunk,
        score_data={
            "raw": {
                "constraint_penalty": candidate.constraint_penalty,
                "goal_distance": candidate.goal_distance,
                "smoothness": candidate.smoothness,
                "consensus_deviation": candidate.consensus_deviation,
            },
            "normalized": candidate.normalized_score_terms,
            "weights": candidate.applied_score_weights,
            "total_score": candidate.total_score,
            "feasible": candidate.feasible,
            "min_clearance": candidate.min_clearance,
            "violation_max": candidate.violation_max,
            "violation_integral": candidate.violation_integral,
            "fallback_key": candidate.fallback_key,
        },
        retained=retained,
        executed_lineage=executed_lineage,
    )


def _write_guided_proposal_record(
    writer: GuidedProposalTraceWriter,
    *,
    action_chunk: ActionChunk,
    score_data: dict[str, Any] | None,
    retained: bool,
    executed_lineage: bool,
) -> None:
    metadata = action_chunk.metadata
    context = dict(metadata.get("proposal_context", {}))
    conditioning_hash = metadata.get("conditioning_hash")
    noise_lineage = metadata.get("noise_lineage")
    if not context or conditioning_hash is None or not isinstance(noise_lineage, dict):
        raise RuntimeError("guided proposal is missing replay metadata")
    itps_keys = ITPSGuidanceConfig().to_json().keys()
    writer.record(
        proposal_id=str(context["proposal_id"]),
        purpose=str(context["purpose"]),
        replan=int(context["replan"]),
        depth=int(context["depth"]),
        parent_id=context.get("parent_id"),
        ancestry=context.get("ancestry", []),
        branch_index=int(context["branch_index"]),
        action_chunk=action_chunk,
        conditioning_hash=str(conditioning_hash),
        itps_config={key: metadata[key] for key in itps_keys if key in metadata},
        noise_lineage=noise_lineage,
        score_data=score_data,
        retained=retained,
        executed_lineage=executed_lineage,
        guidance_geometry_hash=metadata.get("guidance_geometry_hash"),
        guidance_geometry_bundle=metadata.get("guidance_geometry_bundle"),
        guidance_target=metadata.get("guidance_target"),
    )


def _select_guided_candidates(
    candidates: list[CandidateDiagnostics], *, attempted_k: int
) -> ControllerResult:
    """Select exact-feasible guided proposals with deterministic tie breaking."""
    if not candidates:
        raise RuntimeError("guided proposal generation returned no candidates")
    feasible = [candidate for candidate in candidates if candidate.feasible]
    if feasible:
        selected = min(feasible, key=lambda candidate: (candidate.total_score, candidate.index))
        return _controller_result(selected, candidates, [attempted_k], "best_feasible")
    selected = min(candidates, key=_infeasible_candidate_key)
    return _controller_result(selected, candidates, [attempted_k], "least_bad_fallback")


def _itps_eef_path(
    policy: SimpleDP3,
    normalized_trajectory: torch.Tensor,
    world_from_base: torch.Tensor,
) -> torch.Tensor:
    """Unnormalize a denoised action trajectory and run differentiable Panda FK."""
    normalized_actions = normalized_trajectory[..., : policy.action_dim]
    actions = policy.normalizer["action"].unnormalize(normalized_actions)
    return panda_end_effector_position(actions[..., :7], world_from_base)


def _itps_robot_points(
    policy: SimpleDP3,
    normalized_trajectory: torch.Tensor,
    world_from_base: torch.Tensor,
    collision_model: DifferentiablePandaCollisionPoints,
) -> torch.Tensor:
    """Unnormalize an ITPS trajectory and transform sampled Panda collision points."""
    normalized_actions = normalized_trajectory[..., : policy.action_dim]
    actions = policy.normalizer["action"].unnormalize(normalized_actions)
    return collision_model(actions[..., :7], world_from_base)


def _make_itps_collision_model(
    env: Any,
    *,
    policy: SimpleDP3,
    constraint_target: Literal["eef", "robot"],
    methods: list[str],
    config: ITPSGuidanceConfig,
    gripper_open: float,
) -> DifferentiablePandaCollisionPoints | None:
    """Build whole-body ITPS geometry once for the active Panda environment."""
    if constraint_target != "robot" or not any(
        method in {"itps", "itps_reranking", "itps_beam"} for method in methods
    ):
        return None
    unwrapped = getattr(env, "unwrapped", env)
    agent = getattr(unwrapped, "agent", None)
    urdf_path = getattr(agent, "urdf_path", None)
    if urdf_path is None:
        raise RuntimeError("whole-body ITPS requires an active Panda agent with urdf_path")
    template = load_panda_collision_point_template(
        urdf_path,
        point_count=config.robot_points,
        sample_seed=config.robot_sample_seed,
    )
    return DifferentiablePandaCollisionPoints(
        template,
        gripper_open=gripper_open,
    ).to(device=policy.device, dtype=policy.dtype)


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
    score_config: GuidedScoreConfig = _HISTORICAL_SCORE_CONFIG,
    directional_sign: int = 0,
    directional_weight: float = 1.0,
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
            directional_sign=directional_sign,
            directional_weight=directional_weight,
            score_config=score_config,
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
    score_config: GuidedScoreConfig = _HISTORICAL_SCORE_CONFIG,
    directional_sign: int = 0,
    directional_weight: float = 1.0,
) -> CandidateDiagnostics:
    constraint_costs: dict[str, float] = {}
    constraint_satisfied: dict[str, bool] = {}
    min_clearance = float("inf")
    violation_max = 0.0
    violation_integral = 0.0
    for constraint_idx, constraint in enumerate(constraints):
        label = f"{constraint_idx}:{constraint.name}"
        costs = constraint.cost(rollout, scene)
        for key, value in costs.items():
            constraint_costs[_unique_cost_key(constraint_costs, key)] = float(value)
        constraint_satisfied[label] = bool(constraint.satisfied(rollout, scene))
        points = (
            rollout.eef_path
            if constraint.target == "eef"
            else np.concatenate(rollout.robot_point_clouds, axis=0)
        )
        signed_distance = np.asarray(constraint.region.signed_distance(points), dtype=np.float64)
        if signed_distance.size:
            min_clearance = min(min_clearance, float(np.min(signed_distance)))
            violations = np.maximum(float(constraint.margin) - signed_distance, 0.0)
            violation_max = max(violation_max, float(np.max(violations)))
            violation_integral += float(np.sum(violations))
    raw_min_clearance = min_clearance if np.isfinite(min_clearance) else None
    feasible = all(constraint_satisfied.values()) if constraint_satisfied else True
    if raw_min_clearance is not None:
        feasible = feasible and raw_min_clearance >= score_config.effective_hard_clearance_m
    distance = goal_distance(rollout, scene.target_position)
    smoothness = trajectory_smoothness(rollout, order=2)
    penalty = primary_constraint_penalty(constraint_costs)
    directional = (
        directional_preference(rollout, scene.target_position, sign=directional_sign)
        if directional_sign != 0
        else 0.0
    )
    normalized_terms = score_config.terms(
        goal_distance_m=distance,
        min_clearance_m=raw_min_clearance,
        smoothness_rad2=smoothness,
    )
    total_score = score_config.feasible_score(
        normalized_terms,
        avoidance_penalty=penalty,
    )
    if score_config.mode == "avoidance_only":
        total_score += directional_weight * directional
    fallback_key = (
        float(violation_max),
        float(violation_integral),
        float("inf") if distance is None else float(distance),
        int(index),
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
        directional=directional,
        min_clearance=raw_min_clearance,
        violation_max=violation_max,
        violation_integral=violation_integral,
        normalized_score_terms=normalized_terms.to_json(),
        applied_score_weights=score_config.weights.to_json(),
        fallback_key=fallback_key,
    )


def _infeasible_candidate_key(candidate: CandidateDiagnostics) -> tuple[float, float, float, int]:
    return (
        float(candidate.violation_max),
        float(candidate.violation_integral),
        float("inf") if candidate.goal_distance is None else float(candidate.goal_distance),
        int(candidate.index),
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
        "replan_index": replan_index,
        "step": step,
        "selection_reason": decision.selection_reason,
        "candidate_feasible": decision.candidate_feasible,
        "candidate_total": decision.candidate_total,
        "selected_action_sha256": action_sha256(decision.selected_chunk),
        "executed_prefix_sha256": action_sha256(
            np.asarray(decision.selected_chunk.actions[:8], dtype=np.float32)
        ),
    }
    if method in {"itps", "itps_reranking", "itps_beam"}:
        row["itps"] = dict(decision.selected_chunk.metadata)
    if decision.beam_trace is not None:
        row["beam"] = decision.beam_trace.to_json()
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
                "selected_min_clearance": result.selected.min_clearance,
                "selected_violation_max": result.selected.violation_max,
                "selected_violation_integral": result.selected.violation_integral,
                "selected_seed_metadata": {
                    key: value
                    for key, value in result.selected.action_chunk.metadata.items()
                    if key in {"guidance_seed", "candidate_index"}
                },
                "selected_smoothness": result.selected.smoothness,
                "selected_directional": result.selected.directional,
                "selected_constraint_costs": result.selected.constraint_costs,
                "score_min": min(scores) if scores else None,
                "score_mean": float(np.mean(scores)) if scores else None,
            }
        )
    decisions_file.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
    decisions_file.flush()


def _rerun_replan_record(
    decision: EvalDecisionSummary,
    *,
    provider: ManiSkillGhostPandaGeometryProvider,
    current_entry: Entry,
    constraints: list[AvoidRegion | AvoidProjection],
    constraint_target: Literal["eef", "robot"],
    collision_model: DifferentiablePandaCollisionPoints | None,
    step: int,
    replan_index: int,
    timer: TimingRecorder,
) -> dict[str, Any]:
    """Build Rerun-ready candidate and selected EEF paths for one replan."""
    current_tcp = np.asarray(current_entry["tcp_pose"], dtype=np.float32).reshape(-1)[:3]
    candidates: list[dict[str, Any]] = []
    selected_path: np.ndarray
    if decision.result is not None:
        for candidate in decision.result.candidates:
            eef_path = np.concatenate(
                [current_tcp.reshape(1, 3), candidate.rollout.eef_path], axis=0
            )
            candidates.append(
                {
                    "index": int(candidate.index),
                    "eef_path": eef_path,
                    "feasible": bool(candidate.feasible),
                    "score": float(candidate.total_score),
                    "constraint_penalty": float(candidate.constraint_penalty),
                    "min_clearance": candidate.min_clearance,
                    "violation_max": float(candidate.violation_max),
                    "violation_integral": float(candidate.violation_integral),
                    "seed_metadata": {
                        key: value
                        for key, value in candidate.action_chunk.metadata.items()
                        if key in {"guidance_seed", "candidate_index"}
                    },
                }
            )
        selected_path = np.concatenate(
            [current_tcp.reshape(1, 3), decision.result.selected.rollout.eef_path],
            axis=0,
        )
        selected_index: int | None = int(decision.result.selected.index)
    else:
        rollout = _fast_imagine_rollout(
            provider=provider,
            observation=entry_to_world_model_observation(current_entry),
            action_chunk=decision.selected_chunk,
            metadata={"replan_index": replan_index, "artifact": "rerun"},
            timer=timer,
        )
        selected_path = np.concatenate([current_tcp.reshape(1, 3), rollout.eef_path], axis=0)
        selected_index = None
    record = {
        "step": int(step),
        "replan_index": int(replan_index),
        "selection_reason": decision.selection_reason,
        "selected_index": selected_index,
        "selected_eef_path": selected_path,
        "candidates": candidates,
    }
    if decision.beam_trace is not None:
        record["beam"] = decision.beam_trace.to_json()
    if constraint_target == "robot" and collision_model is not None:
        record.update(
            _itps_robot_replan_diagnostics(
                decision.selected_chunk,
                provider=provider,
                collision_model=collision_model,
                constraints=constraints,
                timer=timer,
            )
        )
    return record


def _itps_robot_replan_diagnostics(
    action_chunk: ActionChunk,
    *,
    provider: ManiSkillGhostPandaGeometryProvider,
    collision_model: DifferentiablePandaCollisionPoints,
    constraints: list[AvoidRegion | AvoidProjection],
    timer: TimingRecorder,
) -> dict[str, Any]:
    """Build final selected whole-body geometry without retaining inner MCMC states."""
    device = collision_model.local_points.device
    dtype = collision_model.local_points.dtype
    q = torch.as_tensor(action_chunk.actions[..., :7], device=device, dtype=dtype)
    world_from_base = torch.as_tensor(
        provider.world_from_robot_base(),
        device=device,
        dtype=dtype,
    )
    with timer.time("rerun_itps_robot_geometry"):
        with torch.no_grad():
            robot_points = collision_model(q, world_from_base).detach().cpu().numpy()
    flat_points = robot_points.reshape(-1, 3)
    points_per_step = int(robot_points.shape[1])
    worst_points = []
    for constraint_index, constraint in enumerate(constraints):
        signed_distance = constraint.region.signed_distance(flat_points)
        flat_index = int(np.argmin(signed_distance))
        horizon_index, point_index = divmod(flat_index, points_per_step)
        distance = float(signed_distance[flat_index])
        worst_points.append(
            {
                "constraint_index": constraint_index,
                "constraint_name": constraint.name,
                "horizon_index": horizon_index,
                "point_index": point_index,
                "position": robot_points[horizon_index, point_index],
                "signed_distance": distance,
                "violation": max(0.0, float(constraint.margin) - distance),
            }
        )
    return {
        "itps_robot_points": robot_points.astype(np.float32, copy=False),
        "itps_robot_link_indices": collision_model.link_indices.detach().cpu().numpy(),
        "itps_worst_points": worst_points,
    }


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
    region: SphereRegion | BoxRegion | CylinderRegion,
    robot_points: np.ndarray,
    args: argparse.Namespace,
    *,
    clearance: float,
    name: str,
    preserve_center_z: bool = False,
    max_iter: int = 64,
) -> SphereRegion | BoxRegion | CylinderRegion:
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
    if isinstance(region, CylinderRegion):
        center = region.center.astype(np.float32).copy()
        for _ in range(max_iter):
            candidate = CylinderRegion(
                center=center,
                radius=region.radius,
                half_length=region.half_length,
            )
            signed = candidate.signed_distance(pts)
            min_sd = float(np.min(signed))
            if min_sd >= clearance:
                return candidate
            nearest = pts[int(np.argmin(signed))]
            direction = center - nearest
            if preserve_center_z:
                direction[2] = 0.0
            norm = float(np.linalg.norm(direction))
            if norm < 1e-6:
                direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                norm = 1.0
            center = (center + direction / norm * ((clearance - min_sd) + 1e-3)).astype(np.float32)
        return CylinderRegion(
            center=center,
            radius=region.radius,
            half_length=region.half_length,
        )
    # BoxRegion: translate away from nearest robot point until clear.
    center = region.center.astype(np.float32).copy()
    half_extents = region.half_extents.astype(np.float32)
    for _ in range(max_iter):
        candidate = BoxRegion(center=center, half_extents=half_extents, yaw=region.yaw)
        signed = candidate.signed_distance(pts)
        min_sd = float(np.min(signed))
        if min_sd >= clearance:
            return candidate
        nearest = pts[int(np.argmin(signed))]
        direction = center - nearest
        if preserve_center_z:
            direction[2] = 0.0
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            norm = 1.0
        step = (clearance - min_sd) + 1e-3
        center = (center + direction / norm * step).astype(np.float32)
    print(
        f"warning: could not fully clear avoid box '{name}' from robot "
        f"(min clearance still < {clearance:.3f} m after {max_iter} iters)",
        file=sys.stderr,
    )
    return BoxRegion(center=center, half_extents=half_extents, yaw=region.yaw)


def _ground_embodied_region(
    region: SphereRegion | BoxRegion | CylinderRegion,
    *,
    support_plane_z: float,
) -> SphereRegion | BoxRegion | CylinderRegion:
    """Place a region's bottom face on the configured support plane."""
    center = region.center.astype(np.float32).copy()
    if isinstance(region, BoxRegion):
        center[2] = float(support_plane_z) + float(region.half_extents[2])
        return BoxRegion(center=center, half_extents=region.half_extents, yaw=region.yaw)
    if isinstance(region, CylinderRegion):
        center[2] = float(support_plane_z) + float(region.half_length)
        return CylinderRegion(
            center=center,
            radius=region.radius,
            half_length=region.half_length,
        )
    center[2] = float(support_plane_z) + float(region.radius)
    return SphereRegion(center=center, radius=region.radius)


def _align_cabinet_back_panel_to_path(region: BoxRegion) -> BoxRegion:
    """Shift the cabinet root so its back panel, not open interior, crosses the path."""
    back = next(component for component in CABINET_COMPONENTS if component.name == "back")
    local = np.asarray(back.local_center[:2], dtype=np.float32)
    cos_yaw = float(np.cos(region.yaw))
    sin_yaw = float(np.sin(region.yaw))
    world_offset = np.asarray(
        [
            cos_yaw * local[0] - sin_yaw * local[1],
            sin_yaw * local[0] + cos_yaw * local[1],
        ],
        dtype=np.float32,
    )
    center = region.center.astype(np.float32).copy()
    center[:2] -= world_offset
    return BoxRegion(center=center, half_extents=region.half_extents, yaw=region.yaw)


def _finalize_constraints(
    constraints: list[AvoidRegion],
    *,
    robot_points: np.ndarray | None,
    args: argparse.Namespace,
) -> list[AvoidRegion]:
    """Apply the guidance target and robot-clearance-aware placement to placed regions."""
    enable_clearance = (
        bool(args.robot_clearance_placement) and robot_points is not None and len(robot_points) > 0
    )
    finalized: list[AvoidRegion] = []
    for constraint in constraints:
        region = constraint.region
        if isinstance(region, BoxRegion):
            region = BoxRegion(
                center=region.center,
                half_extents=region.half_extents,
                yaw=np.deg2rad(float(args.obstacle_yaw_deg)),
            )
            if args.embody_obstacle and args.obstacle_family == "cabinet":
                region = _align_cabinet_back_panel_to_path(region)
        grounded = bool(args.embody_obstacle and args.ground_embodied_obstacle)
        if grounded and isinstance(region, (SphereRegion, BoxRegion, CylinderRegion)):
            region = _ground_embodied_region(
                region,
                support_plane_z=float(args.obstacle_support_plane_z),
            )
        if (
            enable_clearance
            and not grounded
            and isinstance(region, (SphereRegion, BoxRegion, CylinderRegion))
        ):
            region = _clear_region_from_robot(
                region,
                robot_points,
                args,
                clearance=float(args.robot_clearance_placement_margin),
                name=constraint.name,
                preserve_center_z=grounded,
            )
        finalized.append(
            replace(
                constraint,
                region=region,
                target=args.constraint_target,
                clearance_scale=float(args.avoid_clearance_scale),
            )
        )
    if args.embody_obstacle and args.obstacle_family in {"cabinet", "u_shape"}:
        family = str(args.obstacle_family)
        if len(finalized) != 1 or not isinstance(finalized[0].region, BoxRegion):
            raise ValueError(f"{family} family requires one root BoxRegion before expansion")
        root = finalized[0]
        root_region = root.region
        components = (
            scaled_cabinet_components(float(root_region.half_extents[2]))
            if family == "cabinet"
            else u_shape_components(root_region.half_extents)
        )
        return [
            replace(
                root,
                name=f"{root.name}/{family}_{component.name}",
                region=BoxRegion(
                    center=component_center,
                    half_extents=component.half_extents,
                    yaw=component_yaw,
                ),
            )
            for component in components
            for component_center, component_yaw in [
                transform_box_component(
                    component,
                    center=root_region.center,
                    yaw=root_region.yaw,
                )
            ]
        ]
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
    policy: SimpleDP3 | None = None,
    adapter: DP3ChunkPolicyAdapter | None = None,
    action_mode: ActionMode = "abs_joint",
    crop_config: PointCloudCropConfig,
    goal_thresh: float = 0.025,
    args: argparse.Namespace,
    zarr_context: dict[str, Any] | None = None,
) -> list[AvoidRegion]:
    if args.no_constraints:
        return []
    fixture_manifest = getattr(args, "_guided_fixture_manifest", None)
    if fixture_manifest is not None:
        constraints = load_episode_constraints(
            fixture_manifest.episode(spec.output_index).constraint_file
        )
        constraints = [
            replace(constraint, margin=float(args.avoid_margin))
            if isinstance(constraint, (AvoidRegion, AvoidProjection))
            else constraint
            for constraint in constraints
        ]
        _validate_precomputed_constraints(constraints, env=env, args=args)
        return constraints
    if args.constraints_dir is not None:
        constraints = load_episode_constraints(
            _precomputed_constraint_path(args.constraints_dir, spec)
        )
        constraints = [
            replace(constraint, margin=float(args.avoid_margin))
            if isinstance(constraint, (AvoidRegion, AvoidProjection))
            else constraint
            for constraint in constraints
        ]
        _validate_precomputed_constraints(
            constraints,
            env=env,
            args=args,
        )
        return constraints
    if policy is None or adapter is None:
        raise ValueError("generated constraints require a policy and DP3 adapter")
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


def _validate_precomputed_constraints(
    constraints: list[AvoidRegion],
    *,
    env: Any,
    args: argparse.Namespace,
) -> None:
    """Validate that serialized constraints match the requested evaluation protocol."""
    mismatched_targets = [
        constraint.target
        for constraint in constraints
        if constraint.target != args.constraint_target
    ]
    if mismatched_targets:
        raise ValueError(
            "precomputed constraint targets do not match --constraint-target "
            f"{args.constraint_target!r}: {mismatched_targets}"
        )
    if args.embody_obstacle:
        if env is None:
            raise ValueError("embodied precomputed constraints require a live environment")
        _validate_embodied_obstacle_geometry(env, constraints)


def _validate_precomputed_initial_clearance(
    constraints: list[AvoidRegion],
    *,
    zarr_context: dict[str, Any] | None,
    minimum_clearance: float | None,
) -> float | None:
    """Reject impossible precomputed episodes before any compared method runs."""
    if minimum_clearance is None or not constraints:
        return None
    clearance = _initial_robot_constraint_clearance(
        constraints,
        zarr_context=zarr_context,
    )
    if clearance + 1e-8 < float(minimum_clearance):
        raise ValueError(
            "precomputed obstacle violates initial robot clearance: "
            f"{clearance:.6f} < {float(minimum_clearance):.6f} m"
        )
    return clearance


def _initial_robot_constraint_clearance(
    constraints: list[AvoidRegion],
    *,
    zarr_context: dict[str, Any] | None,
) -> float:
    """Measure finalized constraints against the stored initial whole-robot cloud."""
    if not constraints:
        raise ValueError("initial robot clearance requires at least one constraint")
    if zarr_context is None:
        raise ValueError("initial robot clearance requires a dataset episode")
    points = np.asarray(zarr_context["point_cloud"], dtype=np.float32).reshape(-1, 3)
    robot_mask = np.asarray(zarr_context["robot_mask"], dtype=bool).reshape(-1)
    valid_mask = np.asarray(zarr_context["point_valid_mask"], dtype=bool).reshape(-1)
    if robot_mask.shape != (points.shape[0],) or valid_mask.shape != (points.shape[0],):
        raise ValueError("initial robot/valid masks do not match the point cloud")
    robot_points = points[robot_mask & valid_mask]
    if not len(robot_points):
        raise ValueError("initial robot point cloud is empty")
    return float(min_constraint_clearance(robot_points, constraints))


def _clearance_safe_candidate_spec(
    spec: RolloutSpec,
    constraints: list[AvoidRegion],
    *,
    zarr_context: dict[str, Any] | None,
    minimum_clearance: float,
    accepted_output_index: int,
) -> tuple[RolloutSpec | None, dict[str, Any]]:
    """Accept an unchanged placement or describe its geometry-only exclusion."""
    clearance = _initial_robot_constraint_clearance(
        constraints,
        zarr_context=zarr_context,
    )
    accepted = clearance + 1e-8 >= float(minimum_clearance)
    record: dict[str, Any] = {
        "source_pool_index": int(spec.output_index),
        "output_index": int(accepted_output_index) if accepted else None,
        "dataset_episode_index": spec.dataset_episode_index,
        "seed": int(spec.seed),
        "initial_robot_clearance": clearance,
        "required_initial_robot_clearance": float(minimum_clearance),
        "exclusion_reason": None if accepted else "insufficient_initial_robot_clearance",
    }
    if not accepted:
        return None, record
    return (
        RolloutSpec(
            output_index=accepted_output_index,
            seed=spec.seed,
            source=spec.source,
            dataset_episode_index=spec.dataset_episode_index,
        ),
        record,
    )


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
                    clearance_scale=args.avoid_clearance_scale,
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
) -> SphereRegion | BoxRegion | CylinderRegion:
    """Build an avoid-region primitive of the configured shape at the given center.

    The center matches exactly where a sphere would be placed; only the shape differs.
    A box uses --avoid-box-half-extents when provided, otherwise isotropic half-extents
    equal to the effective sphere radius so it occupies a comparable volume.
    """
    center = np.asarray(center, dtype=np.float32).reshape(3)
    if getattr(args, "avoid_shape", "sphere") == "cylinder":
        dimensions = (
            np.asarray(args.avoid_box_half_extents, dtype=np.float32)
            if args.avoid_box_half_extents is not None
            else np.full(3, float(radius), dtype=np.float32)
        )
        return CylinderRegion(
            center=center,
            radius=float(dimensions[0]),
            half_length=float(dimensions[2]),
        )
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
            clearance_scale=float(args.avoid_clearance_scale),
            name=name,
        )
    return AvoidRegion(
        region=_avoid_region_of_shape(center, radius, args),
        margin=float(args.avoid_margin),
        weight=float(args.avoid_weight),
        clearance_scale=float(args.avoid_clearance_scale),
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
        name = f"widest_trajectory_avoid_region_{i}" if multi else "widest_trajectory_avoid_region"
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
                clearance_scale=float(args.avoid_clearance_scale),
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
        for xc, yc in zip(xs[:-1], ys[:-1], strict=True):
            ax3d.plot([xc, xc], [yc, yc], [z_lo, z_hi], color="crimson", alpha=0.3, linewidth=0.8)
        ax3d.scatter(cx, cy, 0.5 * (z_lo + z_hi), color="crimson", s=60, zorder=10)
        constraint_legend_label = f"projection {2 * hx:.3f}×{2 * hy:.3f}m"
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
            2 * hx,
            2 * hy,
            linewidth=1.5,
            edgecolor="crimson",
            facecolor="crimson",
            alpha=0.15,
            label=f"projection {2 * hx:.3f}×{2 * hy:.3f}m",
        )
        ax2d.add_patch(rect)
        ax2d.scatter(center[0], center[1], color="crimson", s=60, zorder=10)
    else:
        circle = plt.Circle(
            (center[0], center[1]),
            radius,
            color="crimson",
            fill=False,
            linewidth=1.5,
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
    if args.no_constraints:
        return {"type": "none"}
    if args.constraints_dir is not None:
        return {
            "type": "precomputed",
            "constraints_dir": str(args.constraints_dir),
            "avoid_clearance_scale": float(args.avoid_clearance_scale),
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
        "avoid_clearance_scale": float(args.avoid_clearance_scale),
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
    return {key: value.repeat((k, *([1] * (value.ndim - 1)))) for key, value in batch.items()}


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
        [np.stack([entry["target_position"] for entry in window], axis=0) for window in windows],
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
    tcp = np.asarray(entry["tcp_pose"], dtype=np.float32).reshape(-1)[:3]
    path.append(
        tcp_position=tcp,
        q=np.asarray(entry["agent_pos"], dtype=np.float32),
        target_distance=float(np.asarray(entry["final_distance"], dtype=np.float32).reshape(-1)[0]),
    )


def _whole_robot_clearance_point_clouds(
    path: EpisodePath,
    provider: ManiSkillGhostPandaGeometryProvider,
    *,
    stride: int = 4,
) -> list[np.ndarray]:
    """Sample the URDF/mesh robot point cloud across the executed trajectory.

    Sets the ghost env to each stored qpos (subsampled by ``stride``) and collects the
    mesh-derived robot points, so constraint clearance reflects the whole arm and base
    rather than only the TCP. Keeps one cloud per sampled time for duration/integral
    metrics instead of flattening time.
    """
    q_array = path.q_array
    if q_array.size == 0:
        return []
    stride = max(1, int(stride))
    indices = list(range(0, q_array.shape[0], stride))
    if indices[-1] != q_array.shape[0] - 1:
        indices.append(q_array.shape[0] - 1)
    clouds: list[np.ndarray] = []
    for idx in indices:
        points = np.asarray(provider.robot_point_cloud(q_array[idx]), dtype=np.float32)
        if points.size:
            clouds.append(points.reshape(-1, 3).astype(np.float32, copy=False))
        else:
            clouds.append(np.zeros((0, 3), dtype=np.float32))
    return clouds


def _subsample_robot_clouds(
    clouds: list[np.ndarray],
    *,
    stride: int,
) -> list[np.ndarray]:
    """Subsample online contact clouds with the same endpoint rule as path grading."""
    if not clouds:
        return []
    stride = max(1, int(stride))
    indices = list(range(0, len(clouds), stride))
    if indices[-1] != len(clouds) - 1:
        indices.append(len(clouds) - 1)
    return [np.asarray(clouds[index], dtype=np.float32).reshape(-1, 3) for index in indices]


def _env_kwargs(
    metadata: dict[str, Any],
    *,
    render_mode: str | None,
    obstacle_half_extents: tuple[float, float, float] | None = None,
    obstacle_family: str = "box",
    max_episode_steps: int | None = None,
) -> dict[str, Any]:
    env_kwargs = dict(metadata["env_kwargs"])
    env_kwargs["obs_mode"] = "pointcloud"
    env_kwargs["num_envs"] = 1
    if max_episode_steps is not None:
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        env_kwargs["max_episode_steps"] = int(max_episode_steps)
    if render_mode is None:
        env_kwargs.pop("render_mode", None)
    else:
        env_kwargs["render_mode"] = render_mode
    if obstacle_half_extents is not None:
        env_kwargs["pg3d_obstacle_half_extents"] = obstacle_half_extents
        env_kwargs["pg3d_obstacle_family"] = obstacle_family
    return env_kwargs


def _embodied_obstacle_half_extents(
    args: argparse.Namespace,
) -> tuple[float, float, float] | None:
    if not args.embody_obstacle:
        return None
    values = (
        np.asarray(args.avoid_box_half_extents, dtype=np.float32)
        if args.avoid_box_half_extents is not None
        else np.repeat(float(args.avoid_radius), 3).astype(np.float32)
    )
    return tuple(float(value) for value in values)


def _resolve_grounded_embodied_obstacle_height(
    args: argparse.Namespace,
    *,
    specs: list[RolloutSpec],
    zarr_root: Any | None,
) -> None:
    """Resolve actor height before ManiSkill constructs its collision geometry."""
    if (
        not args.embody_obstacle
        or not args.ground_embodied_obstacle
        or args.constraints_dir is not None
    ):
        return
    top_z = args.obstacle_top_z
    if top_z is None:
        if args.constraint_placement != "direct_path" or zarr_root is None:
            raise ValueError(
                "automatic grounded obstacle height requires dataset direct_path "
                "placement; provide --obstacle-top-z for this configuration"
            )
        fractions = [float(value) for value in args.avoid_path_fractions]
        path_heights: list[float] = []
        for spec in specs:
            if spec.dataset_episode_index is None:
                raise ValueError("dataset obstacle-height resolution requires episode indices")
            context = _zarr_episode_context(zarr_root, spec.dataset_episode_index)
            start_z = float(np.asarray(context["tcp_pose"]).reshape(-1)[2])
            target_z = float(np.asarray(context["target_position"]).reshape(3)[2])
            path_heights.extend(start_z + fraction * (target_z - start_z) for fraction in fractions)
        top_z = max(path_heights) + float(args.obstacle_path_height_margin)
    support_z = float(args.obstacle_support_plane_z)
    required_half_height = 0.5 * (float(top_z) - support_z)
    dimensions = (
        np.asarray(args.avoid_box_half_extents, dtype=np.float32).copy()
        if args.avoid_box_half_extents is not None
        else np.repeat(float(args.avoid_radius), 3).astype(np.float32)
    )
    dimensions[2] = max(float(dimensions[2]), required_half_height)
    args.avoid_box_half_extents = dimensions.astype(float).tolist()
    args.resolved_obstacle_top_z = support_z + 2.0 * float(dimensions[2])


def _embodied_obstacle_reset_options(
    constraints: list[AvoidRegion],
) -> dict[str, list[float] | float]:
    cabinet_shelf = next(
        (
            constraint
            for constraint in constraints
            if constraint.name.endswith("/cabinet_shelf")
            and isinstance(constraint.region, BoxRegion)
        ),
        None,
    )
    if cabinet_shelf is not None:
        return {
            "pg3d_obstacle_center": cabinet_shelf.region.center.astype(float).tolist(),
            "pg3d_obstacle_yaw": float(cabinet_shelf.region.yaw),
        }
    u_left = next(
        (
            constraint
            for constraint in constraints
            if constraint.name.endswith("/u_shape_left_side")
            and isinstance(constraint.region, BoxRegion)
        ),
        None,
    )
    u_right = next(
        (
            constraint
            for constraint in constraints
            if constraint.name.endswith("/u_shape_right_side")
            and isinstance(constraint.region, BoxRegion)
        ),
        None,
    )
    if u_left is not None and u_right is not None:
        root_center = 0.5 * (u_left.region.center + u_right.region.center)
        return {
            "pg3d_obstacle_center": root_center.astype(float).tolist(),
            "pg3d_obstacle_yaw": float(u_left.region.yaw),
        }
    if len(constraints) != 1 or not isinstance(constraints[0].region, (BoxRegion, CylinderRegion)):
        raise ValueError(
            "embodied obstacle evaluation requires exactly one primitive or one supported "
            "composite constraint set"
        )
    yaw = float(constraints[0].region.yaw) if isinstance(constraints[0].region, BoxRegion) else 0.0
    return {
        "pg3d_obstacle_center": constraints[0].region.center.astype(float).tolist(),
        "pg3d_obstacle_yaw": yaw,
    }


def _constraint_bottom_z(constraints: list[AvoidRegion]) -> float:
    return float(
        min(
            constraint.region.center[2]
            - (
                constraint.region.half_extents[2]
                if isinstance(constraint.region, BoxRegion)
                else constraint.region.half_length
            )
            for constraint in constraints
            if isinstance(constraint.region, (BoxRegion, CylinderRegion))
        )
    )


def _constraint_top_z(constraints: list[AvoidRegion]) -> float:
    return float(
        max(
            constraint.region.center[2]
            + (
                constraint.region.half_extents[2]
                if isinstance(constraint.region, BoxRegion)
                else constraint.region.half_length
            )
            for constraint in constraints
            if isinstance(constraint.region, (BoxRegion, CylinderRegion))
        )
    )


def _validate_embodied_obstacle_geometry(
    env: Any,
    constraints: list[AvoidRegion],
) -> None:
    reset_options = _embodied_obstacle_reset_options(constraints)
    unwrapped = getattr(env, "unwrapped", env)
    configured = getattr(unwrapped, "pg3d_obstacle_half_extents", None)
    if configured is None:
        raise ValueError("control environment has no configured pg3d obstacle actor")
    family = getattr(unwrapped, "pg3d_obstacle_family", None)
    if family in {"cabinet", "u_shape"}:
        root_center = np.asarray(reset_options["pg3d_obstacle_center"], dtype=np.float32)
        root_yaw = float(reset_options["pg3d_obstacle_yaw"])
        expected_regions = []
        configured_dimensions = np.asarray(configured, dtype=np.float32)
        components = (
            scaled_cabinet_components(float(configured_dimensions[2]))
            if family == "cabinet"
            else u_shape_components(configured_dimensions)
        )
        for component in components:
            center, yaw = transform_box_component(component, center=root_center, yaw=root_yaw)
            expected_regions.append(
                BoxRegion(
                    center=center,
                    half_extents=component.half_extents,
                    yaw=yaw,
                )
            )
        actual_regions = [constraint.region for constraint in constraints]
        if len(actual_regions) != len(expected_regions) or any(
            not isinstance(actual, BoxRegion)
            or not np.allclose(actual.center, expected.center, atol=1e-7, rtol=0.0)
            or not np.allclose(actual.half_extents, expected.half_extents, atol=1e-7, rtol=0.0)
            or not np.isclose(actual.yaw, expected.yaw, atol=1e-7, rtol=0.0)
            for actual, expected in zip(actual_regions, expected_regions, strict=False)
        ):
            raise ValueError(f"{family} actor components and serialized constraints differ")
        return
    region = constraints[0].region
    expected = (
        np.asarray(region.half_extents, dtype=np.float32)
        if isinstance(region, BoxRegion)
        else np.asarray([region.radius, region.radius, region.half_length], dtype=np.float32)
    )
    actual = np.asarray(configured, dtype=np.float32)
    if actual.shape != (3,) or not np.allclose(actual, expected, atol=1e-7, rtol=0.0):
        raise ValueError(
            "control actor and serialized constraint half-extents differ: "
            f"actor={actual.tolist()} constraint={expected.tolist()}"
        )


def _policy_obstacle_point_count(
    entry: Entry,
    *,
    goal_marker_points: int,
) -> int:
    mask = np.asarray(entry.get("obstacle_mask", []), dtype=bool).copy()
    if goal_marker_points > 0 and mask.size:
        mask[-min(goal_marker_points, mask.size) :] = False
    return int(np.count_nonzero(mask))


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
    constraints: list[AvoidRegion],
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
            half_yaw = 0.5 * float(region.yaw)
            actors.build_box(
                scene,
                half_sizes=region.half_extents.tolist(),
                color=rgba,
                name=name,
                body_type="kinematic",
                add_collision=False,
                initial_pose=sapien.Pose(
                    p=region.center.tolist(),
                    q=[math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)],
                ),
            )
        elif isinstance(region, CylinderRegion):
            actors.build_cylinder(
                scene,
                radius=float(region.radius),
                half_length=float(region.half_length),
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


def _method_config(
    args: argparse.Namespace,
    *,
    method: str,
    itps_config: ITPSGuidanceConfig,
    score_config: GuidedScoreConfig = _HISTORICAL_SCORE_CONFIG,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "planning_horizon_chunks": int(args.planning_horizon_chunks),
        "execution_horizon_chunks": int(args.execution_horizon_chunks),
        "geometry_mode": str(args.geometry_mode),
        "constraint_target": str(args.constraint_target),
        "score": score_config.to_json(),
    }
    if method in {"rejection", "reranking"}:
        config["k_schedule"] = [int(value) for value in args.k_schedule]
    if method == "itps_reranking":
        config["guided_candidates"] = int(args.guided_candidates)
    if method in {"beam", "itps_beam"}:
        config.update(
            {
                "beam_width": int(args.beam_width),
                "beam_branch_factor": int(args.beam_branch_factor),
                "expansion_formula": "B + (D - 1) * W * B",
            }
        )
    if method not in {"itps", "itps_reranking", "itps_beam"}:
        config["ddim_eta"] = float(args.ddim_eta)
    if method in {"itps", "itps_reranking", "itps_beam"}:
        config["itps"] = itps_config.to_json()
    return config


def _write_artifact_manifest(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    run_id: str,
    checkpoint_path: Path,
    dataset_path: Path,
    git_info: dict[str, Any],
) -> dict[str, Any]:
    """Write and validate links between qualitative artifacts and metric rows."""
    artifacts: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        video = row.get("video")
        rerun = row.get("rerun")
        if video is None and rerun is None:
            continue
        if video is not None and rerun is None:
            raise ValueError(
                f"metrics row {row_index} has MP4 output without a matching Rerun file"
            )
        constraint_path = Path(str(row["constraint_path"]))
        files = {
            "video": _artifact_file_record(Path(str(video))) if video is not None else None,
            "rerun": _artifact_file_record(Path(str(rerun))) if rerun is not None else None,
            "policy_pointcloud_bundle": (
                _artifact_file_record(Path(str(row["policy_pointcloud_bundle"])))
                if row.get("policy_pointcloud_bundle") is not None
                else None
            ),
            "policy_pointcloud_metadata": (
                _artifact_file_record(Path(str(row["policy_pointcloud_metadata"])))
                if row.get("policy_pointcloud_metadata") is not None
                else None
            ),
            "constraint": _artifact_file_record(constraint_path),
        }
        artifacts.append(
            {
                "artifact_id": (f"episode_{int(row['episode']):03d}:{str(row['method'])}"),
                "metrics": {
                    "path": str(path.parent / "metrics.jsonl"),
                    "row_index": row_index,
                    "episode": int(row["episode"]),
                    "method": str(row["method"]),
                },
                "paired_identity": {
                    "simulator_seed": int(row["simulator_seed"]),
                    "policy_seed": int(row["policy_seed"]),
                    "dataset_episode_index": row.get("dataset_episode_index"),
                    "constraint_id": str(row["constraint_id"]),
                },
                "obstacle": {
                    "id": row.get("obstacle_id"),
                    "family": row.get("obstacle_family"),
                    "pose": row.get("obstacle_pose"),
                    "collision_geometry": row.get("obstacle_collision_geometry"),
                },
                "embedded_identity": row.get("embedded_artifact_identity"),
                "files": files,
            }
        )
    manifest = {
        "schema_version": "pg3d.artifact_manifest.v1",
        "run_id": run_id,
        "checkpoint": str(checkpoint_path),
        "dataset": str(dataset_path),
        "git": {
            "commit": git_info.get("commit"),
            "dirty": bool(git_info.get("dirty")),
        },
        "rerun_writer_version": "0.35.0",
        "artifacts": artifacts,
    }
    validate_artifact_manifest(manifest, rows=rows, inspect_content=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def _artifact_file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"required artifact does not exist: {path}")
    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f"required artifact is empty: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "size_bytes": size,
        "sha256": digest.hexdigest(),
    }


def validate_artifact_manifest(
    manifest: dict[str, Any],
    *,
    rows: list[dict[str, Any]],
    inspect_content: bool = True,
) -> None:
    if manifest.get("schema_version") != "pg3d.artifact_manifest.v1":
        raise ValueError("unsupported artifact manifest schema")
    seen: set[tuple[int, str]] = set()
    for artifact in manifest.get("artifacts", []):
        metrics = artifact["metrics"]
        row_index = int(metrics["row_index"])
        if not 0 <= row_index < len(rows):
            raise ValueError(f"artifact references invalid metrics row {row_index}")
        row = rows[row_index]
        identity = (int(row["episode"]), str(row["method"]))
        if identity in seen:
            raise ValueError(f"duplicate artifact entry for {identity}")
        seen.add(identity)
        if metrics["episode"] != identity[0] or metrics["method"] != identity[1]:
            raise ValueError(f"artifact metrics selector disagrees with row {row_index}")
        paired = artifact["paired_identity"]
        for key in ("simulator_seed", "policy_seed", "constraint_id"):
            if paired[key] != row[key]:
                raise ValueError(f"artifact identity {key} disagrees with metrics row {row_index}")
        files = artifact["files"]
        if files.get("video") is not None and files.get("rerun") is None:
            raise ValueError(f"artifact {artifact['artifact_id']} has video without Rerun")
        embedded_identity = artifact.get("embedded_identity")
        if embedded_identity is None:
            raise ValueError(f"artifact {artifact['artifact_id']} is missing embedded identity")
        if embedded_identity != row.get("embedded_artifact_identity"):
            raise ValueError(
                f"artifact {artifact['artifact_id']} embedded identity disagrees "
                "with its metrics row"
            )
        expected_embedded = {
            "method": str(row["method"]),
            "episode": int(row["episode"]),
            "simulator_seed": int(row["simulator_seed"]),
            "policy_seed": int(row["policy_seed"]),
            "constraint_id": str(row["constraint_id"]),
            "dataset_episode_index": row.get("dataset_episode_index"),
        }
        for key, expected in expected_embedded.items():
            if embedded_identity.get(key) != expected:
                raise ValueError(
                    f"artifact {artifact['artifact_id']} embedded {key} disagrees "
                    "with its metrics row"
                )
        if files.get("video") is not None and row.get("video_labels_embedded") is not True:
            raise ValueError(f"artifact {artifact['artifact_id']} video has no embedded labels")
        if files.get("rerun") is not None and row.get("rerun_identity_embedded") is not True:
            raise ValueError(f"artifact {artifact['artifact_id']} RRD has no embedded identity")
        metadata_record = files.get("policy_pointcloud_metadata")
        if files.get("rerun") is not None and metadata_record is None:
            raise ValueError(f"artifact {artifact['artifact_id']} has no RRD metadata sidecar")
        if metadata_record is not None:
            metadata = json.loads(Path(str(metadata_record["path"])).read_text(encoding="utf-8"))
            if metadata.get("recording_identity") != embedded_identity:
                raise ValueError(
                    f"artifact {artifact['artifact_id']} RRD identity disagrees "
                    "with its metrics row"
                )
        for record in files.values():
            if record is None:
                continue
            if len(str(record["sha256"])) != 64 or int(record["size_bytes"]) <= 0:
                raise ValueError(f"invalid file record in {artifact['artifact_id']}")
            actual = _artifact_file_record(Path(str(record["path"])))
            if actual != record:
                raise ValueError(
                    f"artifact file changed after hashing in {artifact['artifact_id']}: "
                    f"{record['path']}"
                )
        if inspect_content:
            artifact["validation"] = {
                "video": (
                    _decode_video_artifact(Path(str(files["video"]["path"])))
                    if files.get("video") is not None
                    else None
                ),
                "rerun": (
                    _open_rerun_artifact(Path(str(files["rerun"]["path"])))
                    if files.get("rerun") is not None
                    else None
                ),
            }


def _decode_video_artifact(path: Path) -> dict[str, int | bool]:
    """Decode every video frame so truncated/corrupt MP4s fail the run."""
    import imageio.v2 as imageio

    frame_count = 0
    width: int | None = None
    height: int | None = None
    reader = imageio.get_reader(path)
    try:
        for frame in reader:
            array = np.asarray(frame)
            if array.ndim != 3 or array.shape[2] not in (3, 4):
                raise ValueError(f"decoded video frame has invalid shape {array.shape}: {path}")
            if width is None:
                height, width = int(array.shape[0]), int(array.shape[1])
            elif array.shape[:2] != (height, width):
                raise ValueError(f"video frame dimensions changed while decoding: {path}")
            frame_count += 1
    finally:
        reader.close()
    if frame_count <= 0 or width is None or height is None:
        raise ValueError(f"video contains no decodable frames: {path}")
    return {
        "decoded": True,
        "frame_count": frame_count,
        "width": width,
        "height": height,
    }


def _open_rerun_artifact(path: Path) -> dict[str, bool]:
    """Parse an RRD with the isolated Rerun 0.35 exporter environment."""
    exporter_python = _rerun35_exporter_python()
    exporter = Path(__file__).with_name("export_policy_pointcloud_rerun35.py")
    result = subprocess.run(
        [str(exporter_python), str(exporter), "--validate", str(path)],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else ""
        raise ValueError(f"Rerun could not open {path}: {detail}")
    return {"opened": True}


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
                "ddim_eta": float(args.ddim_eta),
                "itps": {
                    "scheduler": "ddim",
                    "eta": 0.0,
                    "guide_ratio": float(args.itps_guide_ratio),
                    "mcmc_steps": int(args.itps_mcmc_steps),
                    "energy": str(args.itps_energy),
                    "barrier_temperature": float(args.itps_barrier_temperature),
                },
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
            f"episode/{row['method']}/constraint_satisfied": float(row["constraint_satisfied"]),
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


def _episode_policy_seed(base_seed: int, episode_index: int) -> int:
    """Derive an order-independent policy RNG seed shared by paired methods."""
    if base_seed < 0 or episode_index < 0:
        raise ValueError("base_seed and episode_index must be non-negative")
    payload = f"pg3d-policy-seed:{base_seed}:{episode_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


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
