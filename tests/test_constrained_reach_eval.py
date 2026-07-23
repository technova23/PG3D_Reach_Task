from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pg3d.constraints import AvoidRegion, BoxRegion
from pg3d.envs.maniskill_adapter.dataset import PointCloudCropConfig
from pg3d.eval import (
    AvoidOverlayConfig,
    EpisodePath,
    NominalPathAvoidConfig,
    TimingRecorder,
    candidate_feasibility_fraction,
    concatenate_rollouts,
    constraint_clearance_series,
    constraint_fingerprint,
    constraint_violation_metrics,
    direct_path_avoid_region,
    episode_metric_row,
    load_episode_constraints,
    max_joint_velocity,
    min_constraint_clearance,
    nominal_path_avoid_region,
    path_satisfies_constraints,
    progress_series,
    save_episode_constraints,
    scene_context_for_constraints,
    select_artifact_episode_indices,
    should_emit_episode_artifact,
    stable_goal_reached,
    success_rate_ci_rows,
    summarize_metrics,
    trajectory_derivative_mse,
    trajectory_path_length,
    validate_paired_episode_rows,
    validate_planning_horizons,
    wilson_interval,
)
from pg3d.world_model import ActionChunk, ImaginedRollout
from scripts.build_nominal_path_constraints import (
    parse_args as parse_builder_args,
)
from scripts.eval_constrained_reach import (
    DP3ChunkPolicyAdapter,
    _artifact_selection_summary,
    _build_multichunk_candidates,
    _constraint_source_summary,
    _constraints_for_episode,
    _effective_projection_half_extents,
    _embodied_obstacle_half_extents,
    _embodied_obstacle_reset_options,
    _episode_policy_seed,
    _episode_should_stop,
    _episode_step_limit,
    _finalize_constraints,
    _local_path_points_xy,
    _obs_windows_to_torch,
    _point_at_arc_fraction_xy,
    _policy_obstacle_point_count,
    _read_episode_indices_file,
    _seed_torch,
    _validate_embodied_obstacle_geometry,
)
from scripts.eval_constrained_reach import (
    parse_args as parse_eval_args,
)
from scripts.rollout_dp3_reach_policy import RolloutSpec


def test_point_at_arc_fraction_xy_ignores_vertical_lift() -> None:
    # Straight in XY (start x=0 -> goal x=0.3) but with a large vertical arch. The
    # placement must follow XY arc length, so the 0.5 point sits at the XY midpoint
    # regardless of how much arc length the lift consumes.
    t = np.linspace(0.0, 1.0, 41, dtype=np.float32)[:, None]
    path = np.concatenate([t * 0.3, np.zeros_like(t), 0.2 + 0.4 * np.sin(np.pi * t)], axis=1)

    point = _point_at_arc_fraction_xy(path.astype(np.float32), fraction=0.5)

    np.testing.assert_allclose(point[:2], np.asarray([0.15, 0.0], dtype=np.float32), atol=1e-3)


def test_local_path_points_xy_restricts_to_window() -> None:
    path = np.stack(
        [np.linspace(0.0, 1.0, 11), np.zeros(11), np.full(11, 0.2)], axis=1
    ).astype(np.float32)

    local = _local_path_points_xy(path, fraction=0.5, window=0.15)

    # Only points within +/-0.15 arc fraction of the midpoint survive (x in [0.35, 0.65]).
    assert local.shape[1] == 2
    assert float(local[:, 0].min()) >= 0.35 - 1e-6
    assert float(local[:, 0].max()) <= 0.65 + 1e-6


def test_projection_half_extents_local_not_corridor_spanning() -> None:
    # A long, nearly straight reach. Whole-path spread along the travel axis would be
    # huge; the local window must keep the travel-axis half-extent near the floor so
    # the rectangle does not become a corridor-spanning bar.
    t = np.linspace(0.0, 1.0, 51, dtype=np.float32)[:, None]
    paths = [
        np.concatenate([t * 0.4, 0.01 * np.sin(np.pi * t) * sign, np.full_like(t, 0.2)], axis=1)
        for sign in (-1.0, 0.0, 1.0)
    ]
    center_xy = _point_at_arc_fraction_xy(paths[1], fraction=0.5)[:2]

    half = _effective_projection_half_extents(
        center_xy=center_xy,
        paths=[p.astype(np.float32) for p in paths],
        fraction=0.5,
        min_half_extents=np.asarray([0.025, 0.025], dtype=np.float32),
    )

    # Travel-axis (x) extent stays small -- far below the ~0.075 a whole-path
    # 37.5th-percentile would give on a 0.4 m reach.
    assert float(half[0]) <= 0.04
    assert float(half[1]) >= 0.025 - 1e-6


def test_projection_half_extents_floor_at_minimum() -> None:
    # A bundle with essentially no spread floors at the requested minimum half-extents.
    t = np.linspace(0.0, 1.0, 21, dtype=np.float32)[:, None]
    path = np.concatenate([t * 0.3, np.zeros_like(t), np.full_like(t, 0.2)], axis=1)

    half = _effective_projection_half_extents(
        center_xy=_point_at_arc_fraction_xy(path.astype(np.float32), fraction=0.5)[:2],
        paths=[path.astype(np.float32)],
        fraction=0.5,
        min_half_extents=np.asarray([0.03, 0.05], dtype=np.float32),
    )

    np.testing.assert_allclose(half, np.asarray([0.03, 0.05], dtype=np.float32), atol=1e-6)


def test_direct_path_avoid_region_and_json_persistence(tmp_path: Path) -> None:
    constraint = direct_path_avoid_region(
        start_tcp=[0.0, 0.0, 0.2],
        target_position=[0.4, 0.0, 0.2],
        config=AvoidOverlayConfig(radius=0.08),
    )

    np.testing.assert_allclose(constraint.region.center, [0.2, 0.0, 0.2])
    assert constraint.region.radius == pytest.approx(0.08)

    path = tmp_path / "constraints" / "episode_000.json"
    save_episode_constraints(path, [constraint])

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded[0]["type"] == "avoid_region"
    assert loaded[0]["region"]["type"] == "sphere"


def test_direct_path_avoid_region_clamps_radius_for_short_paths() -> None:
    constraint = direct_path_avoid_region(
        start_tcp=[0.0, 0.0, 0.0],
        target_position=[0.1, 0.0, 0.0],
        config=AvoidOverlayConfig(radius=0.08, min_radius=0.02),
    )

    assert constraint.region.radius == pytest.approx(0.045)


def test_nominal_path_avoid_region_uses_arc_length_fraction(tmp_path: Path) -> None:
    tcp_path = np.asarray(
        [
            [0.0, 0.0, 0.2],
            [2.0, 0.0, 0.2],
            [2.0, 2.0, 0.2],
        ],
        dtype=np.float32,
    )

    constraint = nominal_path_avoid_region(
        tcp_path,
        config=NominalPathAvoidConfig(radius=0.03, path_fraction=0.75),
    )

    np.testing.assert_allclose(constraint.region.center, [2.0, 1.0, 0.2])
    assert constraint.region.radius == pytest.approx(0.03)
    assert constraint.name == "nominal_path_avoid_region"

    path = tmp_path / "episode_000.json"
    save_episode_constraints(path, [constraint])
    loaded = load_episode_constraints(path)

    assert len(loaded) == 1
    np.testing.assert_allclose(loaded[0].region.center, [2.0, 1.0, 0.2])


def test_nominal_path_avoid_region_validates_inputs() -> None:
    with pytest.raises(ValueError, match="radius"):
        nominal_path_avoid_region(
            [[0.0, 0.0, 0.0]],
            config=NominalPathAvoidConfig(radius=0.0),
        )
    with pytest.raises(ValueError, match="fraction"):
        nominal_path_avoid_region(
            [[0.0, 0.0, 0.0]],
            config=NominalPathAvoidConfig(path_fraction=1.5),
        )
    with pytest.raises(ValueError, match=r"\[T, 3\]"):
        nominal_path_avoid_region([0.0, 0.0, 0.0])


def test_wilson_interval_bounds_known_center() -> None:
    low, high = wilson_interval(5, 10)

    assert 0.23 < low < 0.24
    assert 0.76 < high < 0.77


def test_constraint_fingerprint_is_stable_across_json_round_trip(tmp_path: Path) -> None:
    constraint = direct_path_avoid_region(
        start_tcp=[0.0, 0.0, 0.0],
        target_position=[1.0, 0.0, 0.0],
    )
    path = tmp_path / "constraint.json"
    save_episode_constraints(path, [constraint])

    assert constraint_fingerprint([constraint]) == constraint_fingerprint(
        load_episode_constraints(path)
    )


def test_episode_policy_seed_is_order_independent_and_episode_specific() -> None:
    assert _episode_policy_seed(7, 3) == _episode_policy_seed(7, 3)
    assert _episode_policy_seed(7, 3) != _episode_policy_seed(7, 4)
    with pytest.raises(ValueError, match="non-negative"):
        _episode_policy_seed(-1, 0)


def test_success_termination_does_not_prevent_hold_collection() -> None:
    assert not _episode_should_stop(terminated=True, truncated=False, success=True)
    assert _episode_should_stop(terminated=True, truncated=False, success=False)
    assert _episode_should_stop(terminated=False, truncated=True, success=True)


def test_post_success_hold_does_not_reduce_nominal_task_horizon() -> None:
    assert (
        _episode_step_limit(
            max_task_steps=80,
            post_success_steps=16,
            first_success_step=None,
        )
        == 80
    )
    assert (
        _episode_step_limit(
            max_task_steps=80,
            post_success_steps=16,
            first_success_step=80,
        )
        == 96
    )


def test_validate_paired_episode_rows_checks_shared_protocol_identity() -> None:
    shared = {
        "episode": 0,
        "simulator_seed": 11,
        "source": "dataset",
        "dataset_episode_index": 4,
        "policy_seed": 22,
        "constraint_id": "constraint",
        "checkpoint_id": "checkpoint",
    }
    rows = [
        {**shared, "method": "base"},
        {**shared, "method": "reranking"},
        {**shared, "method": "itps"},
    ]

    validate_paired_episode_rows(rows, methods=["base", "reranking", "itps"])
    rows[-1] = {**rows[-1], "constraint_id": "different"}
    with pytest.raises(ValueError, match="mismatched constraint_id"):
        validate_paired_episode_rows(rows, methods=["base", "reranking", "itps"])


def test_episode_metric_row_computes_clearance_and_combined_success() -> None:
    constraint = direct_path_avoid_region(
        start_tcp=[0.0, 0.0, 0.0],
        target_position=[1.0, 0.0, 0.0],
        config=AvoidOverlayConfig(radius=0.1),
    )
    path = EpisodePath()
    path.append(tcp_position=[0.0, 0.2, 0.0], q=[0.0, 0.0], target_distance=1.0)
    path.append(tcp_position=[0.5, 0.2, 0.0], q=[0.1, 0.0], target_distance=0.5)
    path.append(tcp_position=[1.0, 0.2, 0.0], q=[0.2, 0.0], target_distance=0.0)

    row = episode_metric_row(
        method="reranking",
        episode=0,
        seed=100,
        path=path,
        constraints=[constraint],
        reach_success=True,
        first_success_step=2,
        steps=2,
        replans=1,
        candidate_feasibility_fraction=0.5,
    )

    assert row["reach_success"] is True
    assert row["constraint_satisfied"] is True
    assert row["combined_success"] is True
    assert row["final_target_distance"] == pytest.approx(0.0)
    assert row["min_clearance"] == pytest.approx(0.1)
    assert row["candidate_feasibility_fraction"] == pytest.approx(0.5)


def test_stable_goal_requires_complete_post_success_hold() -> None:
    distances = [0.5, 0.02, 0.01, 0.02]

    assert stable_goal_reached(
        distances,
        first_success_step=1,
        goal_threshold=0.025,
        hold_steps=2,
    )
    assert not stable_goal_reached(
        distances[:3],
        first_success_step=1,
        goal_threshold=0.025,
        hold_steps=2,
    )
    assert not stable_goal_reached(
        [0.5, 0.02, 0.03, 0.01],
        first_success_step=1,
        goal_threshold=0.025,
        hold_steps=2,
    )


def test_clearance_series_and_violation_metrics_preserve_events() -> None:
    constraint = direct_path_avoid_region(
        start_tcp=[0.0, 0.0, 0.0],
        target_position=[1.0, 0.0, 0.0],
        config=AvoidOverlayConfig(radius=0.1, tolerance=0.0),
    )
    path = np.asarray(
        [
            [0.0, 0.2, 0.0],
            [0.5, 0.05, 0.0],
            [0.5, 0.2, 0.0],
            [0.5, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    clearances = constraint_clearance_series(path, [constraint])
    metrics = constraint_violation_metrics(path, [constraint], dt=0.1)

    np.testing.assert_allclose(clearances, [0.4385165, -0.05, 0.1, -0.1], atol=1e-6)
    assert metrics["max_violation_depth"] == pytest.approx(0.1)
    assert metrics["violation_steps"] == 2
    assert metrics["violation_fraction"] == pytest.approx(0.5)
    assert metrics["integrated_violation"] == pytest.approx(0.015)
    assert metrics["violation_event_count"] == 2


def test_physical_trajectory_metrics_use_control_dt() -> None:
    trajectory = np.asarray([[0.0], [0.1], [0.3], [0.6]], dtype=np.float32)

    assert trajectory_path_length(trajectory) == pytest.approx(0.6)
    assert trajectory_derivative_mse(trajectory, order=2, dt=0.1) == pytest.approx(100.0)
    assert trajectory_derivative_mse(trajectory, order=3, dt=0.1) == pytest.approx(0.0, abs=1e-3)
    assert max_joint_velocity(trajectory, dt=0.1) == pytest.approx(3.0)


def test_episode_metric_row_reports_stable_physical_metrics() -> None:
    path = EpisodePath()
    for idx, distance in enumerate([0.5, 0.02, 0.01, 0.015]):
        path.append(
            tcp_position=[0.1 * idx, 0.0, 0.0],
            q=[0.1 * idx, 0.0],
            target_distance=distance,
        )

    row = episode_metric_row(
        method="base",
        episode=0,
        seed=0,
        path=path,
        constraints=[],
        reach_success=True,
        first_success_step=1,
        steps=3,
        replans=1,
        candidate_feasibility_fraction=None,
        goal_threshold=0.025,
        hold_steps=2,
        control_dt=0.1,
        action_selection_times=[0.1, 0.2, 0.3],
    )

    assert row["stable_goal_reached"] is True
    assert row["stable_combined_success"] is True
    assert row["tcp_path_length"] == pytest.approx(0.3)
    assert row["joint_path_length"] == pytest.approx(0.3)
    assert row["max_joint_velocity"] == pytest.approx(1.0)
    assert row["violation_steps_tcp"] == 0
    assert row["action_selection_time_total"] == pytest.approx(0.6)
    assert row["action_selection_time_median"] == pytest.approx(0.2)
    assert row["action_selection_time_p90"] == pytest.approx(0.28)
    assert row["action_selection_time_p95"] == pytest.approx(0.29)


def test_constraint_satisfaction_fails_for_path_inside_sphere() -> None:
    constraint = direct_path_avoid_region(
        start_tcp=[0.0, 0.0, 0.0],
        target_position=[1.0, 0.0, 0.0],
        config=AvoidOverlayConfig(radius=0.1),
    )
    path = np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float32)

    assert min_constraint_clearance(path, [constraint]) < 0.0
    assert not path_satisfies_constraints(path, [constraint])


def test_validate_planning_horizons() -> None:
    validate_planning_horizons(planning_horizon_chunks=2, execution_horizon_chunks=1)

    with pytest.raises(ValueError, match="planning_horizon_chunks"):
        validate_planning_horizons(planning_horizon_chunks=0, execution_horizon_chunks=1)
    with pytest.raises(ValueError, match="<="):
        validate_planning_horizons(planning_horizon_chunks=1, execution_horizon_chunks=2)


def test_concatenate_rollouts_combines_multichunk_candidate() -> None:
    first = _rollout([[0.1] * 7, [0.2] * 7])
    second = _rollout([[0.3] * 7])

    combined = concatenate_rollouts([first, second], metadata={"candidate": 1})

    assert combined.action_chunk.horizon == 3
    assert combined.q.shape == (3, 9)
    assert combined.eef_path.shape == (3, 3)
    assert len(combined.scene_point_clouds) == 3
    assert combined.metadata["planning_horizon_chunks"] == 2
    assert combined.metadata["candidate"] == 1


def test_summarize_metrics_uses_stable_schema() -> None:
    rows = []
    for idx, success in enumerate([True, False]):
        path = EpisodePath()
        path.append(tcp_position=[0.0, 0.2, 0.0], q=[0.0, 0.0], target_distance=1.0)
        path.append(tcp_position=[1.0, 0.2, 0.0], q=[0.1, 0.0], target_distance=0.1)
        rows.append(
            episode_metric_row(
                method="base",
                episode=idx,
                seed=idx,
                path=path,
                constraints=[],
                reach_success=success,
                first_success_step=1 if success else None,
                steps=1,
                replans=1,
                candidate_feasibility_fraction=None,
            )
        )

    summary = summarize_metrics(rows)

    assert summary["base"]["episodes"] == 2
    assert summary["base"]["reach_success_rate"] == pytest.approx(0.5)
    assert "combined_success_wilson_low" in summary["base"]
    assert "final_target_distance_mean" in summary["base"]


def test_success_rate_ci_rows_accepts_full_summary() -> None:
    summary = {
        "by_method": {
            "base": {
                "reach_success_rate": 0.25,
                "reach_success_wilson_low": 0.1,
                "reach_success_wilson_high": 0.5,
                "constraint_satisfied_rate": 0.75,
                "constraint_satisfied_wilson_low": 0.5,
                "constraint_satisfied_wilson_high": 0.9,
                "combined_success_rate": 0.2,
                "combined_success_wilson_low": 0.05,
                "combined_success_wilson_high": 0.45,
                "stable_combined_success_rate": 0.1,
                "stable_combined_success_wilson_low": 0.01,
                "stable_combined_success_wilson_high": 0.3,
            }
        }
    }

    rows = success_rate_ci_rows(summary)

    assert [row["metric"] for row in rows] == [
        "reach_success",
        "constraint_satisfied",
        "combined_success",
        "stable_combined_success",
    ]
    assert rows[0]["method"] == "base"
    assert rows[0]["err_low"] == pytest.approx(0.15)
    assert rows[0]["err_high"] == pytest.approx(0.25)


def test_candidate_feasibility_fraction_validates_counts() -> None:
    assert candidate_feasibility_fraction(1, 4) == pytest.approx(0.25)
    assert candidate_feasibility_fraction(0, 0) is None
    with pytest.raises(ValueError):
        candidate_feasibility_fraction(2, 1)


def test_timing_recorder_aggregates_json_safe_events() -> None:
    recorder = TimingRecorder(enabled=True)

    with recorder.time("policy_sampling", k=16):
        pass
    with recorder.time("policy_sampling", k=32):
        pass

    summary = recorder.summary()
    events = [event.to_json() for event in recorder.events]

    assert summary["policy_sampling"]["count"] == pytest.approx(2.0)
    assert summary["policy_sampling"]["total"] >= 0.0
    assert events[0]["metadata"]["k"] == 16


def test_periodic_artifact_selection_includes_first_and_interval() -> None:
    assert should_emit_episode_artifact(0, 10)
    assert not should_emit_episode_artifact(8, 10)
    assert should_emit_episode_artifact(9, 10)
    with pytest.raises(ValueError):
        should_emit_episode_artifact(0, 0)


def test_artifact_episode_selection_supports_random_periodic_and_all() -> None:
    episodes = list(range(10))

    random_first = select_artifact_episode_indices(
        episodes,
        selection="random",
        count=5,
        seed=123,
        every_episodes=10,
    )
    random_second = select_artifact_episode_indices(
        episodes,
        selection="random",
        count=5,
        seed=123,
        every_episodes=10,
    )
    periodic = select_artifact_episode_indices(
        episodes,
        selection="periodic",
        count=5,
        seed=123,
        every_episodes=4,
    )
    all_episodes = select_artifact_episode_indices(
        episodes,
        selection="all",
        count=5,
        seed=123,
        every_episodes=10,
    )

    assert random_first == random_second
    assert len(random_first) == 5
    assert len(set(random_first)) == 5
    assert periodic == [0, 3, 7]
    assert all_episodes == episodes


def test_artifact_selection_summary_records_episode_indices_and_seeds() -> None:
    specs = [
        RolloutSpec(
            output_index=idx,
            seed=20000 + idx,
            source="dataset",
            dataset_episode_index=idx,
        )
        for idx in range(4)
    ]
    args = type(
        "Args",
        (),
        {
            "artifact_selection": "random",
            "artifact_episode_count": 2,
            "artifact_selection_seed": 123,
        },
    )()

    summary = _artifact_selection_summary(
        specs,
        video_episode_indices={1, 3},
        rerun_episode_indices={3},
        args=args,
    )

    assert summary["selection"] == "random"
    assert [row["dataset_episode_index"] for row in summary["video"]] == [1, 3]
    assert [row["seed"] for row in summary["rerun"]] == [20003]


def test_disabled_artifact_modes_record_empty_selection(tmp_path: Path) -> None:
    args = parse_eval_args(
        [
            "--checkpoint",
            str(tmp_path / "policy.pt"),
            "--dataset",
            str(tmp_path / "dataset.zarr"),
            "--output-dir",
            str(tmp_path / "eval"),
        ]
    )
    specs = [RolloutSpec(output_index=0, seed=1, source="fresh")]

    summary = _artifact_selection_summary(
        specs,
        video_episode_indices=set(),
        rerun_episode_indices=set(),
        args=args,
    )

    assert summary["video"] == []
    assert summary["rerun"] == []


def test_eval_artifact_selection_seed_defaults_to_run_seed(tmp_path: Path) -> None:
    args = parse_eval_args(
        [
            "--checkpoint",
            str(tmp_path / "policy.pt"),
            "--dataset",
            str(tmp_path / "dataset.zarr"),
            "--output-dir",
            str(tmp_path / "eval"),
            "--seed",
            "13",
        ]
    )

    assert args.artifact_selection == "periodic"
    assert args.artifact_episode_count == 5
    assert args.artifact_selection_seed == 13


def test_eval_episode_indices_file_and_precomputed_constraints(tmp_path: Path) -> None:
    indices_path = tmp_path / "episode_indices.txt"
    indices_path.write_text("# selected base-success episodes\n3\n7\n", encoding="utf-8")
    constraints_dir = tmp_path / "constraints"
    constraint = nominal_path_avoid_region(
        [[0.0, 0.0, 0.2], [0.2, 0.0, 0.2]],
        config=NominalPathAvoidConfig(radius=0.03),
    )
    save_episode_constraints(constraints_dir / "episode_000.json", [constraint])
    args = parse_eval_args(
        [
            "--checkpoint",
            str(tmp_path / "policy.pt"),
            "--dataset",
            str(tmp_path / "dataset.zarr"),
            "--output-dir",
            str(tmp_path / "eval"),
            "--source",
            "dataset",
            "--episode-indices-file",
            str(indices_path),
            "--constraints-dir",
            str(constraints_dir),
        ]
    )

    assert _read_episode_indices_file(indices_path) == [3, 7]
    loaded = _constraints_for_episode(
        None,
        spec=RolloutSpec(
            output_index=0,
            seed=20003,
            source="dataset",
            dataset_episode_index=3,
        ),
        crop_config=PointCloudCropConfig(
            bounds=np.asarray([[-1, 1], [-1, 1], [-1, 1]], dtype=np.float32),
            num_points=4,
        ),
        args=args,
    )

    assert args.constraints_dir == constraints_dir
    assert _constraint_source_summary(args)["type"] == "precomputed"
    np.testing.assert_allclose(loaded[0].region.center, [0.1, 0.0, 0.2])


def test_eval_episode_indices_file_requires_dataset_source(tmp_path: Path) -> None:
    indices_path = tmp_path / "episode_indices.txt"
    indices_path.write_text("0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source dataset"):
        parse_eval_args(
            [
                "--checkpoint",
                str(tmp_path / "policy.pt"),
                "--dataset",
                str(tmp_path / "dataset.zarr"),
                "--output-dir",
                str(tmp_path / "eval"),
                "--source",
                "fresh",
                "--episode-indices-file",
                str(indices_path),
            ]
        )


def test_nominal_path_constraint_builder_defaults(tmp_path: Path) -> None:
    args = parse_builder_args(
        [
            "--checkpoint",
            str(tmp_path / "policy.pt"),
            "--dataset",
            str(tmp_path / "dataset.zarr"),
            "--output-dir",
            str(tmp_path / "constraints"),
        ]
    )

    assert args.episodes == 25
    assert args.avoid_radius == pytest.approx(0.03)
    assert args.path_fraction == pytest.approx(0.5)
    assert args.min_successes == 15


def test_eval_constraint_overlay_flags_parse_and_validate(tmp_path: Path) -> None:
    args = parse_eval_args(
        [
            "--checkpoint",
            str(tmp_path / "policy.pt"),
            "--dataset",
            str(tmp_path / "dataset.zarr"),
            "--output-dir",
            str(tmp_path / "eval"),
            "--no-constraint-overlay-video",
            "--constraint-overlay-alpha",
            "0.4",
            "--constraint-overlay-color",
            "0.8",
            "0.2",
            "0.1",
        ]
    )

    assert args.constraint_overlay_video is False
    assert args.constraint_overlay_alpha == pytest.approx(0.4)
    assert args.constraint_overlay_color == [0.8, 0.2, 0.1]

    with pytest.raises(ValueError, match="constraint-overlay-alpha"):
        parse_eval_args(
            [
                "--checkpoint",
                str(tmp_path / "policy.pt"),
                "--dataset",
                str(tmp_path / "dataset.zarr"),
                "--output-dir",
                str(tmp_path / "eval"),
                "--constraint-overlay-alpha",
                "1.5",
            ]
        )


def test_progress_series_tracks_cumulative_metrics() -> None:
    rows = [
        _metric_row(method="base", episode=0, reach=True, constraint=False),
        _metric_row(method="base", episode=1, reach=True, constraint=True),
    ]

    series = progress_series(rows)

    assert series["base"]["reach_success_rate"] == [1.0, 1.0]
    assert series["base"]["constraint_satisfied_rate"] == [0.0, 0.5]
    assert series["base"]["combined_success_rate"] == [0.0, 0.5]


def test_dp3_adapter_batches_multiple_windows() -> None:
    policy = _FakeDP3Policy(n_action_steps=2, n_obs_steps=2)
    adapter = DP3ChunkPolicyAdapter(
        policy,  # type: ignore[arg-type]
        action_mode="abs_joint",
        device=torch.device("cpu"),
        policy_batch_size=2,
        timer=TimingRecorder(enabled=True),
    )

    chunks = adapter.sample_action_chunks_for_windows([_window(), _window(), _window()])

    assert len(chunks) == 3
    assert policy.batch_sizes == [2, 1]
    assert chunks[0].actions.shape == (2, 7)


def test_eval_no_constraints_mode_is_explicit_and_mutually_exclusive(tmp_path: Path) -> None:
    args = parse_eval_args(
        [
            "--dataset",
            str(tmp_path / "dataset.zarr"),
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--output-dir",
            str(tmp_path / "output"),
            "--no-constraints",
        ]
    )

    assert args.no_constraints is True
    with pytest.raises(ValueError, match="mutually exclusive"):
        parse_eval_args(
            [
                "--dataset",
                str(tmp_path / "dataset.zarr"),
                "--checkpoint",
                str(tmp_path / "checkpoint.pt"),
                "--output-dir",
                str(tmp_path / "output"),
                "--no-constraints",
                "--constraints-dir",
                str(tmp_path / "constraints"),
            ]
        )


def test_no_constraints_mode_returns_empty_program(tmp_path: Path) -> None:
    args = parse_eval_args(
        [
            "--dataset",
            str(tmp_path / "dataset.zarr"),
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--output-dir",
            str(tmp_path / "output"),
            "--no-constraints",
        ]
    )
    spec = RolloutSpec(output_index=0, seed=1, source="fresh")

    constraints = _constraints_for_episode(
        None,
        spec=spec,
        crop_config=PointCloudCropConfig(num_points=4),
        args=args,
    )

    assert constraints == []
    assert _constraint_source_summary(args) == {"type": "none"}


def test_embodied_box_uses_generated_constraint_geometry(tmp_path: Path) -> None:
    args = parse_eval_args(
        [
            "--dataset",
            str(tmp_path / "dataset.zarr"),
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--output-dir",
            str(tmp_path / "output"),
            "--avoid-shape",
            "box",
            "--avoid-box-half-extents",
            "0.04",
            "0.06",
            "0.08",
            "--embody-obstacle",
        ]
    )
    constraint = AvoidRegion(
        region=BoxRegion(center=[0.1, -0.2, 0.3], half_extents=[0.04, 0.06, 0.08])
    )

    assert _embodied_obstacle_half_extents(args) == pytest.approx((0.04, 0.06, 0.08))
    reset_options = _embodied_obstacle_reset_options([constraint])
    assert reset_options["pg3d_obstacle_center"] == pytest.approx([0.1, -0.2, 0.3])
    assert reset_options["pg3d_obstacle_yaw"] == 0.0


def test_embodied_obstacle_rejects_unsupported_geometry(tmp_path: Path) -> None:
    common = [
        "--dataset",
        str(tmp_path / "dataset.zarr"),
        "--checkpoint",
        str(tmp_path / "checkpoint.pt"),
        "--output-dir",
        str(tmp_path / "output"),
        "--embody-obstacle",
    ]
    with pytest.raises(ValueError, match="requires a box or cylinder"):
        parse_eval_args(common)
    with pytest.raises(ValueError, match="exactly one avoid region"):
        parse_eval_args(
            [
                *common,
                "--avoid-shape",
                "box",
                "--avoid-path-fractions",
                "0.3",
                "0.7",
            ]
        )


def test_carton_family_has_reproducible_default_geometry(tmp_path: Path) -> None:
    args = parse_eval_args(
        [
            "--dataset",
            str(tmp_path / "dataset.zarr"),
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--output-dir",
            str(tmp_path / "output"),
            "--avoid-shape",
            "box",
            "--embody-obstacle",
            "--obstacle-family",
            "carton",
        ]
    )

    assert args.avoid_box_half_extents == pytest.approx([0.055, 0.08, 0.16])
    assert _embodied_obstacle_half_extents(args) == pytest.approx(
        (0.055, 0.08, 0.16)
    )


def test_cylinder_family_has_reproducible_default_geometry(tmp_path: Path) -> None:
    args = parse_eval_args(
        [
            "--dataset",
            str(tmp_path / "dataset.zarr"),
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--output-dir",
            str(tmp_path / "output"),
            "--embody-obstacle",
            "--obstacle-family",
            "cylinder",
        ]
    )

    assert args.avoid_shape == "cylinder"
    assert _embodied_obstacle_half_extents(args) == pytest.approx(
        (0.055, 0.055, 0.12)
    )


def test_cabinet_family_expands_root_into_component_constraints(tmp_path: Path) -> None:
    args = parse_eval_args(
        [
            "--dataset",
            str(tmp_path / "dataset.zarr"),
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--output-dir",
            str(tmp_path / "output"),
            "--embody-obstacle",
            "--obstacle-family",
            "cabinet",
            "--obstacle-yaw-deg",
            "15",
        ]
    )
    root = AvoidRegion(
        region=BoxRegion(center=[0.1, -0.2, 0.4], half_extents=[0.08, 0.085, 0.2]),
        name="root",
    )

    constraints = _finalize_constraints([root], robot_points=None, args=args)
    reset = _embodied_obstacle_reset_options(constraints)

    assert len(constraints) == 7
    assert {constraint.name for constraint in constraints} == {
        f"root/cabinet_{name}"
        for name in (
            "left_side",
            "right_side",
            "top",
            "bottom",
            "back",
            "shelf",
            "open_door",
        )
    }
    assert reset["pg3d_obstacle_center"] == pytest.approx([0.1, -0.2, 0.4])
    assert reset["pg3d_obstacle_yaw"] == pytest.approx(np.deg2rad(15))


def test_policy_obstacle_count_excludes_goal_marker_slots() -> None:
    entry = {
        "obstacle_mask": np.asarray([True, False, True, True, True], dtype=bool)
    }

    assert _policy_obstacle_point_count(entry, goal_marker_points=2) == 2


def test_embodied_actor_geometry_must_match_serialized_constraint() -> None:
    constraint = AvoidRegion(
        region=BoxRegion(center=[0.1, -0.2, 0.3], half_extents=[0.04, 0.06, 0.08])
    )
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(pg3d_obstacle_half_extents=(0.04, 0.06, 0.08))
    )
    _validate_embodied_obstacle_geometry(env, [constraint])

    env.unwrapped.pg3d_obstacle_half_extents = (0.04, 0.06, 0.09)
    with pytest.raises(ValueError, match="half-extents differ"):
        _validate_embodied_obstacle_geometry(env, [constraint])


def test_constrained_eval_batch_input_inserts_goal_marker_tail_points() -> None:
    batch = _obs_windows_to_torch(
        [_window()],
        device=torch.device("cpu"),
        goal_marker_points=2,
        goal_marker_radius=0.015,
    )

    points = batch["point_cloud"].cpu().numpy()
    expected = np.broadcast_to(
        np.asarray([[1.0, 0.0, 0.2], [1.015, 0.0, 0.2]], dtype=np.float32),
        (2, 2, 3),
    )
    np.testing.assert_allclose(points[0, :, -2:, :], expected)


def test_seed_torch_controls_policy_sampling_rng() -> None:
    _seed_torch(123)
    first = torch.randn(4)
    _seed_torch(123)
    second = torch.randn(4)

    torch.testing.assert_close(first, second)


def test_fast_multichunk_renders_only_feedback_states() -> None:
    policy = _FakeDP3Policy(n_action_steps=2, n_obs_steps=2)
    adapter = DP3ChunkPolicyAdapter(
        policy,  # type: ignore[arg-type]
        action_mode="abs_joint",
        device=torch.device("cpu"),
        policy_batch_size=8,
        timer=TimingRecorder(enabled=True),
    )
    provider = _FakeFastProvider()
    constraint = direct_path_avoid_region(
        start_tcp=[0.0, 0.0, 0.2],
        target_position=[1.0, 0.0, 0.2],
    )

    candidates = _build_multichunk_candidates(
        adapter=adapter,
        world_model=None,  # type: ignore[arg-type]
        provider=provider,  # type: ignore[arg-type]
        current_entry=_entry(),
        obs_window=_window(),
        scene=scene_context_for_constraints(
            target_position=[1.0, 0.0, 0.2],
            constraints=[constraint],
        ),
        constraints=[constraint],
        crop_config=PointCloudCropConfig(
            bounds=np.asarray([[-1, 2], [-1, 1], [0, 1]], dtype=np.float32),
            num_points=4,
        ),
        goal_thresh=0.01,
        planning_horizon_chunks=2,
        geometry_mode="fast",
        attempted_k=3,
        start_index=0,
        rng=np.random.default_rng(0),
        timer=TimingRecorder(enabled=True),
    )

    assert len(candidates) == 3
    assert provider.eef_calls == 12
    assert provider.robot_cloud_calls == 6


def test_eval_helpers_import_without_heavy_runtime_deps() -> None:
    code = """
import importlib
import sys

importlib.import_module("pg3d.eval")
assert "mani_skill" not in sys.modules
assert "sapien" not in sys.modules
assert "gymnasium" not in sys.modules
assert "rerun" not in sys.modules
assert "wandb" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def _rollout(actions: list[list[float]]) -> ImaginedRollout:
    chunk = ActionChunk(
        actions=np.asarray(actions, dtype=np.float32),
        action_mode="abs_joint",
        dt=1.0,
    )
    horizon = chunk.horizon
    q = np.zeros((horizon, 9), dtype=np.float32)
    q[:, :7] = chunk.actions
    eef = np.stack(
        [
            np.asarray([float(idx), 0.0, 0.2], dtype=np.float32)
            for idx in range(horizon)
        ],
        axis=0,
    )
    return ImaginedRollout(
        q=q,
        eef_path=eef,
        robot_point_clouds=[np.zeros((1, 3), dtype=np.float32) for _ in range(horizon)],
        scene_point_clouds=[np.zeros((2, 3), dtype=np.float32) for _ in range(horizon)],
        robot_masks=[np.asarray([True, False], dtype=bool) for _ in range(horizon)],
        action_chunk=chunk,
    )


def _entry() -> dict[str, np.ndarray | bool | float]:
    return {
        "point_cloud": np.asarray(
            [
                [0.0, 0.0, 0.2],
                [0.1, 0.0, 0.2],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "robot_mask": np.asarray([False, True, False, False], dtype=bool),
        "point_valid_mask": np.asarray([True, True, False, False], dtype=bool),
        "agent_pos": np.asarray([0.0] * 7 + [0.04, 0.04], dtype=np.float32),
        "target_position": np.asarray([1.0, 0.0, 0.2], dtype=np.float32),
        "tcp_pose": np.asarray([0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "success": False,
        "final_distance": 1.0,
    }


def _window() -> list[dict[str, np.ndarray | bool | float]]:
    return [_entry(), _entry()]


def _metric_row(
    *,
    method: str,
    episode: int,
    reach: bool,
    constraint: bool,
) -> dict[str, object]:
    return {
        "method": method,
        "episode": episode,
        "seed": episode,
        "reach_success": reach,
        "constraint_satisfied": constraint,
        "combined_success": reach and constraint,
        "final_target_distance": 0.1,
        "min_clearance": 0.01 if constraint else -0.01,
        "candidate_feasibility_fraction": None,
        "fallback_count": 0,
    }


class _FakeDP3Policy:
    def __init__(self, *, n_action_steps: int, n_obs_steps: int) -> None:
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.batch_sizes: list[int] = []

    def predict_action(self, obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch_size = int(obs_dict["point_cloud"].shape[0])
        self.batch_sizes.append(batch_size)
        base = torch.arange(batch_size, dtype=torch.float32).reshape(batch_size, 1, 1)
        action = torch.ones((batch_size, self.n_action_steps, 7), dtype=torch.float32)
        return {"action": action * (base + 0.1)}


class _FakeFastProvider:
    def __init__(self) -> None:
        self.eef_calls = 0
        self.robot_cloud_calls = 0

    def end_effector_position_only(self, q: np.ndarray) -> np.ndarray:
        self.eef_calls += 1
        return np.asarray([q[0], 0.0, 0.2], dtype=np.float32)

    def robot_point_cloud(self, q: np.ndarray) -> np.ndarray:
        self.robot_cloud_calls += 1
        return np.asarray([[q[0], 0.0, 0.2]], dtype=np.float32)
