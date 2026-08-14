from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pg3d.constraints import AvoidRegion, BoxRegion, CylinderRegion
from pg3d.envs.maniskill_adapter.dataset import PointCloudCropConfig
from pg3d.eval import (
    AvoidOverlayConfig,
    EpisodePath,
    NominalPathAvoidConfig,
    TimingRecorder,
    action_discontinuity_metrics,
    candidate_feasibility_fraction,
    concatenate_rollouts,
    constraint_clearance_series,
    constraint_fingerprint,
    constraint_violation_metrics,
    direct_path_avoid_region,
    episode_metric_row,
    load_episode_constraints,
    max_joint_velocity,
    mcnemar_exact,
    min_constraint_clearance,
    nominal_path_avoid_region,
    paired_bootstrap_difference,
    paired_method_comparisons,
    path_satisfies_constraints,
    point_set_constraint_clearance_series,
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
    _build_constraint,
    _dataset_demo_episode,
    _resolve_shared_grounded_geometry,
)
from scripts.build_nominal_path_constraints import (
    parse_args as parse_builder_args,
)
from scripts.eval_constrained_reach import (
    ComputeOperationCounts,
    DP3ChunkPolicyAdapter,
    ITPSGuidanceConfig,
    _annotate_episode_video_frames,
    _artifact_file_record,
    _artifact_selection_summary,
    _build_multichunk_candidates,
    _clearance_safe_candidate_spec,
    _constraint_bottom_z,
    _constraint_source_summary,
    _constraint_top_z,
    _constraints_for_episode,
    _effective_projection_half_extents,
    _embodied_obstacle_half_extents,
    _embodied_obstacle_reset_options,
    _episode_artifact_identity,
    _episode_policy_seed,
    _episode_should_stop,
    _episode_step_limit,
    _finalize_constraints,
    _ground_embodied_region,
    _local_path_points_xy,
    _obs_windows_to_torch,
    _obstacle_contact_source,
    _point_at_arc_fraction_xy,
    _policy_obstacle_point_count,
    _read_episode_indices_file,
    _resolve_grounded_embodied_obstacle_height,
    _robot_obstacle_contact_pairs,
    _seed_torch,
    _select_decision,
    _termination_reason,
    _validate_embodied_obstacle_geometry,
    _validate_precomputed_initial_clearance,
    _write_artifact_manifest,
    validate_artifact_manifest,
)
from scripts.eval_constrained_reach import (
    parse_args as parse_eval_args,
)
from scripts.rollout_dp3_reach_policy import RolloutSpec
from scripts.run_e3_protocol import (
    builder_command as e3_builder_command,
)
from scripts.run_e3_protocol import (
    evaluation_command as e3_evaluation_command,
)
from scripts.run_e3_protocol import load_constraint_manifest as load_e3_constraint_manifest
from scripts.run_e3_protocol import (
    load_protocol as load_e3_protocol,
)


def test_robot_obstacle_contact_pairs_filters_non_robot_contacts() -> None:
    robot_link = SimpleNamespace(name="panda_link7")
    obstacle = SimpleNamespace(name="pg3d_obstacle")
    table = SimpleNamespace(name="table")
    scene = SimpleNamespace(
        get_contacts=lambda: [
            SimpleNamespace(bodies=[robot_link, obstacle]),
            SimpleNamespace(bodies=[table, obstacle]),
        ]
    )
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            scene=scene,
            agent=SimpleNamespace(robot=SimpleNamespace(links=[robot_link])),
        )
    )

    assert _robot_obstacle_contact_pairs(env) == [["panda_link7", "pg3d_obstacle"]]


def test_termination_reason_prioritizes_physical_collision() -> None:
    assert (
        _termination_reason(
            physical_collision=True,
            terminated_or_truncated=True,
            first_success_step=100,
            observed_post_success_steps=16,
            post_success_steps=16,
            steps=150,
            max_steps=150,
        )
        == "physical_obstacle_collision"
    )
    assert (
        _termination_reason(
            physical_collision=False,
            terminated_or_truncated=False,
            first_success_step=None,
            observed_post_success_steps=0,
            post_success_steps=16,
            steps=150,
            max_steps=150,
        )
        == "task_horizon"
    )
    assert (
        _termination_reason(
            physical_collision=False,
            geometric_collision=True,
            terminated_or_truncated=True,
            first_success_step=None,
            observed_post_success_steps=0,
            post_success_steps=16,
            steps=9,
            max_steps=150,
        )
        == "geometric_obstacle_collision"
    )
    assert (
        _obstacle_contact_source(
            physical_collision=True,
            geometric_collision=True,
        )
        == "physx+geometry"
    )


def test_point_at_arc_fraction_xy_ignores_vertical_lift() -> None:
    # Straight in XY (start x=0 -> goal x=0.3) but with a large vertical arch. The
    # placement must follow XY arc length, so the 0.5 point sits at the XY midpoint
    # regardless of how much arc length the lift consumes.
    t = np.linspace(0.0, 1.0, 41, dtype=np.float32)[:, None]
    path = np.concatenate([t * 0.3, np.zeros_like(t), 0.2 + 0.4 * np.sin(np.pi * t)], axis=1)

    point = _point_at_arc_fraction_xy(path.astype(np.float32), fraction=0.5)

    np.testing.assert_allclose(point[:2], np.asarray([0.15, 0.0], dtype=np.float32), atol=1e-3)


def test_local_path_points_xy_restricts_to_window() -> None:
    path = np.stack([np.linspace(0.0, 1.0, 11), np.zeros(11), np.full(11, 0.2)], axis=1).astype(
        np.float32
    )

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


def test_nominal_path_avoid_region_supports_embodied_box_and_cylinder() -> None:
    path = [[0.0, 0.0, 0.2], [0.2, 0.0, 0.2]]
    box = nominal_path_avoid_region(
        path,
        config=NominalPathAvoidConfig(
            shape="box",
            box_half_extents=(0.04, 0.06, 0.08),
            yaw=np.deg2rad(25.0),
            support_plane_z=0.0,
        ),
    )
    cylinder = nominal_path_avoid_region(
        path,
        config=NominalPathAvoidConfig(
            radius=0.05,
            shape="cylinder",
            cylinder_half_length=0.12,
            support_plane_z=0.0,
        ),
    )

    assert isinstance(box.region, BoxRegion)
    np.testing.assert_allclose(box.region.half_extents, [0.04, 0.06, 0.08])
    assert box.region.center[2] == pytest.approx(0.08)
    assert box.region.yaw == pytest.approx(np.deg2rad(25.0))
    assert isinstance(cylinder.region, CylinderRegion)
    assert cylinder.region.radius == pytest.approx(0.05)
    assert cylinder.region.half_length == pytest.approx(0.12)
    assert cylinder.region.center[2] == pytest.approx(0.12)


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


def test_time_indexed_whole_robot_clearance_preserves_violation_duration() -> None:
    constraint = AvoidRegion(
        region=BoxRegion(center=[0.0, 0.0, 0.0], half_extents=[0.1, 0.1, 0.1]),
        tolerance=0.0,
    )
    point_clouds = [
        np.asarray([[0.3, 0.0, 0.0]], dtype=np.float32),
        np.asarray([[0.05, 0.0, 0.0]], dtype=np.float32),
        np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        np.asarray([[0.3, 0.0, 0.0]], dtype=np.float32),
    ]

    clearances = point_set_constraint_clearance_series(point_clouds, [constraint])

    np.testing.assert_allclose(clearances, [0.2, -0.05, -0.1, 0.2], atol=1e-6)
    path = EpisodePath()
    for index in range(4):
        path.append(
            tcp_position=[0.3, 0.0, 0.0],
            q=[float(index)],
            target_distance=1.0,
        )
    row = episode_metric_row(
        method="base",
        episode=0,
        seed=0,
        path=path,
        constraints=[constraint],
        reach_success=False,
        first_success_step=None,
        steps=3,
        replans=1,
        candidate_feasibility_fraction=None,
        robot_clearance_point_clouds=point_clouds,
        control_dt=0.1,
    )

    assert row["constraint_target"] == "robot"
    assert row["violation_steps"] == 2
    assert row["violation_fraction"] == pytest.approx(0.5)
    assert row["integrated_violation"] == pytest.approx(0.015)
    assert row["violation_event_count"] == 1


def test_action_discontinuity_separates_replan_boundaries() -> None:
    metrics = action_discontinuity_metrics(
        np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 2.0]], dtype=np.float32),
        replan_start_indices=[2],
    )

    assert metrics["action_discontinuity_mean"] == pytest.approx(1.5)
    assert metrics["action_discontinuity_max"] == pytest.approx(2.0)
    assert metrics["replan_boundary_discontinuity_mean"] == pytest.approx(2.0)
    assert metrics["replan_boundary_discontinuity_max"] == pytest.approx(2.0)


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


def test_paired_bootstrap_and_mcnemar_use_episode_differences() -> None:
    rows = _paired_stat_rows()

    continuous = paired_bootstrap_difference(
        rows,
        method_a="itps",
        method_b="reranking",
        metric="final_target_distance",
        samples=2_000,
        seed=17,
    )
    repeated = paired_bootstrap_difference(
        rows,
        method_a="itps",
        method_b="reranking",
        metric="final_target_distance",
        samples=2_000,
        seed=17,
    )
    binary = paired_bootstrap_difference(
        rows,
        method_a="itps",
        method_b="reranking",
        metric="stable_combined_success",
        samples=2_000,
        seed=19,
        binary=True,
    )
    exact = mcnemar_exact(
        rows,
        method_a="itps",
        method_b="reranking",
        metric="stable_combined_success",
    )

    assert continuous == repeated
    assert continuous["paired_episodes"] == 4
    assert continuous["mean_difference"] == pytest.approx(-2.5)
    assert continuous["ci_high"] < 0.0
    assert binary["mean_difference"] == pytest.approx(0.75)
    assert exact["a_success_b_failure"] == 3
    assert exact["a_failure_b_success"] == 0
    assert exact["p_value_two_sided"] == pytest.approx(0.25)


def test_paired_method_comparisons_include_primary_and_conditional_metrics() -> None:
    summary = paired_method_comparisons(
        _paired_stat_rows(),
        methods=["reranking", "itps"],
        bootstrap_samples=500,
        bootstrap_seed=23,
    )

    comparison = summary["comparisons"][0]
    assert comparison["comparison_id"] == "itps_minus_reranking"
    assert comparison["primary_metric"]["metric"] == "stable_combined_success"
    assert comparison["primary_metric"]["mean_difference"] == pytest.approx(0.75)
    assert comparison["primary_metric"]["mcnemar_exact"]["p_value_two_sided"] == (
        pytest.approx(0.25)
    )
    conditional = comparison["conditional_on_both_stable_combined_success"]["tcp_path_length"]
    assert conditional["paired_episodes"] == 1
    assert conditional["excluded_condition_pairs"] == 3


def test_paired_bootstrap_rejects_incomplete_method_pair() -> None:
    rows = _paired_stat_rows()[:-1]

    with pytest.raises(ValueError, match="methods do not match protocol"):
        paired_bootstrap_difference(
            rows,
            method_a="itps",
            method_b="reranking",
            metric="final_target_distance",
            samples=10,
        )


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


def test_compute_operation_counts_measure_actual_batches_and_geometry() -> None:
    counts = ComputeOperationCounts()
    denoiser = torch.nn.Linear(3, 2)
    counts.start_denoiser_tracking(denoiser)

    denoiser(torch.ones((4, 3), dtype=torch.float32))
    denoiser(torch.ones((2, 3), dtype=torch.float32))
    counts.stop_denoiser_tracking()
    denoiser(torch.ones((8, 3), dtype=torch.float32))

    counts.record_provider_delta(
        {
            "end_effector_position_queries": 1,
            "end_effector_position_only_queries": 2,
            "eef_geometry_queries": 3,
            "robot_point_cloud_queries": 4,
            "robot_point_cloud_renders": 2,
        },
        {
            "end_effector_position_queries": 3,
            "end_effector_position_only_queries": 7,
            "eef_geometry_queries": 10,
            "robot_point_cloud_queries": 9,
            "robot_point_cloud_renders": 5,
        },
    )
    counts.record_differentiable_fk(torch.zeros((3, 5, 7), dtype=torch.float32))
    counts.record_differentiable_robot_points(torch.zeros((3, 5, 10, 3), dtype=torch.float32))
    row = counts.to_metric_row(replans=2)

    assert row["denoiser_forward_calls"] == 2
    assert row["denoiser_evaluations"] == 6
    assert row["denoiser_evaluations_per_replan"] == pytest.approx(3.0)
    assert row["differentiable_fk_calls"] == 1
    assert row["differentiable_fk_pose_evaluations"] == 15
    assert row["differentiable_robot_point_calls"] == 1
    assert row["differentiable_robot_point_evaluations"] == 150
    assert row["eef_geometry_queries"] == 7
    assert row["robot_point_cloud_queries"] == 5
    assert row["robot_point_cloud_renders"] == 3
    assert row["geometry_evaluations"] == 25
    assert row["peak_gpu_memory_bytes"] is None


def test_compute_operation_counts_reject_decreasing_provider_counter() -> None:
    counts = ComputeOperationCounts()

    with pytest.raises(ValueError, match="decreased"):
        counts.record_provider_delta(
            {"eef_geometry_queries": 2},
            {"eef_geometry_queries": 1},
        )


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


def test_video_requires_corresponding_rerun_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="video requires --rerun"):
        parse_eval_args(
            [
                "--checkpoint",
                str(tmp_path / "policy.pt"),
                "--dataset",
                str(tmp_path / "dataset.zarr"),
                "--output-dir",
                str(tmp_path / "eval"),
                "--video",
            ]
        )


def test_episode_video_annotation_burns_identity_and_outcome() -> None:
    frame = np.full((96, 320, 3), 220, dtype=np.uint8)
    identity = _episode_artifact_identity(
        row={
            "method": "reranking",
            "episode": 3,
            "seed": 44,
            "obstacle_id": "carton:episode_003",
            "obstacle_family": "carton",
            "reach_success": True,
            "stable_goal_reached": True,
            "constraint_satisfied": True,
            "combined_success": True,
            "stable_combined_success": True,
            "physical_collision": False,
            "termination_reason": "stable_success_hold_complete",
            "min_clearance": 0.012,
            "steps": 71,
        },
        base_identity={
            "dataset_episode_index": 9,
            "simulator_seed": 44,
            "policy_seed": 55,
            "constraint_id": "abc",
        },
    )

    annotated = _annotate_episode_video_frames([frame], identity=identity)

    assert len(annotated) == 1
    assert annotated[0].shape[0] > frame.shape[0]
    assert annotated[0].shape[1:] == frame.shape[1:]
    assert annotated[0].dtype == np.uint8
    assert np.array_equal(frame, np.full_like(frame, 220))
    panel_height = annotated[0].shape[0] - frame.shape[0]
    assert not np.array_equal(
        annotated[0][:panel_height],
        np.full_like(annotated[0][:panel_height], 220),
    )
    np.testing.assert_array_equal(annotated[0][panel_height:], frame)
    assert identity["method"] == "reranking"
    assert identity["stable_combined_success"] is True


def test_artifact_manifest_links_nonempty_files_to_metrics_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "scripts.eval_constrained_reach._decode_video_artifact",
        lambda _path: {"decoded": True, "frame_count": 1, "width": 1, "height": 1},
    )
    monkeypatch.setattr(
        "scripts.eval_constrained_reach._open_rerun_artifact",
        lambda _path: {"opened": True},
    )
    video = tmp_path / "videos" / "base" / "episode_000.mp4"
    rerun = tmp_path / "rerun" / "base" / "episode_000.rrd"
    bundle = tmp_path / "rerun" / "base" / "episode_000.policy_input.npz"
    metadata = tmp_path / "rerun" / "base" / "episode_000.policy_input.json"
    constraint = tmp_path / "constraints" / "episode_000.json"
    for path, content in (
        (video, b"mp4"),
        (rerun, b"rrd"),
        (bundle, b"npz"),
        (constraint, b"[]"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    embedded_identity = {
        "method": "base",
        "episode": 0,
        "simulator_seed": 11,
        "policy_seed": 22,
        "constraint_id": "abc",
        "dataset_episode_index": 3,
    }
    metadata.write_text(
        json.dumps({"recording_identity": embedded_identity}),
        encoding="utf-8",
    )
    row = {
        "episode": 0,
        "method": "base",
        "video": str(video),
        "rerun": str(rerun),
        "constraint_path": str(constraint),
        "constraint_id": "abc",
        "simulator_seed": 11,
        "policy_seed": 22,
        "dataset_episode_index": 3,
        "obstacle_id": "box:episode_000",
        "obstacle_family": "box",
        "obstacle_pose": {"center": [0.0, 0.0, 0.5], "yaw": 0.0},
        "obstacle_collision_geometry": [{"type": "box"}],
        "policy_pointcloud_bundle": str(bundle),
        "policy_pointcloud_metadata": str(metadata),
        "video_labels_embedded": True,
        "rerun_identity_embedded": True,
        "embedded_artifact_identity": embedded_identity,
    }
    manifest_path = tmp_path / "artifact_manifest.json"

    manifest = _write_artifact_manifest(
        manifest_path,
        rows=[row],
        run_id="test-run",
        checkpoint_path=tmp_path / "checkpoint.pt",
        dataset_path=tmp_path / "dataset.zarr",
        git_info={"commit": "deadbeef", "dirty": False},
    )

    assert manifest_path.is_file()
    assert manifest["rerun_writer_version"] == "0.35.0"
    assert len(manifest["artifacts"]) == 1
    artifact = manifest["artifacts"][0]
    assert artifact["metrics"]["row_index"] == 0
    assert artifact["paired_identity"]["constraint_id"] == "abc"
    assert artifact["files"]["video"] == _artifact_file_record(video)
    assert artifact["files"]["rerun"] == _artifact_file_record(rerun)
    assert artifact["embedded_identity"] == embedded_identity
    assert artifact["validation"]["video"]["decoded"] is True
    assert artifact["validation"]["rerun"]["opened"] is True


def test_artifact_manifest_rejects_video_without_rerun(tmp_path: Path) -> None:
    video = tmp_path / "episode.mp4"
    constraint = tmp_path / "constraint.json"
    video.write_bytes(b"mp4")
    constraint.write_bytes(b"[]")
    row = {
        "episode": 0,
        "method": "base",
        "video": str(video),
        "rerun": None,
        "constraint_path": str(constraint),
    }

    with pytest.raises(ValueError, match="without a matching Rerun"):
        _write_artifact_manifest(
            tmp_path / "artifact_manifest.json",
            rows=[row],
            run_id="test-run",
            checkpoint_path=tmp_path / "checkpoint.pt",
            dataset_path=tmp_path / "dataset.zarr",
            git_info={},
        )


def test_artifact_manifest_validator_rejects_identity_mismatch() -> None:
    rows = [
        {
            "episode": 0,
            "method": "base",
            "simulator_seed": 1,
            "policy_seed": 2,
            "constraint_id": "expected",
        }
    ]
    manifest = {
        "schema_version": "pg3d.artifact_manifest.v1",
        "artifacts": [
            {
                "artifact_id": "episode_000:base",
                "metrics": {"row_index": 0, "episode": 0, "method": "base"},
                "paired_identity": {
                    "simulator_seed": 1,
                    "policy_seed": 2,
                    "constraint_id": "wrong",
                },
                "files": {},
            }
        ],
    }

    with pytest.raises(ValueError, match="constraint_id"):
        validate_artifact_manifest(manifest, rows=rows)


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


def test_eval_paired_bootstrap_configuration(tmp_path: Path) -> None:
    args = parse_eval_args(
        [
            "--checkpoint",
            str(tmp_path / "policy.pt"),
            "--dataset",
            str(tmp_path / "dataset.zarr"),
            "--output-dir",
            str(tmp_path / "eval"),
            "--paired-bootstrap-samples",
            "1234",
            "--paired-bootstrap-seed",
            "91",
        ]
    )

    assert args.paired_bootstrap_samples == 1234
    assert args.paired_bootstrap_seed == 91

    with pytest.raises(ValueError, match="paired-bootstrap-samples"):
        parse_eval_args(
            [
                "--checkpoint",
                str(tmp_path / "policy.pt"),
                "--dataset",
                str(tmp_path / "dataset.zarr"),
                "--output-dir",
                str(tmp_path / "eval"),
                "--paired-bootstrap-samples",
                "0",
            ]
        )


def test_target_valid_episodes_requires_all_artifacts(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires --artifact-selection all"):
        parse_eval_args(
            [
                "--checkpoint",
                str(tmp_path / "policy.pt"),
                "--dataset",
                str(tmp_path / "dataset.zarr"),
                "--output-dir",
                str(tmp_path / "eval"),
                "--source",
                "dataset",
                "--constraint-placement",
                "candidate_midpath",
                "--embody-obstacle",
                "--target-valid-episodes",
                "10",
            ]
        )


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


def test_precomputed_box_constraint_can_drive_embodied_actor(tmp_path: Path) -> None:
    constraints_dir = tmp_path / "constraints"
    constraint = AvoidRegion(
        region=BoxRegion(
            center=[0.1, 0.0, 0.3],
            half_extents=[0.04, 0.06, 0.08],
            yaw=np.deg2rad(25.0),
        ),
        target="eef",
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
            "--constraints-dir",
            str(constraints_dir),
            "--embody-obstacle",
            "--avoid-shape",
            "box",
            "--avoid-box-half-extents",
            "0.04",
            "0.06",
            "0.08",
        ]
    )
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            pg3d_obstacle_half_extents=(0.04, 0.06, 0.08),
            pg3d_obstacle_family="box",
        )
    )

    loaded = _constraints_for_episode(
        env,
        spec=RolloutSpec(output_index=0, seed=1, source="dataset"),
        crop_config=PointCloudCropConfig(num_points=4),
        args=args,
    )

    assert isinstance(loaded[0].region, BoxRegion)
    assert loaded[0].region.yaw == pytest.approx(np.deg2rad(25.0))

    env.unwrapped.pg3d_obstacle_half_extents = (0.04, 0.06, 0.09)
    with pytest.raises(ValueError, match="half-extents differ"):
        _constraints_for_episode(
            env,
            spec=RolloutSpec(output_index=0, seed=1, source="dataset"),
            crop_config=PointCloudCropConfig(num_points=4),
            args=args,
        )


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
    assert args.path_source == "policy_success"
    assert args.avoid_radius == pytest.approx(0.03)
    assert args.avoid_shape == "sphere"
    assert args.path_fraction == pytest.approx(0.5)
    assert args.avoid_clearance_scale == pytest.approx(0.05)
    assert args.min_successes == 15


def test_eval_avoid_clearance_scale_parses_and_validates(tmp_path: Path) -> None:
    common = [
        "--checkpoint",
        str(tmp_path / "policy.pt"),
        "--dataset",
        str(tmp_path / "dataset.zarr"),
        "--output-dir",
        str(tmp_path / "eval"),
    ]

    args = parse_eval_args([*common, "--avoid-clearance-scale", "0.08"])

    assert args.avoid_clearance_scale == pytest.approx(0.08)
    assert _constraint_source_summary(args)["avoid_clearance_scale"] == pytest.approx(0.08)
    finalized = _finalize_constraints(
        [
            AvoidRegion(
                region=BoxRegion(center=[0.0, 0.0, 0.2], half_extents=[0.1, 0.1, 0.1]),
                clearance_scale=0.01,
            )
        ],
        robot_points=None,
        args=args,
    )
    assert finalized[0].clearance_scale == pytest.approx(0.08)
    with pytest.raises(ValueError, match="avoid-clearance-scale"):
        parse_eval_args([*common, "--avoid-clearance-scale", "-0.01"])


def test_dataset_demo_path_source_loads_complete_selected_episode() -> None:
    root = {
        "meta": {"episode_ends": np.asarray([3, 7], dtype=np.int64)},
        "data": {
            "tcp_pose": np.asarray(
                [
                    [0.0, 0.0, 0.2, 1, 0, 0, 0],
                    [0.1, 0.0, 0.2, 1, 0, 0, 0],
                    [0.2, 0.0, 0.2, 1, 0, 0, 0],
                    [0.0, 0.0, 0.4, 1, 0, 0, 0],
                    [0.1, 0.0, 0.4, 1, 0, 0, 0],
                    [0.2, 0.0, 0.4, 1, 0, 0, 0],
                    [0.3, 0.0, 0.4, 1, 0, 0, 0],
                ],
                dtype=np.float32,
            ),
            "target_position": np.asarray(
                [[0.2, 0.0, 0.2]] * 3 + [[0.3, 0.0, 0.4]] * 4,
                dtype=np.float32,
            ),
            "success": np.asarray(
                [False, False, True, False, False, True, True],
                dtype=bool,
            ),
            "point_cloud": np.zeros((7, 4, 3), dtype=np.float32),
            "robot_mask": np.asarray([[True, False, False, False]] * 7, dtype=bool),
            "point_valid_mask": np.ones((7, 4), dtype=bool),
        },
    }
    spec = RolloutSpec(
        output_index=0,
        seed=123,
        source="dataset",
        dataset_episode_index=1,
    )

    row = _dataset_demo_episode(root, spec=spec)

    assert row["spec"] == spec
    assert row["success"] is True
    assert row["first_success_step"] == 2
    assert row["tcp_positions"].shape == (4, 3)
    assert row["initial_robot_points"].shape == (1, 3)
    np.testing.assert_allclose(row["tcp_positions"][0], [0.0, 0.0, 0.4])


def test_shared_grounded_geometry_covers_highest_path_anchor() -> None:
    args = SimpleNamespace(
        avoid_box_half_extents=[0.055, 0.08, 0.16],
        avoid_cylinder_half_length=None,
        avoid_radius=0.03,
        avoid_shape="box",
        support_plane_z=0.0,
        path_height_margin=0.02,
        path_fraction=0.5,
        initial_robot_clearance_margin=None,
    )
    rows = [
        {
            "tcp_positions": np.asarray(
                [[0.0, 0.0, height], [0.2, 0.0, height]],
                dtype=np.float32,
            ),
            "first_success_step": 1,
        }
        for height in (0.4, 0.6)
    ]

    resolved = _resolve_shared_grounded_geometry(args, rows)

    assert resolved["box_half_extents"] == pytest.approx((0.055, 0.08, 0.31))
    assert resolved["cylinder_half_length"] is None


def test_clearance_safe_builder_keeps_path_intersection() -> None:
    args = SimpleNamespace(
        avoid_radius=0.03,
        path_fraction=0.5,
        avoid_margin=0.0,
        avoid_weight=1.0,
        avoid_clearance_scale=0.05,
        avoid_tolerance=1e-6,
        avoid_shape="box",
        obstacle_yaw_deg=0.0,
        support_plane_z=0.0,
        initial_robot_clearance_margin=0.02,
        path_fraction_search_min=0.2,
        path_fraction_search_max=0.8,
        path_fraction_search_step=0.05,
        anchor_offset_max_fraction=0.9,
        anchor_offset_step_fraction=0.15,
    )
    path = np.stack(
        [
            np.linspace(0.0, 1.0, 101),
            np.zeros(101),
            np.full(101, 0.4),
        ],
        axis=1,
    ).astype(np.float32)
    row = {
        "dataset_episode_index": 7,
        "initial_robot_points": np.asarray([[0.5, 0.0, 0.3]], dtype=np.float32),
    }

    constraint, placement = _build_constraint(
        args,
        row=row,
        tcp_path=path,
        resolved_geometry={
            "box_half_extents": (0.05, 0.05, 0.21),
            "cylinder_half_length": None,
        },
    )

    assert placement["initial_robot_clearance"] >= 0.02
    assert min_constraint_clearance(path, [constraint]) < 0.0


def test_precomputed_initial_clearance_gate_rejects_impossible_episode() -> None:
    constraint = AvoidRegion(
        region=BoxRegion(
            center=np.asarray([0.0, 0.0, 0.2], dtype=np.float32),
            half_extents=np.asarray([0.1, 0.1, 0.2], dtype=np.float32),
        )
    )
    context = {
        "point_cloud": np.asarray(
            [[0.0, 0.0, 0.2], [0.5, 0.5, 0.5]],
            dtype=np.float32,
        ),
        "robot_mask": np.asarray([True, False]),
        "point_valid_mask": np.asarray([True, True]),
    }

    with pytest.raises(ValueError, match="violates initial robot clearance"):
        _validate_precomputed_initial_clearance(
            [constraint],
            zarr_context=context,
            minimum_clearance=0.02,
        )


def _clearance_context(robot_point: list[float]) -> dict[str, np.ndarray]:
    return {
        "point_cloud": np.asarray([robot_point, [0.8, 0.8, 0.8]], dtype=np.float32),
        "robot_mask": np.asarray([True, False]),
        "point_valid_mask": np.asarray([True, True]),
    }


def test_unsafe_grounded_box_is_rejected_after_final_yaw_and_grounding(
    tmp_path: Path,
) -> None:
    args = parse_eval_args(
        [
            "--checkpoint",
            str(tmp_path / "policy.pt"),
            "--dataset",
            str(tmp_path / "dataset.zarr"),
            "--output-dir",
            str(tmp_path / "eval"),
            "--embody-obstacle",
            "--avoid-shape",
            "box",
            "--avoid-box-half-extents",
            "0.1",
            "0.05",
            "0.2",
            "--obstacle-yaw-deg",
            "30",
        ]
    )
    original = AvoidRegion(region=BoxRegion(center=[0.0, 0.0, 0.8], half_extents=[0.1, 0.05, 0.2]))
    finalized = _finalize_constraints(
        [original],
        robot_points=np.asarray([[0.0, 0.0, 0.2]], dtype=np.float32),
        args=args,
    )

    accepted, record = _clearance_safe_candidate_spec(
        RolloutSpec(output_index=4, seed=17, source="dataset", dataset_episode_index=99),
        finalized,
        zarr_context=_clearance_context([0.0, 0.0, 0.2]),
        minimum_clearance=0.02,
        accepted_output_index=0,
    )

    assert accepted is None
    assert record["exclusion_reason"] == "insufficient_initial_robot_clearance"
    assert record["initial_robot_clearance"] < 0.02
    assert finalized[0].region.center == pytest.approx([0.0, 0.0, 0.2])
    assert finalized[0].region.yaw == pytest.approx(np.deg2rad(30.0))


def test_safe_grounded_placement_is_accepted_without_translation(tmp_path: Path) -> None:
    args = parse_eval_args(
        [
            "--checkpoint",
            str(tmp_path / "policy.pt"),
            "--dataset",
            str(tmp_path / "dataset.zarr"),
            "--output-dir",
            str(tmp_path / "eval"),
            "--embody-obstacle",
            "--avoid-shape",
            "box",
            "--avoid-box-half-extents",
            "0.1",
            "0.05",
            "0.2",
            "--obstacle-yaw-deg",
            "20",
        ]
    )
    original = AvoidRegion(
        region=BoxRegion(center=[0.25, -0.1, 0.8], half_extents=[0.1, 0.05, 0.2])
    )
    finalized = _finalize_constraints(
        [original],
        robot_points=np.asarray([[0.8, 0.8, 0.2]], dtype=np.float32),
        args=args,
    )
    center_before_gate = finalized[0].region.center.copy()

    accepted, record = _clearance_safe_candidate_spec(
        RolloutSpec(output_index=6, seed=23, source="dataset", dataset_episode_index=101),
        finalized,
        zarr_context=_clearance_context([0.8, 0.8, 0.2]),
        minimum_clearance=0.02,
        accepted_output_index=0,
    )

    assert accepted is not None
    assert accepted.output_index == 0
    assert record["exclusion_reason"] is None
    assert record["initial_robot_clearance"] >= 0.02
    np.testing.assert_array_equal(finalized[0].region.center, center_before_gate)
    np.testing.assert_allclose(center_before_gate, [0.25, -0.1, 0.2])


def test_clearance_exclusions_are_replaced_with_contiguous_output_indices() -> None:
    constraint = AvoidRegion(region=BoxRegion(center=[0.0, 0.0, 0.2], half_extents=[0.1, 0.1, 0.1]))
    accepted: list[RolloutSpec] = []
    attempts: list[dict[str, object]] = []
    for pool_index in range(12):
        unsafe = pool_index in {0, 3}
        remapped, record = _clearance_safe_candidate_spec(
            RolloutSpec(
                output_index=pool_index,
                seed=100 + pool_index,
                source="dataset",
                dataset_episode_index=200 + pool_index,
            ),
            [constraint],
            zarr_context=_clearance_context([0.11 if unsafe else 0.13, 0.0, 0.2]),
            minimum_clearance=0.02,
            accepted_output_index=len(accepted),
        )
        attempts.append(record)
        if remapped is not None:
            accepted.append(remapped)
        if len(accepted) == 10:
            break

    assert [spec.output_index for spec in accepted] == list(range(10))
    assert [spec.dataset_episode_index for spec in accepted[:3]] == [201, 202, 204]
    assert [record["source_pool_index"] for record in attempts if record["exclusion_reason"]] == [
        0,
        3,
    ]


def test_candidate_midpath_replacement_pool_respects_locked_partitions() -> None:
    split = json.loads(Path("configs/eval/e3_episode_split.json").read_text(encoding="utf-8"))
    pool = _read_episode_indices_file(
        Path("configs/eval/e3_candidate_midpath_pilot_pool_episode_indices.txt")
    )
    pilot = split["pilot"]["dataset_episode_indices"]
    checkpoint = set(split["checkpoint_gate"]["dataset_episode_indices"])
    definitive_test = set(split["test"]["dataset_episode_indices"])

    assert len(pool) == 40
    assert pool[:10] == pilot
    assert len(set(pool)) == len(pool)
    assert set(pool).isdisjoint(checkpoint)
    assert set(pool).isdisjoint(definitive_test)


def test_frozen_candidate_midpath_75cm_suite_is_internally_consistent() -> None:
    root = Path("configs/eval/e3_candidate_midpath_75cm_frozen_v1")
    fixture = json.loads((root / "fixture.json").read_text(encoding="utf-8"))
    split = json.loads(Path("configs/eval/e3_episode_split.json").read_text(encoding="utf-8"))
    episode_indices = _read_episode_indices_file(root / "episode_indices.txt")
    episodes = fixture["episodes"]

    assert fixture["status"] == "frozen"
    assert fixture["definitive_e3_test_suite"] is False
    assert len(episodes) == 10
    assert [episode["output_index"] for episode in episodes] == list(range(10))
    assert [episode["dataset_episode_index"] for episode in episodes] == episode_indices
    assert split["candidate_midpath_75cm_frozen"]["dataset_episode_indices"] == episode_indices
    assert min(episode["initial_robot_clearance_m"] for episode in episodes) >= 0.02
    assert set(episode_indices).isdisjoint(split["checkpoint_gate"]["dataset_episode_indices"])
    assert set(episode_indices).isdisjoint(split["test"]["dataset_episode_indices"])

    expected_half_extents = fixture["obstacle"]["half_extents_m"]
    for episode in episodes:
        expected_center = episode["center"]
        constraint_file = episode["constraint_file"]
        loaded_by_target = {}
        for target in ("eef", "robot"):
            constraints = load_episode_constraints(root / "constraints" / target / constraint_file)
            assert len(constraints) == 1
            constraint = constraints[0]
            assert isinstance(constraint, AvoidRegion)
            assert isinstance(constraint.region, BoxRegion)
            assert constraint.target == target
            assert constraint.clearance_scale == pytest.approx(0.05)
            np.testing.assert_allclose(constraint.region.center, expected_center)
            np.testing.assert_allclose(constraint.region.half_extents, expected_half_extents)
            assert constraint.region.yaw == pytest.approx(0.0)
            assert constraint.region.center[2] - constraint.region.half_extents[2] == pytest.approx(
                fixture["obstacle"]["support_plane_z_m"]
            )
            loaded_by_target[target] = constraint
        np.testing.assert_allclose(
            loaded_by_target["eef"].region.center,
            loaded_by_target["robot"].region.center,
        )


def test_locked_e3_protocol_requires_all_labeled_video_rerun_pairs() -> None:
    config = load_e3_protocol(Path("configs/eval/e3_protocol.json").resolve())

    assert config["expected_test_episodes"] == 50
    assert config["evaluation"]["methods"] == [
        "base",
        "rejection",
        "reranking",
        "itps",
    ]
    assert config["evaluation"]["max_steps"] == 150
    assert config["evaluation"]["robot_clearance_metric"] is True
    assert config["artifacts"]["selection"] == "all"
    assert config["artifacts"]["require_mp4_rerun_pair_per_method_episode"] is True
    assert config["artifacts"]["require_embedded_identity"] is True


def test_locked_e3_commands_resolve_manifest_geometry(tmp_path: Path) -> None:
    config = load_e3_protocol(Path("configs/eval/e3_protocol.json").resolve())
    population = config["populations"]["full_distribution"]
    constraints_output = tmp_path / "constraints"
    evaluation_output = tmp_path / "evaluation"
    manifest = {
        "constraint_config": {
            "resolved_box_half_extents": [0.055, 0.08, 0.321],
        }
    }

    build = e3_builder_command(config, population, constraints_output)
    evaluate = e3_evaluation_command(
        config,
        population,
        constraints_output,
        evaluation_output,
        manifest,
    )

    assert build[build.index("--path-source") + 1] == "dataset_demo"
    assert build[build.index("--min-successes") + 1] == "50"
    geometry_index = evaluate.index("--avoid-box-half-extents")
    assert evaluate[geometry_index + 1 : geometry_index + 4] == [
        "0.055",
        "0.08",
        "0.321",
    ]
    assert "--robot-clearance-metric" in evaluate
    assert "--terminate-on-obstacle-contact" in evaluate
    assert evaluate[evaluate.index("--geometric-contact-threshold") + 1] == "0.0"
    assert evaluate[evaluate.index("--obstacle-support-plane-z") + 1] == "0.0"
    assert evaluate[evaluate.index("--precomputed-initial-clearance-margin") + 1] == "0.02"
    assert "--video" in evaluate
    assert "--rerun" in evaluate
    assert evaluate[evaluate.index("--artifact-selection") + 1] == "all"


def test_locked_e3_manifest_validation_checks_exact_episode_order(
    tmp_path: Path,
) -> None:
    output = tmp_path / "constraints"
    (output / "constraints").mkdir(parents=True)
    expected_indices = [399, 411]
    manifest = {
        "path_source": "dataset_demo",
        "attempted_episodes": 2,
        "selected_episodes": 2,
        "attempts": [
            {"dataset_episode_index": 399},
            {"dataset_episode_index": 411},
        ],
        "selected": [
            {
                "dataset_episode_index": 399,
                "initial_robot_clearance": 0.02,
                "discrete_min_clearance": -0.01,
            },
            {
                "dataset_episode_index": 411,
                "initial_robot_clearance": 0.03,
                "discrete_min_clearance": -0.02,
            },
        ],
        "constraint_config": {
            "resolved_box_half_extents": [0.055, 0.08, 0.321],
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (output / "episode_indices.txt").write_text("399\n411\n", encoding="utf-8")
    for index in range(2):
        (output / "constraints" / f"episode_{index:03d}.json").write_text(
            "{}",
            encoding="utf-8",
        )

    loaded = load_e3_constraint_manifest(
        output,
        expected_path_source="dataset_demo",
        minimum_selected=2,
        expected_episode_indices=expected_indices,
        minimum_initial_clearance=0.02,
    )

    assert loaded == manifest


def test_nominal_path_constraint_builder_accepts_locked_box_protocol(
    tmp_path: Path,
) -> None:
    indices_path = tmp_path / "episodes.txt"
    indices_path.write_text("286\n297\n", encoding="utf-8")

    args = parse_builder_args(
        [
            "--checkpoint",
            str(tmp_path / "policy.pt"),
            "--dataset",
            str(tmp_path / "dataset.zarr"),
            "--output-dir",
            str(tmp_path / "constraints"),
            "--episode-indices-file",
            str(indices_path),
            "--avoid-shape",
            "box",
            "--avoid-box-half-extents",
            "0.04",
            "0.06",
            "0.08",
            "--obstacle-yaw-deg",
            "25",
            "--support-plane-z",
            "0",
        ]
    )

    assert args.episode_indices_file == indices_path
    assert args.avoid_shape == "box"
    assert args.avoid_box_half_extents == pytest.approx([0.04, 0.06, 0.08])
    assert args.obstacle_yaw_deg == pytest.approx(25.0)
    assert args.support_plane_z == pytest.approx(0.0)


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
            "--embody-obstacle",
            "--obstacle-family",
            "carton",
        ]
    )

    assert args.avoid_shape == "box"
    assert args.avoid_box_half_extents == pytest.approx([0.055, 0.08, 0.16])
    assert args.terminate_on_obstacle_contact
    assert _embodied_obstacle_half_extents(args) == pytest.approx((0.055, 0.08, 0.16))


def test_physical_contact_termination_can_be_disabled_for_ablation(tmp_path: Path) -> None:
    args = parse_eval_args(
        [
            "--dataset",
            str(tmp_path / "dataset.zarr"),
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--output-dir",
            str(tmp_path / "output"),
            "--no-terminate-on-obstacle-contact",
        ]
    )

    assert not args.terminate_on_obstacle_contact


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
    assert _embodied_obstacle_half_extents(args) == pytest.approx((0.055, 0.055, 0.12))


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
    back = next(constraint for constraint in constraints if constraint.name == "root/cabinet_back")
    assert back.region.center[:2] == pytest.approx([0.1, -0.2])
    assert reset["pg3d_obstacle_center"][2] == pytest.approx(0.2)
    assert reset["pg3d_obstacle_yaw"] == pytest.approx(np.deg2rad(15))
    assert _constraint_bottom_z(constraints) == pytest.approx(0.0, abs=1e-7)
    assert _constraint_top_z(constraints) == pytest.approx(0.4)


def test_grounded_embodied_region_keeps_bottom_on_support_plane() -> None:
    box = BoxRegion(center=[0.1, -0.2, 0.45], half_extents=[0.04, 0.06, 0.23])
    grounded = _ground_embodied_region(box, support_plane_z=0.0)

    assert isinstance(grounded, BoxRegion)
    assert grounded.center == pytest.approx([0.1, -0.2, 0.23])
    assert grounded.center[2] - grounded.half_extents[2] == pytest.approx(0.0)
    assert grounded.center[2] + grounded.half_extents[2] == pytest.approx(0.46)


def test_grounded_obstacle_height_covers_direct_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
            "carton",
        ]
    )
    monkeypatch.setattr(
        "scripts.eval_constrained_reach._zarr_episode_context",
        lambda _root, _index: {
            "tcp_pose": np.asarray([0.0, 0.0, 0.6, 1.0, 0.0, 0.0, 0.0]),
            "target_position": np.asarray([0.2, 0.1, 0.3]),
        },
    )

    _resolve_grounded_embodied_obstacle_height(
        args,
        specs=[RolloutSpec(output_index=0, seed=1, source="dataset", dataset_episode_index=4)],
        zarr_root=object(),
    )

    direct_path_z = 0.45
    assert args.avoid_box_half_extents[2] == pytest.approx((direct_path_z + 0.02) / 2)
    assert args.resolved_obstacle_top_z == pytest.approx(direct_path_z + 0.02)


def test_policy_obstacle_count_excludes_goal_marker_slots() -> None:
    entry = {"obstacle_mask": np.asarray([True, False, True, True, True], dtype=bool)}

    assert _policy_obstacle_point_count(entry, goal_marker_points=2) == 2


def test_embodied_actor_geometry_must_match_serialized_constraint() -> None:
    constraint = AvoidRegion(
        region=BoxRegion(center=[0.1, -0.2, 0.3], half_extents=[0.04, 0.06, 0.08])
    )
    env = SimpleNamespace(unwrapped=SimpleNamespace(pg3d_obstacle_half_extents=(0.04, 0.06, 0.08)))
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


def test_exact_single_chunk_selection_uses_supported_controller_arguments() -> None:
    policy = _FakeDP3Policy(n_action_steps=2, n_obs_steps=2)
    adapter = DP3ChunkPolicyAdapter(
        policy,  # type: ignore[arg-type]
        action_mode="abs_joint",
        device=torch.device("cpu"),
    )

    class _ExactWorldModel:
        def imagine(
            self,
            _observation: object,
            action_chunk: ActionChunk,
        ) -> ImaginedRollout:
            return _rollout(action_chunk.actions.tolist())

    decision = _select_decision(
        method="reranking",
        adapter=adapter,
        world_model=_ExactWorldModel(),  # type: ignore[arg-type]
        provider=_FakeFastProvider(),  # type: ignore[arg-type]
        current_entry=_entry(),
        obs_window=_window(),
        scene=scene_context_for_constraints(
            target_position=[1.0, 0.0, 0.2],
            constraints=[],
        ),
        constraints=[],
        crop_config=PointCloudCropConfig(num_points=4),
        goal_thresh=0.01,
        planning_horizon_chunks=1,
        geometry_mode="exact",
        k_schedule=(1,),
        match_current_robot_points=False,
        rng=np.random.default_rng(0),
        timer=TimingRecorder(enabled=False),
        compute_counts=ComputeOperationCounts(),
        itps_config=ITPSGuidanceConfig(),
        itps_collision_model=None,
        constraint_target="eef",
    )

    assert decision.result is not None
    assert decision.selection_reason == "best_feasible"


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
        [np.asarray([float(idx), 0.0, 0.2], dtype=np.float32) for idx in range(horizon)],
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


def _paired_stat_rows() -> list[dict[str, object]]:
    stable = {
        "reranking": [False, False, False, True],
        "itps": [True, True, True, True],
    }
    final_distance = {
        "reranking": [2.0, 4.0, 6.0, 8.0],
        "itps": [1.0, 2.0, 3.0, 4.0],
    }
    path_length = {
        "reranking": [10.0, 11.0, 12.0, 13.0],
        "itps": [8.0, 9.0, 10.0, 11.0],
    }
    rows: list[dict[str, object]] = []
    for episode in range(4):
        for method in ("reranking", "itps"):
            rows.append(
                {
                    "method": method,
                    "episode": episode,
                    "simulator_seed": 100 + episode,
                    "source": "dataset",
                    "dataset_episode_index": episode,
                    "policy_seed": 200 + episode,
                    "constraint_id": f"constraint-{episode}",
                    "checkpoint_id": "checkpoint.pt",
                    "stable_combined_success": stable[method][episode],
                    "combined_success": stable[method][episode],
                    "final_target_distance": final_distance[method][episode],
                    "tcp_path_length": path_length[method][episode],
                }
            )
    return rows


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
