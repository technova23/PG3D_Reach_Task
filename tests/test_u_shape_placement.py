from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pg3d.eval.u_shape_placement import (
    UShapeSearchConfig,
    ValidatedUShapeCandidate,
    box_derived_u_shape_candidate,
    enumerate_u_shape_candidates,
    enumerate_u_shape_trajectory_candidates,
    evaluate_u_shape_geometry,
    half_extents_from_mouth_and_depth,
    opening_toward_point_yaw,
    path_aligned_yaw,
    u_shape_constraints,
    validated_candidate_sort_key,
)
from scripts.run_conventional_u_shape_planner import _position_goal_orientations


def test_box_derived_u_preserves_envelope_and_faces_start() -> None:
    center = np.asarray([0.1, 0.0, 0.375], dtype=np.float32)
    start = np.asarray([0.1, -0.4, 0.3], dtype=np.float32)
    half_extents = np.asarray([0.055, 0.08, 0.375], dtype=np.float32)

    candidate = box_derived_u_shape_candidate(
        center,
        half_extents,
        start=start,
        first_chunk_distance=0.05,
    )

    np.testing.assert_allclose(candidate.root_center, center)
    np.testing.assert_allclose(candidate.half_extents, half_extents)
    assert np.isclose(candidate.yaw, 0.0)
    assert np.isclose(candidate.mouth_width, (10.0 / 7.0) * half_extents[0])
    assert np.isclose(opening_toward_point_yaw(center, start), 0.0)


def test_frozen_source_has_expected_episode_order_and_unique_paired_seeds() -> None:
    fixture = json.loads(
        Path("configs/eval/e3_candidate_midpath_75cm_frozen_v1/fixture.json").read_text(
            encoding="utf-8"
        )
    )
    episodes = fixture["episodes"]

    assert [episode["dataset_episode_index"] for episode in episodes] == [
        305,
        317,
        974,
        986,
        1010,
        1034,
        1069,
        1117,
        1129,
        1138,
    ]
    assert [episode["output_index"] for episode in episodes] == list(range(10))
    assert len({episode["simulator_seed"] for episode in episodes}) == 10
    assert len({episode["policy_seed"] for episode in episodes}) == 10


def test_box_derived_v1_object_geometry_is_finalized() -> None:
    guidance = json.loads(
        Path("configs/eval/e10_u_shape_box_derived_guidance_v1.json").read_text(encoding="utf-8")
    )
    fixture = json.loads(
        Path("configs/eval/e10_u_shape_box_derived_review_v1/fixture.json").read_text(
            encoding="utf-8"
        )
    )

    assert guidance["object_geometry_status"] == "finalized"
    assert fixture["status"] == "object_geometry_finalized"
    assert fixture["object_geometry"]["status"] == "finalized"
    assert fixture["object_geometry"]["finalized_on"] == "2026-08-19"
    assert fixture["episode_count"] == 10
    assert [episode["output_index"] for episode in fixture["episodes"]] == list(range(10))


def test_u_shape_dimensions_invert_scaled_component_proportions() -> None:
    half_extents = half_extents_from_mouth_and_depth(
        0.20,
        0.26,
        full_height=0.75,
    )

    np.testing.assert_allclose(half_extents, [0.14, 0.15, 0.375], atol=1e-7)


def test_path_aligned_candidate_opens_toward_start() -> None:
    start = np.asarray([0.0, -0.30, 0.30], dtype=np.float32)
    goal = np.asarray([0.0, 0.30, 0.30], dtype=np.float32)
    config = UShapeSearchConfig()

    assert np.isclose(path_aligned_yaw(start, goal), 0.0)
    candidates = enumerate_u_shape_candidates(start, goal, 0.08, config=config)
    candidate = next(
        item
        for item in candidates
        if item.mouth_width == 0.20
        and item.mouth_chunk_fraction == 0.5
        and item.back_chunk_ratio == 2.5
        and item.lateral_offset == 0.0
    )

    assert np.isclose(candidate.root_center[1] - candidate.half_extents[1], -0.26)
    assert np.isclose(candidate.mouth_distance, 0.04)
    assert np.isclose(candidate.back_distance, 0.20)


def test_trajectory_candidates_use_actual_chunk_boundary_progress() -> None:
    start = np.asarray([0.0, -0.30, 0.30], dtype=np.float32)
    goal = np.asarray([0.0, 0.50, 0.30], dtype=np.float32)
    path = np.linspace(start, goal, 81, dtype=np.float32)
    config = UShapeSearchConfig(cavity_depths=(0.08,))

    candidates = enumerate_u_shape_trajectory_candidates(
        start,
        goal,
        path,
        config=config,
    )
    candidate = next(
        item
        for item in candidates
        if item.mouth_width == 0.20 and item.back_chunk_ratio == 2.5 and item.lateral_offset == 0.0
    )

    assert np.isclose(candidate.back_distance, 0.20)
    assert np.isclose(candidate.mouth_distance, 0.12)
    assert np.isclose(candidate.root_center[1] - candidate.half_extents[1], -0.18)


def test_trajectory_fallback_advances_back_and_keeps_all_widths() -> None:
    start = np.asarray([0.0, -0.30, 0.30], dtype=np.float32)
    goal = np.asarray([0.0, 0.90, 0.30], dtype=np.float32)
    path = np.linspace(start, goal, 101, dtype=np.float32)

    candidates = enumerate_u_shape_trajectory_candidates(
        start,
        goal,
        path,
        config=UShapeSearchConfig(cavity_depths=(0.08,)),
        fallback=True,
    )

    assert {item.back_chunk_ratio for item in candidates} >= {4.0, 8.0, 12.0}
    assert {item.mouth_width for item in candidates} == {0.18, 0.20, 0.22, 0.24, 0.26, 0.28}


def test_geometry_gate_accepts_clear_path_intersecting_back() -> None:
    start = np.asarray([0.0, -0.30, 0.30], dtype=np.float32)
    goal = np.asarray([0.0, 0.30, 0.30], dtype=np.float32)
    config = UShapeSearchConfig()
    candidate = next(
        item
        for item in enumerate_u_shape_candidates(start, goal, 0.08, config=config)
        if item.mouth_width == 0.20
        and item.mouth_chunk_fraction == 0.5
        and item.back_chunk_ratio == 2.5
        and item.lateral_offset == 0.0
    )
    path = np.linspace(start, goal, 101, dtype=np.float32)

    check = evaluate_u_shape_geometry(
        candidate,
        start_robot_points=start.reshape(1, 3),
        goal_robot_points=goal.reshape(1, 3),
        nominal_tcp_path=path,
        goal=goal,
        config=config,
    )

    assert check.accepted
    assert check.initial_clearance >= 0.03
    assert check.goal_clearance >= 0.03
    assert check.direct_back_clearance <= -0.005


def test_direct_back_gate_does_not_depend_on_curved_rollout_samples() -> None:
    start = np.asarray([0.0, -0.30, 0.30], dtype=np.float32)
    goal = np.asarray([0.0, 0.30, 0.30], dtype=np.float32)
    config = UShapeSearchConfig()
    candidate = next(
        item
        for item in enumerate_u_shape_candidates(start, goal, 0.08, config=config)
        if item.mouth_width == 0.20
        and item.mouth_chunk_fraction == 0.5
        and item.back_chunk_ratio == 2.5
        and item.lateral_offset == 0.0
    )
    curved_nominal = np.asarray(
        [start, [-0.20, -0.05, 0.30], [-0.20, 0.20, 0.30], goal],
        dtype=np.float32,
    )

    check = evaluate_u_shape_geometry(
        candidate,
        start_robot_points=start.reshape(1, 3),
        goal_robot_points=goal.reshape(1, 3),
        nominal_tcp_path=curved_nominal,
        goal=goal,
        config=config,
    )

    assert check.accepted
    assert check.direct_back_clearance <= -0.005


def test_constraints_share_geometry_and_change_only_target() -> None:
    start = np.asarray([0.0, -0.30, 0.30], dtype=np.float32)
    goal = np.asarray([0.0, 0.30, 0.30], dtype=np.float32)
    candidate = enumerate_u_shape_candidates(
        start,
        goal,
        0.08,
        config=UShapeSearchConfig(),
    )[0]

    eef = u_shape_constraints(candidate, target="eef")
    robot = u_shape_constraints(candidate, target="robot")

    assert [item.region.to_json() for item in eef] == [item.region.to_json() for item in robot]
    assert {item.target for item in eef} == {"eef"}
    assert {item.target for item in robot} == {"robot"}


def test_validated_candidate_order_prioritizes_target_then_clearance() -> None:
    start = np.asarray([0.0, -0.30, 0.30], dtype=np.float32)
    goal = np.asarray([0.0, 0.30, 0.30], dtype=np.float32)
    config = UShapeSearchConfig()
    candidates = enumerate_u_shape_candidates(start, goal, 0.08, config=config)
    target = next(item for item in candidates if item.back_chunk_ratio == 2.5)
    off_target = next(item for item in candidates if item.back_chunk_ratio == 2.0)
    path = np.linspace(start, goal, 101, dtype=np.float32)
    target_check = evaluate_u_shape_geometry(
        target,
        start_robot_points=start.reshape(1, 3),
        goal_robot_points=goal.reshape(1, 3),
        nominal_tcp_path=path,
        goal=goal,
        config=config,
    )
    off_check = evaluate_u_shape_geometry(
        off_target,
        start_robot_points=start.reshape(1, 3),
        goal_robot_points=goal.reshape(1, 3),
        nominal_tcp_path=path,
        goal=goal,
        config=config,
    )
    target_validated = ValidatedUShapeCandidate(
        target,
        target_check,
        witness_clearance=0.03,
        witness_path_length=1.0,
        witness_steps=100,
        witness_side="left",
        obstacle_points=64,
    )
    off_validated = ValidatedUShapeCandidate(
        off_target,
        off_check,
        witness_clearance=0.10,
        witness_path_length=0.5,
        witness_steps=50,
        witness_side="right",
        obstacle_points=128,
    )

    assert validated_candidate_sort_key(target_validated) < validated_candidate_sort_key(
        off_validated
    )


def test_position_goal_orientations_are_deterministic_and_normalized() -> None:
    reference = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    orientations = _position_goal_orientations(reference)

    assert len(orientations) == 40
    np.testing.assert_allclose(
        np.linalg.norm(np.stack(orientations), axis=1),
        np.ones(40),
        atol=1e-6,
    )
    np.testing.assert_allclose(orientations[0], reference, atol=1e-7)
