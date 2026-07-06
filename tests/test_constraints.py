from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from pg3d.constraints import (
    AvoidProjection,
    AvoidRegion,
    BoxRegion,
    CartesianPoseConstraint,
    CylinderPassageConstraint,
    CylinderRegion,
    RectRegion2D,
    SmoothnessCost,
    SphereRegion,
    constraint_from_json,
    constraints_from_json,
    constraints_to_json,
    make_obstructing_avoid_region,
    region_from_json,
)
from pg3d.world_model import ActionChunk, ImaginedRollout


def test_sphere_and_box_signed_distance() -> None:
    sphere = SphereRegion(center=[0.0, 0.0, 0.0], radius=1.0)
    np.testing.assert_allclose(
        sphere.signed_distance(np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])),
        np.asarray([-1.0, 1.0], dtype=np.float32),
    )

    box = BoxRegion(center=[0.0, 0.0, 0.0], half_extents=[1.0, 2.0, 3.0])
    distances = box.signed_distance(
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [1.0, 2.0, 3.0],
            ],
            dtype=np.float32,
        )
    )
    np.testing.assert_allclose(distances, np.asarray([-1.0, 1.0, 0.0], dtype=np.float32))


def test_cylinder_signed_distance_and_json_round_trip() -> None:
    cylinder = CylinderRegion(center=[0.0, 0.0, 0.0], axis=[0.0, 0.0, 2.0], radius=0.5, length=2.0)
    distances = cylinder.signed_distance(
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.25, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.2],
            ],
            dtype=np.float32,
        )
    )
    np.testing.assert_allclose(
        distances,
        np.asarray([-0.5, -0.25, 0.5, 0.2], dtype=np.float32),
        atol=1e-6,
    )
    restored = region_from_json(cylinder.to_json())
    assert isinstance(restored, CylinderRegion)
    np.testing.assert_allclose(restored.center, cylinder.center)
    np.testing.assert_allclose(restored.axis, cylinder.axis)
    assert restored.radius == pytest.approx(cylinder.radius)
    assert restored.length == pytest.approx(cylinder.length)


def test_rect2d_signed_distance_ignores_height() -> None:
    rect = RectRegion2D(center=[0.5, 0.0], half_extents=[0.08, 0.08])
    # XY inside the footprint -> negative regardless of z; XY outside -> positive.
    points = np.asarray(
        [
            [0.5, 0.0, 0.05],  # centered, low
            [0.5, 0.0, 0.90],  # centered, very high -> still inside (z ignored)
            [1.0, 0.0, 0.05],  # far in x -> outside
        ],
        dtype=np.float32,
    )
    distances = rect.signed_distance(points)
    np.testing.assert_allclose(distances, np.asarray([-0.08, -0.08, 0.42], dtype=np.float32))
    # Accepts [N, 2] as well and yields the same XY result.
    np.testing.assert_allclose(distances, rect.signed_distance(points[:, :2]))


def test_avoid_projection_penalizes_xy_overflight_at_any_height() -> None:
    # EEF flies high over the restricted footprint; only the XY projection matters.
    rollout = _rollout(eef_path=[[0.2, 0.0, 0.2], [0.5, 0.0, 0.9], [0.8, 0.0, 0.2]])
    constraint = AvoidProjection(region=RectRegion2D(center=[0.5, 0.0], half_extents=[0.08, 0.08]))

    costs = constraint.cost(rollout)

    assert costs["avoid_projection"] > 0.0
    assert costs["avoid_projection/max_violation"] == pytest.approx(0.08)
    assert costs["avoid_projection/inside_count"] == 1.0
    assert costs["avoid_projection/fraction_over"] == pytest.approx(1.0 / 3.0)
    assert not constraint.satisfied(rollout)


def test_avoid_projection_satisfied_when_path_skirts_footprint() -> None:
    # Same x sweep but offset in y so the XY projection never enters the rectangle.
    rollout = _rollout(eef_path=[[0.2, 0.5, 0.2], [0.5, 0.5, 0.2], [0.8, 0.5, 0.2]])
    constraint = AvoidProjection(region=RectRegion2D(center=[0.5, 0.0], half_extents=[0.08, 0.08]))

    costs = constraint.cost(rollout)

    assert costs["avoid_projection"] == 0.0
    assert costs["avoid_projection/inside_count"] == 0.0
    assert constraint.satisfied(rollout)


def test_avoid_projection_json_round_trip() -> None:
    constraint = AvoidProjection(
        region=RectRegion2D(center=[0.4, -0.1], half_extents=[0.1, 0.05]),
        margin=0.01,
        weight=2.0,
        name="candidate_midpath_avoid_projection",
    )
    [restored] = constraints_from_json(constraints_to_json([constraint]))
    assert isinstance(restored, AvoidProjection)
    assert isinstance(restored.region, RectRegion2D)
    assert restored.name == "candidate_midpath_avoid_projection"
    assert restored.weight == pytest.approx(2.0)
    np.testing.assert_allclose(restored.region.center, np.asarray([0.4, -0.1], dtype=np.float32))
    np.testing.assert_allclose(
        restored.region.half_extents, np.asarray([0.1, 0.05], dtype=np.float32)
    )


def test_avoid_region_cost_detects_eef_path_violation() -> None:
    rollout = _rollout(eef_path=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])
    constraint = AvoidRegion(region=SphereRegion(center=[0.5, 0.0, 0.0], radius=0.1))

    costs = constraint.cost(rollout)

    assert costs["avoid_region"] > 0.0
    assert costs["avoid_region/max_violation"] == pytest.approx(0.1)
    assert costs["avoid_region/inside_count"] == 1.0
    assert not constraint.satisfied(rollout)


def test_avoid_region_is_zero_outside_margin() -> None:
    rollout = _rollout(eef_path=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])
    constraint = AvoidRegion(
        region=BoxRegion(center=[0.5, 0.5, 0.0], half_extents=[0.1, 0.1, 0.1]),
        margin=0.0,
    )

    costs = constraint.cost(rollout)

    assert costs["avoid_region"] == 0.0
    assert costs["avoid_region/max_violation"] == 0.0
    assert constraint.satisfied(rollout)


def test_cartesian_pose_constraint_requires_waypoint_pose_match() -> None:
    rollout = _rollout(
        eef_path=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
        eef_orientations=[
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
    )
    constraint = CartesianPoseConstraint(
        target_position=[0.1, 0.0, 0.0],
        target_orientation=[1.0, 0.0, 0.0, 0.0],
        position_tolerance=0.01,
        rotation_tolerance=0.05,
    )
    costs = constraint.cost(rollout)
    assert costs["cartesian_pose"] == 0.0
    assert constraint.satisfied(rollout)


def test_cartesian_pose_constraint_fails_when_position_or_rotation_exceeds_tolerance() -> None:
    rollout = _rollout(
        eef_path=[[0.0, 0.0, 0.0], [0.12, 0.0, 0.0], [0.2, 0.0, 0.0]],
        eef_orientations=[
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
    )
    constraint = CartesianPoseConstraint(
        target_position=[0.11, 0.0, 0.0],
        target_orientation=[1.0, 0.0, 0.0, 0.0],
        position_tolerance=0.01,
        rotation_tolerance=0.05,
    )
    costs = constraint.cost(rollout)
    assert costs["cartesian_pose"] > 0.0
    assert costs["cartesian_pose/best_index"] == pytest.approx(1.0)
    assert costs["cartesian_pose/rotation_violation"] > 0.0
    assert not constraint.satisfied(rollout)


def test_cylinder_passage_constraint_scores_full_traversal() -> None:
    rollout = _rollout(
        eef_path=[
            [-0.6, 0.0, 0.0],
            [-0.2, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.6, 0.0, 0.0],
        ],
        eef_orientations=[[1.0, 0.0, 0.0, 0.0] for _ in range(5)],
    )
    constraint = CylinderPassageConstraint(
        region=CylinderRegion(center=[0.0, 0.0, 0.0], axis=[1.0, 0.0, 0.0], radius=0.25, length=1.0),
        target_orientation=[1.0, 0.0, 0.0, 0.0],
        waypoint_start=0,
        waypoint_end=4,
        position_tolerance=0.05,
        rotation_tolerance=0.05,
    )
    costs = constraint.cost(rollout)
    assert costs["cylinder_passage/passage_violation"] == 0.0
    assert costs["cylinder_passage/rotation_violation"] == 0.0
    assert costs["cylinder_passage/position_violation"] >= 0.0
    assert constraint.satisfied(rollout)


def test_cylinder_passage_constraint_detects_partial_or_missing_traversal() -> None:
    partial = _rollout(
        eef_path=[
            [-0.4, 0.0, 0.0],
            [-0.1, 0.0, 0.0],
            [0.1, 0.0, 0.0],
        ],
        eef_orientations=[[1.0, 0.0, 0.0, 0.0] for _ in range(3)],
    )
    missing = _rollout(
        eef_path=[
            [0.0, 0.4, 0.0],
            [0.1, 0.4, 0.0],
            [0.2, 0.4, 0.0],
        ],
        eef_orientations=[[1.0, 0.0, 0.0, 0.0] for _ in range(3)],
    )
    constraint = CylinderPassageConstraint(
        region=CylinderRegion(center=[0.0, 0.0, 0.0], axis=[1.0, 0.0, 0.0], radius=0.2, length=1.0),
        target_orientation=[1.0, 0.0, 0.0, 0.0],
        waypoint_start=0,
        waypoint_end=2,
        position_tolerance=0.0,
        rotation_tolerance=0.05,
    )
    assert constraint.cost(partial)["cylinder_passage/passage_violation"] > 0.0
    assert not constraint.satisfied(partial)
    assert constraint.cost(missing)["cylinder_passage/passage_violation"] > 0.0
    assert not constraint.satisfied(missing)


def test_make_obstructing_avoid_region_places_sphere_on_direct_path() -> None:
    constraint = make_obstructing_avoid_region(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        radius=0.05,
    )
    rollout = _rollout(eef_path=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])

    assert isinstance(constraint.region, SphereRegion)
    np.testing.assert_allclose(constraint.region.center, np.asarray([0.5, 0.0, 0.0]))
    assert constraint.cost(rollout)["avoid_region"] > 0.0


def test_smoothness_cost_scores_acceleration() -> None:
    straight = _rollout(
        q=[
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        eef_path=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
    )
    bent = _rollout(
        q=[
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        eef_path=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
    )
    smoothness = SmoothnessCost(target="eef", order=2, threshold=0.5)

    assert smoothness.cost(straight)["smoothness"] == 0.0
    assert smoothness.cost(bent)["smoothness"] > 0.5
    assert smoothness.satisfied(straight)
    assert not smoothness.satisfied(bent)


def test_constraint_json_round_trip() -> None:
    constraints = [
        AvoidRegion(
            region=SphereRegion(center=[0.1, 0.2, 0.3], radius=0.05),
            margin=0.01,
            weight=2.0,
            tolerance=0.001,
        ),
        SmoothnessCost(target="q", order=1, weight=0.5, threshold=1.0),
        CartesianPoseConstraint(
            target_position=[0.1, 0.2, 0.3],
            target_orientation=[1.0, 0.0, 0.0, 0.0],
            position_tolerance=0.01,
            rotation_tolerance=0.1,
        ),
        CylinderPassageConstraint(
            region=CylinderRegion(center=[0.0, 0.0, 0.0], axis=[0.0, 0.0, 1.0], radius=0.2, length=1.0),
            target_orientation=[1.0, 0.0, 0.0, 0.0],
            waypoint_start=0,
            waypoint_end=1,
            position_tolerance=0.01,
            rotation_tolerance=0.1,
        ),
    ]

    payload = json.loads(json.dumps(constraints_to_json(constraints)))
    loaded = constraints_from_json(payload)

    assert isinstance(loaded[0], AvoidRegion)
    assert isinstance(loaded[0].region, SphereRegion)
    assert loaded[0].margin == pytest.approx(0.01)
    assert isinstance(loaded[1], SmoothnessCost)
    assert loaded[1].target == "q"
    assert loaded[1].order == 1
    assert isinstance(loaded[2], CartesianPoseConstraint)
    assert isinstance(loaded[3], CylinderPassageConstraint)
    assert loaded[3].region.region_type == "cylinder"


def test_region_and_constraint_json_reject_unknown_types() -> None:
    with pytest.raises(ValueError, match="unknown region"):
        region_from_json({"type": "capsule"})
    with pytest.raises(ValueError, match="unknown constraint"):
        constraint_from_json({"type": "teleport"})


def test_constraints_import_keeps_simulator_deps_lazy() -> None:
    code = """
import importlib
import sys

importlib.import_module("pg3d.constraints")
assert "mani_skill" not in sys.modules
assert "sapien" not in sys.modules
assert "gymnasium" not in sys.modules
assert "rerun" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def _rollout(
    *,
    eef_path: list[list[float]],
    q: list[list[float]] | None = None,
    eef_orientations: list[list[float]] | None = None,
) -> ImaginedRollout:
    eef = np.asarray(eef_path, dtype=np.float32)
    q_array = np.asarray(q if q is not None else eef[:, :2], dtype=np.float32)
    orientations = None if eef_orientations is None else np.asarray(eef_orientations, dtype=np.float32)
    horizon = eef.shape[0]
    action_dim = q_array.shape[1]
    return ImaginedRollout(
        q=q_array,
        eef_path=eef,
        eef_orientations=orientations,
        robot_point_clouds=[np.zeros((0, 3), dtype=np.float32) for _ in range(horizon)],
        scene_point_clouds=[np.zeros((0, 3), dtype=np.float32) for _ in range(horizon)],
        robot_masks=[np.zeros((0,), dtype=bool) for _ in range(horizon)],
        action_chunk=ActionChunk(
            actions=np.zeros((horizon, action_dim), dtype=np.float32),
            action_mode="abs_joint",
            dt=0.1,
        ),
    )
