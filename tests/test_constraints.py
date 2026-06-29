from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from pg3d.constraints import (
    AvoidRegion,
    BoxRegion,
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
    ]

    payload = json.loads(json.dumps(constraints_to_json(constraints)))
    loaded = constraints_from_json(payload)

    assert isinstance(loaded[0], AvoidRegion)
    assert isinstance(loaded[0].region, SphereRegion)
    assert loaded[0].margin == pytest.approx(0.01)
    assert isinstance(loaded[1], SmoothnessCost)
    assert loaded[1].target == "q"
    assert loaded[1].order == 1


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
) -> ImaginedRollout:
    eef = np.asarray(eef_path, dtype=np.float32)
    q_array = np.asarray(q if q is not None else eef[:, :2], dtype=np.float32)
    horizon = eef.shape[0]
    action_dim = q_array.shape[1]
    return ImaginedRollout(
        q=q_array,
        eef_path=eef,
        robot_point_clouds=[np.zeros((0, 3), dtype=np.float32) for _ in range(horizon)],
        scene_point_clouds=[np.zeros((0, 3), dtype=np.float32) for _ in range(horizon)],
        robot_masks=[np.zeros((0,), dtype=bool) for _ in range(horizon)],
        action_chunk=ActionChunk(
            actions=np.zeros((horizon, action_dim), dtype=np.float32),
            action_mode="abs_joint",
            dt=0.1,
        ),
    )
