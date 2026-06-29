from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest
import zarr

from pg3d.envs.maniskill_adapter import Observation, RobotState, SimGroundTruth
from pg3d.envs.maniskill_adapter.dataset import (
    PointCloudCropConfig,
    ReachEpisodeData,
    action_label_from_sim_action,
    crop_point_cloud,
    observation_to_dataset_row,
    write_reach_zarr,
)


def test_crop_point_cloud_preserves_aligned_masks_and_pads() -> None:
    config = PointCloudCropConfig(
        bounds=np.asarray([[-1, 1], [-1, 1], [0, 1]], dtype=np.float32),
        num_points=4,
    )
    points = np.asarray(
        [
            [-2.0, 0.0, 0.5],
            [0.0, 0.0, 0.2],
            [0.2, 0.2, 0.4],
            [0.3, 0.3, 0.6],
            [0.4, 0.4, 0.8],
            [2.0, 0.0, 0.5],
        ],
        dtype=np.float32,
    )
    robot_mask = np.asarray([True, True, False, True, False, True])

    cropped = crop_point_cloud(points, robot_mask=robot_mask, config=config)

    np.testing.assert_allclose(
        cropped["point_cloud"],
        np.asarray(
            [
                [0.0, 0.0, 0.2],
                [0.2, 0.2, 0.4],
                [0.3, 0.3, 0.6],
                [0.4, 0.4, 0.8],
            ],
            dtype=np.float32,
        ),
    )
    assert cropped["robot_mask"].tolist() == [True, False, True, False]
    assert cropped["point_valid_mask"].tolist() == [True, True, True, True]


def test_crop_point_cloud_downsamples_deterministically() -> None:
    config = PointCloudCropConfig(
        bounds=np.asarray([[-1, 1], [-1, 1], [0, 1]], dtype=np.float32),
        num_points=3,
    )
    points = np.asarray([[idx * 0.1, 0.0, 0.5] for idx in range(5)], dtype=np.float32)

    cropped = crop_point_cloud(points, config=config)

    np.testing.assert_allclose(
        cropped["point_cloud"],
        np.asarray([[0.0, 0.0, 0.5], [0.2, 0.0, 0.5], [0.4, 0.0, 0.5]], dtype=np.float32),
    )


def test_crop_point_cloud_downsamples_with_robot_quota() -> None:
    config = PointCloudCropConfig(
        bounds=np.asarray([[-1, 2], [-1, 1], [0, 1]], dtype=np.float32),
        num_points=8,
        robot_point_fraction=0.5,
    )
    points = np.asarray([[idx * 0.1, 0.0, 0.5] for idx in range(12)], dtype=np.float32)
    robot_mask = np.asarray([True, True, True, True] + [False] * 8)

    cropped = crop_point_cloud(points, robot_mask=robot_mask, config=config)

    assert cropped["point_cloud"].shape == (8, 3)
    assert int(cropped["robot_mask"].sum()) == 4
    assert cropped["point_valid_mask"].tolist() == [True] * 8


def test_action_label_conversion_supports_abs_and_delta() -> None:
    sim_action = np.asarray([1, 2, 3, 4, 5, 6, 7, 0.04], dtype=np.float32)
    state = np.asarray([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 0.04, 0.04], dtype=np.float32)

    np.testing.assert_allclose(
        action_label_from_sim_action(sim_action, state, action_mode="abs_joint"),
        sim_action[:7],
    )
    np.testing.assert_allclose(
        action_label_from_sim_action(sim_action, state, action_mode="delta_joint"),
        np.asarray([0.5, 1, 1.5, 2, 2.5, 3, 3.5], dtype=np.float32),
    )


def test_observation_to_dataset_row_keeps_eval_fields_separate() -> None:
    observation = Observation(
        point_cloud=np.asarray([[0.0, 0.0, 0.2], [2.0, 0.0, 0.2]], dtype=np.float32),
        point_features={},
        robot_mask=np.asarray([True, False]),
        robot_state=RobotState(
            joint_positions=np.zeros(9, dtype=np.float32),
            tcp_pose=np.asarray([0, 0, 0.2, 1, 0, 0, 0], dtype=np.float32),
        ),
        sim_gt=SimGroundTruth(
            task_name="PG3DReach-Narrow-v0",
            target_position=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        ),
    )
    row = observation_to_dataset_row(
        observation,
        sim_action=np.arange(8, dtype=np.float32),
        action_mode="abs_joint",
        crop_config=PointCloudCropConfig(
            bounds=np.asarray([[-1, 1], [-1, 1], [0, 1]], dtype=np.float32),
            num_points=2,
        ),
    )

    assert set(row) == {
        "action",
        "point_cloud",
        "point_valid_mask",
        "robot_mask",
        "sim_action",
        "state",
        "target_position",
        "tcp_pose",
    }
    assert row["action"].shape == (7,)
    assert row["target_position"].tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert row["point_valid_mask"].tolist() == [True, False]


def test_write_reach_zarr_schema_and_metadata(tmp_path) -> None:
    episode = ReachEpisodeData(
        state=np.zeros((2, 9), dtype=np.float32),
        action=np.ones((2, 7), dtype=np.float32),
        sim_action=np.ones((2, 8), dtype=np.float32),
        point_cloud=np.zeros((2, 4, 3), dtype=np.float32),
        robot_mask=np.zeros((2, 4), dtype=bool),
        point_valid_mask=np.ones((2, 4), dtype=bool),
        target_position=np.zeros((2, 3), dtype=np.float32),
        tcp_pose=np.zeros((2, 7), dtype=np.float32),
        success=np.asarray([False, True]),
        metadata={"seed": 3, "success": True},
    )

    output = tmp_path / "reach.zarr"
    summary = write_reach_zarr(
        output,
        [episode],
        metadata={
            "env_id": "PG3DReach-Narrow-v0",
            "env_kwargs": {"obs_mode": "pointcloud"},
            "action_mode": "abs_joint",
        },
    )

    root = zarr.open_group(str(output), mode="r")
    assert root["data"]["state"].shape == (2, 9)
    assert root["data"]["action"].shape == (2, 7)
    assert root["data"]["point_cloud"].shape == (2, 4, 3)
    assert root["meta"]["episode_ends"][:].tolist() == [2]
    assert summary["num_steps"] == 2
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == "pg3d.reach.zarr.v1"
    assert metadata["episodes"][0]["seed"] == 3


def test_reach_dataset_imports_keep_simulator_lazy() -> None:
    code = """
import importlib
import sys

importlib.import_module("pg3d.envs.maniskill_adapter.dataset")
importlib.import_module("pg3d.envs.maniskill_adapter.registration")
assert "mani_skill" not in sys.modules
assert "sapien" not in sys.modules
assert "gymnasium" not in sys.modules
assert "rerun" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)
