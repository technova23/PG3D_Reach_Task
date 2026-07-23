from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import zarr

from scripts.convert_xarm_m1_to_m2 import (
    CONVERTED_FROM,
    COORDINATE_FRAME,
    convert_constraints_dir,
    convert_zarr,
)

SHIFT = np.asarray([0.615, 0.0, 0.0], dtype=np.float32)


def test_convert_zarr_shifts_only_supported_cartesian_fields(tmp_path: Path) -> None:
    input_zarr = tmp_path / "m1.zarr"
    output_zarr = tmp_path / "m2.zarr"
    _write_tiny_zarr(input_zarr)

    convert_zarr(input_zarr, output_zarr)

    root_in = zarr.open_group(str(input_zarr), mode="r")
    root_out = zarr.open_group(str(output_zarr), mode="r")
    data_in = root_in["data"]
    data_out = root_out["data"]

    np.testing.assert_allclose(
        data_out["point_cloud"][:],
        data_in["point_cloud"][:] + SHIFT.reshape(1, 1, 3),
    )
    np.testing.assert_allclose(
        data_out["target_position"][:],
        data_in["target_position"][:] + SHIFT.reshape(1, 3),
    )
    np.testing.assert_allclose(
        data_out["goal_pos"][:],
        data_in["goal_pos"][:] + SHIFT.reshape(1, 3),
    )
    expected_tcp = data_in["tcp_pose"][:]
    expected_tcp[:, :3] += SHIFT
    np.testing.assert_allclose(data_out["tcp_pose"][:], expected_tcp)

    np.testing.assert_allclose(data_out["state"][:], data_in["state"][:])
    np.testing.assert_allclose(data_out["action"][:], data_in["action"][:])
    np.testing.assert_allclose(data_out["sim_action"][:], data_in["sim_action"][:])
    np.testing.assert_array_equal(data_out["robot_mask"][:], data_in["robot_mask"][:])
    np.testing.assert_array_equal(
        data_out["point_valid_mask"][:],
        data_in["point_valid_mask"][:],
    )

    assert root_out.attrs["coordinate_frame"] == COORDINATE_FRAME
    assert root_out.attrs["converted_from"] == CONVERTED_FROM
    metadata = json.loads((output_zarr / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["coordinate_frame"] == COORDINATE_FRAME
    assert metadata["converted_from"] == CONVERTED_FROM
    np.testing.assert_allclose(
        metadata["crop"]["bounds"],
        [[0.515, 1.215], [-0.55, 0.55], [-0.02, 0.60]],
    )
    np.testing.assert_allclose(metadata["env_kwargs"]["goal_center"], [0.34, 0.0, 0.21])
    np.testing.assert_allclose(metadata["env_kwargs"]["goal_regions"][0]["center"], [0.34, 0, 0.21])
    np.testing.assert_allclose(metadata["episodes"][0]["target_position"], [0.715, 0.2, 0.3])
    np.testing.assert_allclose(
        metadata["episodes"][0]["start_tcp_pose"],
        [0.815, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        metadata["episodes"][0]["goal_pose"],
        [0.915, 0.1, 0.5, 1.0, 0.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        metadata["episodes"][0]["start_sampling"]["actual_position"],
        [0.765, 0.1, 0.2],
    )
    np.testing.assert_allclose(
        metadata["episodes"][0]["start_sampling"]["sampled_position"],
        [0.775, 0.1, 0.2],
    )
    np.testing.assert_allclose(
        metadata["episodes"][0]["trajectory_waypoints"],
        [[0.815, 0.1, 0.3], [0.915, -0.1, 0.4]],
    )
    assert "crop.bounds" in metadata["shifted_metadata_fields"]


def test_convert_constraints_dir_shifts_centers_and_cartesian_targets(tmp_path: Path) -> None:
    input_dir = tmp_path / "constraints_m1"
    output_dir = tmp_path / "constraints_m2"
    input_dir.mkdir()
    payload = [
        {
            "type": "avoid_region",
            "target": "eef",
            "region": {"type": "sphere", "center": [0.1, 0.2, 0.3], "radius": 0.05},
            "margin": 0.01,
            "weight": 2.0,
            "tolerance": 1e-6,
            "name": "sphere_keepout",
        },
        {
            "type": "avoid_projection",
            "target": "robot",
            "region": {"type": "rect2d", "center": [0.4, -0.2], "half_extents": [0.1, 0.2]},
            "margin": 0.02,
            "weight": 3.0,
            "tolerance": 1e-6,
            "name": "projection_keepout",
        },
        {
            "type": "cartesian_pose",
            "target": "eef",
            "target_position": [0.2, 0.0, 0.4],
            "target_orientation": [1.0, 0.0, 0.0, 0.0],
            "position_tolerance": 0.03,
            "rotation_tolerance": 0.4,
            "weight": 1.0,
            "name": "pose",
        },
        {
            "type": "cylinder_passage",
            "region": {
                "type": "cylinder",
                "center": [0.3, 0.1, 0.2],
                "axis": [0.0, 0.0, 1.0],
                "radius": 0.2,
                "length": 0.5,
            },
            "target_orientation": [1.0, 0.0, 0.0, 0.0],
            "waypoint_start": 0,
            "waypoint_end": 2,
            "position_tolerance": 0.01,
            "rotation_tolerance": 0.2,
            "weight": 1.5,
            "name": "passage",
        },
    ]
    (input_dir / "episode_000.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    convert_constraints_dir(input_dir, output_dir)

    shifted = json.loads((output_dir / "episode_000.json").read_text(encoding="utf-8"))
    np.testing.assert_allclose(shifted[0]["region"]["center"], [0.715, 0.2, 0.3])
    assert shifted[0]["region"]["radius"] == payload[0]["region"]["radius"]
    np.testing.assert_allclose(shifted[1]["region"]["center"], [1.015, -0.2])
    assert shifted[1]["region"]["half_extents"] == payload[1]["region"]["half_extents"]
    np.testing.assert_allclose(shifted[2]["target_position"], [0.815, 0.0, 0.4])
    assert shifted[2]["target_orientation"] == payload[2]["target_orientation"]
    assert shifted[2]["metadata"]["coordinate_frame"] == COORDINATE_FRAME
    assert shifted[2]["metadata"]["converted_from"] == CONVERTED_FROM
    np.testing.assert_allclose(shifted[3]["region"]["center"], [0.915, 0.1, 0.2])
    assert shifted[3]["region"]["axis"] == payload[3]["region"]["axis"]
    assert shifted[3]["region"]["radius"] == payload[3]["region"]["radius"]
    assert shifted[3]["region"]["length"] == payload[3]["region"]["length"]


def _write_tiny_zarr(path: Path) -> None:
    root = zarr.group(store=zarr.DirectoryStore(str(path)), overwrite=True)
    data = root.create_group("data")
    meta = root.create_group("meta")
    data.array(
        name="point_cloud",
        data=np.asarray(
            [
                [[0.0, 0.0, 0.1], [0.2, 0.0, 0.2]],
                [[0.1, 0.1, 0.1], [0.3, 0.0, 0.2]],
            ],
            dtype=np.float32,
        ),
    )
    data.array(
        name="target_position",
        data=np.asarray([[0.4, 0.0, 0.2], [0.5, 0.0, 0.2]], dtype=np.float32),
    )
    data.array(
        name="goal_pos",
        data=np.asarray([[0.4, 0.0, 0.2], [0.5, 0.0, 0.2]], dtype=np.float32),
    )
    data.array(
        name="tcp_pose",
        data=np.asarray(
            [
                [0.1, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0],
                [0.2, 0.0, 0.2, 0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    data.array(name="state", data=np.ones((2, 7), dtype=np.float32))
    data.array(name="action", data=np.ones((2, 7), dtype=np.float32) * 2.0)
    data.array(name="sim_action", data=np.ones((2, 8), dtype=np.float32) * 3.0)
    data.array(name="robot_mask", data=np.asarray([[True, False], [False, True]]))
    data.array(name="point_valid_mask", data=np.ones((2, 2), dtype=bool))
    meta.array(name="episode_ends", data=np.asarray([2], dtype=np.int64))
    (path / "metadata.json").write_text(
        json.dumps(
            {
                "name": "tiny_m1",
                "crop": {
                    "bounds": [[-0.1, 0.6], [-0.55, 0.55], [-0.02, 0.60]],
                },
                "env_kwargs": {
                    "goal_center": [-0.275, 0.0, 0.21],
                    "goal_regions": [{"center": [-0.275, 0.0, 0.21]}],
                },
                "episodes": [
                    {
                        "target_position": [0.1, 0.2, 0.3],
                        "start_tcp_pose": [0.2, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
                        "goal_pose": [0.3, 0.1, 0.5, 1.0, 0.0, 0.0, 0.0],
                        "start_sampling": {
                            "actual_position": [0.15, 0.1, 0.2],
                            "sampled_position": [0.16, 0.1, 0.2],
                        },
                        "trajectory_waypoints": [
                            [0.2, 0.1, 0.3],
                            [0.3, -0.1, 0.4],
                        ],
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
