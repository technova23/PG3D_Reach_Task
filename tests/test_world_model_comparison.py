from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from pg3d.envs.maniskill_adapter.dataset import PointCloudCropConfig
from pg3d.policies.dp3.checkpoint import latest_reach_checkpoint
from pg3d.world_model import ActionChunk, ImaginedRollout
from scripts.compare_world_model_rollout import (
    entry_to_world_model_observation,
    rerun_path_for_episode,
    resolve_checkpoint_path,
    world_model_entry_from_rollout_step,
)


def test_latest_reach_checkpoint_prefers_final_checkpoint_at_latest_step(
    tmp_path: Path,
) -> None:
    for name in [
        "step_00005000.pt",
        "step_00010000.pt",
        "final_step_00010000.pt",
        "final_step_00009000.pt",
        "notes.pt",
    ]:
        (tmp_path / name).touch()

    assert latest_reach_checkpoint(tmp_path) == tmp_path / "final_step_00010000.pt"
    assert resolve_checkpoint_path(None, tmp_path) == tmp_path / "final_step_00010000.pt"

    explicit = tmp_path / "custom.pt"
    assert resolve_checkpoint_path(explicit, None) == explicit


def test_rerun_path_for_episode_uses_per_episode_comparison_names(tmp_path: Path) -> None:
    paths = [rerun_path_for_episode(tmp_path, episode_idx) for episode_idx in range(2)]

    assert paths == [
        tmp_path / "episode_000_comparison.rrd",
        tmp_path / "episode_001_comparison.rrd",
    ]


def test_entry_to_world_model_observation_strips_invalid_padding() -> None:
    entry = _entry()
    observation = entry_to_world_model_observation(entry)

    assert observation.point_cloud.shape == (3, 3)
    assert observation.robot_mask is not None
    assert observation.robot_mask.tolist() == [False, True, False]
    np.testing.assert_allclose(observation.robot_state.joint_positions, entry["agent_pos"])
    assert observation.sim_gt is not None
    np.testing.assert_allclose(observation.sim_gt.target_position, entry["target_position"])


def test_world_model_entry_from_rollout_step_recrops_imagined_scene() -> None:
    previous = _entry()
    chunk = ActionChunk(
        actions=np.asarray([[0.1] * 7, [0.2] * 7], dtype=np.float32),
        action_mode="abs_joint",
        dt=1.0,
    )
    rollout = ImaginedRollout(
        q=np.asarray(
            [
                [0.1] * 7 + [0.04, 0.04],
                [0.2] * 7 + [0.04, 0.04],
            ],
            dtype=np.float32,
        ),
        eef_path=np.asarray([[0.0, 0.0, 0.2], [0.0, 0.0, 0.35]], dtype=np.float32),
        robot_point_clouds=[np.zeros((1, 3), dtype=np.float32)] * 2,
        scene_point_clouds=[
            np.asarray([[0.0, 0.0, 0.2], [3.0, 3.0, 3.0]], dtype=np.float32),
            np.asarray([[0.0, 0.0, 0.35], [0.1, 0.1, 0.1]], dtype=np.float32),
        ],
        robot_masks=[
            np.asarray([True, False], dtype=bool),
            np.asarray([True, False], dtype=bool),
        ],
        action_chunk=chunk,
    )
    crop_config = PointCloudCropConfig(
        bounds=np.asarray([[-1.0, 1.0], [-1.0, 1.0], [0.0, 1.0]], dtype=np.float32),
        num_points=4,
    )

    entry = world_model_entry_from_rollout_step(
        rollout,
        1,
        previous_entry=previous,
        crop_config=crop_config,
        goal_thresh=0.01,
    )

    assert entry["point_cloud"].shape == (4, 3)
    assert entry["point_valid_mask"].tolist() == [True, True, False, False]
    assert entry["robot_mask"].tolist() == [True, False, False, False]
    assert entry["success"] is True
    np.testing.assert_allclose(entry["agent_pos"], rollout.q[1])
    np.testing.assert_allclose(entry["tcp_pose"][:3], rollout.eef_path[1])


def test_comparison_script_import_keeps_simulator_deps_lazy() -> None:
    code = """
import importlib
import sys

importlib.import_module("scripts.compare_world_model_rollout")
assert "mani_skill" not in sys.modules
assert "sapien" not in sys.modules
assert "gymnasium" not in sys.modules
assert "rerun" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def _entry() -> dict[str, np.ndarray | bool | float]:
    return {
        "point_cloud": np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "robot_mask": np.asarray([False, True, False, False], dtype=bool),
        "point_valid_mask": np.asarray([True, True, True, False], dtype=bool),
        "agent_pos": np.asarray([0.0] * 7 + [0.04, 0.04], dtype=np.float32),
        "target_position": np.asarray([0.0, 0.0, 0.35], dtype=np.float32),
        "tcp_pose": np.asarray([0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "success": False,
        "final_distance": 0.15,
    }
