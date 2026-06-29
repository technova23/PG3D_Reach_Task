from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from pg3d.envs.maniskill_adapter.types import Observation, RobotState
from pg3d.world_model import (
    ActionChunk,
    GeometricWorldModel,
    compose_robot_cloud,
    interpret_joint_chunk,
    static_scene_from_robot_mask,
)
from pg3d.world_model.types import Array


class DeterministicGeometry:
    def end_effector_position(self, q: Array) -> Array:
        return np.asarray([q[0], q[1], q[2]], dtype=np.float32)

    def robot_point_cloud(self, q: Array) -> Array:
        return np.asarray(
            [
                [q[0], 0.0, 0.0],
                [0.0, q[1], 0.0],
            ],
            dtype=np.float32,
        )


def test_abs_joint_chunk_updates_controlled_prefix_and_holds_tail() -> None:
    chunk = ActionChunk(
        actions=np.asarray(
            [
                [0.1, 0.2, 0.3, -1.0, 0.0, 1.0, 0.5],
                [0.2, 0.3, 0.4, -0.9, 0.1, 0.9, 0.4],
            ],
            dtype=np.float32,
        ),
        action_mode="abs_joint",
        dt=0.125,
    )
    start_q = np.asarray([0.0, 0.0, 0.0, -1.1, 0.0, 1.1, 0.6, 0.04, 0.04])

    q = interpret_joint_chunk(chunk, start_q)

    np.testing.assert_allclose(q[:, :7], chunk.actions)
    np.testing.assert_allclose(q[:, 7:], np.asarray([[0.04, 0.04], [0.04, 0.04]]))


def test_delta_joint_chunk_is_cumulative() -> None:
    chunk = ActionChunk(
        actions=np.asarray(
            [
                [0.1, 0.0, 0.0],
                [0.0, 0.2, 0.0],
                [-0.05, 0.0, 0.3],
            ],
            dtype=np.float32,
        ),
        action_mode="delta_joint",
        dt=0.1,
    )
    start_q = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    q = interpret_joint_chunk(chunk, start_q)

    np.testing.assert_allclose(
        q,
        np.asarray(
            [
                [1.1, 2.0, 3.0, 4.0],
                [1.1, 2.2, 3.0, 4.0],
                [1.05, 2.2, 3.3, 4.0],
            ],
            dtype=np.float32,
        ),
        rtol=1e-6,
    )


def test_action_chunk_rejects_unsupported_and_invalid_inputs() -> None:
    with pytest.raises(NotImplementedError, match="ee_pose"):
        ActionChunk(actions=np.zeros((1, 6), dtype=np.float32), action_mode="ee_pose", dt=0.1)
    with pytest.raises(ValueError, match="at least one timestep"):
        ActionChunk(actions=np.zeros((0, 7), dtype=np.float32), action_mode="abs_joint", dt=0.1)
    with pytest.raises(ValueError, match="finite"):
        ActionChunk(
            actions=np.asarray([[np.nan]], dtype=np.float32),
            action_mode="abs_joint",
            dt=0.1,
        )
    with pytest.raises(ValueError, match="positive finite"):
        ActionChunk(actions=np.zeros((1, 7), dtype=np.float32), action_mode="abs_joint", dt=0.0)


def test_action_dim_cannot_exceed_robot_dof() -> None:
    chunk = ActionChunk(actions=np.zeros((1, 7), dtype=np.float32), action_mode="abs_joint", dt=0.1)

    with pytest.raises(ValueError, match="cannot exceed"):
        interpret_joint_chunk(chunk, np.zeros((6,), dtype=np.float32))


def test_compositor_removes_current_robot_points_and_inserts_future_robot_points() -> None:
    point_cloud = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    static_scene = static_scene_from_robot_mask(
        point_cloud,
        np.asarray([True, False, True], dtype=bool),
    )
    future_robot = np.asarray([[9.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)

    scene, robot_mask = compose_robot_cloud(static_scene, future_robot)

    np.testing.assert_allclose(static_scene, np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(
        scene,
        np.asarray(
            [
                [1.0, 0.0, 0.0],
                [9.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    assert robot_mask.tolist() == [False, True, True]


def test_world_model_imagines_robot_only_rollout() -> None:
    observation = _observation()
    chunk = ActionChunk(
        actions=np.asarray([[0.1, 0.2, 0.3], [0.2, 0.4, 0.6]], dtype=np.float32),
        action_mode="abs_joint",
        dt=0.1,
    )

    rollout = GeometricWorldModel(DeterministicGeometry()).imagine(observation, chunk)

    np.testing.assert_allclose(
        rollout.q,
        np.asarray(
            [
                [0.1, 0.2, 0.3, 0.04, 0.04],
                [0.2, 0.4, 0.6, 0.04, 0.04],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(rollout.eef_path, rollout.q[:, :3])
    assert len(rollout.robot_point_clouds) == 2
    assert rollout.scene_point_clouds[0].shape == (4, 3)
    assert rollout.robot_masks[0].tolist() == [False, False, True, True]
    assert rollout.metadata["static_scene_points"] == 2
    assert rollout.metadata["current_robot_points"] == 1


def test_world_model_requires_robot_mask() -> None:
    observation = Observation(
        point_cloud=np.zeros((2, 3), dtype=np.float32),
        point_features={},
        robot_mask=None,
        robot_state=RobotState(joint_positions=np.zeros((5,), dtype=np.float32)),
    )
    chunk = ActionChunk(actions=np.zeros((1, 3), dtype=np.float32), action_mode="abs_joint", dt=0.1)

    with pytest.raises(ValueError, match="robot_mask is required"):
        GeometricWorldModel(DeterministicGeometry()).imagine(observation, chunk)


def test_world_model_import_keeps_simulator_deps_lazy() -> None:
    code = """
import importlib
import sys

importlib.import_module("pg3d.world_model")
assert "mani_skill" not in sys.modules
assert "sapien" not in sys.modules
assert "gymnasium" not in sys.modules
assert "rerun" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def _observation() -> Observation:
    return Observation(
        point_cloud=np.asarray(
            [
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        point_features={},
        robot_mask=np.asarray([False, False, True], dtype=bool),
        robot_state=RobotState(
            joint_positions=np.asarray([0.0, 0.0, 0.0, 0.04, 0.04], dtype=np.float32),
        ),
    )
