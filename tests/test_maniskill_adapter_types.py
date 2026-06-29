from __future__ import annotations

import numpy as np
import pytest

from pg3d.envs.maniskill_adapter import (
    Observation,
    RobotState,
    SegmentationContext,
    SimGroundTruth,
    adapt_observation,
    segmentation_context_from_env,
)


def test_policy_inputs_exclude_masks_and_sim_ground_truth_by_default() -> None:
    observation = Observation(
        point_cloud=np.zeros((2, 3), dtype=np.float32),
        point_features={
            "rgb": np.asarray([[255, 0, 0], [0, 255, 0]], dtype=np.uint8),
            "segmentation": np.asarray([1, 2], dtype=np.int64),
        },
        robot_mask=np.asarray([True, False]),
        object_masks={"cube": np.asarray([False, True])},
        robot_state=RobotState(
            joint_positions=np.arange(7, dtype=np.float32),
            joint_velocities=np.ones(7, dtype=np.float32),
            tcp_pose=np.zeros(7, dtype=np.float32),
        ),
        sim_gt=SimGroundTruth(
            task_name="PickCube-v1",
            target_position=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
            success=False,
        ),
    )

    policy_inputs = observation.as_policy_inputs()

    assert set(policy_inputs) == {"point_cloud", "agent_pos"}
    assert policy_inputs["point_cloud"].shape == (2, 3)
    assert policy_inputs["agent_pos"].tolist() == list(np.arange(7, dtype=np.float32))

    rgb_policy_inputs = observation.as_policy_inputs(include_rgb=True)
    assert rgb_policy_inputs["point_cloud"].shape == (2, 6)
    assert np.max(rgb_policy_inputs["point_cloud"][:, 3:]) <= 1.0


def test_observation_validation_rejects_feature_length_mismatch() -> None:
    with pytest.raises(ValueError, match="point_features\\['rgb'\\] first dimension"):
        Observation(
            point_cloud=np.zeros((3, 3), dtype=np.float32),
            point_features={"rgb": np.zeros((2, 3), dtype=np.uint8)},
            robot_state=RobotState(joint_positions=np.zeros(7, dtype=np.float32)),
        )


def test_observation_validation_rejects_robot_mask_length_mismatch() -> None:
    with pytest.raises(ValueError, match="robot_mask must have shape"):
        Observation(
            point_cloud=np.zeros((3, 3), dtype=np.float32),
            point_features={},
            robot_mask=np.zeros(2, dtype=bool),
            robot_state=RobotState(joint_positions=np.zeros(7, dtype=np.float32)),
        )


def test_state_dict_adapter_keeps_policy_and_eval_fields_separate() -> None:
    observation = adapt_observation(
        {
            "agent": {
                "qpos": np.asarray([[1, 2, 3, 4, 5, 6, 7, 0.04, 0.04]], dtype=np.float32),
                "qvel": np.zeros((1, 9), dtype=np.float32),
            },
            "extra": {
                "tcp_pose": np.asarray([[0, 0, 1, 0, 0, 0, 1]], dtype=np.float32),
                "goal_pos": np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32),
            },
        },
        info={"success": np.asarray([True])},
        task_name="PickCube-v1",
    )

    assert observation.point_cloud.shape == (0, 3)
    assert observation.robot_state.joint_positions.shape == (9,)
    assert observation.robot_state.gripper_open == pytest.approx(0.04)
    assert observation.sim_gt is not None
    assert observation.sim_gt.target_position.tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert observation.sim_gt.success is True

    policy_inputs = observation.as_policy_inputs()
    assert set(policy_inputs) == {"point_cloud", "agent_pos"}
    assert "target_position" not in policy_inputs
    assert "success" not in policy_inputs


def test_pointcloud_adapter_converts_features_and_masks_from_context() -> None:
    observation = adapt_observation(
        {
            "agent": {
                "qpos": np.zeros((1, 9), dtype=np.float32),
                "qvel": np.zeros((1, 9), dtype=np.float32),
            },
            "extra": {},
            "pointcloud": {
                "xyzw": np.asarray(
                    [[[1.0, 2.0, 3.0, 1.0], [4.0, 5.0, 6.0, 1.0], [7.0, 8.0, 9.0, 1.0]]],
                    dtype=np.float32,
                ),
                "rgb": np.asarray([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8),
                "segmentation": np.asarray([[[1], [18], [19]]], dtype=np.int16),
            },
        },
        segmentation_context=SegmentationContext(
            robot_ids=frozenset({1}),
            object_ids={
                "cube": frozenset({18}),
                "goal_site": frozenset({19}),
            },
        ),
        task_name="PickCube-v1",
    )

    assert observation.point_cloud.dtype == np.float32
    np.testing.assert_allclose(
        observation.point_cloud,
        np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32),
    )
    assert observation.point_features["rgb"].dtype == np.uint8
    assert observation.point_features["segmentation"].tolist() == [1, 18, 19]
    assert observation.robot_mask is not None
    assert observation.robot_mask.tolist() == [True, False, False]
    assert observation.object_masks["cube"].tolist() == [False, True, False]
    assert observation.object_masks["goal_site"].tolist() == [False, False, True]

    policy_inputs = observation.as_policy_inputs()
    assert policy_inputs["point_cloud"].shape == (3, 3)
    assert "segmentation" not in policy_inputs


def test_pointcloud_adapter_leaves_masks_absent_without_context() -> None:
    observation = adapt_observation(
        {
            "agent": {
                "qpos": np.zeros((1, 9), dtype=np.float32),
                "qvel": np.zeros((1, 9), dtype=np.float32),
            },
            "extra": {},
            "pointcloud": {
                "xyzw": np.zeros((1, 2, 4), dtype=np.float32),
                "segmentation": np.asarray([[[1], [18]]], dtype=np.int16),
            },
        },
        task_name="PickCube-v1",
    )

    assert observation.robot_mask is None
    assert observation.object_masks == {}
    assert observation.point_features["segmentation"].tolist() == [1, 18]


def test_segmentation_context_from_env_uses_robot_and_task_actor_ids() -> None:
    fake_env = _FakeEnv()

    context = segmentation_context_from_env(fake_env)

    assert context.robot_ids == frozenset({1, 2})
    assert context.object_ids["cube"] == frozenset({18})
    assert context.object_ids["goal_site"] == frozenset({19})


class _FakeActor:
    def __init__(self, value: int) -> None:
        self.per_scene_id = np.asarray([value], dtype=np.int32)


class _FakeRobot:
    links = [_FakeActor(1), _FakeActor(2)]


class _FakeAgent:
    robot = _FakeRobot()


class _FakeEnv:
    agent = _FakeAgent()
    cube = _FakeActor(18)
    goal_site = _FakeActor(19)
    unwrapped = None

    def __init__(self) -> None:
        self.unwrapped = self
