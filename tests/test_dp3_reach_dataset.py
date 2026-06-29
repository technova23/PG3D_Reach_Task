from __future__ import annotations

import subprocess
import sys

import numpy as np
import torch

from pg3d.envs.maniskill_adapter.dataset import ReachEpisodeData, write_reach_zarr
from pg3d.policies.dp3 import SimpleDP3
from pg3d.policies.dp3.goal_markers import (
    DEFAULT_GOAL_MARKER_RADIUS,
    goal_marker_offsets,
    insert_goal_marker_points,
)
from pg3d.policies.dp3.reach_dataset import (
    ReachDatasetConfig,
    ReachSequenceDataset,
    create_sequence_indices,
    normalizer_step_indices,
    reach_shape_meta,
    validation_episode_mask,
)


def test_reach_sequence_dataset_returns_policy_visible_batch(tmp_path) -> None:
    dataset_path = _write_reach_dataset(tmp_path, num_episodes=2, episode_length=4)
    dataset = ReachSequenceDataset(
        ReachDatasetConfig(
            dataset_path=dataset_path,
            horizon=4,
            n_obs_steps=2,
            goal_marker_points=0,
        ),
        split="train",
    )

    sample = dataset[0]

    assert set(sample) == {"obs", "action"}
    assert set(sample["obs"]) == {"point_cloud", "agent_pos"}
    assert sample["obs"]["point_cloud"].shape == (4, 4, 3)
    assert sample["obs"]["agent_pos"].shape == (4, 9)
    assert sample["action"].shape == (4, 7)
    assert "target_position" not in sample["obs"]
    assert dataset.shape_meta == reach_shape_meta(num_points=4, state_dim=9, action_dim=7)


def test_goal_marker_insertion_uses_ordered_tail_points(tmp_path) -> None:
    dataset_path = _write_reach_dataset(tmp_path, num_episodes=2, episode_length=4)
    dataset = ReachSequenceDataset(
        ReachDatasetConfig(
            dataset_path=dataset_path,
            horizon=4,
            n_obs_steps=2,
            goal_marker_points=2,
            goal_marker_radius=DEFAULT_GOAL_MARKER_RADIUS,
        ),
        split="train",
    )
    raw_dataset = ReachSequenceDataset(
        ReachDatasetConfig(
            dataset_path=dataset_path,
            horizon=4,
            n_obs_steps=2,
            goal_marker_points=0,
        ),
        split="train",
    )

    sample = dataset[0]
    raw_sample = raw_dataset[0]
    point_cloud = sample["obs"]["point_cloud"].numpy()

    assert set(sample["obs"]) == {"point_cloud", "agent_pos"}
    np.testing.assert_allclose(point_cloud[..., -2:, :], 0.0)
    np.testing.assert_allclose(
        point_cloud[..., :-2, :],
        raw_sample["obs"]["point_cloud"][..., :-2, :],
    )


def test_goal_marker_helper_supports_fixed_16_point_layout() -> None:
    point_cloud = np.zeros((1, 20, 3), dtype=np.float32)
    target = np.asarray([[0.2, -0.1, 0.3]], dtype=np.float32)

    transformed = insert_goal_marker_points(point_cloud, target, num_points=16, radius=0.015)
    expected = target[:, None, :] + goal_marker_offsets(num_points=16, radius=0.015)

    np.testing.assert_allclose(transformed[:, -16:, :], expected)
    np.testing.assert_allclose(transformed[:, :4, :], 0.0)


def test_reach_dataset_validation_split_is_deterministic(tmp_path) -> None:
    dataset_path = _write_reach_dataset(tmp_path, num_episodes=3, episode_length=4)
    config = ReachDatasetConfig(
        dataset_path=dataset_path,
        horizon=3,
        n_obs_steps=2,
        val_ratio=0.34,
        seed=7,
        goal_marker_points=0,
    )

    train = ReachSequenceDataset(config, split="train")
    val = ReachSequenceDataset(config, split="val")
    mask = validation_episode_mask(3, val_ratio=0.34, seed=7)

    assert mask.sum() == 1
    assert len(train) > 0
    assert len(val) > 0
    assert len(train) + len(val) == len(ReachSequenceDataset(config, split="all"))


def test_sequence_indices_pad_episode_start() -> None:
    indices = create_sequence_indices(
        np.asarray([4], dtype=np.int64),
        sequence_length=3,
        episode_mask=np.asarray([True]),
        pad_before=1,
    )

    assert indices[0].tolist() == [0, 2, 1, 3]


def test_sequence_indices_pad_episode_end_for_terminal_chunks() -> None:
    indices = create_sequence_indices(
        np.asarray([4], dtype=np.int64),
        sequence_length=3,
        episode_mask=np.asarray([True]),
        pad_after=2,
    )

    assert indices[-1].tolist() == [3, 4, 0, 1]


def test_reach_dataset_normalizer_supports_dp3_loss(tmp_path) -> None:
    dataset_path = _write_reach_dataset(tmp_path, num_episodes=2, episode_length=4)
    dataset = ReachSequenceDataset(
        ReachDatasetConfig(
            dataset_path=dataset_path,
            horizon=4,
            n_obs_steps=2,
            goal_marker_points=2,
        ),
        split="train",
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    batch = next(iter(loader))
    policy = SimpleDP3(
        shape_meta=dataset.shape_meta,
        horizon=4,
        n_obs_steps=2,
        n_action_steps=1,
        goal_marker_points=2,
        goal_marker_radius=DEFAULT_GOAL_MARKER_RADIUS,
        num_inference_steps=2,
        encoder_output_dim=16,
        diffusion_step_embed_dim=32,
        down_dims=(32, 64),
        kernel_size=3,
        n_groups=8,
        pointcloud_encoder_cfg={
            "out_channels": 16,
            "use_layernorm": True,
            "final_norm": "layernorm",
        },
    )
    policy.set_normalizer(dataset.get_normalizer())

    loss, loss_dict = policy.compute_loss(batch)

    assert torch.isfinite(loss)
    assert loss_dict["bc_loss"] >= 0.0


def test_reach_dataset_normalizer_uses_bounded_rows(tmp_path) -> None:
    dataset_path = _write_reach_dataset(tmp_path, num_episodes=2, episode_length=5)
    dataset = ReachSequenceDataset(
        ReachDatasetConfig(
            dataset_path=dataset_path,
            horizon=4,
            n_obs_steps=2,
            goal_marker_points=0,
            normalizer_max_steps=3,
        ),
        split="train",
    )

    normalizer = dataset.get_normalizer()

    assert normalizer["action"].scale.shape == (7,)
    np.testing.assert_array_equal(
        normalizer_step_indices(total_steps=10, max_steps=3),
        np.asarray([0, 4, 9], dtype=np.int64),
    )


def test_reach_dataset_imports_keep_simulator_and_external_dp3_lazy() -> None:
    code = """
import importlib
import sys

importlib.import_module("pg3d.policies.dp3.reach_dataset")
assert "mani_skill" not in sys.modules
assert "sapien" not in sys.modules
assert "gymnasium" not in sys.modules
assert "diffusion_policy_3d" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def _write_reach_dataset(tmp_path, *, num_episodes: int, episode_length: int):
    episodes = []
    for episode_idx in range(num_episodes):
        state = np.full((episode_length, 9), episode_idx, dtype=np.float32)
        state[:, :7] += np.linspace(0.0, 0.3, episode_length, dtype=np.float32).reshape(-1, 1)
        action = state[:, :7] + 0.1
        point_cloud = np.zeros((episode_length, 4, 3), dtype=np.float32)
        point_cloud[..., 0] = np.linspace(0.0, 0.2, episode_length, dtype=np.float32).reshape(
            -1, 1
        )
        point_cloud[..., 1] = np.arange(4, dtype=np.float32)
        episodes.append(
            ReachEpisodeData(
                state=state,
                action=action.astype(np.float32),
                sim_action=np.concatenate(
                    [action.astype(np.float32), np.zeros((episode_length, 1), dtype=np.float32)],
                    axis=1,
                ),
                point_cloud=point_cloud,
                robot_mask=np.zeros((episode_length, 4), dtype=bool),
                point_valid_mask=np.ones((episode_length, 4), dtype=bool),
                target_position=np.zeros((episode_length, 3), dtype=np.float32),
                tcp_pose=np.zeros((episode_length, 7), dtype=np.float32),
                success=np.ones((episode_length,), dtype=bool),
                metadata={"seed": episode_idx, "success": True},
            )
        )
    output = tmp_path / "reach.zarr"
    write_reach_zarr(
        output,
        episodes,
        metadata={"env_id": "PG3DReach-Narrow-v0", "env_kwargs": {}, "action_mode": "abs_joint"},
    )
    return output
