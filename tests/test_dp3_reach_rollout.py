from __future__ import annotations

import subprocess
import sys

import numpy as np
import torch

from scripts.rollout_dp3_reach_policy import (
    _distance_drift,
    append_obs_window,
    make_initial_obs_window,
    obs_window_to_torch,
    policy_action_to_sim_action,
    rollout_spec_video_stem,
    select_mixed_rollout_specs,
    select_random_dataset_rollout_specs,
    select_rollout_specs,
)


def test_policy_action_to_sim_action_supports_abs_and_delta() -> None:
    action = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)
    state = np.asarray([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 0.04, 0.04], dtype=np.float32)

    abs_action = policy_action_to_sim_action(
        action,
        state,
        action_mode="abs_joint",
        sim_action_dim=8,
        low=np.full((8,), -10.0, dtype=np.float32),
        high=np.full((8,), 10.0, dtype=np.float32),
        gripper_open=0.04,
    )
    delta_action = policy_action_to_sim_action(
        action,
        state,
        action_mode="delta_joint",
        sim_action_dim=8,
        low=np.full((8,), -10.0, dtype=np.float32),
        high=np.full((8,), 10.0, dtype=np.float32),
        gripper_open=0.04,
    )

    np.testing.assert_allclose(abs_action[:7], action)
    assert abs_action[-1] == np.float32(0.04)
    np.testing.assert_allclose(delta_action[:7], state[:7] + action)
    assert delta_action[-1] == np.float32(0.04)

    np.testing.assert_allclose(
        policy_action_to_sim_action(
            action,
            state,
            action_mode="abs_joint",
            sim_action_dim=7,
            low=np.full((7,), -10.0, dtype=np.float32),
            high=np.full((7,), 10.0, dtype=np.float32),
        ),
        action,
    )


def test_policy_action_to_sim_action_clips_bounds() -> None:
    sim_action = policy_action_to_sim_action(
        np.asarray([2.0] * 7, dtype=np.float32),
        np.zeros((9,), dtype=np.float32),
        action_mode="abs_joint",
        sim_action_dim=8,
        low=np.full((8,), -1.0, dtype=np.float32),
        high=np.full((8,), 1.0, dtype=np.float32),
        gripper_open=0.04,
    )

    np.testing.assert_allclose(sim_action, np.asarray([1.0] * 7 + [0.04], dtype=np.float32))


def test_observation_window_pads_and_rolls_without_aliasing() -> None:
    first = _entry(1.0)
    window = make_initial_obs_window(first, n_obs_steps=2)
    first["agent_pos"][0] = 99.0

    assert len(window) == 2
    assert window[0]["agent_pos"][0] == 1.0
    assert window[1]["agent_pos"][0] == 1.0

    window = append_obs_window(window, _entry(2.0), n_obs_steps=2)

    assert len(window) == 2
    assert window[0]["agent_pos"][0] == 1.0
    assert window[1]["agent_pos"][0] == 2.0


def test_select_rollout_specs_dataset_and_fresh_seed_skipping() -> None:
    dataset_specs = select_rollout_specs(
        source="dataset",
        dataset_episode_seeds=[5, 6, 7],
        episodes=2,
        episode_indices=None,
    )
    indexed_specs = select_rollout_specs(
        source="dataset",
        dataset_episode_seeds=[5, 6, 7],
        episodes=2,
        episode_indices=[2, 0],
    )
    fresh_specs = select_rollout_specs(
        source="fresh",
        dataset_episode_seeds=[10000, 10001],
        episodes=3,
        seed_start=10000,
    )

    assert [spec.seed for spec in dataset_specs] == [5, 6]
    assert [spec.dataset_episode_index for spec in indexed_specs] == [2, 0]
    assert [spec.seed for spec in indexed_specs] == [7, 5]
    assert [spec.seed for spec in fresh_specs] == [10002, 10003, 10004]


def test_select_mixed_rollout_specs_defaults_to_three_dataset_two_fresh() -> None:
    specs = select_mixed_rollout_specs(
        dataset_episode_seeds=[1, 2, 3, 10000],
        total_count=5,
        seed_start=10000,
    )

    assert [spec.source for spec in specs] == ["dataset", "dataset", "dataset", "fresh", "fresh"]
    assert [spec.seed for spec in specs] == [1, 2, 3, 10001, 10002]


def test_select_mixed_rollout_specs_seeded_is_diverse_and_deterministic() -> None:
    seeds = list(range(100))

    legacy = select_mixed_rollout_specs(
        dataset_episode_seeds=seeds, total_count=5, seed_start=10000
    )
    seeded = select_mixed_rollout_specs(
        dataset_episode_seeds=seeds, total_count=5, seed_start=10000, selection_seed=7
    )
    seeded_again = select_mixed_rollout_specs(
        dataset_episode_seeds=seeds, total_count=5, seed_start=10000, selection_seed=7
    )
    next_step = select_mixed_rollout_specs(
        dataset_episode_seeds=seeds, total_count=5, seed_start=10000, selection_seed=8
    )

    legacy_dataset_idx = [s.dataset_episode_index for s in legacy if s.source == "dataset"]
    seeded_dataset_idx = [s.dataset_episode_index for s in seeded if s.source == "dataset"]
    # Legacy keeps the first N; the seeded draw spreads across the whole dataset.
    assert legacy_dataset_idx == [0, 1, 2]
    assert seeded_dataset_idx != [0, 1, 2]
    assert len(set(seeded_dataset_idx)) == 3
    # Same seed reproduces; the next checkpoint (seed+1) resamples a new subset.
    assert [s.seed for s in seeded] == [s.seed for s in seeded_again]
    assert [s.dataset_episode_index for s in seeded] != [
        s.dataset_episode_index for s in next_step
    ]


def test_select_random_dataset_rollout_specs_is_deterministic_and_clamped() -> None:
    seeds = [10, 11, 12, 13, 14, 15]

    first = select_random_dataset_rollout_specs(
        dataset_episode_seeds=seeds,
        total_count=3,
        seed=7,
    )
    second = select_random_dataset_rollout_specs(
        dataset_episode_seeds=seeds,
        total_count=3,
        seed=7,
    )
    clamped = select_random_dataset_rollout_specs(
        dataset_episode_seeds=seeds[:2],
        total_count=5,
        seed=7,
    )

    assert [spec.dataset_episode_index for spec in first] == [
        spec.dataset_episode_index for spec in second
    ]
    assert len({spec.dataset_episode_index for spec in first}) == 3
    assert len(clamped) == 2
    assert all(spec.source == "dataset" for spec in first)


def test_rollout_spec_video_stem_includes_validation_episode_identity() -> None:
    spec = select_random_dataset_rollout_specs(
        dataset_episode_seeds=[20000, 20001, 20002],
        total_count=1,
        seed=0,
    )[0]

    stem = rollout_spec_video_stem(spec, validation=True)

    assert stem.startswith("validation_episode_")
    assert stem.endswith(f"_seed_{spec.seed}")


def test_distance_drift_ignores_non_finite_values() -> None:
    assert np.isclose(_distance_drift([0.02, float("nan"), 0.05, 0.03]), 0.03)


def test_obs_window_to_torch_inserts_goal_marker_tail_points() -> None:
    window = [_entry(0.0), _entry(1.0)]

    batch = obs_window_to_torch(
        window,
        device=torch.device("cpu"),
        goal_marker_points=2,
        goal_marker_radius=0.015,
    )

    points = batch["point_cloud"].cpu().numpy()
    np.testing.assert_allclose(points[0, 0, -2:, :], np.zeros((2, 3), dtype=np.float32))
    np.testing.assert_allclose(points[0, 1, -2:, :], np.ones((2, 3), dtype=np.float32))


def test_rollout_script_import_keeps_simulator_lazy() -> None:
    code = """
import importlib
import sys

importlib.import_module("scripts.rollout_dp3_reach_policy")
assert "mani_skill" not in sys.modules
assert "sapien" not in sys.modules
assert "gymnasium" not in sys.modules
assert "rerun" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def _entry(value: float) -> dict[str, np.ndarray | bool | float]:
    return {
        "point_cloud": np.full((4, 3), value, dtype=np.float32),
        "robot_mask": np.zeros((4,), dtype=bool),
        "point_valid_mask": np.ones((4,), dtype=bool),
        "agent_pos": np.full((9,), value, dtype=np.float32),
        "target_position": np.full((3,), value, dtype=np.float32),
        "tcp_pose": np.full((7,), value, dtype=np.float32),
        "success": False,
        "final_distance": value,
    }
