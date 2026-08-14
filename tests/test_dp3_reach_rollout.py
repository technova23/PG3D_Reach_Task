from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from pg3d.constraints import AvoidRegion, SphereRegion
from pg3d.policies.dp3.goal_markers import goal_marker_offsets
from scripts.rollout_dp3_reach_policy import (
    _distance_drift,
    _policy_point_cloud_semantics,
    _rerun_tcp_clearance,
    append_obs_window,
    make_initial_obs_window,
    obs_window_to_torch,
    policy_action_to_sim_action,
    rollout_spec_video_stem,
    save_rerun_timeline,
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
    assert [s.dataset_episode_index for s in seeded] != [s.dataset_episode_index for s in next_step]


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
    offsets = goal_marker_offsets(num_points=2, radius=0.015)
    np.testing.assert_allclose(points[0, 0, -2:, :], offsets)
    np.testing.assert_allclose(points[0, 1, -2:, :], 1.0 + offsets)


def test_rerun_semantics_match_final_policy_tensor_and_marker_overwrite() -> None:
    entry = _entry(0.0)
    entry["robot_mask"] = np.asarray([True, False, False, False])
    entry["obstacle_mask"] = np.asarray([False, False, True, True])
    entry["target_position"] = np.asarray([0.4, 0.2, 0.3], dtype=np.float32)

    semantics = _policy_point_cloud_semantics(
        entry,
        goal_marker_points=2,
        goal_marker_radius=0.015,
    )
    policy = obs_window_to_torch(
        [entry],
        device=torch.device("cpu"),
        goal_marker_points=2,
        goal_marker_radius=0.015,
    )

    np.testing.assert_allclose(
        semantics["all_points"],
        policy["point_cloud"].cpu().numpy()[0, 0],
    )
    assert semantics["robot_mask"].tolist() == [True, False, False, False]
    assert semantics["obstacle_mask"].tolist() == [False, False, False, False]
    assert semantics["goal_mask"].tolist() == [False, False, True, True]


def test_rerun_tcp_clearance_uses_constraint_margin() -> None:
    constraint = AvoidRegion(
        SphereRegion(center=[0.0, 0.0, 0.0], radius=0.1),
        margin=0.02,
    )

    assert np.isclose(
        _rerun_tcp_clearance(np.asarray([0.15, 0.0, 0.0]), [constraint]),
        0.03,
    )


def test_rerun35_export_retains_exact_policy_tensor_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = _entry(1.0)
    entry["robot_mask"][0] = True
    entry["obstacle_mask"] = np.asarray([False, True, False, False])
    entry["point_valid_mask"][-1] = False
    output = tmp_path / "episode_000.rrd"
    fake_python = tmp_path / "python"
    fake_python.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        "scripts.rollout_dp3_reach_policy._rerun35_exporter_python",
        lambda: fake_python,
    )

    def fake_run(command: list[str], *, check: bool) -> None:
        assert check is True
        Path(command[command.index("--output") + 1]).write_bytes(b"rrd35")

    monkeypatch.setattr("scripts.rollout_dp3_reach_policy.subprocess.run", fake_run)

    save_rerun_timeline(
        output,
        [entry],
        goal_marker_points=2,
        goal_marker_radius=0.015,
        recording_identity={
            "method": "reranking",
            "episode": 0,
            "simulator_seed": 123,
        },
        replans=[
            {
                "step": 0,
                "replan_index": 0,
                "itps_robot_points": np.zeros((2, 10, 3), dtype=np.float32),
                "itps_robot_link_indices": np.arange(10, dtype=np.int64),
                "itps_worst_points": [
                    {
                        "constraint_index": 0,
                        "position": [0.0, 0.0, 0.0],
                    }
                ],
            }
        ],
    )

    with np.load(output.with_suffix(".policy_input.npz"), allow_pickle=False) as bundle:
        semantics = _policy_point_cloud_semantics(
            entry,
            goal_marker_points=2,
            goal_marker_radius=0.015,
        )
        np.testing.assert_array_equal(bundle["point_cloud"][0], semantics["all_points"])
        np.testing.assert_array_equal(bundle["valid_mask"][0], semantics["all_valid_mask"])
        np.testing.assert_array_equal(bundle["obstacle_mask"][0], semantics["all_obstacle_mask"])
        np.testing.assert_array_equal(bundle["scene_mask"][0], semantics["all_scene_mask"])
        assert bundle["point_cloud"].shape == (1, 4, 3)
        assert bundle["itps_robot_points"].shape == (1, 2, 10, 3)
        np.testing.assert_array_equal(bundle["itps_robot_link_indices"], np.arange(10))
    assert output.read_bytes() == b"rrd35"
    metadata = json.loads(output.with_suffix(".policy_input.json").read_text(encoding="utf-8"))
    assert metadata["rerun_writer_version"] == "0.35.0"
    assert metadata["recording_identity"] == {
        "episode": 0,
        "method": "reranking",
        "simulator_seed": 123,
    }
    assert "itps_robot_points" not in metadata["replans"][0]
    assert metadata["replans"][0]["itps_robot_points_bundle_index"] == 0


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
