from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from pg3d.envs.maniskill_adapter.dataset import PointCloudCropConfig
from pg3d.eval import (
    AvoidOverlayConfig,
    EpisodePath,
    NominalPathAvoidConfig,
    TimingRecorder,
    candidate_feasibility_fraction,
    concatenate_rollouts,
    direct_path_avoid_region,
    episode_metric_row,
    load_episode_constraints,
    min_constraint_clearance,
    nominal_path_avoid_region,
    path_satisfies_constraints,
    progress_series,
    save_episode_constraints,
    scene_context_for_constraints,
    select_artifact_episode_indices,
    should_emit_episode_artifact,
    success_rate_ci_rows,
    summarize_metrics,
    validate_planning_horizons,
    wilson_interval,
)
from pg3d.world_model import ActionChunk, ImaginedRollout
from scripts.build_nominal_path_constraints import (
    parse_args as parse_builder_args,
)
from scripts.eval_constrained_reach import (
    DP3ChunkPolicyAdapter,
    _artifact_selection_summary,
    _build_multichunk_candidates,
    _constraint_source_summary,
    _constraints_for_episode,
    _obs_windows_to_torch,
    _read_episode_indices_file,
    _seed_torch,
)
from scripts.eval_constrained_reach import (
    parse_args as parse_eval_args,
)
from scripts.rollout_dp3_reach_policy import RolloutSpec


def test_direct_path_avoid_region_and_json_persistence(tmp_path: Path) -> None:
    constraint = direct_path_avoid_region(
        start_tcp=[0.0, 0.0, 0.2],
        target_position=[0.4, 0.0, 0.2],
        config=AvoidOverlayConfig(radius=0.08),
    )

    np.testing.assert_allclose(constraint.region.center, [0.2, 0.0, 0.2])
    assert constraint.region.radius == pytest.approx(0.08)

    path = tmp_path / "constraints" / "episode_000.json"
    save_episode_constraints(path, [constraint])

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded[0]["type"] == "avoid_region"
    assert loaded[0]["region"]["type"] == "sphere"


def test_direct_path_avoid_region_clamps_radius_for_short_paths() -> None:
    constraint = direct_path_avoid_region(
        start_tcp=[0.0, 0.0, 0.0],
        target_position=[0.1, 0.0, 0.0],
        config=AvoidOverlayConfig(radius=0.08, min_radius=0.02),
    )

    assert constraint.region.radius == pytest.approx(0.045)


def test_nominal_path_avoid_region_uses_arc_length_fraction(tmp_path: Path) -> None:
    tcp_path = np.asarray(
        [
            [0.0, 0.0, 0.2],
            [2.0, 0.0, 0.2],
            [2.0, 2.0, 0.2],
        ],
        dtype=np.float32,
    )

    constraint = nominal_path_avoid_region(
        tcp_path,
        config=NominalPathAvoidConfig(radius=0.03, path_fraction=0.75),
    )

    np.testing.assert_allclose(constraint.region.center, [2.0, 1.0, 0.2])
    assert constraint.region.radius == pytest.approx(0.03)
    assert constraint.name == "nominal_path_avoid_region"

    path = tmp_path / "episode_000.json"
    save_episode_constraints(path, [constraint])
    loaded = load_episode_constraints(path)

    assert len(loaded) == 1
    np.testing.assert_allclose(loaded[0].region.center, [2.0, 1.0, 0.2])


def test_nominal_path_avoid_region_validates_inputs() -> None:
    with pytest.raises(ValueError, match="radius"):
        nominal_path_avoid_region(
            [[0.0, 0.0, 0.0]],
            config=NominalPathAvoidConfig(radius=0.0),
        )
    with pytest.raises(ValueError, match="fraction"):
        nominal_path_avoid_region(
            [[0.0, 0.0, 0.0]],
            config=NominalPathAvoidConfig(path_fraction=1.5),
        )
    with pytest.raises(ValueError, match=r"\[T, 3\]"):
        nominal_path_avoid_region([0.0, 0.0, 0.0])


def test_wilson_interval_bounds_known_center() -> None:
    low, high = wilson_interval(5, 10)

    assert 0.23 < low < 0.24
    assert 0.76 < high < 0.77


def test_episode_metric_row_computes_clearance_and_combined_success() -> None:
    constraint = direct_path_avoid_region(
        start_tcp=[0.0, 0.0, 0.0],
        target_position=[1.0, 0.0, 0.0],
        config=AvoidOverlayConfig(radius=0.1),
    )
    path = EpisodePath()
    path.append(tcp_position=[0.0, 0.2, 0.0], q=[0.0, 0.0], target_distance=1.0)
    path.append(tcp_position=[0.5, 0.2, 0.0], q=[0.1, 0.0], target_distance=0.5)
    path.append(tcp_position=[1.0, 0.2, 0.0], q=[0.2, 0.0], target_distance=0.0)

    row = episode_metric_row(
        method="reranking",
        episode=0,
        seed=100,
        path=path,
        constraints=[constraint],
        reach_success=True,
        first_success_step=2,
        steps=2,
        replans=1,
        candidate_feasibility_fraction=0.5,
    )

    assert row["reach_success"] is True
    assert row["constraint_satisfied"] is True
    assert row["combined_success"] is True
    assert row["final_target_distance"] == pytest.approx(0.0)
    assert row["min_clearance"] == pytest.approx(0.1)
    assert row["candidate_feasibility_fraction"] == pytest.approx(0.5)


def test_constraint_satisfaction_fails_for_path_inside_sphere() -> None:
    constraint = direct_path_avoid_region(
        start_tcp=[0.0, 0.0, 0.0],
        target_position=[1.0, 0.0, 0.0],
        config=AvoidOverlayConfig(radius=0.1),
    )
    path = np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float32)

    assert min_constraint_clearance(path, [constraint]) < 0.0
    assert not path_satisfies_constraints(path, [constraint])


def test_validate_planning_horizons() -> None:
    validate_planning_horizons(planning_horizon_chunks=2, execution_horizon_chunks=1)

    with pytest.raises(ValueError, match="planning_horizon_chunks"):
        validate_planning_horizons(planning_horizon_chunks=0, execution_horizon_chunks=1)
    with pytest.raises(ValueError, match="<="):
        validate_planning_horizons(planning_horizon_chunks=1, execution_horizon_chunks=2)


def test_concatenate_rollouts_combines_multichunk_candidate() -> None:
    first = _rollout([[0.1] * 7, [0.2] * 7])
    second = _rollout([[0.3] * 7])

    combined = concatenate_rollouts([first, second], metadata={"candidate": 1})

    assert combined.action_chunk.horizon == 3
    assert combined.q.shape == (3, 9)
    assert combined.eef_path.shape == (3, 3)
    assert len(combined.scene_point_clouds) == 3
    assert combined.metadata["planning_horizon_chunks"] == 2
    assert combined.metadata["candidate"] == 1


def test_summarize_metrics_uses_stable_schema() -> None:
    rows = []
    for idx, success in enumerate([True, False]):
        path = EpisodePath()
        path.append(tcp_position=[0.0, 0.2, 0.0], q=[0.0, 0.0], target_distance=1.0)
        path.append(tcp_position=[1.0, 0.2, 0.0], q=[0.1, 0.0], target_distance=0.1)
        rows.append(
            episode_metric_row(
                method="base",
                episode=idx,
                seed=idx,
                path=path,
                constraints=[],
                reach_success=success,
                first_success_step=1 if success else None,
                steps=1,
                replans=1,
                candidate_feasibility_fraction=None,
            )
        )

    summary = summarize_metrics(rows)

    assert summary["base"]["episodes"] == 2
    assert summary["base"]["reach_success_rate"] == pytest.approx(0.5)
    assert "combined_success_wilson_low" in summary["base"]
    assert "final_target_distance_mean" in summary["base"]


def test_success_rate_ci_rows_accepts_full_summary() -> None:
    summary = {
        "by_method": {
            "base": {
                "reach_success_rate": 0.25,
                "reach_success_wilson_low": 0.1,
                "reach_success_wilson_high": 0.5,
                "constraint_satisfied_rate": 0.75,
                "constraint_satisfied_wilson_low": 0.5,
                "constraint_satisfied_wilson_high": 0.9,
                "combined_success_rate": 0.2,
                "combined_success_wilson_low": 0.05,
                "combined_success_wilson_high": 0.45,
            }
        }
    }

    rows = success_rate_ci_rows(summary)

    assert [row["metric"] for row in rows] == [
        "reach_success",
        "constraint_satisfied",
        "combined_success",
    ]
    assert rows[0]["method"] == "base"
    assert rows[0]["err_low"] == pytest.approx(0.15)
    assert rows[0]["err_high"] == pytest.approx(0.25)


def test_candidate_feasibility_fraction_validates_counts() -> None:
    assert candidate_feasibility_fraction(1, 4) == pytest.approx(0.25)
    assert candidate_feasibility_fraction(0, 0) is None
    with pytest.raises(ValueError):
        candidate_feasibility_fraction(2, 1)


def test_timing_recorder_aggregates_json_safe_events() -> None:
    recorder = TimingRecorder(enabled=True)

    with recorder.time("policy_sampling", k=16):
        pass
    with recorder.time("policy_sampling", k=32):
        pass

    summary = recorder.summary()
    events = [event.to_json() for event in recorder.events]

    assert summary["policy_sampling"]["count"] == pytest.approx(2.0)
    assert summary["policy_sampling"]["total"] >= 0.0
    assert events[0]["metadata"]["k"] == 16


def test_periodic_artifact_selection_includes_first_and_interval() -> None:
    assert should_emit_episode_artifact(0, 10)
    assert not should_emit_episode_artifact(8, 10)
    assert should_emit_episode_artifact(9, 10)
    with pytest.raises(ValueError):
        should_emit_episode_artifact(0, 0)


def test_artifact_episode_selection_supports_random_periodic_and_all() -> None:
    episodes = list(range(10))

    random_first = select_artifact_episode_indices(
        episodes,
        selection="random",
        count=5,
        seed=123,
        every_episodes=10,
    )
    random_second = select_artifact_episode_indices(
        episodes,
        selection="random",
        count=5,
        seed=123,
        every_episodes=10,
    )
    periodic = select_artifact_episode_indices(
        episodes,
        selection="periodic",
        count=5,
        seed=123,
        every_episodes=4,
    )
    all_episodes = select_artifact_episode_indices(
        episodes,
        selection="all",
        count=5,
        seed=123,
        every_episodes=10,
    )

    assert random_first == random_second
    assert len(random_first) == 5
    assert len(set(random_first)) == 5
    assert periodic == [0, 3, 7]
    assert all_episodes == episodes


def test_artifact_selection_summary_records_episode_indices_and_seeds() -> None:
    specs = [
        RolloutSpec(
            output_index=idx,
            seed=20000 + idx,
            source="dataset",
            dataset_episode_index=idx,
        )
        for idx in range(4)
    ]
    args = type(
        "Args",
        (),
        {
            "artifact_selection": "random",
            "artifact_episode_count": 2,
            "artifact_selection_seed": 123,
        },
    )()

    summary = _artifact_selection_summary(
        specs,
        video_episode_indices={1, 3},
        rerun_episode_indices={3},
        args=args,
    )

    assert summary["selection"] == "random"
    assert [row["dataset_episode_index"] for row in summary["video"]] == [1, 3]
    assert [row["seed"] for row in summary["rerun"]] == [20003]


def test_eval_artifact_selection_seed_defaults_to_run_seed(tmp_path: Path) -> None:
    args = parse_eval_args(
        [
            "--checkpoint",
            str(tmp_path / "policy.pt"),
            "--dataset",
            str(tmp_path / "dataset.zarr"),
            "--output-dir",
            str(tmp_path / "eval"),
            "--seed",
            "13",
        ]
    )

    assert args.artifact_selection == "periodic"
    assert args.artifact_episode_count == 5
    assert args.artifact_selection_seed == 13


def test_eval_episode_indices_file_and_precomputed_constraints(tmp_path: Path) -> None:
    indices_path = tmp_path / "episode_indices.txt"
    indices_path.write_text("# selected base-success episodes\n3\n7\n", encoding="utf-8")
    constraints_dir = tmp_path / "constraints"
    constraint = nominal_path_avoid_region(
        [[0.0, 0.0, 0.2], [0.2, 0.0, 0.2]],
        config=NominalPathAvoidConfig(radius=0.03),
    )
    save_episode_constraints(constraints_dir / "episode_000.json", [constraint])
    args = parse_eval_args(
        [
            "--checkpoint",
            str(tmp_path / "policy.pt"),
            "--dataset",
            str(tmp_path / "dataset.zarr"),
            "--output-dir",
            str(tmp_path / "eval"),
            "--source",
            "dataset",
            "--episode-indices-file",
            str(indices_path),
            "--constraints-dir",
            str(constraints_dir),
        ]
    )

    assert _read_episode_indices_file(indices_path) == [3, 7]
    loaded = _constraints_for_episode(
        None,
        spec=RolloutSpec(
            output_index=0,
            seed=20003,
            source="dataset",
            dataset_episode_index=3,
        ),
        crop_config=PointCloudCropConfig(
            bounds=np.asarray([[-1, 1], [-1, 1], [-1, 1]], dtype=np.float32),
            num_points=4,
        ),
        args=args,
    )

    assert args.constraints_dir == constraints_dir
    assert _constraint_source_summary(args)["type"] == "precomputed"
    np.testing.assert_allclose(loaded[0].region.center, [0.1, 0.0, 0.2])


def test_eval_episode_indices_file_requires_dataset_source(tmp_path: Path) -> None:
    indices_path = tmp_path / "episode_indices.txt"
    indices_path.write_text("0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source dataset"):
        parse_eval_args(
            [
                "--checkpoint",
                str(tmp_path / "policy.pt"),
                "--dataset",
                str(tmp_path / "dataset.zarr"),
                "--output-dir",
                str(tmp_path / "eval"),
                "--source",
                "fresh",
                "--episode-indices-file",
                str(indices_path),
            ]
        )


def test_nominal_path_constraint_builder_defaults(tmp_path: Path) -> None:
    args = parse_builder_args(
        [
            "--checkpoint",
            str(tmp_path / "policy.pt"),
            "--dataset",
            str(tmp_path / "dataset.zarr"),
            "--output-dir",
            str(tmp_path / "constraints"),
        ]
    )

    assert args.episodes == 25
    assert args.avoid_radius == pytest.approx(0.03)
    assert args.path_fraction == pytest.approx(0.5)
    assert args.min_successes == 15


def test_eval_constraint_overlay_flags_parse_and_validate(tmp_path: Path) -> None:
    args = parse_eval_args(
        [
            "--checkpoint",
            str(tmp_path / "policy.pt"),
            "--dataset",
            str(tmp_path / "dataset.zarr"),
            "--output-dir",
            str(tmp_path / "eval"),
            "--no-constraint-overlay-video",
            "--constraint-overlay-alpha",
            "0.4",
            "--constraint-overlay-color",
            "0.8",
            "0.2",
            "0.1",
        ]
    )

    assert args.constraint_overlay_video is False
    assert args.constraint_overlay_alpha == pytest.approx(0.4)
    assert args.constraint_overlay_color == [0.8, 0.2, 0.1]

    with pytest.raises(ValueError, match="constraint-overlay-alpha"):
        parse_eval_args(
            [
                "--checkpoint",
                str(tmp_path / "policy.pt"),
                "--dataset",
                str(tmp_path / "dataset.zarr"),
                "--output-dir",
                str(tmp_path / "eval"),
                "--constraint-overlay-alpha",
                "1.5",
            ]
        )


def test_progress_series_tracks_cumulative_metrics() -> None:
    rows = [
        _metric_row(method="base", episode=0, reach=True, constraint=False),
        _metric_row(method="base", episode=1, reach=True, constraint=True),
    ]

    series = progress_series(rows)

    assert series["base"]["reach_success_rate"] == [1.0, 1.0]
    assert series["base"]["constraint_satisfied_rate"] == [0.0, 0.5]
    assert series["base"]["combined_success_rate"] == [0.0, 0.5]


def test_dp3_adapter_batches_multiple_windows() -> None:
    policy = _FakeDP3Policy(n_action_steps=2, n_obs_steps=2)
    adapter = DP3ChunkPolicyAdapter(
        policy,  # type: ignore[arg-type]
        action_mode="abs_joint",
        device=torch.device("cpu"),
        policy_batch_size=2,
        timer=TimingRecorder(enabled=True),
    )

    chunks = adapter.sample_action_chunks_for_windows([_window(), _window(), _window()])

    assert len(chunks) == 3
    assert policy.batch_sizes == [2, 1]
    assert chunks[0].actions.shape == (2, 7)


def test_constrained_eval_batch_input_inserts_goal_marker_tail_points() -> None:
    batch = _obs_windows_to_torch(
        [_window()],
        device=torch.device("cpu"),
        goal_marker_points=2,
        goal_marker_radius=0.015,
    )

    points = batch["point_cloud"].cpu().numpy()
    expected = np.broadcast_to(
        np.asarray([1.0, 0.0, 0.2], dtype=np.float32),
        (2, 2, 3),
    )
    np.testing.assert_allclose(points[0, :, -2:, :], expected)


def test_seed_torch_controls_policy_sampling_rng() -> None:
    _seed_torch(123)
    first = torch.randn(4)
    _seed_torch(123)
    second = torch.randn(4)

    torch.testing.assert_close(first, second)


def test_fast_multichunk_renders_only_feedback_states() -> None:
    policy = _FakeDP3Policy(n_action_steps=2, n_obs_steps=2)
    adapter = DP3ChunkPolicyAdapter(
        policy,  # type: ignore[arg-type]
        action_mode="abs_joint",
        device=torch.device("cpu"),
        policy_batch_size=8,
        timer=TimingRecorder(enabled=True),
    )
    provider = _FakeFastProvider()
    constraint = direct_path_avoid_region(
        start_tcp=[0.0, 0.0, 0.2],
        target_position=[1.0, 0.0, 0.2],
    )

    candidates = _build_multichunk_candidates(
        adapter=adapter,
        world_model=None,  # type: ignore[arg-type]
        provider=provider,  # type: ignore[arg-type]
        current_entry=_entry(),
        obs_window=_window(),
        scene=scene_context_for_constraints(
            target_position=[1.0, 0.0, 0.2],
            constraints=[constraint],
        ),
        constraints=[constraint],
        crop_config=PointCloudCropConfig(
            bounds=np.asarray([[-1, 2], [-1, 1], [0, 1]], dtype=np.float32),
            num_points=4,
        ),
        goal_thresh=0.01,
        planning_horizon_chunks=2,
        geometry_mode="fast",
        attempted_k=3,
        start_index=0,
        rng=np.random.default_rng(0),
        timer=TimingRecorder(enabled=True),
    )

    assert len(candidates) == 3
    assert provider.eef_calls == 12
    assert provider.robot_cloud_calls == 6


def test_eval_helpers_import_without_heavy_runtime_deps() -> None:
    code = """
import importlib
import sys

importlib.import_module("pg3d.eval")
assert "mani_skill" not in sys.modules
assert "sapien" not in sys.modules
assert "gymnasium" not in sys.modules
assert "rerun" not in sys.modules
assert "wandb" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def _rollout(actions: list[list[float]]) -> ImaginedRollout:
    chunk = ActionChunk(
        actions=np.asarray(actions, dtype=np.float32),
        action_mode="abs_joint",
        dt=1.0,
    )
    horizon = chunk.horizon
    q = np.zeros((horizon, 9), dtype=np.float32)
    q[:, :7] = chunk.actions
    eef = np.stack(
        [
            np.asarray([float(idx), 0.0, 0.2], dtype=np.float32)
            for idx in range(horizon)
        ],
        axis=0,
    )
    return ImaginedRollout(
        q=q,
        eef_path=eef,
        robot_point_clouds=[np.zeros((1, 3), dtype=np.float32) for _ in range(horizon)],
        scene_point_clouds=[np.zeros((2, 3), dtype=np.float32) for _ in range(horizon)],
        robot_masks=[np.asarray([True, False], dtype=bool) for _ in range(horizon)],
        action_chunk=chunk,
    )


def _entry() -> dict[str, np.ndarray | bool | float]:
    return {
        "point_cloud": np.asarray(
            [
                [0.0, 0.0, 0.2],
                [0.1, 0.0, 0.2],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "robot_mask": np.asarray([False, True, False, False], dtype=bool),
        "point_valid_mask": np.asarray([True, True, False, False], dtype=bool),
        "agent_pos": np.asarray([0.0] * 7 + [0.04, 0.04], dtype=np.float32),
        "target_position": np.asarray([1.0, 0.0, 0.2], dtype=np.float32),
        "tcp_pose": np.asarray([0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "success": False,
        "final_distance": 1.0,
    }


def _window() -> list[dict[str, np.ndarray | bool | float]]:
    return [_entry(), _entry()]


def _metric_row(
    *,
    method: str,
    episode: int,
    reach: bool,
    constraint: bool,
) -> dict[str, object]:
    return {
        "method": method,
        "episode": episode,
        "seed": episode,
        "reach_success": reach,
        "constraint_satisfied": constraint,
        "combined_success": reach and constraint,
        "final_target_distance": 0.1,
        "min_clearance": 0.01 if constraint else -0.01,
        "candidate_feasibility_fraction": None,
        "fallback_count": 0,
    }


class _FakeDP3Policy:
    def __init__(self, *, n_action_steps: int, n_obs_steps: int) -> None:
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.batch_sizes: list[int] = []

    def predict_action(self, obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch_size = int(obs_dict["point_cloud"].shape[0])
        self.batch_sizes.append(batch_size)
        base = torch.arange(batch_size, dtype=torch.float32).reshape(batch_size, 1, 1)
        action = torch.ones((batch_size, self.n_action_steps, 7), dtype=torch.float32)
        return {"action": action * (base + 0.1)}


class _FakeFastProvider:
    def __init__(self) -> None:
        self.eef_calls = 0
        self.robot_cloud_calls = 0

    def end_effector_position_only(self, q: np.ndarray) -> np.ndarray:
        self.eef_calls += 1
        return np.asarray([q[0], 0.0, 0.2], dtype=np.float32)

    def robot_point_cloud(self, q: np.ndarray) -> np.ndarray:
        self.robot_cloud_calls += 1
        return np.asarray([[q[0], 0.0, 0.2]], dtype=np.float32)
