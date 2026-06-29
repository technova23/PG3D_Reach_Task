from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import scripts.train_dp3_reach as train
from pg3d.envs.maniskill_adapter.dataset import ReachEpisodeData, write_reach_zarr
from pg3d.policies.dp3 import SimpleDP3
from pg3d.policies.dp3.checkpoint import (
    checkpoint_path_for_step,
    load_reach_policy_from_checkpoint,
    save_reach_policy_checkpoint,
    should_save_checkpoint,
)
from pg3d.policies.dp3.normalizer import LinearNormalizer
from pg3d.policies.dp3.reach_dataset import reach_shape_meta


def test_lr_scale_uses_warmup_then_cosine_decay() -> None:
    assert train.lr_scale_for_step(0, warmup_steps=5, total_steps=10, min_lr_scale=0.0) == 0.2
    assert train.lr_scale_for_step(4, warmup_steps=5, total_steps=10, min_lr_scale=0.0) == 1.0
    assert train.lr_scale_for_step(10, warmup_steps=5, total_steps=10, min_lr_scale=0.0) == 0.0


def test_clip_gradients_reduces_norm() -> None:
    model = torch.nn.Linear(4, 1)
    loss = model(torch.ones((2, 4))).sum()
    loss.backward()

    before = train._grad_norm(model)
    after = train._clip_gradients(model, 0.1)

    assert before > 0.1
    assert after <= 0.1001


def test_checkpoint_saves_and_loads_ema_by_default(tmp_path: Path) -> None:
    policy_kwargs = _tiny_policy_kwargs()
    policy = SimpleDP3(**policy_kwargs)
    policy.set_normalizer(
        LinearNormalizer.identity_for_keys(["action", "point_cloud", "agent_pos"])
    )
    ema_policy = copy.deepcopy(policy)
    with torch.no_grad():
        for param in ema_policy.parameters():
            if param.numel():
                param.add_(0.5)
                break

    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    checkpoint = tmp_path / "policy.pt"
    save_reach_policy_checkpoint(
        checkpoint,
        policy=policy,
        ema_policy=ema_policy,
        optimizer=optimizer,
        scheduler=None,
        policy_kwargs=policy_kwargs,
        args=argparse.Namespace(dataset=tmp_path / "dataset.zarr"),
        step=3,
        best_val_loss=0.1,
    )

    raw = load_reach_policy_from_checkpoint(
        checkpoint,
        device=torch.device("cpu"),
        prefer_ema=False,
    )
    ema = load_reach_policy_from_checkpoint(
        checkpoint,
        device=torch.device("cpu"),
        prefer_ema=True,
    )

    raw_first = _first_nonempty_parameter(raw)
    ema_first = _first_nonempty_parameter(ema)
    assert not torch.allclose(raw_first, ema_first)


def test_checkpoint_path_helper_uses_step_filenames(tmp_path: Path) -> None:
    assert checkpoint_path_for_step(tmp_path, 5000) == tmp_path / "step_00005000.pt"
    assert (
        checkpoint_path_for_step(tmp_path, 123, final=True)
        == tmp_path / "final_step_00000123.pt"
    )
    assert should_save_checkpoint(10, 5)
    assert not should_save_checkpoint(10, 0)


def test_checkpoint_dir_replaces_checkpoint_out_cli(tmp_path: Path) -> None:
    args = train.parse_args(
        [
            "--max-steps",
            "1",
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--checkpoint-rollout-dataset",
            str(tmp_path / "val.zarr"),
            "--checkpoint-rollout-selection-seed",
            "11",
        ]
    )

    assert args.checkpoint_dir == tmp_path / "checkpoints"
    assert args.checkpoint_rollout_dataset == tmp_path / "val.zarr"
    assert args.checkpoint_rollout_selection_seed == 11
    assert not hasattr(args, "checkpoint_out")
    with pytest.raises(SystemExit):
        train.parse_args(["--checkpoint-out", str(tmp_path / "policy.pt")])

    default_seed_args = train.parse_args(["--max-steps", "1", "--seed", "9"])
    assert default_seed_args.checkpoint_rollout_selection_seed == 9
    assert default_seed_args.goal_marker_points == 16
    assert default_seed_args.goal_marker_radius == 0.015


def test_trainer_writes_periodic_and_final_checkpoints(tmp_path: Path) -> None:
    dataset_path = _write_tiny_reach_dataset(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints"

    result = train.main(
        [
            "--dataset",
            str(dataset_path),
            "--device",
            "cpu",
            "--max-steps",
            "2",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--val-ratio",
            "0",
            "--horizon",
            "4",
            "--n-obs-steps",
            "2",
            "--n-action-steps",
            "1",
            "--goal-marker-points",
            "2",
            "--encoder-output-dim",
            "16",
            "--diffusion-step-embed-dim",
            "32",
            "--down-dims",
            "32",
            "64",
            "--kernel-size",
            "3",
            "--num-inference-steps",
            "2",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--checkpoint-every",
            "1",
            "--no-checkpoint-rollout-videos",
        ]
    )

    assert result == 0
    assert (checkpoint_dir / "step_00000001.pt").exists()
    assert (checkpoint_dir / "step_00000002.pt").exists()
    assert (checkpoint_dir / "final_step_00000002.pt").exists()


def test_checkpoint_rollout_failure_is_nonfatal(tmp_path: Path, monkeypatch) -> None:
    class FakeRun:
        def __init__(self) -> None:
            self.logged: list[tuple[dict[str, float], int]] = []

        def log(self, metrics, *, step: int) -> None:
            self.logged.append((metrics, step))

    def fail_rollout(*args, **kwargs) -> None:
        raise RuntimeError("rendering unavailable")

    run = FakeRun()
    args = argparse.Namespace(
        checkpoint_dir=tmp_path,
        checkpoint_rollout_videos=True,
    )
    monkeypatch.setattr(train, "_log_checkpoint_rollouts", fail_rollout)

    train._maybe_log_checkpoint_rollouts(
        run,
        args,
        train_dataset=object(),
        policy=object(),
        device=torch.device("cpu"),
        step=7,
        rollout_attempted_steps=set(),
    )

    assert run.logged == [({"rollout/skipped": 1.0}, 7)]


def test_checkpoint_rollout_dataset_selects_validation_specs(tmp_path: Path) -> None:
    validation_path = tmp_path / "val.zarr"
    validation_path.mkdir()
    (validation_path / "metadata.json").write_text(
        json.dumps(
            {
                "env_id": "PG3DReach-Workspace-v0",
                "env_kwargs": {},
                "episodes": [{"seed": 20000 + idx} for idx in range(10)],
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        checkpoint_rollout_dataset=validation_path,
        checkpoint_rollout_count=5,
        checkpoint_rollout_selection_seed=3,
        checkpoint_rollout_seed_start=10000,
        checkpoint_rollout_resample_per_step=False,
    )
    train_dataset = type(
        "Dataset",
        (),
        {"metadata": {"episodes": [{"seed": 0}, {"seed": 1}, {"seed": 10000}]}},
    )()

    metadata, specs, using_validation_dataset = train._checkpoint_rollout_metadata_and_specs(
        args,
        train_dataset=train_dataset,  # type: ignore[arg-type]
        step=0,
    )

    assert using_validation_dataset
    assert metadata["env_id"] == "PG3DReach-Workspace-v0"
    assert len(specs) == 5
    assert all(spec.source == "dataset" for spec in specs)
    assert all(spec.seed >= 20000 for spec in specs)
    assert len({spec.dataset_episode_index for spec in specs}) == 5


def test_checkpoint_rollout_without_validation_dataset_keeps_mixed_specs() -> None:
    args = argparse.Namespace(
        checkpoint_rollout_dataset=None,
        checkpoint_rollout_count=5,
        checkpoint_rollout_selection_seed=3,
        checkpoint_rollout_seed_start=10000,
        checkpoint_rollout_resample_per_step=False,
    )
    train_dataset = type(
        "Dataset",
        (),
        {"metadata": {"episodes": [{"seed": 1}, {"seed": 2}, {"seed": 3}, {"seed": 10000}]}},
    )()

    _metadata, specs, using_validation_dataset = train._checkpoint_rollout_metadata_and_specs(
        args,
        train_dataset=train_dataset,  # type: ignore[arg-type]
        step=0,
    )

    # Mixed specs now draw a seeded, diverse dataset subset (not the first N) plus
    # fresh seeds that avoid the training seeds.
    assert not using_validation_dataset
    assert [spec.source for spec in specs] == ["dataset", "dataset", "dataset", "fresh", "fresh"]
    dataset_specs = specs[:3]
    fresh_specs = specs[3:]
    training_seeds = {1, 2, 3, 10000}
    assert len({spec.dataset_episode_index for spec in dataset_specs}) == 3
    assert all(0 <= spec.dataset_episode_index < 4 for spec in dataset_specs)
    assert all(spec.seed not in training_seeds for spec in fresh_specs)
    assert len({spec.seed for spec in fresh_specs}) == 2


def test_train_script_import_keeps_checkpoint_rollout_deps_lazy() -> None:
    code = """
import importlib
import sys

importlib.import_module("scripts.train_dp3_reach")
assert "mani_skill" not in sys.modules
assert "sapien" not in sys.modules
assert "gymnasium" not in sys.modules
assert "rerun" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def _tiny_policy_kwargs() -> dict:
    return {
        "shape_meta": reach_shape_meta(num_points=4, state_dim=9, action_dim=7),
        "horizon": 4,
        "n_obs_steps": 2,
        "n_action_steps": 1,
        "num_inference_steps": 2,
        "encoder_output_dim": 16,
        "diffusion_step_embed_dim": 32,
        "down_dims": (32, 64),
        "kernel_size": 3,
        "n_groups": 8,
        "pointcloud_encoder_cfg": {
            "out_channels": 16,
            "use_layernorm": True,
            "final_norm": "layernorm",
        },
    }


def _first_nonempty_parameter(model: torch.nn.Module) -> torch.Tensor:
    for param in model.parameters():
        if param.numel():
            return param.detach()
    raise AssertionError("model has no non-empty parameters")


def _write_tiny_reach_dataset(tmp_path: Path) -> Path:
    episodes = []
    for episode_idx in range(2):
        episode_length = 4
        state = np.full((episode_length, 9), episode_idx, dtype=np.float32)
        state[:, :7] += np.linspace(0.0, 0.2, episode_length, dtype=np.float32).reshape(-1, 1)
        action = state[:, :7] + 0.05
        point_cloud = np.zeros((episode_length, 4, 3), dtype=np.float32)
        point_cloud[..., 0] = np.linspace(0.0, 0.1, episode_length, dtype=np.float32).reshape(
            -1, 1
        )
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
