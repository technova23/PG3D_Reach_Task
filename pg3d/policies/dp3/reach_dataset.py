from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import zarr

from pg3d.policies.dp3.goal_markers import (
    DEFAULT_GOAL_MARKER_POINTS,
    DEFAULT_GOAL_MARKER_RADIUS,
    insert_goal_marker_points,
)
from pg3d.policies.dp3.normalizer import LinearNormalizer
from pg3d.policies.dp3.policy import DP3Batch

Split = Literal["train", "val", "all"]


@dataclass(frozen=True)
class ReachDatasetConfig:
    """Configuration for sampling DP3 action chunks from a pg3d reach Zarr dataset."""

    dataset_path: Path
    horizon: int = 16
    n_obs_steps: int = 2
    pad_before: int | None = None
    pad_after: int = 0
    val_ratio: float = 0.0
    seed: int = 42
    max_train_episodes: int | None = None
    goal_marker_points: int = DEFAULT_GOAL_MARKER_POINTS
    goal_marker_radius: float = DEFAULT_GOAL_MARKER_RADIUS
    use_goal_encoder: bool = False
    normalizer_max_steps: int | None = 4_096
    normalize_mode: Literal["standardize", "minmax"] = "minmax"

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.n_obs_steps <= 0:
            raise ValueError("n_obs_steps must be positive")
        if self.n_obs_steps > self.horizon:
            raise ValueError("n_obs_steps must be <= horizon")
        if not 0.0 <= self.val_ratio < 1.0:
            raise ValueError("val_ratio must be in [0, 1)")
        if self.max_train_episodes is not None and self.max_train_episodes <= 0:
            raise ValueError("max_train_episodes must be positive")
        if self.goal_marker_points < 0:
            raise ValueError("goal_marker_points must be non-negative")
        if self.goal_marker_radius < 0:
            raise ValueError("goal_marker_radius must be non-negative")
        if self.normalizer_max_steps is not None and self.normalizer_max_steps <= 0:
            raise ValueError("normalizer_max_steps must be positive or None")
        if self.normalize_mode not in {"standardize", "minmax"}:
            raise ValueError("normalize_mode must be 'standardize' or 'minmax'")
        object.__setattr__(self, "dataset_path", Path(self.dataset_path))

    @property
    def resolved_pad_before(self) -> int:
        return self.n_obs_steps - 1 if self.pad_before is None else self.pad_before


class ReachSequenceDataset(torch.utils.data.Dataset):
    """Torch dataset for pg3d reach Zarr files.

    Policy observations intentionally contain only DP3-visible fields:
    point cloud and robot joint state. Simulator ground truth, success flags,
    point-valid masks, robot masks, and TCP/target debug arrays stay out of
    the training sample.
    """

    def __init__(
        self,
        config: ReachDatasetConfig | Path | str,
        *,
        split: Split = "train",
    ) -> None:
        super().__init__()
        self.config = (
            config
            if isinstance(config, ReachDatasetConfig)
            else ReachDatasetConfig(dataset_path=Path(config))
        )
        if split not in {"train", "val", "all"}:
            raise ValueError(f"unsupported split {split!r}")
        self.split = split
        self.root = zarr.open_group(str(self.config.dataset_path), mode="r")
        self.metadata = _load_metadata(self.config.dataset_path)
        self.episode_ends = np.asarray(self.root["meta"]["episode_ends"][:], dtype=np.int64)
        self.target_position_key = _target_position_key(self.root["data"])
        self._validate_arrays()
        episode_mask = self._episode_mask(split)
        self.indices = create_sequence_indices(
            self.episode_ends,
            sequence_length=self.config.horizon,
            episode_mask=episode_mask,
            pad_before=self.config.resolved_pad_before,
            pad_after=self.config.pad_after,
        )

    @property
    def shape_meta(self) -> dict[str, dict[str, dict[str, list[int]]]]:
        """Return the DP3 shape metadata implied by the Zarr arrays."""
        point_cloud = self.root["data"]["point_cloud"]
        state = self.root["data"]["state"]
        action = self.root["data"]["action"]
        return reach_shape_meta(
            num_points=int(point_cloud.shape[1]),
            point_dim=int(point_cloud.shape[2]),
            state_dim=int(state.shape[1]),
            action_dim=int(action.shape[1]),
            use_goal_encoder=self.config.use_goal_encoder,
        )

    @property
    def num_episodes(self) -> int:
        return int(len(self.episode_ends))

    def get_validation_dataset(self) -> ReachSequenceDataset:
        """Return a validation view using the same opened Zarr path and config."""
        return ReachSequenceDataset(copy.copy(self.config), split="val")

    def get_normalizer(self) -> LinearNormalizer:
        """Fit policy-field normalizers from deterministic Zarr rows.

        The generated reach datasets can contain tens of thousands of 1024-point
        observations. Reading all point clouds only to estimate XYZ statistics is
        slow and memory-hungry, so large datasets use an evenly spaced subset of
        timesteps while small smoke datasets remain exact.
        """
        data = self.root["data"]
        row_indices = normalizer_step_indices(
            total_steps=int(data["point_cloud"].shape[0]),
            max_steps=self.config.normalizer_max_steps,
        )
        point_cloud = np.asarray(data["point_cloud"].get_orthogonal_selection(row_indices))
        if self.config.goal_marker_points:
            point_cloud = insert_goal_marker_points(
                point_cloud,
                np.asarray(data[self.target_position_key].get_orthogonal_selection(row_indices)),
                num_points=self.config.goal_marker_points,
                radius=self.config.goal_marker_radius,
            )
        normalizer_data = {
            "point_cloud": point_cloud,
            "agent_pos": np.asarray(data["state"].get_orthogonal_selection(row_indices)),
            "action": np.asarray(data["action"].get_orthogonal_selection(row_indices)),
        }
        if self.config.use_goal_encoder:
            normalizer_data["goal_xyz"] = np.asarray(
                data[self.target_position_key].get_orthogonal_selection(row_indices)
            )
        # Only the action field interacts with the scheduler's clip_sample=[-1, 1]
        # clamp, so it gets the configurable mode (minmax by default). Observation
        # fields keep standardization.
        modes = {"action": self.config.normalize_mode}
        return LinearNormalizer.from_data(normalizer_data, modes=modes)

    def get_all_actions(self) -> torch.Tensor:
        """Return all action labels as a torch tensor for diagnostics."""
        return torch.from_numpy(np.asarray(self.root["data"]["action"][:], dtype=np.float32))

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> DP3Batch:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        sample = self._sample_sequence(idx)
        point_cloud = sample["point_cloud"].astype(np.float32)
        if self.config.goal_marker_points:
            point_cloud = insert_goal_marker_points(
                point_cloud,
                sample["target_position"].astype(np.float32),
                num_points=self.config.goal_marker_points,
                radius=self.config.goal_marker_radius,
            )
        obs = {
            "point_cloud": torch.from_numpy(point_cloud),
            "agent_pos": torch.from_numpy(sample["state"].astype(np.float32)),
        }
        if self.config.use_goal_encoder:
            obs["goal_xyz"] = torch.from_numpy(sample["target_position"].astype(np.float32))
            obs["ee_position"] = torch.from_numpy(sample["tcp_pose"].astype(np.float32)[..., :3])
        return {
            "obs": obs,
            "action": torch.from_numpy(sample["action"].astype(np.float32)),
        }

    def _validate_arrays(self) -> None:
        data = self.root["data"]
        required = {"point_cloud", "state", "action"}
        if self.config.goal_marker_points or self.config.use_goal_encoder:
            required.add(self.target_position_key)
        if self.config.use_goal_encoder:
            required.add("tcp_pose")
        missing = required.difference(data.keys())
        if missing:
            raise ValueError(f"dataset missing required arrays: {sorted(missing)}")
        total_steps = int(self.episode_ends[-1]) if len(self.episode_ends) else 0
        for key in required:
            if data[key].shape[0] != total_steps:
                raise ValueError(
                    f"/data/{key} length {data[key].shape[0]} != total steps {total_steps}"
                )
        if data["point_cloud"].ndim != 3 or data["point_cloud"].shape[2] != 3:
            raise ValueError("/data/point_cloud must have shape [T, N, 3]")
        if (
            self.config.goal_marker_points
            and self.config.goal_marker_points >= data["point_cloud"].shape[1]
        ):
            raise ValueError(
                "goal_marker_points must be smaller than the stored point-cloud point count"
            )
        if data["state"].ndim != 2:
            raise ValueError("/data/state must have shape [T, state_dim]")
        if data["action"].ndim != 2:
            raise ValueError("/data/action must have shape [T, action_dim]")
        if self.config.goal_marker_points and (
            data[self.target_position_key].ndim != 2
            or data[self.target_position_key].shape[1] != 3
        ):
            raise ValueError(f"/data/{self.target_position_key} must have shape [T, 3]")
        if self.config.use_goal_encoder and (
            data["tcp_pose"].ndim != 2 or data["tcp_pose"].shape[1] < 3
        ):
            raise ValueError("/data/tcp_pose must have shape [T, >=3]")

    def _episode_mask(self, split: Split) -> np.ndarray:
        if split == "all":
            return np.ones(self.num_episodes, dtype=bool)
        val_mask = validation_episode_mask(
            self.num_episodes,
            val_ratio=self.config.val_ratio,
            seed=self.config.seed,
        )
        if split == "val":
            return val_mask
        train_mask = ~val_mask
        return downsample_episode_mask(
            train_mask,
            max_episodes=self.config.max_train_episodes,
            seed=self.config.seed,
        )

    def _sample_sequence(self, idx: int) -> dict[str, np.ndarray]:
        buffer_start, buffer_end, sample_start, sample_end = self.indices[idx]
        keys = ["point_cloud", "state", "action"]
        if self.config.goal_marker_points or self.config.use_goal_encoder:
            keys.append(self.target_position_key)
        if self.config.use_goal_encoder:
            keys.append("tcp_pose")
        keys = tuple(keys)
        sample = {
            key: sample_padded_sequence(
                self.root["data"][key],
                buffer_start_idx=int(buffer_start),
                buffer_end_idx=int(buffer_end),
                sample_start_idx=int(sample_start),
                sample_end_idx=int(sample_end),
                sequence_length=self.config.horizon,
            )
            for key in keys
        }
        if self.config.goal_marker_points and self.target_position_key != "target_position":
            sample["target_position"] = sample[self.target_position_key]
        return sample


def reach_shape_meta(
    *,
    num_points: int = 512,
    point_dim: int = 3,
    state_dim: int = 9,
    action_dim: int = 7,
    use_goal_encoder: bool = False,
) -> dict[str, dict[str, dict[str, list[int]]]]:
    """Shape metadata for pg3d-native DP3 on the ManiSkill reach schema."""
    obs = {
        "point_cloud": {"shape": [num_points, point_dim]},
        "agent_pos": {"shape": [state_dim]},
    }
    if use_goal_encoder:
        obs["goal_xyz"] = {"shape": [3]}
        obs["ee_position"] = {"shape": [3]}
    return {
        "obs": obs,
        "action": {"shape": [action_dim]},
    }


def validation_episode_mask(n_episodes: int, *, val_ratio: float, seed: int) -> np.ndarray:
    """Return a deterministic validation mask with at least one train episode left."""
    mask = np.zeros(n_episodes, dtype=bool)
    if n_episodes <= 1 or val_ratio <= 0:
        return mask
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    rng = np.random.default_rng(seed)
    mask[rng.choice(n_episodes, size=n_val, replace=False)] = True
    return mask


def normalizer_step_indices(*, total_steps: int, max_steps: int | None) -> np.ndarray:
    """Return deterministic row indices for fitting dataset normalizers."""
    if total_steps < 0:
        raise ValueError("total_steps must be non-negative")
    if total_steps == 0:
        return np.zeros((0,), dtype=np.int64)
    if max_steps is None or total_steps <= max_steps:
        return np.arange(total_steps, dtype=np.int64)
    if max_steps <= 0:
        raise ValueError("max_steps must be positive or None")
    return np.unique(np.linspace(0, total_steps - 1, max_steps, dtype=np.int64))


def downsample_episode_mask(
    episode_mask: np.ndarray,
    *,
    max_episodes: int | None,
    seed: int,
) -> np.ndarray:
    """Optionally cap the number of selected train episodes."""
    mask = np.asarray(episode_mask, dtype=bool).copy()
    selected = np.flatnonzero(mask)
    if max_episodes is None or selected.size <= max_episodes:
        return mask
    rng = np.random.default_rng(seed)
    keep = rng.choice(selected, size=max_episodes, replace=False)
    mask[:] = False
    mask[keep] = True
    return mask


def create_sequence_indices(
    episode_ends: np.ndarray,
    *,
    sequence_length: int,
    episode_mask: np.ndarray,
    pad_before: int = 0,
    pad_after: int = 0,
) -> np.ndarray:
    """Create padded sequence indices using the same convention as upstream DP3."""
    episode_ends = np.asarray(episode_ends, dtype=np.int64)
    episode_mask = np.asarray(episode_mask, dtype=bool)
    if episode_mask.shape != episode_ends.shape:
        raise ValueError("episode_mask must have the same shape as episode_ends")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    pad_before = min(max(int(pad_before), 0), sequence_length - 1)
    pad_after = min(max(int(pad_after), 0), sequence_length - 1)

    indices: list[list[int]] = []
    for episode_idx, enabled in enumerate(episode_mask):
        if not enabled:
            continue
        episode_start = 0 if episode_idx == 0 else int(episode_ends[episode_idx - 1])
        episode_end = int(episode_ends[episode_idx])
        episode_length = episode_end - episode_start
        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after
        for start_offset in range(min_start, max_start + 1):
            buffer_start = max(start_offset, 0) + episode_start
            buffer_end = min(start_offset + sequence_length, episode_length) + episode_start
            sample_start = buffer_start - (start_offset + episode_start)
            sample_end = sequence_length - (
                (start_offset + sequence_length + episode_start) - buffer_end
            )
            indices.append([buffer_start, buffer_end, sample_start, sample_end])
    return np.asarray(indices, dtype=np.int64).reshape(-1, 4)


def sample_padded_sequence(
    array: np.ndarray,
    *,
    buffer_start_idx: int,
    buffer_end_idx: int,
    sample_start_idx: int,
    sample_end_idx: int,
    sequence_length: int,
) -> np.ndarray:
    """Read one sequence and pad episode boundaries by repeating endpoint values."""
    sample = np.asarray(array[buffer_start_idx:buffer_end_idx])
    if sample.shape[0] == sequence_length and sample_start_idx == 0:
        return sample
    if sample.shape[0] == 0:
        raise ValueError("cannot pad an empty sequence")
    result = np.zeros((sequence_length,) + sample.shape[1:], dtype=sample.dtype)
    if sample_start_idx > 0:
        result[:sample_start_idx] = sample[0]
    if sample_end_idx < sequence_length:
        result[sample_end_idx:] = sample[-1]
    result[sample_start_idx:sample_end_idx] = sample
    return result


def _load_metadata(dataset_path: Path) -> dict[str, object]:
    metadata_path = dataset_path / "metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _target_position_key(data: Any) -> str:
    if "target_position" in data:
        return "target_position"
    if "goal_pos" in data:
        return "goal_pos"
    return "target_position"
