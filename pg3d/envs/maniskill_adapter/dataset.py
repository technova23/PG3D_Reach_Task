from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import zarr

from pg3d.envs.maniskill_adapter.types import Observation
from pg3d.utils.serialization import jsonable

Array = np.ndarray
ActionMode = Literal["abs_joint", "delta_joint"]

DEFAULT_WORKSPACE_BOUNDS = np.asarray(
    [[-0.9, 0.7], [-0.6, 0.6], [0.0, 1.1]],
    dtype=np.float32,
)


@dataclass(frozen=True)
class PointCloudCropConfig:
    """Fixed-size point-cloud crop policy used by the first reach dataset."""

    bounds: Array = field(default_factory=lambda: DEFAULT_WORKSPACE_BOUNDS.copy())
    num_points: int = 1024
    robot_point_fraction: float = 0.25

    def __post_init__(self) -> None:
        bounds = np.asarray(self.bounds, dtype=np.float32)
        if bounds.shape != (3, 2):
            raise ValueError(f"bounds must have shape (3, 2), got {bounds.shape}")
        if np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("each point-cloud bound must have min < max")
        if self.num_points <= 0:
            raise ValueError("num_points must be positive")
        if not 0.0 <= self.robot_point_fraction <= 1.0:
            raise ValueError("robot_point_fraction must be between 0 and 1")
        object.__setattr__(self, "bounds", bounds)

    def to_json(self) -> dict[str, Any]:
        return {
            "bounds": self.bounds.astype(float).tolist(),
            "num_points": int(self.num_points),
            "robot_point_fraction": float(self.robot_point_fraction),
        }


@dataclass
class ReachEpisodeData:
    """One flattened reach episode ready to append to the Zarr dataset."""

    state: Array
    action: Array
    sim_action: Array
    point_cloud: Array
    robot_mask: Array
    point_valid_mask: Array
    target_position: Array
    tcp_pose: Array
    success: Array
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.state = _as_array(self.state, name="state", dtype=np.float32, ndim=2)
        self.action = _as_array(self.action, name="action", dtype=np.float32, ndim=2)
        self.sim_action = _as_array(self.sim_action, name="sim_action", dtype=np.float32, ndim=2)
        self.point_cloud = _as_array(
            self.point_cloud, name="point_cloud", dtype=np.float32, ndim=3
        )
        self.robot_mask = _as_array(self.robot_mask, name="robot_mask", dtype=bool, ndim=2)
        self.point_valid_mask = _as_array(
            self.point_valid_mask, name="point_valid_mask", dtype=bool, ndim=2
        )
        self.target_position = _as_array(
            self.target_position, name="target_position", dtype=np.float32, ndim=2
        )
        self.tcp_pose = _as_array(self.tcp_pose, name="tcp_pose", dtype=np.float32, ndim=2)
        self.success = _as_array(self.success, name="success", dtype=bool, ndim=1)
        self._validate_shapes()

    def _validate_shapes(self) -> None:
        length = self.state.shape[0]
        arrays = {
            "action": self.action,
            "sim_action": self.sim_action,
            "point_cloud": self.point_cloud,
            "robot_mask": self.robot_mask,
            "point_valid_mask": self.point_valid_mask,
            "target_position": self.target_position,
            "tcp_pose": self.tcp_pose,
            "success": self.success,
        }
        for name, array in arrays.items():
            if array.shape[0] != length:
                raise ValueError(f"{name} length {array.shape[0]} != state length {length}")
        # State is the robot qpos; its width is robot-dependent (Panda: 9 = 7 arm + 2
        # gripper, xArm7 no-gripper: 7). Require at least the 7 arm joints rather than
        # a fixed 9 so non-Panda arms are accepted; the DP3 loader reads state_dim from
        # the zarr, so any consistent width works downstream.
        if self.state.ndim != 2 or self.state.shape[1] < 7:
            raise ValueError(f"state must be [T, D] with D >= 7, got {self.state.shape}")
        if self.action.shape[1] != 7:
            raise ValueError(f"action must have shape [T, 7], got {self.action.shape}")
        if self.point_cloud.shape[2] != 3:
            raise ValueError(f"point_cloud must have shape [T, N, 3], got {self.point_cloud.shape}")
        point_shape = self.point_cloud.shape[:2]
        if self.robot_mask.shape != point_shape:
            raise ValueError(
                f"robot_mask must have shape {point_shape}, got {self.robot_mask.shape}"
            )
        if self.point_valid_mask.shape != point_shape:
            raise ValueError(
                f"point_valid_mask must have shape {point_shape}, got {self.point_valid_mask.shape}"
            )
        if self.target_position.shape[1] != 3:
            raise ValueError(
                f"target_position must have shape [T, 3], got {self.target_position.shape}"
            )
        if self.tcp_pose.shape[1] != 7:
            raise ValueError(f"tcp_pose must have shape [T, 7], got {self.tcp_pose.shape}")


def crop_point_cloud(
    point_cloud: Array,
    *,
    robot_mask: Array | None = None,
    config: PointCloudCropConfig | None = None,
) -> dict[str, Array]:
    """Crop, deterministically downsample, and pad a point cloud with aligned masks."""
    config = config or PointCloudCropConfig()
    points = _as_array(point_cloud, name="point_cloud", dtype=np.float32, ndim=2)
    if points.shape[1] != 3:
        raise ValueError(f"point_cloud must have shape [N, 3], got {points.shape}")
    source_robot_mask = (
        np.zeros(points.shape[0], dtype=bool)
        if robot_mask is None
        else _as_array(robot_mask, name="robot_mask", dtype=bool, ndim=1)
    )
    if source_robot_mask.shape != (points.shape[0],):
        raise ValueError(
            f"robot_mask must have shape {(points.shape[0],)}, got {source_robot_mask.shape}"
        )

    in_bounds = np.all(
        (points >= config.bounds[:, 0]) & (points <= config.bounds[:, 1]),
        axis=1,
    )
    cropped_indices = np.flatnonzero(in_bounds)
    if config.robot_point_fraction >= 1.0:
        # Robot-only: drop scene/background points entirely rather than letting
        # them backfill leftover slots when robot points fall short of num_points.
        cropped_indices = cropped_indices[source_robot_mask[cropped_indices]]
    if cropped_indices.size > config.num_points:
        cropped_indices = _downsample_with_robot_quota(
            cropped_indices,
            robot_mask=source_robot_mask,
            num_points=config.num_points,
            robot_point_fraction=config.robot_point_fraction,
        )

    out_points = np.zeros((config.num_points, 3), dtype=np.float32)
    out_robot_mask = np.zeros((config.num_points,), dtype=bool)
    out_valid_mask = np.zeros((config.num_points,), dtype=bool)
    count = min(cropped_indices.size, config.num_points)
    if count > 0:
        selected_indices = cropped_indices[:count]
        out_points[:count] = points[selected_indices]
        out_robot_mask[:count] = source_robot_mask[selected_indices]
        out_valid_mask[:count] = True
    return {
        "point_cloud": out_points,
        "robot_mask": out_robot_mask,
        "point_valid_mask": out_valid_mask,
    }


def _downsample_with_robot_quota(
    indices: Array,
    *,
    robot_mask: Array,
    num_points: int,
    robot_point_fraction: float,
) -> Array:
    robot_indices = indices[robot_mask[indices]]
    scene_indices = indices[~robot_mask[indices]]
    target_robot = min(robot_indices.size, int(np.ceil(num_points * robot_point_fraction)))
    target_scene = min(scene_indices.size, num_points - target_robot)
    target_robot = min(robot_indices.size, num_points - target_scene)
    selected = np.concatenate(
        [
            _linspace_select(robot_indices, target_robot),
            _linspace_select(scene_indices, target_scene),
        ],
        axis=0,
    )
    if selected.size < num_points:
        remaining = np.setdiff1d(indices, selected, assume_unique=True)
        selected = np.concatenate(
            [selected, _linspace_select(remaining, num_points - selected.size)],
            axis=0,
        )
    return np.sort(selected.astype(np.int64, copy=False))


def _linspace_select(values: Array, count: int) -> Array:
    values = np.asarray(values, dtype=np.int64)
    if count <= 0 or values.size == 0:
        return np.zeros((0,), dtype=np.int64)
    if values.size <= count:
        return values.astype(np.int64, copy=True)
    selected = np.linspace(0, values.size - 1, count)
    return values[np.rint(selected).astype(np.int64)]


def observation_to_dataset_row(
    observation: Observation,
    *,
    sim_action: Array,
    action_mode: ActionMode,
    crop_config: PointCloudCropConfig | None = None,
) -> dict[str, Array]:
    """Convert one adapted observation and simulator action into dataset arrays."""
    state = observation.robot_state.as_agent_pos()
    sim_action = _as_array(sim_action, name="sim_action", dtype=np.float32, ndim=1)
    label = action_label_from_sim_action(sim_action, state, action_mode=action_mode)
    cropped = crop_point_cloud(
        observation.point_cloud,
        robot_mask=observation.robot_mask,
        config=crop_config,
    )
    target_position = (
        np.zeros((3,), dtype=np.float32)
        if observation.sim_gt is None or observation.sim_gt.target_position is None
        else observation.sim_gt.target_position.astype(np.float32, copy=True)
    )
    tcp_pose = (
        np.zeros((7,), dtype=np.float32)
        if observation.robot_state.tcp_pose is None
        else observation.robot_state.tcp_pose.astype(np.float32, copy=True)
    )
    return {
        "state": state,
        "action": label,
        "sim_action": sim_action,
        "target_position": target_position,
        "tcp_pose": tcp_pose,
        **cropped,
    }


def action_label_from_sim_action(
    sim_action: Array,
    state: Array,
    *,
    action_mode: ActionMode,
) -> Array:
    """Extract the 7-DoF Panda arm label expected by pg3d-native DP3."""
    sim_action = _as_array(sim_action, name="sim_action", dtype=np.float32, ndim=1)
    state = _as_array(state, name="state", dtype=np.float32, ndim=1)
    if sim_action.shape[0] < 7:
        raise ValueError(f"sim_action must have at least 7 values, got {sim_action.shape}")
    if state.shape[0] < 7:
        raise ValueError(f"state must have at least 7 values, got {state.shape}")
    arm_action = sim_action[:7].astype(np.float32, copy=True)
    if action_mode == "abs_joint":
        return arm_action
    if action_mode == "delta_joint":
        return (arm_action - state[:7]).astype(np.float32, copy=False)
    raise ValueError(f"unsupported action_mode {action_mode!r}")


def write_reach_zarr(
    output_path: Path | str,
    episodes: list[ReachEpisodeData],
    *,
    metadata: dict[str, Any],
    overwrite: bool = False,
    append: bool = False,
) -> dict[str, Any]:
    """Write reach episodes to a DP3-style Zarr replay-buffer layout.

    If append=True, concatenates with existing zarr at output_path.
    If overwrite=True, replaces existing zarr.
    If neither, raises FileExistsError if output_path exists.
    """
    if not episodes:
        raise ValueError("at least one episode is required")
    output_path = Path(output_path)

    if append and output_path.exists():
        existing_episodes = _load_zarr_episodes(output_path)
        episodes = existing_episodes + episodes
    elif output_path.exists():
        if not overwrite:
            raise FileExistsError(f"{output_path} already exists; pass overwrite=True or append=True")
        shutil.rmtree(output_path)

    arrays = _stack_episodes(episodes)
    root = zarr.group(store=zarr.DirectoryStore(str(output_path)), overwrite=True)
    data_group = root.create_group("data")
    meta_group = root.create_group("meta")
    for key, value in arrays["data"].items():
        chunks = (min(max(1, value.shape[0]), 1024),) + value.shape[1:]
        data_group.array(name=key, data=value, chunks=chunks)
    meta_group.array(
        name="episode_ends",
        data=arrays["episode_ends"],
        chunks=arrays["episode_ends"].shape,
    )

    summary = dataset_summary_from_arrays(arrays["data"], arrays["episode_ends"])
    json_metadata = {
        **metadata,
        "schema_version": "pg3d.reach.zarr.v1",
        "summary": summary,
        "episodes": [jsonable(episode.metadata) for episode in episodes],
    }
    (output_path / "metadata.json").write_text(
        json.dumps(jsonable(json_metadata), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _load_zarr_episodes(dataset_path: Path | str) -> list[ReachEpisodeData]:
    """Load all episodes from an existing reach zarr dataset."""
    dataset_path = Path(dataset_path)
    root = zarr.open_group(str(dataset_path), mode="r")
    data_group = root["data"]
    meta_group = root["meta"]
    episode_ends = np.asarray(meta_group["episode_ends"][:], dtype=np.int64)

    episodes: list[ReachEpisodeData] = []
    start_idx = 0
    for end_idx in episode_ends:
        episode_data = {
            key: np.asarray(data_group[key][start_idx:end_idx], dtype=data_group[key].dtype)
            for key in data_group.keys()
        }
        metadata_list = load_reach_metadata(dataset_path).get("episodes", [])
        episode_metadata = metadata_list[len(episodes)] if len(episodes) < len(metadata_list) else {}
        episodes.append(
            ReachEpisodeData(
                state=episode_data.get("state", np.zeros((0, 9))),
                action=episode_data.get("action", np.zeros((0, 7))),
                sim_action=episode_data.get("sim_action", np.zeros((0, 9))),
                point_cloud=episode_data.get("point_cloud", np.zeros((0, 1024, 3))),
                robot_mask=episode_data.get("robot_mask", np.zeros((0, 1024), dtype=bool)),
                point_valid_mask=episode_data.get("point_valid_mask", np.zeros((0, 1024), dtype=bool)),
                target_position=episode_data.get("target_position", np.zeros((0, 3))),
                tcp_pose=episode_data.get("tcp_pose", np.zeros((0, 7))),
                success=episode_data.get("success", np.zeros(0, dtype=bool)),
                metadata=episode_metadata,
            )
        )
        start_idx = end_idx
    return episodes


def load_reach_metadata(dataset_path: Path | str) -> dict[str, Any]:
    path = Path(dataset_path) / "metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_reach_zarr(dataset_path: Path | str) -> dict[str, Any]:
    root = zarr.open_group(str(dataset_path), mode="r")
    data = {key: root["data"][key] for key in root["data"].keys()}
    return dataset_summary_from_arrays(data, root["meta"]["episode_ends"][:])


def dataset_summary_from_arrays(data: dict[str, Any], episode_ends: Array) -> dict[str, Any]:
    return {
        "num_episodes": int(len(episode_ends)),
        "num_steps": int(episode_ends[-1]) if len(episode_ends) else 0,
        "episode_ends": np.asarray(episode_ends, dtype=np.int64).tolist(),
        "arrays": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in sorted(data.items())
        },
    }


def git_commit_info(repo_path: Path | str) -> dict[str, Any]:
    """Return commit and dirty status without failing if git metadata is unavailable."""
    repo_path = Path(repo_path)

    def run_git(args: list[str]) -> str | None:
        result = subprocess.run(
            ["git", "-C", str(repo_path), *args],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    status = run_git(["status", "--short"])
    return {
        "path": str(repo_path),
        "commit": run_git(["rev-parse", "HEAD"]),
        "dirty": bool(status),
        "status_short": status or "",
    }


def _stack_episodes(episodes: list[ReachEpisodeData]) -> dict[str, Any]:
    data: dict[str, Array] = {
        "state": np.concatenate([episode.state for episode in episodes], axis=0),
        "action": np.concatenate([episode.action for episode in episodes], axis=0),
        "sim_action": np.concatenate([episode.sim_action for episode in episodes], axis=0),
        "point_cloud": np.concatenate([episode.point_cloud for episode in episodes], axis=0),
        "robot_mask": np.concatenate([episode.robot_mask for episode in episodes], axis=0),
        "point_valid_mask": np.concatenate(
            [episode.point_valid_mask for episode in episodes], axis=0
        ),
        "target_position": np.concatenate(
            [episode.target_position for episode in episodes], axis=0
        ),
        "tcp_pose": np.concatenate([episode.tcp_pose for episode in episodes], axis=0),
        "success": np.concatenate([episode.success for episode in episodes], axis=0),
    }
    episode_lengths = [episode.state.shape[0] for episode in episodes]
    episode_ends = np.cumsum(episode_lengths, dtype=np.int64)
    return {"data": data, "episode_ends": episode_ends}


def _as_array(
    value: Any,
    *,
    name: str,
    dtype: np.dtype | type,
    ndim: int,
) -> Array:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got shape {array.shape}")
    return array
