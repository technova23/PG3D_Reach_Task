from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

Array = np.ndarray


def _as_array(
    value: Any,
    *,
    name: str,
    dtype: np.dtype | type | None = np.float32,
    ndim: int | None = None,
) -> Array:
    array = np.asarray(value, dtype=dtype)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got shape {array.shape}")
    return array


def _optional_array(
    value: Any,
    *,
    name: str,
    dtype: np.dtype | type | None = np.float32,
    ndim: int | None = None,
) -> Array | None:
    if value is None:
        return None
    return _as_array(value, name=name, dtype=dtype, ndim=ndim)


def _array_summary(array: Array) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }
    if array.size > 0 and np.issubdtype(array.dtype, np.number):
        finite = array[np.isfinite(array)] if np.issubdtype(array.dtype, np.floating) else array
        if finite.size > 0:
            summary["min"] = float(np.min(finite))
            summary["max"] = float(np.max(finite))
    if array.dtype == np.bool_:
        summary["true_count"] = int(np.count_nonzero(array))
    return summary


@dataclass
class RobotState:
    """Robot proprioception extracted from a ManiSkill observation."""

    joint_positions: Array
    joint_velocities: Array | None = None
    gripper_open: float | None = None
    tcp_pose: Array | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.joint_positions = _as_array(
            self.joint_positions, name="joint_positions", dtype=np.float32, ndim=1
        )
        self.joint_velocities = _optional_array(
            self.joint_velocities, name="joint_velocities", dtype=np.float32, ndim=1
        )
        self.tcp_pose = _optional_array(self.tcp_pose, name="tcp_pose", dtype=np.float32, ndim=1)
        if self.gripper_open is not None:
            self.gripper_open = float(self.gripper_open)

    def as_agent_pos(self) -> Array:
        """Return the default DP3-visible low-dimensional state."""
        return self.joint_positions.astype(np.float32, copy=True)

    def summary(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "joint_positions": _array_summary(self.joint_positions),
            "metadata": dict(self.metadata),
        }
        if self.joint_velocities is not None:
            data["joint_velocities"] = _array_summary(self.joint_velocities)
        if self.tcp_pose is not None:
            data["tcp_pose"] = _array_summary(self.tcp_pose)
        if self.gripper_open is not None:
            data["gripper_open"] = self.gripper_open
        return data


@dataclass
class SimGroundTruth:
    """Simulator-only context for evaluation, debugging, and artifact labeling."""

    task_name: str
    target_position: Array | None = None
    success: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.target_position = _optional_array(
            self.target_position, name="target_position", dtype=np.float32, ndim=1
        )
        if self.target_position is not None and self.target_position.shape != (3,):
            raise ValueError(
                "target_position must have shape (3,), "
                f"got {self.target_position.shape}"
            )
        if self.success is not None:
            self.success = bool(self.success)

    def summary(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "task_name": self.task_name,
            "metadata": dict(self.metadata),
        }
        if self.target_position is not None:
            data["target_position"] = self.target_position.astype(float).tolist()
        if self.success is not None:
            data["success"] = self.success
        return data


@dataclass
class Observation:
    """pg3d observation object with explicit policy and simulator boundaries."""

    point_cloud: Array
    point_features: dict[str, Array]
    robot_state: RobotState
    robot_mask: Array | None = None
    object_masks: dict[str, Array] = field(default_factory=dict)
    sim_gt: SimGroundTruth | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.point_cloud = _as_array(
            self.point_cloud, name="point_cloud", dtype=np.float32, ndim=2
        )
        if self.point_cloud.shape[1] != 3:
            raise ValueError(f"point_cloud must have shape [N, 3], got {self.point_cloud.shape}")
        if not np.all(np.isfinite(self.point_cloud)):
            raise ValueError("point_cloud must contain only finite points")

        num_points = self.point_cloud.shape[0]
        self.point_features = {
            key: np.asarray(value) for key, value in self.point_features.items()
        }
        for key, value in self.point_features.items():
            if value.ndim == 0:
                raise ValueError(f"point_features[{key!r}] must be at least 1D")
            if value.shape[0] != num_points:
                raise ValueError(
                    f"point_features[{key!r}] first dimension must be {num_points}, "
                    f"got shape {value.shape}"
                )

        self.robot_mask = self._validate_mask(self.robot_mask, name="robot_mask")
        self.object_masks = {
            key: self._validate_mask(value, name=f"object_masks[{key!r}]")
            for key, value in self.object_masks.items()
        }

    def _validate_mask(self, value: Any, *, name: str) -> Array | None:
        if value is None:
            return None
        mask = np.asarray(value, dtype=bool)
        expected = (self.point_cloud.shape[0],)
        if mask.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {mask.shape}")
        return mask

    def as_policy_inputs(self, *, include_rgb: bool = False) -> dict[str, Array]:
        """Return the default DP3-visible observation dictionary."""
        point_cloud = self.point_cloud
        if include_rgb and "rgb" in self.point_features:
            rgb = self.point_features["rgb"].astype(np.float32)
            if rgb.size > 0 and np.max(rgb) > 1.0:
                rgb = rgb / 255.0
            point_cloud = np.concatenate([point_cloud, rgb], axis=1).astype(np.float32)
        return {
            "point_cloud": point_cloud.astype(np.float32, copy=True),
            "agent_pos": self.robot_state.as_agent_pos(),
        }

    def summary(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "point_cloud": _array_summary(self.point_cloud),
            "point_features": {
                key: _array_summary(value) for key, value in self.point_features.items()
            },
            "robot_mask": (
                _array_summary(self.robot_mask)
                if self.robot_mask is not None
                else {"available": False}
            ),
            "object_masks": {
                key: _array_summary(value) for key, value in self.object_masks.items()
            },
            "robot_state": self.robot_state.summary(),
            "metadata": dict(self.metadata),
        }
        if self.sim_gt is not None:
            data["sim_gt"] = self.sim_gt.summary()
        return data
