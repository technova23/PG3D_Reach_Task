from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import numpy as np

Array = np.ndarray
ActionMode = Literal["abs_joint", "delta_joint", "ee_pose"]


def as_float_array(value: Any, *, name: str, ndim: int | None = None) -> Array:
    """Convert a value to finite `float32` NumPy data with optional rank validation."""
    array = np.asarray(value, dtype=np.float32)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


@dataclass
class ActionChunk:
    """Joint-action chunk proposed by a policy for world-model imagination."""

    actions: Array
    action_mode: ActionMode
    dt: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.actions = as_float_array(self.actions, name="actions", ndim=2)
        if self.actions.shape[0] <= 0:
            raise ValueError("actions must contain at least one timestep")
        if self.actions.shape[1] <= 0:
            raise ValueError("actions must contain at least one action dimension")
        if self.action_mode == "ee_pose":
            raise NotImplementedError("ee_pose action chunks are deferred for P07")
        if self.action_mode not in {"abs_joint", "delta_joint"}:
            raise ValueError(f"unsupported action_mode {self.action_mode!r}")
        self.dt = float(self.dt)
        if self.dt <= 0.0 or not np.isfinite(self.dt):
            raise ValueError("dt must be a positive finite value")
        self.metadata = dict(self.metadata)

    @property
    def horizon(self) -> int:
        """Number of imagined action timesteps."""
        return int(self.actions.shape[0])

    @property
    def action_dim(self) -> int:
        """Number of controlled action dimensions in each timestep."""
        return int(self.actions.shape[1])


class RobotGeometryProvider(Protocol):
    """Simulator-free robot geometry interface used by the world model.

    Implementations may use a test double, cached mesh samples, or a future ManiSkill/SAPIEN
    adapter, but the outputs are always world-frame NumPy arrays.
    """

    def end_effector_position(self, q: Array) -> Array:
        """Return the world-frame end-effector position for one joint state as `[3]`."""
        ...

    def robot_point_cloud(self, q: Array) -> Array:
        """Return the world-frame robot point cloud for one joint state as `[N, 3]`."""
        ...


@dataclass
class ImaginedRollout:
    """Robot-only imagined rollout produced from one action chunk."""

    q: Array
    eef_path: Array
    robot_point_clouds: list[Array]
    scene_point_clouds: list[Array]
    robot_masks: list[Array]
    action_chunk: ActionChunk
    metadata: dict[str, Any] = field(default_factory=dict)
    eef_orientations: Array | None = None

    def __post_init__(self) -> None:
        self.q = as_float_array(self.q, name="q", ndim=2)
        self.eef_path = as_float_array(self.eef_path, name="eef_path", ndim=2)
        if self.eef_path.shape != (self.q.shape[0], 3):
            raise ValueError(
                f"eef_path must have shape {(self.q.shape[0], 3)}, "
                f"got {self.eef_path.shape}"
            )
        if self.eef_orientations is not None:
            self.eef_orientations = as_float_array(
                self.eef_orientations, name="eef_orientations", ndim=2
            )
            if self.eef_orientations.shape[0] != self.q.shape[0]:
                raise ValueError(
                    "eef_orientations must have the same horizon as q when provided"
                )
            if self.eef_orientations.shape[1] not in (4, 9):
                raise ValueError("eef_orientations must have shape [T, 4] or [T, 9]")

        horizon = int(self.q.shape[0])
        if self.action_chunk.horizon != horizon:
            raise ValueError(
                f"action_chunk horizon {self.action_chunk.horizon} "
                f"does not match q horizon {horizon}"
            )
        self.robot_point_clouds = [
            _point_cloud(array, name=f"robot_point_clouds[{idx}]")
            for idx, array in enumerate(self.robot_point_clouds)
        ]
        self.scene_point_clouds = [
            _point_cloud(array, name=f"scene_point_clouds[{idx}]")
            for idx, array in enumerate(self.scene_point_clouds)
        ]
        self.robot_masks = [np.asarray(mask, dtype=bool) for mask in self.robot_masks]
        if not (
            len(self.robot_point_clouds)
            == len(self.scene_point_clouds)
            == len(self.robot_masks)
            == horizon
        ):
            raise ValueError("rollout point-cloud lists must match q horizon")
        for idx, (scene, mask) in enumerate(
            zip(self.scene_point_clouds, self.robot_masks, strict=True)
        ):
            if mask.shape != (scene.shape[0],):
                raise ValueError(
                    f"robot_masks[{idx}] must have shape {(scene.shape[0],)}, got {mask.shape}"
                )
        self.metadata = dict(self.metadata)


def _point_cloud(value: Any, *, name: str) -> Array:
    cloud = as_float_array(value, name=name, ndim=2)
    if cloud.shape[1] != 3:
        raise ValueError(f"{name} must have shape [N, 3], got {cloud.shape}")
    return cloud
