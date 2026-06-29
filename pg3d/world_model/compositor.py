from __future__ import annotations

from typing import Any

import numpy as np

from pg3d.world_model.types import Array, as_float_array


def static_scene_from_robot_mask(point_cloud: Array, robot_mask: Array | None) -> Array:
    """Remove current robot points from a scene point cloud."""
    if robot_mask is None:
        raise ValueError("robot_mask is required for robot-only point-cloud compositing")
    points = _point_cloud(point_cloud, name="point_cloud")
    mask = np.asarray(robot_mask, dtype=bool)
    if mask.shape != (points.shape[0],):
        raise ValueError(f"robot_mask must have shape {(points.shape[0],)}, got {mask.shape}")
    return points[~mask].astype(np.float32, copy=True)


def compose_robot_cloud(static_scene: Array, robot_point_cloud: Array) -> tuple[Array, Array]:
    """Append future robot points to a static scene and return the aligned robot mask."""
    static = _point_cloud(static_scene, name="static_scene")
    robot = _point_cloud(robot_point_cloud, name="robot_point_cloud")
    scene = np.concatenate([static, robot], axis=0).astype(np.float32, copy=False)
    robot_mask = np.concatenate(
        [
            np.zeros((static.shape[0],), dtype=bool),
            np.ones((robot.shape[0],), dtype=bool),
        ],
        axis=0,
    )
    return scene, robot_mask


def _point_cloud(value: Any, *, name: str) -> Array:
    points = as_float_array(value, name=name, ndim=2)
    if points.shape[1] != 3:
        raise ValueError(f"{name} must have shape [N, 3], got {points.shape}")
    return points
