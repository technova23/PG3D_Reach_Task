from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from pg3d.constraints import AvoidRegion, SphereRegion
from pg3d.envs.maniskill_adapter.dataset import PointCloudCropConfig, crop_point_cloud

Array = np.ndarray

# Points sampled per avoid-region sphere surface when --obstacle-realism real.
DEFAULT_OBSTACLE_POINTS_PER_REGION = 96


def sphere_surface_points(center: Array, radius: float, num_points: int) -> Array:
    """Deterministic, near-even points on a sphere surface (Fibonacci lattice).

    Stands in for what a depth camera would return for a real spherical
    obstacle, since the physics engine can't host an unattached floating
    collision body for the avoid-region sphere itself.
    """
    if radius <= 0.0 or not np.isfinite(radius):
        raise ValueError("radius must be a positive finite value")
    if num_points < 0:
        raise ValueError("num_points must be non-negative")
    center = np.asarray(center, dtype=np.float32).reshape(3)
    if num_points == 0:
        return np.zeros((0, 3), dtype=np.float32)

    # Fibonacci/golden-angle lattice: near-uniform coverage without the
    # polar clustering of a naive lat/long grid.
    indices = np.arange(num_points, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * indices / num_points)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    theta = golden_angle * indices
    offsets = np.stack(
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)],
        axis=1,
    ).astype(np.float32) * np.float32(radius)
    return center.reshape(1, 3) + offsets


def avoid_region_sphere_points(
    constraints: list[Any],
    *,
    num_points_per_region: int = DEFAULT_OBSTACLE_POINTS_PER_REGION,
) -> Array:
    """Sample surface points for every spherical ``AvoidRegion`` in ``constraints``.

    Only ``AvoidRegion`` constraints whose ``region`` is a ``SphereRegion`` are
    sampled; other constraint/region types are ignored (real-obstacle
    injection currently only supports avoid-region spheres).
    """
    regions = [
        constraint.region
        for constraint in constraints
        if isinstance(constraint, AvoidRegion) and isinstance(constraint.region, SphereRegion)
    ]
    if not regions:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(
        [
            sphere_surface_points(region.center, region.radius, num_points_per_region)
            for region in regions
        ],
        axis=0,
    )


def insert_real_obstacle_points(
    point_cloud: Array,
    robot_mask: Array,
    point_valid_mask: Array,
    obstacle_points: Array,
    *,
    crop_config: PointCloudCropConfig,
) -> tuple[Array, Array, Array]:
    """Mix real-obstacle points into the scene and re-run the standard crop.

    Drops the array's existing padding rows (``point_valid_mask`` False),
    appends ``obstacle_points`` (tagged non-robot, exactly as a real
    segmentation would label an external object) to the surviving real
    points, and hands the combined cloud back to ``crop_point_cloud`` --
    the same bounds-filter/quota-downsample/pad routine every real camera
    point already passes through. The obstacle therefore competes for the
    fixed point budget like any other object instead of being force-written
    into reserved slots: how many of its points actually survive depends on
    scene density, exactly as it would for a real object the camera saw.

    Exception: when ``crop_config.robot_point_fraction >= 1.0`` (a
    robot-only-trained checkpoint), ``crop_point_cloud`` drops every
    non-robot point unconditionally before any competition can happen (see
    its "Robot-only" branch), so an obstacle point would never survive that
    path -- it would always be discarded, regardless of scene density or
    obstacle size. For that regime we instead reserve exactly
    ``len(obstacle_points)`` slots for the obstacle (guaranteed survival)
    and run only the *remaining* budget through the normal robot-only crop.
    This is knowingly out-of-distribution input for such a checkpoint (it
    has never seen a non-robot point in training), but is the only way to
    make an obstacle visible to it at all.
    """
    points = np.asarray(point_cloud, dtype=np.float32)
    mask = np.asarray(robot_mask, dtype=bool)
    valid = np.asarray(point_valid_mask, dtype=bool)
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"point_cloud must have shape [N, 3], got {points.shape}")
    if points.shape[:1] != mask.shape or points.shape[:1] != valid.shape:
        raise ValueError("point_cloud/robot_mask/point_valid_mask shapes must align")

    obstacle_points = np.asarray(obstacle_points, dtype=np.float32)
    num_obstacle_points = int(obstacle_points.shape[0])
    if num_obstacle_points == 0:
        return points.copy(), mask.copy(), valid.copy()

    real_points = points[valid]
    real_mask = mask[valid]

    if crop_config.robot_point_fraction >= 1.0:
        if num_obstacle_points >= crop_config.num_points:
            raise ValueError(
                f"{num_obstacle_points} obstacle points would consume the entire "
                f"{crop_config.num_points}-point budget, leaving no room for the robot; "
                "lower --obstacle-points-per-region"
            )
        remaining_config = replace(
            crop_config, num_points=crop_config.num_points - num_obstacle_points
        )
        cropped = crop_point_cloud(real_points, robot_mask=real_mask, config=remaining_config)
        out_points = np.concatenate([cropped["point_cloud"], obstacle_points], axis=0)
        out_mask = np.concatenate(
            [cropped["robot_mask"], np.zeros((num_obstacle_points,), dtype=bool)], axis=0
        )
        out_valid = np.concatenate(
            [cropped["point_valid_mask"], np.ones((num_obstacle_points,), dtype=bool)], axis=0
        )
        return out_points, out_mask, out_valid

    combined_points = np.concatenate([real_points, obstacle_points], axis=0)
    combined_mask = np.concatenate(
        [real_mask, np.zeros((num_obstacle_points,), dtype=bool)], axis=0
    )
    cropped = crop_point_cloud(combined_points, robot_mask=combined_mask, config=crop_config)
    return cropped["point_cloud"], cropped["robot_mask"], cropped["point_valid_mask"]
