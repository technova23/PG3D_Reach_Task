from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from pg3d.constraints import AvoidRegion, BoxRegion, CylinderRegion, SphereRegion
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


def box_surface_points(
    center: Array,
    half_extents: Array,
    num_points: int,
    *,
    rotation: Array | None = None,
) -> Array:
    """Deterministic points spread over a box's six faces.

    Stands in for what a depth camera would return for a real box obstacle,
    for the same reason as ``sphere_surface_points`` -- the physics engine
    can't host an unattached floating collision body for the avoid-region
    box itself. Points per face are allocated proportional to face area and
    drawn from a fixed-seed RNG so the output is deterministic given the
    inputs. Faces are sampled in the box's local frame (axis-aligned) and
    then rotated into world coordinates by ``rotation`` if given -- same
    ``(3, 3)``, columns-are-local-axes convention as ``BoxRegion.rotation``.
    """
    center = np.asarray(center, dtype=np.float32).reshape(3)
    half_extents = np.asarray(half_extents, dtype=np.float32).reshape(3)
    if np.any(half_extents <= 0.0) or not np.all(np.isfinite(half_extents)):
        raise ValueError("half_extents must be positive finite values")
    if num_points < 0:
        raise ValueError("num_points must be non-negative")
    if num_points == 0:
        return np.zeros((0, 3), dtype=np.float32)

    # (fixed_axis, u_axis, v_axis, sign) for each of the 6 faces.
    face_specs = [
        (0, 1, 2, 1.0),
        (0, 1, 2, -1.0),
        (1, 0, 2, 1.0),
        (1, 0, 2, -1.0),
        (2, 0, 1, 1.0),
        (2, 0, 1, -1.0),
    ]
    face_areas = np.array(
        [
            half_extents[u] * half_extents[v]
            for _fixed, u, v, _sign in face_specs
        ],
        dtype=np.float64,
    )
    total_area = float(np.sum(face_areas))
    counts = np.floor(face_areas / total_area * num_points).astype(int)
    counts[-1] += num_points - int(counts.sum())

    rng = np.random.default_rng(0)
    batches: list[Array] = []
    for count, (fixed_axis, u_axis, v_axis, sign) in zip(counts, face_specs):
        if count <= 0:
            continue
        face_points = np.zeros((count, 3), dtype=np.float32)
        face_points[:, fixed_axis] = sign * half_extents[fixed_axis]
        face_points[:, u_axis] = rng.uniform(-half_extents[u_axis], half_extents[u_axis], size=count)
        face_points[:, v_axis] = rng.uniform(-half_extents[v_axis], half_extents[v_axis], size=count)
        batches.append(face_points)
    combined = np.concatenate(batches, axis=0) if batches else np.zeros((0, 3), dtype=np.float32)
    if rotation is not None:
        rotation = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
        combined = combined @ rotation.T
    return center.reshape(1, 3) + combined


def cylinder_surface_points(
    center: Array,
    axis: Array,
    radius: float,
    length: float,
    num_points: int,
) -> Array:
    """Deterministic points on a finite cylinder's lateral surface and end caps.

    Stands in for what a depth camera would return for a real cylindrical
    obstacle, for the same reason as ``sphere_surface_points``. ``axis``
    need not be axis-aligned or normalized. Lateral vs. cap point counts are
    allocated proportional to surface area; draws come from a fixed-seed
    RNG so the output is deterministic given the inputs.
    """
    center = np.asarray(center, dtype=np.float32).reshape(3)
    axis = np.asarray(axis, dtype=np.float32).reshape(3)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 0.0 or not np.isfinite(axis_norm):
        raise ValueError("axis must be a non-zero finite vector")
    axis = axis / axis_norm
    if radius <= 0.0 or not np.isfinite(radius):
        raise ValueError("radius must be a positive finite value")
    if length <= 0.0 or not np.isfinite(length):
        raise ValueError("length must be a positive finite value")
    if num_points < 0:
        raise ValueError("num_points must be non-negative")
    if num_points == 0:
        return np.zeros((0, 3), dtype=np.float32)

    reference = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(reference, axis))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    u_axis = np.cross(axis, reference)
    u_axis = (u_axis / np.linalg.norm(u_axis)).astype(np.float32)
    v_axis = np.cross(axis, u_axis).astype(np.float32)

    half_length = float(length) * 0.5
    lateral_area = 2.0 * np.pi * radius * length
    cap_area = np.pi * radius**2
    total_area = lateral_area + 2.0 * cap_area
    num_lateral = int(round(num_points * lateral_area / total_area))
    num_lateral = min(num_lateral, num_points)
    remaining = num_points - num_lateral
    cap_counts = (remaining // 2, remaining - remaining // 2)

    rng = np.random.default_rng(0)
    batches: list[Array] = []
    if num_lateral > 0:
        theta = rng.uniform(0.0, 2.0 * np.pi, size=num_lateral)
        h = rng.uniform(-half_length, half_length, size=num_lateral)
        batches.append(
            radius * np.cos(theta)[:, None] * u_axis.reshape(1, 3)
            + radius * np.sin(theta)[:, None] * v_axis.reshape(1, 3)
            + h[:, None] * axis.reshape(1, 3)
        )
    for cap_count, sign in zip(cap_counts, (1.0, -1.0)):
        if cap_count <= 0:
            continue
        r = radius * np.sqrt(rng.uniform(0.0, 1.0, size=cap_count))
        theta = rng.uniform(0.0, 2.0 * np.pi, size=cap_count)
        batches.append(
            r[:, None] * np.cos(theta)[:, None] * u_axis.reshape(1, 3)
            + r[:, None] * np.sin(theta)[:, None] * v_axis.reshape(1, 3)
            + (sign * half_length) * axis.reshape(1, 3)
        )
    combined = np.concatenate(batches, axis=0).astype(np.float32)
    return center.reshape(1, 3) + combined


def avoid_region_surface_points(
    constraints: list[Any],
    *,
    num_points_per_region: int = DEFAULT_OBSTACLE_POINTS_PER_REGION,
) -> Array:
    """Sample surface points for every supported ``AvoidRegion`` in ``constraints``.

    Supports ``SphereRegion``, ``BoxRegion``, and ``CylinderRegion``; other
    constraint/region types are ignored.
    """
    batches: list[Array] = []
    for constraint in constraints:
        if not isinstance(constraint, AvoidRegion):
            continue
        region = constraint.region
        if isinstance(region, SphereRegion):
            batches.append(sphere_surface_points(region.center, region.radius, num_points_per_region))
        elif isinstance(region, BoxRegion):
            batches.append(
                box_surface_points(
                    region.center,
                    region.half_extents,
                    num_points_per_region,
                    rotation=region.rotation,
                )
            )
        elif isinstance(region, CylinderRegion):
            batches.append(
                cylinder_surface_points(
                    region.center,
                    region.axis,
                    region.radius,
                    region.length,
                    num_points_per_region,
                )
            )
    if not batches:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(batches, axis=0)


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
