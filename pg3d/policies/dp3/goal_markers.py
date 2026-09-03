from __future__ import annotations

import numpy as np

Array = np.ndarray

# Match the dataset bake (`--goal-marker-points 192 --goal-marker-radius 0.055`).
# The encoder splits off the trailing `goal_marker_points` slots as the goal
# branch, so this MUST equal the number of goal slots baked by the dataset
# writer or baked markers leak into the PointNet scene branch.
DEFAULT_GOAL_MARKER_POINTS = 192
DEFAULT_GOAL_MARKER_RADIUS = 0.055


def goal_marker_offsets(
    *,
    num_points: int = DEFAULT_GOAL_MARKER_POINTS,
    radius: float = DEFAULT_GOAL_MARKER_RADIUS,
) -> Array:
    """Return deterministic points on a sphere shell, used for target-centered goal tokens.

    Fibonacci/golden-angle spiral placement -- purely a function of `num_points`,
    no RNG, so `offsets[k]` means the exact same thing on every call. That
    determinism matters beyond just reproducibility: these offsets also feed
    `PointNetEncoderXYZ.goal_marker_mlp` (pg3d/policies/dp3/modules.py), which
    flattens the marker points into one fixed-length vector rather than pooling
    them order-invariantly -- index k has to be the same offset direction every
    call, or the MLP sees a different function of the same target position each
    time. Matches the golden-angle construction already used for the TCP marker
    in dataset_generation/write_maniskill_reach_dataset.py::_marker_sphere_points
    (minus that function's extra radial "ring" jitter -- every point here sits
    exactly on the sphere shell at `radius`).

    A sphere is rotationally symmetric (SO(3)-invariant): unlike an asymmetric
    marker, rotating it by the target orientation would produce an identical
    point set, so this shape carries no orientation information by construction
    -- same as the cross+ring pattern it replaces, which also never varied with
    target orientation despite being asymmetric in a fixed, world-frame sense.
    """
    if num_points < 0:
        raise ValueError("num_points must be non-negative")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if num_points == 0:
        return np.zeros((0, 3), dtype=np.float32)

    r = np.float32(radius)
    if r == 0:
        return np.zeros((num_points, 3), dtype=np.float32)

    indices = np.arange(num_points, dtype=np.float64)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (indices + 0.5) / float(num_points)
    radial = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = indices * golden_angle
    shell = np.stack([radial * np.cos(theta), radial * np.sin(theta), z], axis=1)
    return (r * shell).astype(np.float32)


def goal_marker_points(
    target_position: Array,
    *,
    num_points: int = DEFAULT_GOAL_MARKER_POINTS,
    radius: float = DEFAULT_GOAL_MARKER_RADIUS,
) -> Array:
    """Return fixed ordered marker points centered at each target position."""
    target = np.asarray(target_position, dtype=np.float32)
    if target.shape[-1:] != (3,):
        raise ValueError(f"target_position must end with shape [3], got {target.shape}")
    offsets = goal_marker_offsets(num_points=num_points, radius=radius)
    if num_points == 0:
        return np.zeros((*target.shape[:-1], 0, 3), dtype=np.float32)
    return target[..., None, :] + offsets.reshape((1,) * (target.ndim - 1) + offsets.shape)


def insert_goal_marker_points(
    point_cloud: Array,
    target_position: Array,
    *,
    num_points: int = DEFAULT_GOAL_MARKER_POINTS,
    radius: float = DEFAULT_GOAL_MARKER_RADIUS,
) -> Array:
    """Overwrite the final ``num_points`` point-cloud slots with ordered goal tokens."""
    points = np.asarray(point_cloud, dtype=np.float32)
    if points.shape[-1:] != (3,):
        raise ValueError(f"point_cloud must end with shape [*, 3], got {points.shape}")
    if points.ndim < 2:
        raise ValueError(f"point_cloud must have at least 2 dimensions, got {points.shape}")
    if num_points < 0:
        raise ValueError("num_points must be non-negative")
    if num_points == 0:
        return points.astype(np.float32, copy=True)
    if num_points >= points.shape[-2]:
        raise ValueError(
            "num_points must be smaller than the point-cloud point count "
            f"({num_points} >= {points.shape[-2]})"
        )

    marker = goal_marker_points(target_position, num_points=num_points, radius=radius)
    expected_marker_shape = (*points.shape[:-2], num_points, 3)
    try:
        marker = np.broadcast_to(marker, expected_marker_shape)
    except ValueError as exc:
        raise ValueError(
            f"target_position shape {np.asarray(target_position).shape} cannot broadcast "
            f"to point_cloud shape {points.shape}"
        ) from exc

    output = points.astype(np.float32, copy=True)
    output[..., -num_points:, :] = marker
    return output
