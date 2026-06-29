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
    """Return deterministic structured offsets used for target-centered goal tokens."""
    if num_points < 0:
        raise ValueError("num_points must be non-negative")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if num_points == 0:
        return np.zeros((0, 3), dtype=np.float32)

    r = np.float32(radius)
    if r == 0:
        return np.zeros((num_points, 3), dtype=np.float32)

    pattern: list[np.ndarray] = [np.zeros(3, dtype=np.float32)]
    cross_dirs = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float32,
    )
    for direction in cross_dirs:
        pattern.append(r * direction)

    ring_count = max(0, num_points - len(pattern))
    for idx in range(ring_count):
        angle = 2.0 * np.pi * idx / max(ring_count, 1)
        ring_radius = r * (0.70 if idx % 2 == 0 else 1.00)
        z_offset = r * 0.25 * (1.0 if idx % 4 in {0, 1} else -1.0)
        pattern.append(
            np.asarray(
                [ring_radius * np.cos(angle), ring_radius * np.sin(angle), z_offset],
                dtype=np.float32,
            )
        )

    if num_points <= len(pattern):
        return np.asarray(pattern[:num_points], dtype=np.float32)
    repeats = int(np.ceil(num_points / len(pattern)))
    return np.tile(np.asarray(pattern, dtype=np.float32), (repeats, 1))[:num_points]


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
