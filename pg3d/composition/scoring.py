from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from pg3d.constraints.core import mean_squared_norm
from pg3d.world_model import ActionChunk, ImaginedRollout


def goal_distance(rollout: ImaginedRollout, target_position: np.ndarray | None) -> float | None:
    """Return final EEF distance to the target when a target is available."""
    if target_position is None:
        return None
    return float(np.linalg.norm(rollout.eef_path[-1] - target_position))


def trajectory_smoothness(rollout: ImaginedRollout, *, order: int = 2) -> float:
    """Return mean squared joint trajectory finite-difference norm."""
    if order not in {1, 2}:
        raise ValueError("order must be 1 or 2")
    if rollout.q.shape[0] <= order:
        return 0.0
    return mean_squared_norm(np.diff(rollout.q, n=order, axis=0))


def directional_preference(
    rollout: ImaginedRollout,
    target_position: np.ndarray | None,
    *,
    sign: int,
) -> float:
    """Return a lower-is-better cost that prefers TCP paths bowing to one side.

    "Left" and "right" are defined relative to the sightline from the path start
    toward the goal, projected onto the ground (XY) plane: forward points at the
    goal, world +Z is up, and ``left = up x forward`` (a 90 deg CCW turn of the
    forward vector). The signed lateral offset of every TCP point from the start is
    averaged over the path; ``sign=+1`` rewards positive (left) offsets, ``sign=-1``
    rewards right, and ``sign=0`` disables the term. Start and goal are shared across
    candidates, so this only ranks how far a candidate bows to the chosen side.
    """
    if sign == 0 or target_position is None:
        return 0.0
    path = np.asarray(rollout.eef_path, dtype=np.float32).reshape(-1, 3)
    if path.shape[0] < 2:
        return 0.0
    start = path[0]
    forward = np.asarray(target_position, dtype=np.float32).reshape(3)[:2] - start[:2]
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm <= 1e-8:
        return 0.0
    forward /= forward_norm
    left_dir = np.array([-forward[1], forward[0]], dtype=np.float32)
    lateral = (path[:, :2] - start[:2].reshape(1, 2)) @ left_dir
    return float(-sign * np.mean(lateral))


def consensus_deviations(chunks: list[ActionChunk]) -> list[float]:
    """Return per-chunk mean squared deviation from compatible candidate consensus."""
    deviations = [0.0 for _ in chunks]
    groups: dict[tuple[str, tuple[int, ...]], list[int]] = defaultdict(list)
    for idx, chunk in enumerate(chunks):
        groups[(chunk.action_mode, tuple(chunk.actions.shape))].append(idx)

    for indices in groups.values():
        if len(indices) <= 1:
            continue
        stack = np.stack([chunks[idx].actions for idx in indices], axis=0)
        mean = np.mean(stack, axis=0, dtype=np.float32)
        for idx in indices:
            deviations[idx] = float(np.mean((chunks[idx].actions - mean) ** 2))
    return deviations


def primary_constraint_penalty(costs: dict[str, float]) -> float:
    """Sum primary constraint terms while ignoring detailed slash-qualified diagnostics."""
    primary = [
        float(value)
        for key, value in costs.items()
        if "/" not in key and np.isfinite(float(value))
    ]
    if primary:
        return float(sum(primary))
    return float(
        sum(
            max(float(value), 0.0)
            for value in costs.values()
            if np.isfinite(float(value))
        )
    )


def optional_policy_surrogate(
    policy: Any,
    policy_input: Any,
    chunks: list[ActionChunk],
) -> list[float | None]:
    """Return optional lower-is-better policy surrogate scores."""
    score_fn = getattr(policy, "score_surrogate", None)
    if score_fn is None:
        return [None for _ in chunks]
    scores = list(score_fn(policy_input, chunks))
    if len(scores) != len(chunks):
        raise ValueError(
            f"score_surrogate returned {len(scores)} scores for {len(chunks)} chunks"
        )
    return [float(score) for score in scores]
