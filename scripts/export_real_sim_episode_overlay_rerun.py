from __future__ import annotations

import numpy as np


def episode_ranges(episode_ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert cumulative exclusive episode ends into aligned start/end arrays."""
    ends = np.asarray(episode_ends, dtype=np.int64)
    if ends.ndim != 1 or ends.size == 0:
        raise ValueError("episode_ends must be a non-empty 1D array")
    if np.any(ends <= 0) or np.any(np.diff(ends) <= 0):
        raise ValueError("episode_ends must be strictly increasing and positive")
    starts = np.concatenate((np.zeros(1, dtype=np.int64), ends[:-1]))
    return starts, ends.copy()


def synchronized_indices(real_length: int, sim_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Return endpoint-preserving nearest indices on a shared normalized timeline."""
    if real_length <= 0 or sim_length <= 0:
        raise ValueError("episode lengths must be positive")
    timeline_length = max(real_length, sim_length)
    real = np.rint(np.linspace(0, real_length - 1, timeline_length)).astype(np.int64)
    sim = np.rint(np.linspace(0, sim_length - 1, timeline_length)).astype(np.int64)
    return real, sim
