from __future__ import annotations

import numpy as np

from scripts.export_real_sim_episode_overlay_rerun import episode_ranges, synchronized_indices


def test_episode_ranges() -> None:
    starts, ends = episode_ranges(np.asarray([3, 8, 10], dtype=np.int64))
    np.testing.assert_array_equal(starts, [0, 3, 8])
    np.testing.assert_array_equal(ends, [3, 8, 10])


def test_synchronized_indices_span_both_full_episodes() -> None:
    real, sim = synchronized_indices(3, 5)
    np.testing.assert_array_equal(real, [0, 0, 1, 2, 2])
    np.testing.assert_array_equal(sim, [0, 1, 2, 3, 4])
    assert real[0] == sim[0] == 0
    assert real[-1] == 2
    assert sim[-1] == 4
