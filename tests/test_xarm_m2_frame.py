from __future__ import annotations

import numpy as np

from pg3d.envs.xarm_adapter.reach_config import (
    M1_TO_M2_XYZ_SHIFT,
    ROBOT_BASE_POSITION,
    XARM7_BALANCED_GOAL_REGIONS,
    XARM7_CROP_BOX_BASE,
    XARM7_M1_SIM_BASE_POSITION,
    XARM7_REACH_BOX_BASE,
    XARM7_REACH_GOAL_CENTER,
    XARM7_REACH_GOAL_HALF_EXTENTS,
    XARM7_REACH_WORKSPACE_BOUNDS,
    XARM7_WORKSPACE_BOUNDS,
    bounds_center_half_extents,
)


def test_xarm_m2_base_frame_constants() -> None:
    np.testing.assert_allclose(ROBOT_BASE_POSITION, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(XARM7_M1_SIM_BASE_POSITION, [-0.615, 0.0, 0.0])
    np.testing.assert_allclose(M1_TO_M2_XYZ_SHIFT, [0.615, 0.0, 0.0])


def test_xarm_world_bounds_are_base_frame_bounds_under_m2() -> None:
    np.testing.assert_allclose(XARM7_REACH_WORKSPACE_BOUNDS, XARM7_REACH_BOX_BASE)
    np.testing.assert_allclose(XARM7_WORKSPACE_BOUNDS, XARM7_CROP_BOX_BASE)


def test_xarm_goal_defaults_are_derived_from_base_frame_reach_box() -> None:
    center, half_extents = bounds_center_half_extents(XARM7_REACH_BOX_BASE)

    np.testing.assert_allclose(XARM7_REACH_GOAL_CENTER, center)
    np.testing.assert_allclose(XARM7_REACH_GOAL_HALF_EXTENTS, half_extents)
    np.testing.assert_allclose(XARM7_REACH_GOAL_CENTER, [0.34, 0.0, 0.21])
    np.testing.assert_allclose(XARM7_REACH_GOAL_HALF_EXTENTS, [0.16, 0.42, 0.16])


def test_xarm_balanced_goal_regions_stay_inside_reach_box() -> None:
    assert [region.name for region in XARM7_BALANCED_GOAL_REGIONS] == [
        "core_practical",
        "outer_practical",
    ]
    assert [region.weight for region in XARM7_BALANCED_GOAL_REGIONS] == [0.70, 0.30]
    for region in XARM7_BALANCED_GOAL_REGIONS:
        bounds = np.asarray(region.bounds, dtype=np.float32)
        assert np.all(bounds[:, 0] >= XARM7_REACH_BOX_BASE[:, 0] - 1e-6)
        assert np.all(bounds[:, 1] <= XARM7_REACH_BOX_BASE[:, 1] + 1e-6)
