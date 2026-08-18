from __future__ import annotations

import numpy as np
import pytest

from pg3d.envs.obstacles import (
    CABINET_COMPONENTS,
    transform_box_component,
    u_shape_components,
)


def test_cabinet_contains_body_shelf_and_open_door_components() -> None:
    names = {component.name for component in CABINET_COMPONENTS}

    assert {
        "left_side",
        "right_side",
        "top",
        "bottom",
        "back",
        "shelf",
        "open_door",
    } == names
    door = next(component for component in CABINET_COMPONENTS if component.name == "open_door")
    assert door.yaw_offset == np.deg2rad(70.0)


def test_cabinet_component_transform_applies_root_pose() -> None:
    shelf = next(component for component in CABINET_COMPONENTS if component.name == "shelf")
    left = next(component for component in CABINET_COMPONENTS if component.name == "left_side")
    root = np.asarray([0.2, -0.1, 0.4], dtype=np.float32)

    shelf_center, shelf_yaw = transform_box_component(
        shelf, center=root, yaw=np.pi / 2
    )
    left_center, left_yaw = transform_box_component(
        left, center=root, yaw=np.pi / 2
    )

    np.testing.assert_allclose(shelf_center, root)
    assert shelf_yaw == np.pi / 2
    np.testing.assert_allclose(left_center, [0.2, -0.175, 0.4], atol=1e-6)
    assert left_yaw == np.pi / 2


def test_u_shape_has_open_front_and_closed_back() -> None:
    components = u_shape_components((0.14, 0.15, 0.30))

    assert {component.name for component in components} == {
        "left_side",
        "right_side",
        "back",
    }
    left = next(component for component in components if component.name == "left_side")
    right = next(component for component in components if component.name == "right_side")
    back = next(component for component in components if component.name == "back")
    assert left.local_center[0] == -right.local_center[0]
    assert left.local_center[1] == right.local_center[1] == 0.0
    assert back.local_center[1] > 0.0
    assert back.half_extents[0] == pytest.approx(0.14)
    assert [component.half_extents[2] for component in components] == pytest.approx(
        [0.30, 0.30, 0.30]
    )
