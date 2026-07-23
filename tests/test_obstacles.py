from __future__ import annotations

import numpy as np

from pg3d.envs.obstacles import CABINET_COMPONENTS, transform_box_component


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
