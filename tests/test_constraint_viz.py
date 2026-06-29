from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from pg3d.constraints import AvoidRegion, BoxRegion, SmoothnessCost, SphereRegion
from pg3d.viz.constraints import (
    avoid_region_line_visuals,
    box_wireframe,
    sphere_wireframe,
)


def test_sphere_wireframe_has_expected_radius_and_shape() -> None:
    center = np.asarray([0.1, -0.2, 0.3], dtype=np.float32)
    strips = sphere_wireframe(center, 0.25, segments=16)

    assert len(strips) == 3
    for strip in strips:
        assert strip.shape == (17, 3)
        assert np.all(np.isfinite(strip))
        np.testing.assert_allclose(strip[0], strip[-1], atol=1e-6)
        distances = np.linalg.norm(strip - center.reshape(1, 3), axis=1)
        np.testing.assert_allclose(distances, 0.25, atol=1e-6)


def test_box_wireframe_matches_center_and_half_extents() -> None:
    center = np.asarray([0.2, 0.3, 0.4], dtype=np.float32)
    half_extents = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)

    strips = box_wireframe(center, half_extents)
    vertices = np.concatenate(strips, axis=0)

    assert len(strips) == 12
    assert all(strip.shape == (2, 3) for strip in strips)
    np.testing.assert_allclose(vertices.min(axis=0), center - half_extents)
    np.testing.assert_allclose(vertices.max(axis=0), center + half_extents)


def test_avoid_region_line_visuals_ignore_non_avoid_constraints() -> None:
    sphere = AvoidRegion(region=SphereRegion(center=[0.0, 0.0, 0.0], radius=0.1))
    box = AvoidRegion(
        region=BoxRegion(center=[1.0, 0.0, 0.0], half_extents=[0.1, 0.2, 0.3])
    )

    visuals = avoid_region_line_visuals([SmoothnessCost(), sphere, box], sphere_segments=16)

    assert [visual.name for visual in visuals] == ["avoid_region_0", "avoid_region_1"]
    assert len(visuals[0].line_strips) == 3
    assert len(visuals[1].line_strips) == 12


def test_constraint_viz_import_keeps_runtime_deps_lazy() -> None:
    code = """
import importlib
import sys

importlib.import_module("pg3d.viz.constraints")
assert "mani_skill" not in sys.modules
assert "sapien" not in sys.modules
assert "gymnasium" not in sys.modules
assert "rerun" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_sphere_wireframe_validates_inputs() -> None:
    with pytest.raises(ValueError, match="segments"):
        sphere_wireframe([0.0, 0.0, 0.0], 0.1, segments=4)
    with pytest.raises(ValueError, match="radius"):
        sphere_wireframe([0.0, 0.0, 0.0], 0.0)
