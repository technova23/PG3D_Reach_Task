from __future__ import annotations

import numpy as np

from scripts.validate_obstacle_collision_geometry import (
    _family_regions,
    _intersecting_probe_center,
    parse_args,
)


def test_collision_validation_geometry_has_inside_and_outside_probe_cases() -> None:
    center = np.asarray([0.0, 0.0, 0.75], dtype=np.float32)
    dimensions_by_family = {
        "box": (0.04, 0.06, 0.08),
        "carton": (0.055, 0.08, 0.16),
        "cylinder": (0.055, 0.055, 0.12),
        "cabinet": (0.08, 0.085, 0.20),
    }

    for family, dimensions in dimensions_by_family.items():
        regions = _family_regions(family, center=center, dimensions=dimensions)
        inside_point = _intersecting_probe_center(
            family,
            center=center,
            dimensions=dimensions,
            probe_radius=0.01,
        )
        inside = (
            min(
                region.signed_distance(inside_point.reshape(1, 3))[0]
                for region in regions
            )
            - 0.01
        )
        outside_point = center + [0.5, 0.0, 0.0]
        outside = min(
            region.signed_distance(outside_point.reshape(1, 3))[0]
            for region in regions
        )
        assert inside < 0.0
        assert outside > 0.0


def test_collision_validation_cli_defaults_cover_every_family() -> None:
    args = parse_args([])

    assert args.families == ["box", "cabinet", "carton", "cylinder"]
