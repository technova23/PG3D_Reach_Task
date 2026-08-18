from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from pg3d.constraints import BoxRegion, CylinderRegion
from pg3d.envs.maniskill_adapter import register_pg3d_reach_envs
from pg3d.envs.obstacles import (
    CABINET_COMPONENTS,
    transform_box_component,
    u_shape_components,
)
from pg3d.utils.arrays import to_numpy

_FAMILY_DIMENSIONS = {
    "box": (0.04, 0.06, 0.08),
    "carton": (0.055, 0.08, 0.16),
    "cylinder": (0.055, 0.055, 0.12),
    "cabinet": (0.08, 0.085, 0.20),
    "u_shape": (0.14, 0.15, 0.30),
}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import gymnasium as gym

    register_pg3d_reach_envs()
    results = []
    for family in args.families:
        dimensions = _FAMILY_DIMENSIONS[family]
        env = gym.make(
            args.env_id,
            num_envs=1,
            obs_mode="state",
            pg3d_obstacle_half_extents=dimensions,
            pg3d_obstacle_family=family,
            pg3d_collision_probe_radius=args.probe_radius,
        )
        try:
            center = np.asarray([0.0, 0.0, 0.75], dtype=np.float32)
            unwrapped = env.unwrapped
            regions = _family_regions(family, center=center, dimensions=dimensions)
            for label, probe_center in (
                (
                    "intersecting",
                    _intersecting_probe_center(
                        family,
                        center=center,
                        dimensions=dimensions,
                        probe_radius=args.probe_radius,
                    ),
                ),
                ("separated", center + np.asarray([0.5, 0.0, 0.0], dtype=np.float32)),
            ):
                env.reset(
                    seed=args.seed,
                    options={
                        "reconfigure": False,
                        "pg3d_obstacle_center": center.tolist(),
                        "pg3d_obstacle_yaw": 0.0,
                        "pg3d_collision_probe_center": probe_center.tolist(),
                    },
                )
                unwrapped = env.unwrapped
                probe = unwrapped.pg3d_collision_probe
                actors_for_family = list(unwrapped.pg3d_obstacle_actors)
                contact_force = 0.0
                contact = False
                for _ in range(3):
                    unwrapped.scene.step()
                    contact = contact or _has_obstacle_probe_contact(unwrapped.scene)
                    contact_force = max(
                        contact_force,
                        sum(
                            float(
                                np.linalg.norm(
                                    to_numpy(
                                        unwrapped.scene.get_pairwise_contact_forces(
                                            actor, probe
                                        )
                                    )
                                )
                            )
                            for actor in actors_for_family
                        ),
                    )
                min_clearance = min(
                    float(region.signed_distance(probe_center.reshape(1, 3))[0])
                    - args.probe_radius
                    for region in regions
                )
                geometry_intersection = min_clearance < 0.0
                results.append(
                    {
                        "family": family,
                        "case": label,
                        "probe_center": probe_center.astype(float).tolist(),
                        "probe_radius": args.probe_radius,
                        "min_signed_clearance": min_clearance,
                        "contact_force_norm": contact_force,
                        "simulator_contact": contact,
                        "geometry_intersection": geometry_intersection,
                        "agree": contact == geometry_intersection,
                    }
                )
        finally:
            env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    failures = [result for result in results if not result["agree"]]
    print(json.dumps({"cases": len(results), "failures": failures}, indent=2))
    return 1 if failures else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare embodied-obstacle simulator contacts with serialized SDF geometry."
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=sorted(_FAMILY_DIMENSIONS),
        default=sorted(_FAMILY_DIMENSIONS),
    )
    parser.add_argument("--env-id", default="PG3DReach-BalancedWorkspace-v0")
    parser.add_argument("--probe-radius", type=float, default=0.01)
    parser.add_argument("--contact-force-tolerance", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/e2-obstacle-collision-validation/results.json"),
    )
    args = parser.parse_args(argv)
    if args.probe_radius <= 0.0:
        raise ValueError("--probe-radius must be positive")
    if args.contact_force_tolerance < 0.0:
        raise ValueError("--contact-force-tolerance must be non-negative")
    return args


def _family_regions(
    family: str,
    *,
    center: np.ndarray,
    dimensions: tuple[float, float, float],
) -> list[BoxRegion | CylinderRegion]:
    if family == "cylinder":
        return [
            CylinderRegion(
                center=center,
                radius=dimensions[0],
                half_length=dimensions[2],
            )
        ]
    if family in {"cabinet", "u_shape"}:
        components = (
            CABINET_COMPONENTS
            if family == "cabinet"
            else u_shape_components(dimensions)
        )
        return [
            BoxRegion(
                center=component_center,
                half_extents=component.half_extents,
                yaw=component_yaw,
            )
            for component in components
            for component_center, component_yaw in [
                transform_box_component(component, center=center, yaw=0.0)
            ]
        ]
    return [BoxRegion(center=center, half_extents=dimensions)]


def _intersecting_probe_center(
    family: str,
    *,
    center: np.ndarray,
    dimensions: tuple[float, float, float],
    probe_radius: float,
) -> np.ndarray:
    """Place the probe across an exterior collision surface, not fully contained."""
    if family in {"cabinet", "u_shape"}:
        components = (
            CABINET_COMPONENTS
            if family == "cabinet"
            else u_shape_components(dimensions)
        )
        probe_component = next(
            component
            for component in components
            if component.name == ("shelf" if family == "cabinet" else "back")
        )
        component_center, _ = transform_box_component(
            probe_component, center=center, yaw=0.0
        )
        return component_center + np.asarray(
            [probe_component.half_extents[0] + 0.5 * probe_radius, 0.0, 0.0],
            dtype=np.float32,
        )
    radial_extent = dimensions[0]
    return center + np.asarray(
        [radial_extent + 0.5 * probe_radius, 0.0, 0.0],
        dtype=np.float32,
    )


def _has_obstacle_probe_contact(scene: Any) -> bool:
    for contact in scene.get_contacts():
        names = {_contact_body_name(body) for body in contact.bodies}
        if any("pg3d_collision_probe" in name for name in names) and any(
            "pg3d_obstacle" in name for name in names
        ):
            return True
    return False


def _contact_body_name(body: Any) -> str:
    name = getattr(body, "name", None)
    if name is None and hasattr(body, "get_name"):
        name = body.get_name()
    return str(name or "")


if __name__ == "__main__":
    raise SystemExit(main())
