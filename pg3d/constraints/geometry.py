from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np

from pg3d.world_model.types import Array, as_float_array

RegionType = Literal["sphere", "box", "rect2d"]


class Region(Protocol):
    """Simple keep-out region primitive."""

    region_type: RegionType

    def signed_distance(self, points: Array) -> Array:
        """Return positive outside, zero on boundary, and negative inside."""
        ...

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-safe region config."""
        ...


@dataclass(frozen=True)
class SphereRegion:
    """Spherical keep-out region."""

    center: Array
    radius: float
    region_type: RegionType = "sphere"

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _vector3(self.center, name="center"))
        radius = float(self.radius)
        if radius <= 0.0 or not np.isfinite(radius):
            raise ValueError("radius must be a positive finite value")
        object.__setattr__(self, "radius", radius)

    def signed_distance(self, points: Array) -> Array:
        points = _points(points)
        return np.linalg.norm(points - self.center.reshape(1, 3), axis=1) - self.radius

    def to_json(self) -> dict[str, Any]:
        return {
            "type": self.region_type,
            "center": self.center.tolist(),
            "radius": self.radius,
        }


@dataclass(frozen=True)
class BoxRegion:
    """Axis-aligned box keep-out region."""

    center: Array
    half_extents: Array
    region_type: RegionType = "box"

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _vector3(self.center, name="center"))
        half_extents = _vector3(self.half_extents, name="half_extents")
        if np.any(half_extents <= 0.0):
            raise ValueError("half_extents must be positive")
        object.__setattr__(self, "half_extents", half_extents)

    def signed_distance(self, points: Array) -> Array:
        points = _points(points)
        q = np.abs(points - self.center.reshape(1, 3)) - self.half_extents.reshape(1, 3)
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(np.max(q, axis=1), 0.0)
        return outside + inside

    def to_json(self) -> dict[str, Any]:
        return {
            "type": self.region_type,
            "center": self.center.tolist(),
            "half_extents": self.half_extents.tolist(),
        }


@dataclass(frozen=True)
class RectRegion2D:
    """Axis-aligned rectangle keep-out in the XY plane, infinite in Z.

    Used for the ``avoid_projection`` (no-overflight) constraint: only the XY
    footprint matters, so the keep-out is a 2-D rectangle that extends through
    all heights. ``signed_distance`` accepts ``[N, 2]`` or ``[N, 3]`` points and
    silently projects onto XY (dropping the Z column), so the same primitive can
    score imagined EEF paths and feed the executed-trajectory clearance metric
    without any caller changes.
    """

    center: Array
    half_extents: Array
    region_type: RegionType = "rect2d"

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _vector2(self.center, name="center"))
        half_extents = _vector2(self.half_extents, name="half_extents")
        if np.any(half_extents <= 0.0):
            raise ValueError("half_extents must be positive")
        object.__setattr__(self, "half_extents", half_extents)

    def signed_distance(self, points: Array) -> Array:
        points = _points_xy(points)
        q = np.abs(points - self.center.reshape(1, 2)) - self.half_extents.reshape(1, 2)
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(np.max(q, axis=1), 0.0)
        return outside + inside

    def to_json(self) -> dict[str, Any]:
        return {
            "type": self.region_type,
            "center": self.center.tolist(),
            "half_extents": self.half_extents.tolist(),
        }


def region_from_json(config: dict[str, Any]) -> Region:
    """Load a region primitive from a JSON-safe config."""
    region_type = config.get("type")
    if region_type == "sphere":
        return SphereRegion(center=config["center"], radius=float(config["radius"]))
    if region_type == "box":
        return BoxRegion(center=config["center"], half_extents=config["half_extents"])
    if region_type == "rect2d":
        return RectRegion2D(center=config["center"], half_extents=config["half_extents"])
    raise ValueError(f"unknown region type {region_type!r}")


def _vector3(value: Any, *, name: str) -> Array:
    array = as_float_array(value, name=name, ndim=1)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}")
    return array


def _vector2(value: Any, *, name: str) -> Array:
    array = as_float_array(value, name=name, ndim=1)
    if array.shape != (2,):
        raise ValueError(f"{name} must have shape (2,), got {array.shape}")
    return array


def _points(value: Any) -> Array:
    points = as_float_array(value, name="points", ndim=2)
    if points.shape[1] != 3:
        raise ValueError(f"points must have shape [N, 3], got {points.shape}")
    return points


def _points_xy(value: Any) -> Array:
    """Return the XY columns of ``[N, 2]`` or ``[N, 3]`` points for 2-D scoring."""
    points = as_float_array(value, name="points", ndim=2)
    if points.shape[1] not in (2, 3):
        raise ValueError(f"points must have shape [N, 2] or [N, 3], got {points.shape}")
    return points[:, :2]
