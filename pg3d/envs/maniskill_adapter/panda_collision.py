from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from pg3d.world_model.panda_collision import PandaCollisionPointTemplate
from pg3d.world_model.panda_fk import PANDA_MOVABLE_COLLISION_LINKS


def load_panda_collision_point_template(
    urdf_path: str | Path,
    *,
    point_count: int = 1024,
    sample_seed: int = 0,
    min_points_per_link: int = 32,
) -> PandaCollisionPointTemplate:
    """Load and deterministically sample movable Panda collision geometry from a URDF."""
    if point_count <= 0:
        raise ValueError("point_count must be positive")
    if sample_seed < 0:
        raise ValueError("sample_seed must be non-negative")
    if min_points_per_link <= 0:
        raise ValueError("min_points_per_link must be positive")
    minimum_total = min_points_per_link * len(PANDA_MOVABLE_COLLISION_LINKS)
    if point_count < minimum_total:
        raise ValueError(
            f"point_count must be at least {minimum_total} to allocate "
            f"{min_points_per_link} points per movable link"
        )

    path = Path(urdf_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Panda URDF does not exist: {path}")
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"invalid Panda URDF XML: {path}") from exc

    link_elements = {element.get("name"): element for element in root.findall("link")}
    missing = [name for name in PANDA_MOVABLE_COLLISION_LINKS if name not in link_elements]
    if missing:
        raise ValueError(f"Panda URDF is missing movable collision links: {missing}")

    meshes = [
        _link_collision_mesh(link_elements[name], urdf_path=path)
        for name in PANDA_MOVABLE_COLLISION_LINKS
    ]
    areas = np.asarray([float(mesh.area) for mesh in meshes], dtype=np.float64)
    if not np.isfinite(areas).all() or np.any(areas <= 0.0):
        raise ValueError("every movable Panda link must have positive collision surface area")
    counts = _allocate_point_counts(
        areas,
        point_count=point_count,
        min_points_per_link=min_points_per_link,
    )

    local_points = []
    link_indices = []
    for link_index, (mesh, count) in enumerate(zip(meshes, counts, strict=True)):
        link_seed = int(np.random.SeedSequence([sample_seed, link_index]).generate_state(1)[0])
        points = _sample_surface_exact(mesh, int(count), seed=link_seed)
        local_points.append(points.astype(np.float32, copy=False))
        link_indices.append(np.full(int(count), link_index, dtype=np.int64))

    return PandaCollisionPointTemplate(
        local_points=np.concatenate(local_points, axis=0),
        link_indices=np.concatenate(link_indices, axis=0),
        link_counts=tuple(int(value) for value in counts),
        sample_seed=sample_seed,
    )


def _link_collision_mesh(link: ET.Element, *, urdf_path: Path):
    import trimesh

    collisions = link.findall("collision")
    link_name = link.get("name", "<unnamed>")
    if not collisions:
        raise ValueError(f"Panda link {link_name!r} has no collision geometry")
    meshes = []
    for collision in collisions:
        geometry = collision.find("geometry")
        if geometry is None:
            raise ValueError(f"Panda link {link_name!r} has collision without geometry")
        origin = _origin_matrix(collision.find("origin"))
        mesh_element = geometry.find("mesh")
        box_element = geometry.find("box")
        if mesh_element is not None:
            mesh = _load_mesh(mesh_element, urdf_path=urdf_path)
            mesh.apply_transform(origin)
        elif box_element is not None:
            size = _parse_vector(box_element.get("size"), length=3, name="box size")
            if np.any(size <= 0.0):
                raise ValueError(f"Panda link {link_name!r} has non-positive box size")
            mesh = trimesh.creation.box(extents=size, transform=origin)
        else:
            children = [child.tag for child in geometry]
            raise ValueError(
                f"Panda link {link_name!r} uses unsupported collision geometry {children}"
            )
        meshes.append(mesh)
    return trimesh.util.concatenate(meshes)


def _load_mesh(element: ET.Element, *, urdf_path: Path):
    import trimesh

    filename = element.get("filename")
    if not filename:
        raise ValueError("Panda collision mesh is missing filename")
    if filename.startswith("package://"):
        filename = filename.removeprefix("package://")
    mesh_path = Path(filename)
    if not mesh_path.is_absolute():
        mesh_path = urdf_path.parent / mesh_path
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Panda collision mesh does not exist: {mesh_path}")
    loaded = trimesh.load(mesh_path, force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"Panda collision asset is not a triangle mesh: {mesh_path}")
    mesh = loaded.copy()
    scale_text = element.get("scale")
    if scale_text is not None:
        scale = _parse_vector(scale_text, length=3, name="mesh scale")
        if np.any(scale <= 0.0):
            raise ValueError(f"Panda collision mesh has non-positive scale: {mesh_path}")
        mesh.vertices = np.asarray(mesh.vertices) * scale.reshape(1, 3)
    return mesh


def _origin_matrix(element: ET.Element | None) -> np.ndarray:
    xyz = (
        np.zeros(3, dtype=np.float64)
        if element is None or element.get("xyz") is None
        else _parse_vector(element.get("xyz"), length=3, name="origin xyz")
    )
    rpy = (
        np.zeros(3, dtype=np.float64)
        if element is None or element.get("rpy") is None
        else _parse_vector(element.get("rpy"), length=3, name="origin rpy")
    )
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    matrix = np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, xyz[0]],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, xyz[1]],
            [-sp, cp * sr, cp * cr, xyz[2]],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return matrix


def _parse_vector(value: str | None, *, length: int, name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"missing {name}")
    try:
        result = np.asarray([float(item) for item in value.split()], dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"invalid {name}: {value!r}") from exc
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {length} finite values")
    return result


def _allocate_point_counts(
    areas: np.ndarray,
    *,
    point_count: int,
    min_points_per_link: int,
) -> np.ndarray:
    remaining = point_count - min_points_per_link * areas.shape[0]
    shares = remaining * areas / float(areas.sum())
    extras = np.floor(shares).astype(np.int64)
    leftover = int(remaining - int(extras.sum()))
    if leftover:
        fractional = shares - extras
        order = np.argsort(-fractional, kind="stable")
        extras[order[:leftover]] += 1
    counts = extras + min_points_per_link
    if int(counts.sum()) != point_count:
        raise RuntimeError("collision-point allocation did not preserve the requested total")
    return counts


def _sample_surface_exact(mesh, count: int, *, seed: int) -> np.ndarray:
    import trimesh

    points, _ = trimesh.sample.sample_surface_even(mesh, count, seed=seed)
    if points.shape[0] < count:
        missing = count - points.shape[0]
        fill, _ = trimesh.sample.sample_surface(mesh, missing, seed=seed + 1)
        points = np.concatenate((points, fill), axis=0)
    if points.shape != (count, 3) or not np.isfinite(points).all():
        raise RuntimeError(f"collision sampling returned invalid shape {points.shape}")
    return np.asarray(points, dtype=np.float64)
