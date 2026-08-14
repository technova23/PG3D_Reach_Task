from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import trimesh

from pg3d.envs.maniskill_adapter.panda_collision import (
    load_panda_collision_point_template,
)
from pg3d.world_model.panda_collision import (
    DifferentiablePandaCollisionPoints,
    PandaCollisionPointTemplate,
)
from pg3d.world_model.panda_fk import PANDA_MOVABLE_COLLISION_LINKS


def _write_test_urdf(tmp_path: Path, *, unsupported: bool = False) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.creation.box(extents=(0.1, 0.2, 0.3))
    mesh.export(tmp_path / "link1.stl")
    links = []
    for index, name in enumerate(PANDA_MOVABLE_COLLISION_LINKS):
        if index == 0:
            geometry = '<mesh filename="link1.stl" scale="2 1 1"/>'
        elif unsupported and index == 1:
            geometry = '<sphere radius="0.1"/>'
        else:
            size = 0.02 * (index + 1)
            geometry = f'<box size="{size} {size + 0.01} {size + 0.02}"/>'
        links.append(
            f"""
            <link name="{name}">
              <collision>
                <origin xyz="{0.01 * index} 0 0" rpy="0 0 0"/>
                <geometry>{geometry}</geometry>
              </collision>
            </link>
            """
        )
    path = tmp_path / "panda.urdf"
    path.write_text(f'<robot name="panda">{"".join(links)}</robot>', encoding="utf-8")
    return path


def test_collision_sampling_is_deterministic_and_allocates_every_link(tmp_path: Path) -> None:
    urdf_path = _write_test_urdf(tmp_path)
    first = load_panda_collision_point_template(urdf_path, point_count=400, sample_seed=7)
    second = load_panda_collision_point_template(urdf_path, point_count=400, sample_seed=7)

    assert first.point_count == 400
    assert sum(first.link_counts) == 400
    assert min(first.link_counts) >= 32
    assert set(first.allocation()) == set(PANDA_MOVABLE_COLLISION_LINKS)
    np.testing.assert_array_equal(first.local_points, second.local_points)
    np.testing.assert_array_equal(first.link_indices, second.link_indices)
    assert first.link_counts == second.link_counts


def test_collision_sampling_applies_mesh_scale_and_collision_origin(tmp_path: Path) -> None:
    template = load_panda_collision_point_template(
        _write_test_urdf(tmp_path),
        point_count=320,
        sample_seed=3,
    )
    link1_points = template.local_points[template.link_indices == 0]
    half_extents = np.asarray([0.1, 0.1, 0.15], dtype=np.float32)
    distances_to_faces = np.abs(np.abs(link1_points) - half_extents)

    assert np.all(np.min(distances_to_faces, axis=1) < 1e-5)
    link2_points = template.local_points[template.link_indices == 1]
    centered = link2_points - np.asarray([0.01, 0.0, 0.0], dtype=np.float32)
    link2_half_extents = np.asarray([0.02, 0.025, 0.03], dtype=np.float32)
    assert np.all(np.min(np.abs(np.abs(centered) - link2_half_extents), axis=1) < 1e-5)


def test_differentiable_collision_points_batch_shape_and_gradient(tmp_path: Path) -> None:
    template = load_panda_collision_point_template(
        _write_test_urdf(tmp_path),
        point_count=320,
        sample_seed=0,
    )
    model = DifferentiablePandaCollisionPoints(template, gripper_open=0.04).to(torch.float64)
    q = torch.tensor(
        [[[0.1, 0.3, -0.2, -1.5, 0.2, 1.7, 0.4]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    points = model(q)
    loss = points[..., 0].sum()
    gradient = torch.autograd.grad(loss, q)[0]

    assert points.shape == (1, 1, 320, 3)
    assert gradient.shape == q.shape
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient).item() > 0


def test_collision_template_and_loader_validate_inputs(tmp_path: Path) -> None:
    urdf_path = _write_test_urdf(tmp_path)
    with pytest.raises(ValueError, match="at least 320"):
        load_panda_collision_point_template(urdf_path, point_count=319)
    with pytest.raises(ValueError, match="unsupported collision geometry"):
        load_panda_collision_point_template(
            _write_test_urdf(tmp_path / "unsupported", unsupported=True),
            point_count=320,
        )
    with pytest.raises(ValueError, match="link_counts must sum"):
        PandaCollisionPointTemplate(
            local_points=np.zeros((10, 3), dtype=np.float32),
            link_indices=np.arange(10, dtype=np.int64),
            link_counts=(2,) * 10,
            sample_seed=0,
        )
    with pytest.raises(ValueError, match="gripper_open"):
        DifferentiablePandaCollisionPoints(
            load_panda_collision_point_template(urdf_path, point_count=320),
            gripper_open=0.05,
        )
