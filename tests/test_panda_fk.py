from __future__ import annotations

import pytest
import torch

from pg3d.world_model.panda_fk import panda_end_effector_position


def test_panda_fk_matches_urdf_reference_positions() -> None:
    q = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.3926991, 0.0, -1.9634954, 0.0, 2.3561945, 0.7853982],
        ],
        dtype=torch.float64,
    )
    expected = torch.tensor(
        [
            [0.088, 0.0, 0.8226],
            [0.61501336, 0.0, 0.1697819],
        ],
        dtype=torch.float64,
    )

    torch.testing.assert_close(
        panda_end_effector_position(q),
        expected,
        atol=1e-7,
        rtol=1e-7,
    )


def test_panda_fk_applies_broadcast_world_transform() -> None:
    q = torch.zeros((2, 3, 7), dtype=torch.float64)
    world_from_base = torch.eye(4, dtype=torch.float64)
    world_from_base[:3, 3] = torch.tensor([-0.615, 0.2, 0.1], dtype=torch.float64)

    positions = panda_end_effector_position(q, world_from_base)

    assert positions.shape == (2, 3, 3)
    expected = torch.tensor([-0.527, 0.2, 0.9226], dtype=torch.float64)
    torch.testing.assert_close(positions, expected.expand_as(positions))


def test_panda_fk_gradient_matches_finite_difference() -> None:
    q = torch.tensor(
        [0.1, 0.3, -0.2, -1.5, 0.2, 1.7, 0.4],
        dtype=torch.float64,
        requires_grad=True,
    )
    position = panda_end_effector_position(q)
    analytic = torch.autograd.functional.jacobian(panda_end_effector_position, q)

    epsilon = 1e-6
    columns = []
    for joint_index in range(7):
        offset = torch.zeros_like(q)
        offset[joint_index] = epsilon
        plus = panda_end_effector_position((q + offset).detach())
        minus = panda_end_effector_position((q - offset).detach())
        columns.append((plus - minus) / (2.0 * epsilon))
    numeric = torch.stack(columns, dim=1)

    assert position.requires_grad
    torch.testing.assert_close(analytic, numeric, atol=1e-6, rtol=1e-5)


def test_panda_fk_validates_shapes() -> None:
    with pytest.raises(ValueError, match="q must have shape"):
        panda_end_effector_position(torch.zeros(6))
    with pytest.raises(ValueError, match="world_from_base"):
        panda_end_effector_position(torch.zeros(7), torch.eye(3))
    with pytest.raises(ValueError, match="not broadcastable"):
        panda_end_effector_position(torch.zeros((2, 7)), torch.eye(4).repeat(3, 1, 1))
