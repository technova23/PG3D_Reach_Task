from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from pg3d.constraints.geometry import BoxRegion, RectRegion2D, SphereRegion
from pg3d.constraints.programs import AvoidProjection, AvoidRegion
from pg3d.constraints.torch_geometry import avoidance_energy
from pg3d.policies.dp3.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from pg3d.world_model.panda_fk import panda_end_effector_position
from scripts.eval_constrained_reach import _itps_eef_path
from scripts.eval_constrained_reach import parse_args as parse_eval_args


def test_hinge_energy_matches_weighted_margin_for_every_region_shape() -> None:
    path = torch.tensor([[[0.5, 0.0, 0.0]]], dtype=torch.float64)
    constraints = [
        AvoidRegion(
            SphereRegion(center=[0.0, 0.0, 0.0], radius=1.0),
            margin=0.2,
            weight=2.0,
        ),
        AvoidRegion(
            BoxRegion(center=[0.0, 0.0, 0.0], half_extents=[1.0, 1.0, 1.0]),
        ),
        AvoidProjection(
            RectRegion2D(center=[0.0, 0.0], half_extents=[1.0, 1.0]),
        ),
    ]

    energy = avoidance_energy(path, constraints, mode="hinge")

    # Sphere: 2 * (0.2 - -0.5) = 1.4; box and rectangle contribute 0.5 each.
    torch.testing.assert_close(energy, torch.tensor([2.4], dtype=torch.float64))


def test_smooth_energy_guides_before_penetration_and_backpropagates() -> None:
    path = torch.tensor([[[1.2, 0.1, 0.0], [1.4, 0.2, 0.0]]], requires_grad=True)
    constraint = AvoidRegion(SphereRegion(center=[0.0, 0.0, 0.0], radius=1.0))

    hinge = avoidance_energy(path, [constraint], mode="hinge")
    smooth = avoidance_energy(path, [constraint], mode="smooth", temperature=0.01)
    smooth.sum().backward()

    torch.testing.assert_close(hinge, torch.zeros_like(hinge))
    assert smooth.item() > 0.0
    assert path.grad is not None
    assert torch.count_nonzero(path.grad).item() > 0


def test_avoidance_energy_validates_inputs_and_target() -> None:
    path = torch.zeros((1, 2, 3))
    robot_constraint = AvoidRegion(
        SphereRegion(center=[0.0, 0.0, 0.0], radius=1.0),
        target="robot",
    )

    with pytest.raises(ValueError, match="target='eef'"):
        avoidance_energy(path, [robot_constraint])
    with pytest.raises(ValueError, match="temperature"):
        avoidance_energy(path, [], temperature=0.0)
    with pytest.raises(ValueError, match="shape"):
        avoidance_energy(torch.zeros((2, 3)), [])


def test_itps_eef_path_unnormalizes_actions_with_gradients() -> None:
    rest_q = torch.tensor(
        [0.0, 0.3926991, 0.0, -1.9634954, 0.0, 2.3561945, 0.7853982]
    )
    normalizer = LinearNormalizer(
        {
            "action": SingleFieldLinearNormalizer.create_manual(
                scale=torch.full((7,), 2.0),
                offset=-2.0 * rest_q,
            )
        }
    )
    policy = SimpleNamespace(action_dim=7, normalizer=normalizer)
    normalized = torch.zeros((1, 2, 7), requires_grad=True)
    world_from_base = torch.eye(4)
    world_from_base[0, 3] = -0.615

    path = _itps_eef_path(policy, normalized, world_from_base)  # type: ignore[arg-type]
    expected = panda_end_effector_position(rest_q, world_from_base)
    path.sum().backward()

    torch.testing.assert_close(path, expected.view(1, 1, 3).expand_as(path))
    assert normalized.grad is not None
    assert torch.count_nonzero(normalized.grad).item() > 0


def test_itps_cli_defaults_and_validation() -> None:
    args = parse_eval_args(_base_args())
    assert args.itps_guide_ratio == 60.0
    assert args.itps_mcmc_steps == 4
    assert args.itps_energy == "smooth"
    assert args.itps_barrier_temperature == 0.01

    selected = parse_eval_args(
        [*_base_args(), "--itps-energy", "hinge", "--itps-mcmc-steps", "2"]
    )
    assert selected.itps_energy == "hinge"
    assert selected.itps_mcmc_steps == 2

    with pytest.raises(ValueError, match="itps-guide-ratio"):
        parse_eval_args([*_base_args(), "--itps-guide-ratio", "-1"])
    with pytest.raises(ValueError, match="constraint-target eef"):
        parse_eval_args(
            [*_base_args(), "--methods", "itps", "--constraint-target", "robot"]
        )


def _base_args() -> list[str]:
    return [
        "--checkpoint",
        "checkpoint.pt",
        "--dataset",
        "dataset.zarr",
        "--output-dir",
        "/tmp/pg3d-itps-test-output",
    ]
