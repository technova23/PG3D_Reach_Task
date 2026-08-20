from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pg3d.constraints.geometry import (
    BoxRegion,
    CylinderRegion,
    RectRegion2D,
    SphereRegion,
)
from pg3d.constraints.programs import AvoidProjection, AvoidRegion
from pg3d.constraints.torch_geometry import avoidance_energy
from pg3d.eval import TimingRecorder
from pg3d.policies.dp3.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from pg3d.world_model import ActionChunk
from pg3d.world_model.panda_collision import (
    DifferentiablePandaCollisionPoints,
    PandaCollisionPointTemplate,
)
from pg3d.world_model.panda_fk import panda_end_effector_position
from scripts.eval_constrained_reach import (
    ComputeOperationCounts,
    ITPSGuidanceConfig,
    _itps_eef_path,
    _itps_robot_points,
    _itps_robot_replan_diagnostics,
    _select_itps_chunk,
)
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
                margin=0.0,
            ),
            AvoidProjection(
                RectRegion2D(center=[0.0, 0.0], half_extents=[1.0, 1.0]),
                margin=0.0,
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


def test_rotated_box_torch_energy_matches_numpy_geometry() -> None:
    region = BoxRegion(
        center=[0.0, 0.0, 0.0],
        half_extents=[1.0, 0.25, 0.5],
        yaw=torch.pi / 2,
    )
    points = torch.tensor([[[0.0, 0.75, 0.0], [0.75, 0.0, 0.0]]], dtype=torch.float64)

    energy = avoidance_energy(points, [AvoidRegion(region, margin=0.0)], mode="hinge")

    # First point is 0.25 m inside; second is outside.
    torch.testing.assert_close(energy, torch.tensor([0.25], dtype=torch.float64))


def test_cylinder_torch_energy_matches_numpy_geometry() -> None:
    region = CylinderRegion(center=[0.0, 0.0, 0.5], radius=0.2, half_length=0.4)
    points = torch.tensor([[[0.0, 0.0, 0.5], [0.3, 0.0, 0.5]]], dtype=torch.float64)

    energy = avoidance_energy(points, [AvoidRegion(region, margin=0.0)], mode="hinge")

    torch.testing.assert_close(energy, torch.tensor([0.2], dtype=torch.float64))


def test_avoidance_energy_validates_inputs_and_target() -> None:
    path = torch.zeros((1, 2, 3))
    robot_constraint = AvoidRegion(
        SphereRegion(center=[0.0, 0.0, 0.0], radius=1.0),
        target="robot",
    )

    with pytest.raises(ValueError, match="does not match"):
        avoidance_energy(path, [robot_constraint])
    with pytest.raises(ValueError, match="does not match"):
        avoidance_energy(
            path[:, :, None, :],
            [AvoidRegion(SphereRegion(center=[0.0, 0.0, 0.0], radius=1.0))],
            target="robot",
        )
    with pytest.raises(ValueError, match="temperature"):
        avoidance_energy(path, [], temperature=0.0)
    with pytest.raises(ValueError, match="shape"):
        avoidance_energy(torch.zeros((2, 3)), [])
    with pytest.raises(ValueError, match="shape"):
        avoidance_energy(path, [], target="robot")


def test_robot_energy_takes_exact_worst_point_across_horizon_and_points() -> None:
    points = torch.tensor(
        [
            [
                [[1.4, 0.0, 0.0], [0.8, 0.0, 0.0]],
                [[0.3, 0.0, 0.0], [1.2, 0.0, 0.0]],
            ]
        ],
        dtype=torch.float64,
    )
    constraint = AvoidRegion(
        SphereRegion(center=[0.0, 0.0, 0.0], radius=1.0),
        target="robot",
        margin=0.1,
        weight=2.0,
    )

    energy = avoidance_energy(points, [constraint], target="robot", mode="hinge")

    # The unique worst point is 0.7 m inside the sphere plus the 0.1 m margin.
    torch.testing.assert_close(energy, torch.tensor([1.6], dtype=torch.float64))


@pytest.mark.parametrize(
    "constraint",
    [
        AvoidRegion(
            SphereRegion(center=[0.0, 0.0, 0.0], radius=1.0),
            target="robot",
        ),
        AvoidRegion(
            BoxRegion(
                center=[0.0, 0.0, 0.0],
                half_extents=[1.0, 0.25, 0.5],
                yaw=torch.pi / 4,
            ),
            target="robot",
        ),
        AvoidRegion(
            CylinderRegion(center=[0.0, 0.0, 0.0], radius=0.5, half_length=0.5),
            target="robot",
        ),
        AvoidProjection(
            RectRegion2D(center=[0.0, 0.0], half_extents=[0.5, 0.5]),
            target="robot",
        ),
    ],
)
def test_robot_energy_supports_every_region_shape(constraint: AvoidRegion) -> None:
    points = torch.tensor([[[[0.1, 0.1, 0.1], [2.0, 2.0, 2.0]]]], dtype=torch.float64)

    energy = avoidance_energy(points, [constraint], target="robot", mode="hinge")

    assert energy.shape == (1,)
    assert energy.item() > 0.0


def test_robot_energy_gradient_matches_finite_difference() -> None:
    points = torch.tensor(
        [[[[0.25, 0.1, 0.05], [1.5, 0.0, 0.0]]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    constraint = AvoidRegion(
        SphereRegion(center=[0.0, 0.0, 0.0], radius=1.0),
        target="robot",
    )

    def energy_fn(value: torch.Tensor) -> torch.Tensor:
        return avoidance_energy(value, [constraint], target="robot", mode="hinge").sum()

    analytic = torch.autograd.grad(energy_fn(points), points)[0]
    epsilon = 1e-6
    numeric = torch.zeros_like(points)
    for coordinate in range(3):
        offset = torch.zeros_like(points)
        offset[0, 0, 0, coordinate] = epsilon
        numeric[0, 0, 0, coordinate] = (
            energy_fn((points + offset).detach()) - energy_fn((points - offset).detach())
        ) / (2.0 * epsilon)

    torch.testing.assert_close(analytic, numeric, atol=1e-6, rtol=1e-5)


def test_itps_eef_path_unnormalizes_actions_with_gradients() -> None:
    rest_q = torch.tensor([0.0, 0.3926991, 0.0, -1.9634954, 0.0, 2.3561945, 0.7853982])
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


def test_itps_robot_points_unnormalize_actions_with_gradients() -> None:
    normalizer = LinearNormalizer(
        {
            "action": SingleFieldLinearNormalizer.create_manual(
                scale=torch.ones(7),
                offset=torch.zeros(7),
            )
        }
    )
    policy = SimpleNamespace(action_dim=7, normalizer=normalizer)
    template = PandaCollisionPointTemplate(
        local_points=np.zeros((10, 3), dtype=np.float32),
        link_indices=np.arange(10, dtype=np.int64),
        link_counts=(1,) * 10,
        sample_seed=0,
    )
    collision_model = DifferentiablePandaCollisionPoints(template)
    normalized = torch.tensor(
        [[[0.1, 0.3, -0.2, -1.5, 0.2, 1.7, 0.4]]],
        requires_grad=True,
    )

    points = _itps_robot_points(  # type: ignore[arg-type]
        policy,
        normalized,
        torch.eye(4),
        collision_model,
    )
    points[..., 0].sum().backward()

    assert points.shape == (1, 1, 10, 3)
    assert normalized.grad is not None
    assert torch.count_nonzero(normalized.grad).item() > 0


def test_itps_chunk_uses_whole_body_guidance_with_mocked_policy() -> None:
    normalizer = LinearNormalizer(
        {
            "action": SingleFieldLinearNormalizer.create_manual(
                scale=torch.ones(7),
                offset=torch.zeros(7),
            )
        }
    )

    class MockPolicy:
        device = torch.device("cpu")
        dtype = torch.float32
        action_dim = 7
        goal_marker_points = 0
        goal_marker_radius = 0.015

        def __init__(self) -> None:
            self.normalizer = normalizer
            self.guidance_gradient: torch.Tensor | None = None

        def predict_action_itps(self, _obs_batch, **kwargs):
            trajectory = torch.tensor(
                [[[0.1, 0.3, -0.2, -1.5, 0.2, 1.7, 0.4]]],
                requires_grad=True,
            )
            energy = kwargs["guidance_fn"](trajectory)
            self.guidance_gradient = torch.autograd.grad(energy.sum(), trajectory)[0]
            return {"action": trajectory.detach(), "action_pred": trajectory.detach()}

    class MockProvider:
        @staticmethod
        def world_from_robot_base() -> np.ndarray:
            return np.eye(4, dtype=np.float32)

    local_points = np.stack(
        [np.asarray([0.01 * (index + 1), 0.02, 0.03], dtype=np.float32) for index in range(10)]
    )
    collision_model = DifferentiablePandaCollisionPoints(
        PandaCollisionPointTemplate(
            local_points=local_points,
            link_indices=np.arange(10, dtype=np.int64),
            link_counts=(1,) * 10,
            sample_seed=0,
        )
    )
    policy = MockPolicy()
    entry = {
        "point_cloud": np.zeros((4, 3), dtype=np.float32),
        "agent_pos": np.zeros(9, dtype=np.float32),
        "target_position": np.asarray([0.5, 0.0, 0.4], dtype=np.float32),
        "tcp_pose": np.asarray([0.0, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    }
    constraint = AvoidRegion(
        SphereRegion(center=[0.4, 0.1, 0.4], radius=0.3),
        target="robot",
    )

    compute_counts = ComputeOperationCounts()
    chunk = _select_itps_chunk(
        policy=policy,  # type: ignore[arg-type]
        provider=MockProvider(),  # type: ignore[arg-type]
        obs_window=[entry],  # type: ignore[list-item]
        constraints=[constraint],
        rng=np.random.default_rng(0),
        config=ITPSGuidanceConfig(energy="smooth"),
        compute_counts=compute_counts,
        collision_model=collision_model,
        constraint_target="robot",
    )

    assert chunk.actions.shape == (1, 7)
    assert chunk.metadata["guidance_target"] == "robot"
    assert chunk.metadata["collision_geometry_source"] == "maniskill_panda_urdf"
    assert chunk.metadata["excluded_collision_links"] == ["panda_link0"]
    assert sum(chunk.metadata["collision_link_allocation"].values()) == 10
    assert compute_counts.differentiable_robot_point_calls == 1
    assert compute_counts.differentiable_robot_point_evaluations == 10
    assert policy.guidance_gradient is not None
    assert torch.count_nonzero(policy.guidance_gradient).item() > 0


def test_itps_robot_replan_diagnostics_capture_cloud_and_worst_point() -> None:
    local_points = np.stack(
        [np.asarray([0.01 * index, 0.02, 0.03], dtype=np.float32) for index in range(10)]
    )
    collision_model = DifferentiablePandaCollisionPoints(
        PandaCollisionPointTemplate(
            local_points=local_points,
            link_indices=np.arange(10, dtype=np.int64),
            link_counts=(1,) * 10,
            sample_seed=0,
        )
    )

    class MockProvider:
        @staticmethod
        def world_from_robot_base() -> np.ndarray:
            return np.eye(4, dtype=np.float32)

    diagnostics = _itps_robot_replan_diagnostics(
        ActionChunk(
            actions=np.zeros((2, 7), dtype=np.float32),
            action_mode="abs_joint",
            dt=1.0,
        ),
        provider=MockProvider(),  # type: ignore[arg-type]
        collision_model=collision_model,
        constraints=[
            AvoidRegion(
                SphereRegion(center=[0.1, 0.0, 0.4], radius=0.2),
                target="robot",
            )
        ],
        timer=TimingRecorder(enabled=False),
    )

    assert diagnostics["itps_robot_points"].shape == (2, 10, 3)
    np.testing.assert_array_equal(diagnostics["itps_robot_link_indices"], np.arange(10))
    worst = diagnostics["itps_worst_points"][0]
    assert 0 <= worst["horizon_index"] < 2
    assert 0 <= worst["point_index"] < 10
    assert np.asarray(worst["position"]).shape == (3,)


def test_itps_cli_defaults_and_validation() -> None:
    args = parse_eval_args(_base_args())
    assert args.ddim_eta == 0.0
    assert args.itps_guide_ratio == 60.0
    assert args.itps_mcmc_steps == 4
    assert args.itps_energy == "smooth"
    assert args.itps_barrier_temperature == 0.01
    assert args.itps_robot_points == 1024
    assert args.itps_robot_sample_seed == 0

    selected = parse_eval_args([*_base_args(), "--itps-energy", "hinge", "--itps-mcmc-steps", "2"])
    assert selected.itps_energy == "hinge"
    assert selected.itps_mcmc_steps == 2

    with pytest.raises(ValueError, match="itps-guide-ratio"):
        parse_eval_args([*_base_args(), "--itps-guide-ratio", "-1"])
    selected = parse_eval_args([*_base_args(), "--ddim-eta", "1"])
    assert selected.ddim_eta == 1.0
    with pytest.raises(ValueError, match="ddim-eta"):
        parse_eval_args([*_base_args(), "--ddim-eta", "1.1"])
    robot = parse_eval_args([*_base_args(), "--methods", "itps", "--constraint-target", "robot"])
    assert robot.constraint_target == "robot"
    assert robot.geometry_mode == "fast"
    with pytest.raises(ValueError, match="at least 320"):
        parse_eval_args([*_base_args(), "--itps-robot-points", "319"])
    with pytest.raises(ValueError, match="sample-seed"):
        parse_eval_args([*_base_args(), "--itps-robot-sample-seed", "-1"])
    with pytest.raises(ValueError, match="gripper-open"):
        parse_eval_args(
            [
                *_base_args(),
                "--methods",
                "itps",
                "--constraint-target",
                "robot",
                "--gripper-open",
                "0.05",
            ]
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
