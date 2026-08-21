from __future__ import annotations

import numpy as np
import pytest

from pg3d.composition import (
    AdaptiveWeightState,
    ConvexScoreWeights,
    FeasibleMassPosterior,
    ScoreIntervalNode,
    active_beam_width,
    add_mass_weight,
    allocate_uncertain_boundary_node,
    feasible_mass_risk,
    route_descriptor,
    route_distance,
)


def test_beta_posterior_and_conservative_mass_risk() -> None:
    prior = FeasibleMassPosterior()
    viable = prior.update(True)
    failed = prior.update(False)

    assert viable.alpha == 2.0 and viable.beta == 1.0
    assert viable.mean == pytest.approx(2.0 / 3.0)
    assert viable.quantile(0.1) > prior.quantile(0.1)
    assert feasible_mass_risk(viable.quantile(0.1)) < feasible_mass_risk(
        failed.quantile(0.1)
    )


def test_equal_fraction_with_fewer_probes_has_lower_conservative_mass() -> None:
    few = FeasibleMassPosterior(viable=1, probes=2)
    many = FeasibleMassPosterior(viable=10, probes=20)

    assert few.mean == many.mean
    assert few.quantile(0.1) < many.quantile(0.1)


def test_mass_weight_proportionally_rescales_task_weights() -> None:
    result = add_mass_weight(
        ConvexScoreWeights(goal=0.5, clearance=0.4, smoothness=0.1),
        0.25,
    )

    assert result.as_tuple() == pytest.approx((0.375, 0.3, 0.075, 0.25))


def test_adaptive_update_is_projected_and_forces_fallback_risk() -> None:
    state = AdaptiveWeightState.from_weights(
        ConvexScoreWeights(goal=0.5, clearance=0.3, smoothness=0.1, mass=0.1),
        rho=0.5,
        temperature=1.0,
    )
    update = state.update([0.0, 0.0, 0.0, 0.0], infeasible_fallback=True)
    weights = state.weights()

    assert update["errors"] == [-1.0, 1.0, -1.0, 1.0]
    assert all(0.0 <= value <= 4.0 for value in state.logits)
    assert min(weights.as_tuple()) >= 0.02
    assert sum(weights.as_tuple()) == pytest.approx(1.0)


def test_uncertainty_allocation_uses_widest_boundary_interval_then_ancestry() -> None:
    nodes = [
        ScoreIntervalNode("b", 0.1, 0.5),
        ScoreIntervalNode("a", 0.1, 0.5),
        ScoreIntervalNode("c", 0.7, 0.8),
    ]

    selected = allocate_uncertain_boundary_node(nodes, retention_width=1)

    assert selected is not None and selected.ancestry == "a"


def test_dynamic_width_contracts_only_for_decisive_pessimistic_winner() -> None:
    decisive = [ScoreIntervalNode("a", 0.1, 0.2), ScoreIntervalNode("b", 0.3, 0.4)]
    overlap = [ScoreIntervalNode("a", 0.1, 0.3), ScoreIntervalNode("b", 0.2, 0.4)]

    assert active_beam_width(decisive) == 1
    assert active_beam_width(overlap) == 2


def test_route_descriptor_is_arc_length_resampled_and_geometry_agnostic() -> None:
    first = route_descriptor(np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.float32))
    second = route_descriptor(np.asarray([[0, 0, 0], [0, 1, 0]], dtype=np.float32))

    assert first.shape == (16, 3)
    assert route_distance(first, first) == 0.0
    assert route_distance(first, second) > 0.05
