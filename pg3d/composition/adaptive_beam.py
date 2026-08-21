from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import beta as beta_distribution

from pg3d.composition.guided_scoring import ConvexScoreWeights


@dataclass(frozen=True)
class FeasibleMassPosterior:
    viable: int = 0
    probes: int = 0
    prior_alpha: float = 1.0
    prior_beta: float = 1.0

    def __post_init__(self) -> None:
        if self.probes < 0 or self.viable < 0 or self.viable > self.probes:
            raise ValueError("mass counts must satisfy 0 <= viable <= probes")
        if self.prior_alpha <= 0.0 or self.prior_beta <= 0.0:
            raise ValueError("beta prior parameters must be positive")

    @property
    def alpha(self) -> float:
        return self.prior_alpha + self.viable

    @property
    def beta(self) -> float:
        return self.prior_beta + self.probes - self.viable

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def quantile(self, probability: float) -> float:
        if not 0.0 < probability < 1.0:
            raise ValueError("beta quantile probability must be in (0, 1)")
        return float(beta_distribution.ppf(probability, self.alpha, self.beta))

    def update(self, viable: bool) -> FeasibleMassPosterior:
        return FeasibleMassPosterior(
            viable=self.viable + int(viable),
            probes=self.probes + 1,
            prior_alpha=self.prior_alpha,
            prior_beta=self.prior_beta,
        )

    def to_json(self) -> dict[str, float | int]:
        return {
            "viable": int(self.viable),
            "probes": int(self.probes),
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "mean": float(self.mean),
            "lower_0.10": self.quantile(0.10),
            "upper_0.90": self.quantile(0.90),
        }


def feasible_mass_risk(probability: float, *, epsilon: float = 1e-3) -> float:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("feasible mass probability must be in [0, 1]")
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be in (0, 1)")
    return float(np.clip(-np.log(epsilon + probability) / -np.log(epsilon), 0.0, 1.0))


def add_mass_weight(
    task_weights: ConvexScoreWeights,
    mass_weight: float,
) -> ConvexScoreWeights:
    if task_weights.mass != 0.0:
        raise ValueError("task_weights must have zero mass before proportional rescaling")
    if not 0.0 <= mass_weight < 1.0:
        raise ValueError("mass_weight must be in [0, 1)")
    scale = 1.0 - mass_weight
    return ConvexScoreWeights(
        goal=task_weights.goal * scale,
        clearance=task_weights.clearance * scale,
        smoothness=task_weights.smoothness * scale,
        mass=mass_weight,
    )


@dataclass
class AdaptiveWeightState:
    logits: np.ndarray
    rho: float
    temperature: float
    weight_floor: float = 0.02
    logit_min: float = 0.0
    logit_max: float = 4.0

    @classmethod
    def from_weights(
        cls,
        weights: ConvexScoreWeights,
        *,
        rho: float,
        temperature: float,
        weight_floor: float = 0.02,
    ) -> AdaptiveWeightState:
        if rho <= 0.0 or temperature <= 0.0:
            raise ValueError("adaptive rate and temperature must be positive")
        initial = np.maximum(np.asarray(weights.as_tuple(), dtype=np.float64), weight_floor)
        initial /= np.sum(initial)
        logits = temperature * np.log(initial)
        logits -= np.min(logits)
        return cls(
            logits=np.clip(logits, 0.0, 4.0),
            rho=float(rho),
            temperature=float(temperature),
            weight_floor=float(weight_floor),
        )

    def weights(self) -> ConvexScoreWeights:
        shifted = self.logits / self.temperature
        exponent = np.exp(shifted - np.max(shifted))
        softmax = exponent / np.sum(exponent)
        weights = self.weight_floor + (1.0 - 4.0 * self.weight_floor) * softmax
        return ConvexScoreWeights(*map(float, weights))

    def update(
        self,
        normalized_costs: Sequence[float],
        *,
        infeasible_fallback: bool,
    ) -> dict[str, object]:
        costs = np.asarray(normalized_costs, dtype=np.float64)
        if costs.shape != (4,) or not np.isfinite(costs).all():
            raise ValueError("adaptive update requires four finite normalized costs")
        errors = 2.0 * np.clip(costs, 0.0, 1.0) - 1.0
        if infeasible_fallback:
            errors[1] = 1.0
            errors[3] = 1.0
        before = self.logits.copy()
        self.logits = np.clip(
            self.logits + self.rho * errors,
            self.logit_min,
            self.logit_max,
        )
        return {
            "logits_before": before.tolist(),
            "errors": errors.tolist(),
            "logits_after": self.logits.tolist(),
            "weights_after": self.weights().to_json(),
        }


@dataclass(frozen=True)
class ScoreIntervalNode:
    ancestry: str
    optimistic_score: float
    pessimistic_score: float

    @property
    def width(self) -> float:
        return self.pessimistic_score - self.optimistic_score


def allocate_uncertain_boundary_node(
    nodes: Iterable[ScoreIntervalNode],
    *,
    retention_width: int,
) -> ScoreIntervalNode | None:
    ordered = sorted(nodes, key=lambda node: (node.optimistic_score, node.ancestry))
    if not ordered or retention_width <= 0:
        return None
    boundary_score = ordered[min(retention_width, len(ordered)) - 1].pessimistic_score
    eligible = [
        node
        for node in ordered
        if node.optimistic_score <= boundary_score and node.pessimistic_score >= boundary_score
    ]
    return (
        max(eligible, key=lambda node: (node.width, _reverse_lex_key(node.ancestry)))
        if eligible
        else None
    )


def active_beam_width(
    nodes: Iterable[ScoreIntervalNode],
    *,
    margin: float = 0.05,
    minimum: int = 1,
    maximum: int = 2,
) -> int:
    ordered = sorted(nodes, key=lambda node: (node.pessimistic_score, node.ancestry))
    if len(ordered) <= minimum:
        return minimum
    best = ordered[0]
    decisive = all(
        best.pessimistic_score + margin < node.optimistic_score for node in ordered[1:]
    )
    return minimum if decisive else min(maximum, len(ordered))


def route_descriptor(path: np.ndarray, *, points: int = 16) -> np.ndarray:
    values = np.asarray(path, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not len(values):
        raise ValueError("route path must have non-empty shape [T, 3]")
    if points <= 1:
        raise ValueError("route descriptor requires at least two points")
    if len(values) == 1:
        return np.repeat(values, points, axis=0)
    lengths = np.linalg.norm(np.diff(values, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    if cumulative[-1] <= 1e-12:
        return np.repeat(values[:1], points, axis=0)
    samples = np.linspace(0.0, cumulative[-1], points)
    return np.stack(
        [np.interp(samples, cumulative, values[:, axis]) for axis in range(3)], axis=1
    )


def route_distance(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape:
        raise ValueError("route descriptors must have equal shape")
    return float(np.mean(np.linalg.norm(first - second, axis=1)))


def _reverse_lex_key(value: str) -> tuple[int, ...]:
    return tuple(-ord(character) for character in value)
