from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

GuidedScoreMode = Literal[
    "avoidance_only",
    "fixed_task",
    "mass_mean",
    "mass_lcb",
    "adaptive_mass",
]


@dataclass(frozen=True)
class ConvexScoreWeights:
    """Convex weights used only to rank exactly feasible prefixes."""

    goal: float = 0.5
    clearance: float = 0.4
    smoothness: float = 0.1
    mass: float = 0.0

    def __post_init__(self) -> None:
        values = np.asarray(self.as_tuple(), dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError("score weights must be finite and non-negative")
        if not np.isclose(float(np.sum(values)), 1.0, atol=1e-8):
            raise ValueError("score weights must sum to one")

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.goal, self.clearance, self.smoothness, self.mass)

    def to_json(self) -> dict[str, float]:
        return {
            "goal": float(self.goal),
            "clearance": float(self.clearance),
            "smoothness": float(self.smoothness),
            "mass": float(self.mass),
        }


@dataclass(frozen=True)
class NormalizedScoreTerms:
    goal: float
    clearance: float
    smoothness: float
    mass: float = 0.0

    def __post_init__(self) -> None:
        values = np.asarray(self.as_tuple(), dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError("normalized score terms must be finite and in [0, 1]")

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.goal, self.clearance, self.smoothness, self.mass)

    def to_json(self) -> dict[str, float]:
        return {
            "goal": float(self.goal),
            "clearance": float(self.clearance),
            "smoothness": float(self.smoothness),
            "mass": float(self.mass),
        }


@dataclass(frozen=True)
class GuidedScoreConfig:
    """Physical normalizers and mode for feasible-prefix scoring."""

    mode: GuidedScoreMode = "avoidance_only"
    goal_reference_m: float = 0.75
    hard_clearance_m: float = 0.03
    clearance_buffer_m: float = 0.08
    smoothness_reference_rad2: float = 0.001
    verification_buffer_m: float = 0.0
    weights: ConvexScoreWeights = ConvexScoreWeights()

    def __post_init__(self) -> None:
        if self.goal_reference_m <= 0.0:
            raise ValueError("goal_reference_m must be positive")
        if self.hard_clearance_m < 0.0:
            raise ValueError("hard_clearance_m must be non-negative")
        if self.clearance_buffer_m <= self.hard_clearance_m + self.verification_buffer_m:
            raise ValueError("clearance_buffer_m must exceed effective hard clearance")
        if self.smoothness_reference_rad2 <= 0.0:
            raise ValueError("smoothness_reference_rad2 must be positive")
        if self.verification_buffer_m < 0.0:
            raise ValueError("verification_buffer_m must be non-negative")
        if self.mode == "fixed_task" and self.weights.mass != 0.0:
            raise ValueError("fixed_task scoring requires zero mass weight")

    @property
    def effective_hard_clearance_m(self) -> float:
        return self.hard_clearance_m + self.verification_buffer_m

    def terms(
        self,
        *,
        goal_distance_m: float | None,
        min_clearance_m: float | None,
        smoothness_rad2: float,
        mass_risk: float = 0.0,
    ) -> NormalizedScoreTerms:
        goal = 0.0 if goal_distance_m is None else goal_distance_m / self.goal_reference_m
        clearance = (
            1.0
            if min_clearance_m is None
            else (self.clearance_buffer_m - min_clearance_m)
            / (self.clearance_buffer_m - self.effective_hard_clearance_m)
        )
        return NormalizedScoreTerms(
            goal=_clip01(goal),
            clearance=_clip01(clearance),
            smoothness=_clip01(smoothness_rad2 / self.smoothness_reference_rad2),
            mass=_clip01(mass_risk),
        )

    def feasible_score(
        self,
        terms: NormalizedScoreTerms,
        *,
        avoidance_penalty: float,
        weights: ConvexScoreWeights | None = None,
    ) -> float:
        if self.mode == "avoidance_only":
            return float(avoidance_penalty)
        active = weights or self.weights
        return float(np.dot(active.as_tuple(), terms.as_tuple()))

    def to_json(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "goal_reference_m": float(self.goal_reference_m),
            "hard_clearance_m": float(self.hard_clearance_m),
            "clearance_buffer_m": float(self.clearance_buffer_m),
            "smoothness_reference_rad2": float(self.smoothness_reference_rad2),
            "verification_buffer_m": float(self.verification_buffer_m),
            "effective_hard_clearance_m": float(self.effective_hard_clearance_m),
            "weights": self.weights.to_json(),
        }

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        mode: GuidedScoreMode | None = None,
        weights: ConvexScoreWeights | None = None,
        verification_buffer_m: float | None = None,
    ) -> GuidedScoreConfig:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw = dict(payload.get("scoring", payload))
        raw_weights = dict(raw.get("default_fixed_weights", raw.get("weights", {})))
        configured_weights = weights or ConvexScoreWeights(
            goal=float(raw_weights.get("goal", 0.5)),
            clearance=float(raw_weights.get("clearance", 0.4)),
            smoothness=float(raw_weights.get("smoothness", 0.1)),
            mass=float(raw_weights.get("mass", 0.0)),
        )
        return cls(
            mode=mode or raw.get("mode", "fixed_task"),
            goal_reference_m=float(raw.get("goal_reference_m", 0.75)),
            hard_clearance_m=float(raw.get("hard_clearance_m", 0.03)),
            clearance_buffer_m=float(raw.get("clearance_buffer_m", 0.08)),
            smoothness_reference_rad2=float(raw.get("smoothness_reference_rad2", 0.001)),
            verification_buffer_m=(
                float(raw.get("verification_buffer_m", 0.0))
                if verification_buffer_m is None
                else float(verification_buffer_m)
            ),
            weights=configured_weights,
        )


def simplex_weights(step: float = 0.25) -> list[ConvexScoreWeights]:
    """Return a deterministic goal/clearance/smoothness simplex with zero mass."""
    if step <= 0.0 or step > 1.0 or not np.isclose(round(1.0 / step) * step, 1.0):
        raise ValueError("step must divide one exactly")
    units = int(round(1.0 / step))
    result = []
    for goal_units in range(units + 1):
        for clearance_units in range(units - goal_units + 1):
            smoothness_units = units - goal_units - clearance_units
            result.append(
                ConvexScoreWeights(
                    goal=goal_units / units,
                    clearance=clearance_units / units,
                    smoothness=smoothness_units / units,
                    mass=0.0,
                )
            )
    return result


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))
