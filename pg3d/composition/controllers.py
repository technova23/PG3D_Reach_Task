from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from pg3d.composition.scoring import (
    consensus_deviations,
    directional_preference,
    goal_distance,
    optional_policy_surrogate,
    primary_constraint_penalty,
    trajectory_smoothness,
)
from pg3d.composition.types import (
    CandidateDiagnostics,
    ControllerInput,
    ControllerResult,
    Policy,
    ScoreWeights,
    WorldModel,
)
from pg3d.constraints import Constraint
from pg3d.world_model import ActionChunk, ImaginedRollout


class BaseController:
    """Shared sampling, imagination, and scoring logic for composition controllers."""

    def __init__(
        self,
        *,
        policy: Policy,
        world_model: WorldModel,
        constraints: Iterable[Constraint] = (),
        k_schedule: tuple[int, ...] = (16, 32, 64),
        score_weights: ScoreWeights | None = None,
        smoothness_order: int = 2,
        directional_sign: int = 0,
    ) -> None:
        self.policy = policy
        self.world_model = world_model
        self.constraints = list(constraints)
        self.k_schedule = _validate_k_schedule(k_schedule)
        self.score_weights = score_weights or ScoreWeights()
        if smoothness_order not in {1, 2}:
            raise ValueError("smoothness_order must be 1 or 2")
        self.smoothness_order = smoothness_order
        if directional_sign not in {-1, 0, 1}:
            raise ValueError("directional_sign must be -1, 0, or 1")
        self.directional_sign = directional_sign

    def select(
        self,
        controller_input: ControllerInput,
        *,
        rng: np.random.Generator | None = None,
    ) -> ControllerResult:
        """Select one candidate action chunk."""
        raise NotImplementedError

    def _sample_and_score(
        self,
        controller_input: ControllerInput,
        *,
        attempted_k: int,
        start_index: int,
        rng: np.random.Generator | None,
    ) -> list[CandidateDiagnostics]:
        policy_input = controller_input.input_for_policy()
        chunks = self.policy.sample_action_chunks(policy_input, k=attempted_k, rng=rng)
        if not chunks:
            return []
        surrogates = optional_policy_surrogate(self.policy, policy_input, chunks)
        consensus = consensus_deviations(chunks)
        diagnostics: list[CandidateDiagnostics] = []
        for local_idx, chunk in enumerate(chunks):
            rollout = self.world_model.imagine(controller_input.observation, chunk)
            diagnostics.append(
                self._score_candidate(
                    controller_input,
                    action_chunk=chunk,
                    rollout=rollout,
                    attempted_k=attempted_k,
                    index=start_index + local_idx,
                    consensus_deviation=consensus[local_idx],
                    policy_surrogate=surrogates[local_idx],
                )
            )
        return diagnostics

    def _score_candidate(
        self,
        controller_input: ControllerInput,
        *,
        action_chunk: ActionChunk,
        rollout: ImaginedRollout,
        attempted_k: int,
        index: int,
        consensus_deviation: float,
        policy_surrogate: float | None,
    ) -> CandidateDiagnostics:
        constraint_costs: dict[str, float] = {}
        constraint_satisfied: dict[str, bool] = {}
        for constraint_idx, constraint in enumerate(self.constraints):
            label = _constraint_label(constraint, constraint_idx)
            costs = constraint.cost(rollout, controller_input.scene)
            for key, value in costs.items():
                constraint_costs[_unique_cost_key(constraint_costs, key)] = float(value)
            constraint_satisfied[label] = bool(
                constraint.satisfied(rollout, controller_input.scene)
            )

        feasible = all(constraint_satisfied.values()) if constraint_satisfied else True
        constraint_penalty = primary_constraint_penalty(constraint_costs)
        distance = goal_distance(rollout, controller_input.scene.target_position)
        smoothness = trajectory_smoothness(rollout, order=self.smoothness_order)
        directional = (
            directional_preference(
                rollout,
                controller_input.scene.target_position,
                sign=self.directional_sign,
            )
            if self.directional_sign != 0
            else 0.0
        )
        total_score = (
            self.score_weights.constraint * constraint_penalty
            + self.score_weights.goal_distance * (0.0 if distance is None else distance)
            + self.score_weights.smoothness * smoothness
            + self.score_weights.consensus * consensus_deviation
            + self.score_weights.policy_surrogate
            * (0.0 if policy_surrogate is None else policy_surrogate)
            + self.score_weights.directional * directional
        )
        return CandidateDiagnostics(
            index=index,
            attempted_k=attempted_k,
            action_chunk=action_chunk,
            rollout=rollout,
            constraint_costs=constraint_costs,
            constraint_satisfied=constraint_satisfied,
            feasible=feasible,
            goal_distance=distance,
            constraint_penalty=constraint_penalty,
            smoothness=smoothness,
            consensus_deviation=consensus_deviation,
            policy_surrogate=policy_surrogate,
            total_score=float(total_score),
            directional=directional,
        )


class RejectionController(BaseController):
    """Policy-order preserving rejection/filtering controller."""

    def select(
        self,
        controller_input: ControllerInput,
        *,
        rng: np.random.Generator | None = None,
    ) -> ControllerResult:
        candidates: list[CandidateDiagnostics] = []
        attempted: list[int] = []
        for k in self.k_schedule:
            attempted.append(k)
            batch = self._sample_and_score(
                controller_input,
                attempted_k=k,
                start_index=len(candidates),
                rng=rng,
            )
            candidates.extend(batch)
            feasible = [candidate for candidate in batch if candidate.feasible]
            if feasible:
                return _result(feasible[0], candidates, attempted, "first_feasible")
        return _fallback_result(candidates, attempted)


class RerankingController(BaseController):
    """Score candidates and select the best feasible imagined rollout."""

    def select(
        self,
        controller_input: ControllerInput,
        *,
        rng: np.random.Generator | None = None,
    ) -> ControllerResult:
        candidates: list[CandidateDiagnostics] = []
        attempted: list[int] = []
        for k in self.k_schedule:
            attempted.append(k)
            batch = self._sample_and_score(
                controller_input,
                attempted_k=k,
                start_index=len(candidates),
                rng=rng,
            )
            candidates.extend(batch)
            feasible = [candidate for candidate in candidates if candidate.feasible]
            if feasible:
                selected = min(feasible, key=lambda candidate: candidate.total_score)
                return _result(selected, candidates, attempted, "best_feasible")
        return _fallback_result(candidates, attempted)


def _result(
    selected: CandidateDiagnostics,
    candidates: list[CandidateDiagnostics],
    attempted: list[int],
    reason: str,
) -> ControllerResult:
    selected.selection_reason = reason
    return ControllerResult(
        selected=selected,
        candidates=candidates,
        attempted_k_values=list(attempted),
        selection_reason=reason,
    )


def _fallback_result(
    candidates: list[CandidateDiagnostics],
    attempted: list[int],
) -> ControllerResult:
    if not candidates:
        raise RuntimeError("policy returned no candidate action chunks")
    # No candidate fully satisfies the constraints, so rank by constraint penalty
    # first and break ties by total_score. Ranking on total_score alone lets the
    # (much larger) goal_distance term dominate, which would pick the candidate
    # that barrels straight through the keep-out region whenever that path is the
    # shortest to the goal -- the "avoids none" failure. Constraint-first keeps the
    # fallback choosing the least-violating candidate it has.
    selected = min(
        candidates,
        key=lambda candidate: (candidate.constraint_penalty, candidate.total_score),
    )
    return _result(selected, candidates, attempted, "least_bad_fallback")


def _validate_k_schedule(k_schedule: tuple[int, ...]) -> tuple[int, ...]:
    if not k_schedule:
        raise ValueError("k_schedule must not be empty")
    if any(k <= 0 for k in k_schedule):
        raise ValueError("all k_schedule values must be positive")
    return tuple(int(k) for k in k_schedule)


def _constraint_label(constraint: Constraint, idx: int) -> str:
    name = getattr(constraint, "name", None) or getattr(constraint, "constraint_type", None)
    return f"{idx}:{name or 'constraint'}"


def _unique_cost_key(costs: dict[str, float], key: str) -> str:
    if key not in costs:
        return key
    suffix = 1
    while f"{key}#{suffix}" in costs:
        suffix += 1
    return f"{key}#{suffix}"
