from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from pg3d.constraints import Constraint, SceneContext
from pg3d.envs.maniskill_adapter.types import Observation
from pg3d.world_model import ActionChunk, ImaginedRollout


@dataclass
class ControllerInput:
    """Inputs shared by rejection and reranking controllers."""

    observation: Observation
    scene: SceneContext
    policy_input: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.metadata = dict(self.metadata)

    def input_for_policy(self) -> Any:
        """Return policy-specific input, falling back to the pg3d observation."""
        return self.observation if self.policy_input is None else self.policy_input


class Policy(Protocol):
    """Policy adapter interface used by composition controllers."""

    def sample_action_chunks(
        self,
        policy_input: Any,
        *,
        k: int,
        rng: np.random.Generator | None = None,
    ) -> list[ActionChunk]:
        """Sample up to `k` candidate action chunks."""
        ...


class WorldModel(Protocol):
    """World-model interface used by composition controllers."""

    def imagine(self, observation: Observation, action_chunk: ActionChunk) -> ImaginedRollout:
        """Imagine one candidate action chunk from the current observation."""
        ...


@dataclass(frozen=True)
class ScoreWeights:
    """Weights for soft candidate scoring terms."""

    goal_distance: float = 1.0
    constraint: float = 1.0
    smoothness: float = 0.1
    consensus: float = 0.01
    policy_surrogate: float = 1.0


@dataclass
class CandidateDiagnostics:
    """Per-candidate rollout, costs, feasibility, and selection metadata."""

    index: int
    attempted_k: int
    action_chunk: ActionChunk
    rollout: ImaginedRollout
    constraint_costs: dict[str, float]
    constraint_satisfied: dict[str, bool]
    feasible: bool
    goal_distance: float | None
    constraint_penalty: float
    smoothness: float
    consensus_deviation: float
    policy_surrogate: float | None
    total_score: float
    selection_reason: str | None = None


@dataclass
class ControllerResult:
    """Selection result returned by composition controllers."""

    selected: CandidateDiagnostics
    candidates: list[CandidateDiagnostics]
    attempted_k_values: list[int]
    selection_reason: str

    @property
    def action_chunk(self) -> ActionChunk:
        """Selected action chunk convenience accessor."""
        return self.selected.action_chunk

    @property
    def rollout(self) -> ImaginedRollout:
        """Selected imagined rollout convenience accessor."""
        return self.selected.rollout


ConstraintList = list[Constraint]
