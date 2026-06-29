from __future__ import annotations

import subprocess
import sys
from typing import Any

import numpy as np

from pg3d.composition import (
    ControllerInput,
    RejectionController,
    RerankingController,
    ScoreWeights,
)
from pg3d.constraints import SceneContext, SphereRegion
from pg3d.constraints.programs import AvoidRegion
from pg3d.envs.maniskill_adapter.types import Observation, RobotState
from pg3d.world_model import ActionChunk, ImaginedRollout


def test_rejection_selects_first_feasible_candidate() -> None:
    unsafe = _chunk("unsafe", [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])
    safe = _chunk("safe", [[0.0, 0.0, 0.0], [0.5, 0.3, 0.0], [1.0, 0.0, 0.0]])
    controller = RejectionController(
        policy=_FakePolicy({2: [unsafe, safe]}),
        world_model=_FakeWorldModel(),
        constraints=[_avoid_center()],
        k_schedule=(2,),
    )

    result = controller.select(_controller_input())

    assert result.selected.action_chunk.metadata["name"] == "safe"
    assert result.selection_reason == "first_feasible"
    assert result.attempted_k_values == [2]
    assert [candidate.feasible for candidate in result.candidates] == [False, True]


def test_reranking_prefers_feasible_candidate_over_lower_cost_unsafe_candidate() -> None:
    unsafe = _chunk("short_unsafe", [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])
    safe = _chunk("safe", [[0.0, 0.0, 0.0], [0.5, 0.3, 0.0], [1.0, 0.0, 0.0]])
    controller = RerankingController(
        policy=_FakePolicy({2: [unsafe, safe]}),
        world_model=_FakeWorldModel(),
        constraints=[_avoid_center()],
        k_schedule=(2,),
        score_weights=ScoreWeights(
            goal_distance=1.0,
            constraint=1.0,
            smoothness=0.0,
            consensus=0.0,
        ),
    )

    result = controller.select(_controller_input())

    assert result.selected.action_chunk.metadata["name"] == "safe"
    assert result.selection_reason == "best_feasible"


def test_fallback_k_schedule_retries_until_feasible_candidate() -> None:
    unsafe = _chunk("unsafe", [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])
    safe = _chunk("safe", [[0.0, 0.0, 0.0], [0.5, 0.3, 0.0], [1.0, 0.0, 0.0]])
    policy = _FakePolicy({2: [unsafe], 4: [safe]})
    controller = RerankingController(
        policy=policy,
        world_model=_FakeWorldModel(),
        constraints=[_avoid_center()],
        k_schedule=(2, 4, 8),
    )

    result = controller.select(_controller_input())

    assert policy.requested_k == [2, 4]
    assert result.attempted_k_values == [2, 4]
    assert result.selected.action_chunk.metadata["name"] == "safe"


def test_all_violating_candidates_return_least_bad_fallback() -> None:
    through_center = _chunk(
        "through_center",
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]],
    )
    near_edge = _chunk(
        "near_edge",
        [[0.0, 0.0, 0.0], [0.5, 0.05, 0.0], [1.0, 0.0, 0.0]],
    )
    controller = RerankingController(
        policy=_FakePolicy({2: [through_center, near_edge]}),
        world_model=_FakeWorldModel(),
        constraints=[_avoid_center()],
        k_schedule=(2,),
        score_weights=ScoreWeights(
            goal_distance=0.0,
            constraint=1.0,
            smoothness=0.0,
            consensus=0.0,
            policy_surrogate=0.0,
        ),
    )

    result = controller.select(_controller_input())

    assert result.selection_reason == "least_bad_fallback"
    assert result.selected.action_chunk.metadata["name"] == "near_edge"
    assert not result.selected.feasible


def test_candidate_diagnostics_include_soft_score_terms() -> None:
    first = _chunk("first", [[0.0, 0.0, 0.0], [0.4, 0.2, 0.0], [0.9, 0.0, 0.0]])
    second = _chunk("second", [[0.0, 0.0, 0.0], [0.4, 0.4, 0.0], [0.8, 0.0, 0.0]])
    controller = RerankingController(
        policy=_FakePolicy({2: [first, second]}, surrogate={2: [0.2, 0.1]}),
        world_model=_FakeWorldModel(),
        constraints=[],
        k_schedule=(2,),
    )

    result = controller.select(_controller_input())

    diagnostic = result.candidates[0]
    assert diagnostic.goal_distance is not None
    assert diagnostic.smoothness >= 0.0
    assert diagnostic.consensus_deviation > 0.0
    assert diagnostic.policy_surrogate == 0.2
    assert diagnostic.total_score >= 0.0


def test_controller_passes_policy_specific_input_for_future_dp3_adapter() -> None:
    policy_input = {"obs_window": object()}
    safe = _chunk("safe", [[0.0, 0.0, 0.0], [0.5, 0.3, 0.0], [1.0, 0.0, 0.0]])
    policy = _FakePolicy({1: [safe]})
    controller = RejectionController(
        policy=policy,
        world_model=_FakeWorldModel(),
        constraints=[_avoid_center()],
        k_schedule=(1,),
    )

    result = controller.select(_controller_input(policy_input=policy_input))

    assert result.selected.action_chunk.metadata["name"] == "safe"
    assert policy.seen_policy_inputs == [policy_input]


def test_composition_import_keeps_heavy_deps_lazy() -> None:
    code = """
import importlib
import sys

importlib.import_module("pg3d.composition")
assert "mani_skill" not in sys.modules
assert "sapien" not in sys.modules
assert "gymnasium" not in sys.modules
assert "rerun" not in sys.modules
assert "torch" not in sys.modules
assert "pg3d.policies.dp3" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


class _FakePolicy:
    def __init__(
        self,
        batches: dict[int, list[ActionChunk]],
        *,
        surrogate: dict[int, list[float]] | None = None,
    ) -> None:
        self.batches = batches
        self.surrogate = surrogate or {}
        self.requested_k: list[int] = []
        self.seen_policy_inputs: list[Any] = []

    def sample_action_chunks(
        self,
        policy_input: Any,
        *,
        k: int,
        rng: np.random.Generator | None = None,
    ) -> list[ActionChunk]:
        self.requested_k.append(k)
        self.seen_policy_inputs.append(policy_input)
        return list(self.batches.get(k, []))

    def score_surrogate(self, policy_input: Any, chunks: list[ActionChunk]) -> list[float]:
        return list(self.surrogate.get(len(chunks), [0.0 for _ in chunks]))


class _FakeWorldModel:
    def imagine(self, observation: Observation, action_chunk: ActionChunk) -> ImaginedRollout:
        eef_path = action_chunk.actions[:, :3]
        horizon = action_chunk.horizon
        return ImaginedRollout(
            q=action_chunk.actions.astype(np.float32, copy=True),
            eef_path=eef_path.astype(np.float32, copy=True),
            robot_point_clouds=[np.zeros((0, 3), dtype=np.float32) for _ in range(horizon)],
            scene_point_clouds=[np.zeros((0, 3), dtype=np.float32) for _ in range(horizon)],
            robot_masks=[np.zeros((0,), dtype=bool) for _ in range(horizon)],
            action_chunk=action_chunk,
        )


def _chunk(name: str, points: list[list[float]]) -> ActionChunk:
    return ActionChunk(
        actions=np.asarray(points, dtype=np.float32),
        action_mode="abs_joint",
        dt=0.1,
        metadata={"name": name},
    )


def _avoid_center() -> AvoidRegion:
    return AvoidRegion(region=SphereRegion(center=[0.5, 0.0, 0.0], radius=0.1))


def _controller_input(policy_input: Any | None = None) -> ControllerInput:
    return ControllerInput(
        observation=Observation(
            point_cloud=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
            point_features={},
            robot_mask=np.asarray([False], dtype=bool),
            robot_state=RobotState(joint_positions=np.zeros((3,), dtype=np.float32)),
        ),
        scene=SceneContext(target_position=np.asarray([1.0, 0.0, 0.0], dtype=np.float32)),
        policy_input=policy_input,
    )
