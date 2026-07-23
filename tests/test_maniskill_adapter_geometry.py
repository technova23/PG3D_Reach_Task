from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import pg3d.envs.maniskill_adapter.geometry as geometry_module
from pg3d.envs.maniskill_adapter.geometry import (
    ManiSkillGhostPandaGeometryProvider,
)


class _FakeRobot:
    def __init__(self) -> None:
        self.qpos = np.zeros((1, 9), dtype=np.float32)

    def get_qpos(self) -> np.ndarray:
        return self.qpos

    def set_qpos(self, qpos: np.ndarray) -> None:
        self.qpos = np.asarray(qpos, dtype=np.float32).copy()

    def set_qvel(self, _qvel: np.ndarray) -> None:
        pass


class _FakeEnv:
    def __init__(self) -> None:
        self.unwrapped = self
        self.scene = SimpleNamespace()
        self.agent = SimpleNamespace(
            robot=_FakeRobot(),
            tcp_pose=SimpleNamespace(
                p=np.asarray([[0.4, -0.1, 0.3]], dtype=np.float32)
            ),
        )

    def evaluate(self) -> dict[str, object]:
        return {}

    def get_obs(self, _info: dict[str, object]) -> dict[str, object]:
        return {}


def test_ghost_geometry_provider_counts_queries_and_cache_misses(monkeypatch) -> None:
    adapted = SimpleNamespace(
        point_cloud=np.asarray(
            [[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]],
            dtype=np.float32,
        ),
        robot_mask=np.asarray([False, True], dtype=bool),
        robot_state=SimpleNamespace(
            tcp_pose=np.asarray([0.4, -0.1, 0.3, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        ),
    )
    monkeypatch.setattr(
        geometry_module,
        "adapt_observation",
        lambda *_args, **_kwargs: adapted,
    )
    provider = ManiSkillGhostPandaGeometryProvider(_FakeEnv())
    q0 = np.zeros((9,), dtype=np.float32)
    q1 = np.ones((9,), dtype=np.float32)
    q2 = np.full((9,), 2.0, dtype=np.float32)

    provider.end_effector_position(q0)
    provider.robot_point_cloud(q0)
    provider.robot_point_cloud(q1)
    provider.end_effector_position_only(q2)

    assert provider.counter_snapshot() == {
        "end_effector_position_queries": 1,
        "end_effector_position_only_queries": 1,
        "eef_geometry_queries": 2,
        "robot_point_cloud_queries": 2,
        "robot_point_cloud_renders": 2,
    }
