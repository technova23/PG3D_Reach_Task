from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from pg3d.envs.maniskill_adapter.types import Observation, RobotState
from pg3d.utils.serialization import jsonable
from pg3d.world_model import ActionChunk, GeometricWorldModel
from pg3d.world_model.types import Array


class SyntheticRobotGeometry:
    """Small deterministic geometry provider for world-model smoke artifacts."""

    def end_effector_position(self, q: Array) -> Array:
        return np.asarray([q[0], q[1], 0.2 + 0.1 * np.mean(q[:7])], dtype=np.float32)

    def robot_point_cloud(self, q: Array) -> Array:
        offset = np.asarray([q[0], q[1], 0.05 + q[2]], dtype=np.float32)
        template = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.03, 0.0, 0.02],
                [0.0, 0.03, 0.02],
                [0.02, 0.02, 0.04],
            ],
            dtype=np.float32,
        )
        return template + offset


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    observation = _synthetic_observation()
    action_chunk = _synthetic_action_chunk(steps=args.steps, dt=args.dt)
    rollout = GeometricWorldModel(SyntheticRobotGeometry()).imagine(
        observation,
        action_chunk,
        metadata={"source": "synthetic_visualization"},
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = _summary(rollout)
    (args.output_dir / "summary.json").write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    np.savez_compressed(
        args.output_dir / "rollout.npz",
        q=rollout.q,
        eef_path=rollout.eef_path,
        scene_point_cloud=np.stack(rollout.scene_point_clouds, axis=0),
        robot_mask=np.stack(rollout.robot_masks, axis=0),
    )
    if args.rerun_path is not None:
        _save_rerun(args.rerun_path, rollout)

    print(f"saved summary: {args.output_dir / 'summary.json'}")
    print(f"saved arrays: {args.output_dir / 'rollout.npz'}")
    if args.rerun_path is not None:
        print(f"saved rerun: {args.rerun_path}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a synthetic robot-only world-model rollout artifact."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/world_model_rollout"))
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--dt", type=float, default=0.125)
    parser.add_argument("--rerun-path", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.dt <= 0:
        raise ValueError("--dt must be positive")
    return args


def _synthetic_observation() -> Observation:
    scene_points = np.asarray(
        [
            [-0.2, 0.0, 0.0],
            [0.0, -0.2, 0.0],
            [0.2, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, 0.05],
            [0.03, 0.0, 0.07],
        ],
        dtype=np.float32,
    )
    robot_mask = np.asarray([False, False, False, False, True, True], dtype=bool)
    return Observation(
        point_cloud=scene_points,
        point_features={},
        robot_mask=robot_mask,
        robot_state=RobotState(
            joint_positions=np.asarray([0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.5, 0.04, 0.04]),
        ),
    )


def _synthetic_action_chunk(*, steps: int, dt: float) -> ActionChunk:
    actions = np.zeros((steps, 7), dtype=np.float32)
    actions[:, 0] = np.linspace(0.0, 0.2, steps, dtype=np.float32)
    actions[:, 1] = np.linspace(0.0, 0.1, steps, dtype=np.float32)
    actions[:, 2] = np.linspace(0.0, 0.05, steps, dtype=np.float32)
    actions[:, 3] = -1.0
    actions[:, 5] = 1.0
    actions[:, 6] = 0.5
    return ActionChunk(actions=actions, action_mode="abs_joint", dt=dt)


def _summary(rollout: Any) -> dict[str, Any]:
    return {
        "horizon": int(rollout.q.shape[0]),
        "dof": int(rollout.q.shape[1]),
        "eef_start": rollout.eef_path[0].tolist(),
        "eef_end": rollout.eef_path[-1].tolist(),
        "scene_points_per_step": [int(points.shape[0]) for points in rollout.scene_point_clouds],
        "robot_points_per_step": [int(points.shape[0]) for points in rollout.robot_point_clouds],
        "metadata": rollout.metadata,
    }


def _save_rerun(path: Path, rollout: Any) -> None:
    try:
        import rerun as rr
    except Exception as exc:
        raise RuntimeError(
            "Rerun export requires: "
            "uv sync --extra cu129 --extra maniskill --extra viz --group dev --group notebooks"
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    rr.init("pg3d_world_model_rollout", spawn=False)
    rr.save(str(path))
    for step_idx, (scene, mask) in enumerate(
        zip(rollout.scene_point_clouds, rollout.robot_masks, strict=True)
    ):
        rr.set_time_sequence("step", step_idx)
        rr.log("world/scene_points", rr.Points3D(scene[~mask], colors=[180, 180, 180]))
        rr.log("world/robot_points", rr.Points3D(scene[mask], colors=[0, 128, 255]))
        rr.log(
            "world/eef",
            rr.Points3D(rollout.eef_path[step_idx].reshape(1, 3), colors=[255, 220, 0]),
        )
    rr.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
