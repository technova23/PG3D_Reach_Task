from __future__ import annotations

import argparse

import numpy as np
import torch

from pg3d.world_model.panda_fk import panda_end_effector_position


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare differentiable Panda FK with the live ManiSkill TCP pose."
    )
    parser.add_argument("--env-id", default="PG3DReach-Narrow-v0")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.samples < 0:
        raise ValueError("samples must be non-negative")
    if args.tolerance <= 0.0:
        raise ValueError("tolerance must be positive")

    import gymnasium as gym

    from pg3d.envs.maniskill_adapter import register_pg3d_reach_envs
    from pg3d.envs.maniskill_adapter.geometry import ManiSkillGhostPandaGeometryProvider

    register_pg3d_reach_envs()
    env = gym.make(args.env_id, obs_mode="state", num_envs=1)
    try:
        env.reset(seed=args.seed, options={"reconfigure": True})
        provider = ManiSkillGhostPandaGeometryProvider(env, task_name=args.env_id)
        unwrapped = env.unwrapped
        robot = unwrapped.agent.robot
        base_matrix = robot.pose.to_transformation_matrix()[0].detach().cpu()
        current_q = robot.get_qpos()[0, :7].detach().cpu().numpy()
        limits = robot.get_qlimits()[0, :7].detach().cpu().numpy()
        rng = np.random.default_rng(args.seed)
        configurations = [current_q]
        configurations.extend(
            rng.uniform(limits[:, 0], limits[:, 1]).astype(np.float32)
            for _ in range(args.samples)
        )

        errors = []
        for q_numpy in configurations:
            expected = provider.end_effector_position_only(q_numpy)
            q = torch.as_tensor(q_numpy, dtype=torch.float32)
            actual = panda_end_effector_position(q, base_matrix).detach().cpu().numpy()
            errors.append(float(np.linalg.norm(actual - expected)))
        max_error = max(errors)
        print(
            f"checked={len(errors)} max_error_m={max_error:.8f} "
            f"tolerance_m={args.tolerance:.8f}"
        )
        if max_error > args.tolerance:
            print("Panda FK check failed")
            return 1
        print("Panda FK check passed")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
