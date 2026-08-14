from __future__ import annotations

import argparse

import numpy as np
import torch

from pg3d.world_model.panda_fk import (
    PANDA_MOVABLE_COLLISION_LINKS,
    panda_end_effector_position,
    panda_link_transforms,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare differentiable Panda FK with the live ManiSkill TCP pose."
    )
    parser.add_argument("--env-id", default="PG3DReach-Narrow-v0")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--rotation-tolerance", type=float, default=1e-4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.samples < 0:
        raise ValueError("samples must be non-negative")
    if args.tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if args.rotation_tolerance <= 0.0:
        raise ValueError("rotation-tolerance must be positive")

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
            rng.uniform(limits[:, 0], limits[:, 1]).astype(np.float32) for _ in range(args.samples)
        )

        links = {link.name: link for link in robot.get_links()}
        missing_links = [name for name in PANDA_MOVABLE_COLLISION_LINKS if name not in links]
        if missing_links:
            raise RuntimeError(f"live Panda is missing expected links: {missing_links}")

        tcp_errors = []
        link_position_errors = []
        link_rotation_errors = []
        for q_numpy in configurations:
            expected = provider.end_effector_position_only(q_numpy)
            q = torch.as_tensor(q_numpy, dtype=torch.float32)
            actual = panda_end_effector_position(q, base_matrix).detach().cpu().numpy()
            tcp_errors.append(float(np.linalg.norm(actual - expected)))
            actual_links = panda_link_transforms(q, base_matrix).detach().cpu().numpy()
            for link_index, link_name in enumerate(PANDA_MOVABLE_COLLISION_LINKS):
                expected_link = (
                    links[link_name].pose.to_transformation_matrix()[0].detach().cpu().numpy()
                )
                link_position_errors.append(
                    float(np.linalg.norm(actual_links[link_index, :3, 3] - expected_link[:3, 3]))
                )
                link_rotation_errors.append(
                    float(np.max(np.abs(actual_links[link_index, :3, :3] - expected_link[:3, :3])))
                )
        max_tcp_error = max(tcp_errors)
        max_link_position_error = max(link_position_errors)
        max_link_rotation_error = max(link_rotation_errors)
        print(
            f"checked={len(tcp_errors)} max_tcp_error_m={max_tcp_error:.8f} "
            f"max_link_position_error_m={max_link_position_error:.8f} "
            f"max_link_rotation_error={max_link_rotation_error:.8f} "
            f"position_tolerance_m={args.tolerance:.8f} "
            f"rotation_tolerance={args.rotation_tolerance:.8f}"
        )
        if (
            max_tcp_error > args.tolerance
            or max_link_position_error > args.tolerance
            or max_link_rotation_error > args.rotation_tolerance
        ):
            print("Panda FK check failed")
            return 1
        print("Panda FK check passed")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
