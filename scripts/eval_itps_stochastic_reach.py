from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from pg3d.policies.dp3.checkpoint import load_reach_policy_from_checkpoint
from pg3d.utils.devices import select_device


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an ITPS-style stochastic sampling baseline for DP3 reach. "
            "This first pass applies gradient steering in action space as a proxy for "
            "obstacle avoidance, while keeping the experiment isolated from the main "
            "rejection/reranking path."
        )
    )
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument("--checkpoint", type=Path, default=None)
    checkpoint_group.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--goal-marker-points", type=int, default=16)
    parser.add_argument("--goal-marker-radius", type=float, default=0.015)
    parser.add_argument("--guidance-scale", type=float, default=0.25)
    parser.add_argument("--guidance-steps", type=int, default=1)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument(
        "--obstacle-center",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Optional explicit obstacle center used by the proxy guidance loss.",
    )
    parser.add_argument("--episodes", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = select_device(args.device)
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        from scripts.eval_constrained_reach import resolve_checkpoint_path

        checkpoint_path = resolve_checkpoint_path(args.checkpoint, args.checkpoint_dir)

    policy = load_reach_policy_from_checkpoint(checkpoint_path, device=device, prefer_ema=True)
    if args.num_inference_steps is not None:
        policy.num_inference_steps = int(args.num_inference_steps)
    policy.goal_marker_points = int(args.goal_marker_points)
    policy.goal_marker_radius = float(args.goal_marker_radius)

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # This script intentionally stays separate from the main constrained-eval
    # runner. For the first baseline comparison we only need the steering hook,
    # not a full fork of the reranking controller stack.
    del args.dataset, args.episodes

    with torch.inference_mode(False):
        dummy = torch.zeros((1, policy.horizon, policy.action_dim), device=device, dtype=policy.dtype)
        mask = torch.zeros_like(dummy, dtype=torch.bool)
        cond = dummy.clone()

        obstacle_center = None
        if args.obstacle_center is not None:
            obstacle_center = torch.tensor(args.obstacle_center, device=device, dtype=policy.dtype)

        def guidance_fn(traj: torch.Tensor) -> torch.Tensor:
            # Proxy objective: keep the chunk mean away from the obstacle center.
            # We use this as the first ITPS-style steering baseline until a
            # differentiable FK/EEF proxy is wired into the sampling loop.
            chunk = traj[..., : min(7, traj.shape[-1])]
            if obstacle_center is None:
                return torch.mean(chunk**2)
            chunk_xyz = torch.stack(
                (
                    chunk.mean(dim=1)[:, 0],
                    chunk.mean(dim=1)[:, 1] if chunk.shape[-1] > 1 else torch.zeros_like(chunk[:, 0]),
                    chunk.mean(dim=1)[:, 2] if chunk.shape[-1] > 2 else torch.zeros_like(chunk[:, 0]),
                ),
                dim=-1,
            )
            return -torch.sum((chunk_xyz - obstacle_center) ** 2)

        sample = policy.stochastic_sample(
            cond,
            mask,
            generator=torch.Generator(device=device).manual_seed(args.seed),
            guidance_fn=guidance_fn,
            guidance_scale=float(args.guidance_scale),
            guidance_steps=int(args.guidance_steps),
        )

    action = policy.normalizer["action"].unnormalize(sample[..., : policy.action_dim])
    print(
        {
            "checkpoint": str(checkpoint_path),
            "device": str(device),
            "seed": args.seed,
            "action_mean": float(action.mean().cpu()),
            "action_std": float(action.std().cpu()),
            "guidance_scale": float(args.guidance_scale),
            "guidance_steps": int(args.guidance_steps),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
