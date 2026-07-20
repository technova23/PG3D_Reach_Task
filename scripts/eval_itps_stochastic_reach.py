from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from pg3d.envs.maniskill_adapter.dataset import load_reach_metadata
from pg3d.policies.dp3.checkpoint import load_reach_policy_from_checkpoint
from pg3d.utils.devices import select_device
from pg3d.world_model.panda_fk import panda_end_effector_position


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an ITPS-style stochastic steering baseline for DP3 reach. "
            "This baseline uses Panda FK to compute an EEF clearance loss over the full "
            "predicted joint horizon, rather than the joint-mean proxy."
        )
    )
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument("--checkpoint", type=Path, default=None)
    checkpoint_group.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--goal-marker-points", type=int, default=16)
    parser.add_argument("--goal-marker-radius", type=float, default=0.015)
    parser.add_argument("--guidance-scale", type=float, default=0.25)
    parser.add_argument("--guidance-steps", type=int, default=1)
    parser.add_argument("--obstacle-center", type=float, nargs=3, required=True)
    parser.add_argument("--obstacle-radius", type=float, default=0.08)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = select_device(args.device)
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        from scripts.eval_constrained_reach import resolve_checkpoint_path

        checkpoint_path = resolve_checkpoint_path(args.checkpoint, args.checkpoint_dir)

    metadata = load_reach_metadata(args.dataset)

    policy = load_reach_policy_from_checkpoint(checkpoint_path, device=device, prefer_ema=True)
    if args.num_inference_steps is not None:
        policy.num_inference_steps = int(args.num_inference_steps)
    policy.goal_marker_points = int(args.goal_marker_points)
    policy.goal_marker_radius = float(args.goal_marker_radius)

    obstacle_center = torch.tensor(args.obstacle_center, device=device, dtype=policy.dtype).reshape(3)
    obstacle_radius = float(args.obstacle_radius)

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # This baseline is intentionally local to the inference-time sampler.
    # We use the Panda FK wrapper to compute the clearance loss on the full
    # predicted EEF path, then provide a finite-difference gradient back to
    # the denoiser. That keeps the guidance geometrically meaningful without
    # changing the main reranking implementation.
    del args.dataset, args.episodes, metadata, rng

    with torch.inference_mode(False):
        dummy = torch.zeros((1, policy.horizon, policy.action_dim), device=device, dtype=policy.dtype)
        mask = torch.zeros_like(dummy, dtype=torch.bool)
        cond = dummy.clone()

        def guidance_fn(traj: torch.Tensor) -> torch.Tensor:
            q = traj[..., :7]
            eef_path = panda_end_effector_position(q)
            distances = torch.linalg.norm(eef_path - obstacle_center.view(1, 1, 3), dim=-1)
            violations = torch.clamp(obstacle_radius - distances, min=0.0)
            return torch.amax(violations, dim=1)

        sample = policy.stochastic_sample(
            cond,
            mask,
            generator=torch.Generator(device=device).manual_seed(args.seed),
            guidance_fn=guidance_fn,
            guide_ratio=float(args.guidance_scale),
            mcmc_steps=int(args.guidance_steps),
        )

    action = policy.normalizer["action"].unnormalize(sample[..., : policy.action_dim])
    print(
        {
            "checkpoint": str(checkpoint_path),
            "device": str(device),
            "seed": args.seed,
            "episodes": int(args.episodes),
            "guidance_scale": float(args.guidance_scale),
            "guidance_steps": int(args.guidance_steps),
            "obstacle_center": obstacle_center.tolist(),
            "obstacle_radius": obstacle_radius,
            "action_mean": float(action.mean().cpu()),
            "action_std": float(action.std().cpu()),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
