from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from pg3d.policies.dp3 import ReachDatasetConfig, ReachSequenceDataset, SimpleDP3
from pg3d.policies.dp3.checkpoint import load_reach_policy_from_checkpoint
from pg3d.policies.dp3.goal_markers import (
    DEFAULT_GOAL_MARKER_POINTS,
    DEFAULT_GOAL_MARKER_RADIUS,
)
from pg3d.policies.dp3.utils import dict_apply
from pg3d.utils.devices import select_device


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = select_device(args.device)
    policy = (
        load_reach_policy_from_checkpoint(
            args.checkpoint,
            device=device,
            prefer_ema=args.checkpoint_model == "ema",
        )
        if args.checkpoint is not None
        else None
    )
    goal_marker_points = (
        int(policy.goal_marker_points) if policy is not None else args.goal_marker_points
    )
    goal_marker_radius = (
        float(policy.goal_marker_radius) if policy is not None else args.goal_marker_radius
    )
    dataset = ReachSequenceDataset(
        ReachDatasetConfig(
            dataset_path=args.dataset,
            horizon=args.horizon,
            n_obs_steps=args.n_obs_steps,
            val_ratio=0.0,
            seed=args.seed,
            goal_marker_points=goal_marker_points,
            goal_marker_radius=goal_marker_radius,
        ),
        split="all",
    )
    if len(dataset) == 0:
        raise RuntimeError("dataset has no sequences")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    if policy is None:
        policy = _build_untrained_policy(args, dataset=dataset, device=device)

    total_mse = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = _batch_to(batch, device)
            output = policy.predict_action(batch["obs"])
            target = batch["action"][
                :,
                args.n_obs_steps - 1 : args.n_obs_steps - 1 + args.n_action_steps,
            ]
            mse = torch.nn.functional.mse_loss(output["action"], target)
            total_mse += float(mse.detach().cpu())
            total_batches += 1
            print(
                f"batch={total_batches} action_shape={tuple(output['action'].shape)} "
                f"demo_mse={float(mse.detach().cpu()):.6f}"
            )
            if total_batches >= args.max_batches:
                break

    summary = {
        "dataset": str(args.dataset),
        "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
        "batches": total_batches,
        "mean_demo_mse": total_mse / max(total_batches, 1),
        "goal_marker_points": goal_marker_points,
        "goal_marker_radius": goal_marker_radius,
        "device": str(device),
    }
    print("summary: " + json.dumps(summary, sort_keys=True))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dataset-only DP3 inference sanity checks for pg3d reach."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("artifacts/reach-dataset-smoke/pg3d-reach-smoke.zarr"),
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-model", choices=["ema", "raw"], default="ema")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--n-obs-steps", type=int, default=2)
    parser.add_argument("--n-action-steps", type=int, default=8)
    parser.add_argument("--goal-marker-points", type=int, default=DEFAULT_GOAL_MARKER_POINTS)
    parser.add_argument("--goal-marker-radius", type=float, default=DEFAULT_GOAL_MARKER_RADIUS)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--encoder-output-dim", type=int, default=32)
    parser.add_argument("--diffusion-step-embed-dim", type=int, default=64)
    parser.add_argument("--down-dims", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--n-groups", type=int, default=8)
    args = parser.parse_args(argv)
    if args.goal_marker_points < 0:
        raise ValueError("--goal-marker-points must be non-negative")
    if args.goal_marker_radius < 0:
        raise ValueError("--goal-marker-radius must be non-negative")
    return args


def _build_untrained_policy(
    args: argparse.Namespace,
    *,
    dataset: ReachSequenceDataset,
    device: torch.device,
) -> SimpleDP3:
    policy = SimpleDP3(
        shape_meta=dataset.shape_meta,
        horizon=args.horizon,
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
        num_inference_steps=args.num_inference_steps,
        encoder_output_dim=args.encoder_output_dim,
        diffusion_step_embed_dim=args.diffusion_step_embed_dim,
        down_dims=tuple(args.down_dims),
        kernel_size=args.kernel_size,
        n_groups=args.n_groups,
        goal_marker_points=args.goal_marker_points,
        goal_marker_radius=args.goal_marker_radius,
        pointcloud_encoder_cfg={
            "out_channels": args.encoder_output_dim,
            "use_layernorm": True,
            "final_norm": "layernorm",
        },
    )
    policy.set_normalizer(dataset.get_normalizer())
    policy.to(device)
    policy.eval()
    return policy


def _batch_to(batch: Any, device: torch.device) -> Any:
    return dict_apply(batch, lambda tensor: tensor.to(device=device, dtype=torch.float32))


if __name__ == "__main__":
    raise SystemExit(main())
