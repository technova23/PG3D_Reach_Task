from __future__ import annotations

import argparse
import sys

import torch

from pg3d.policies.dp3 import make_synthetic_batch, make_tiny_policy


def main() -> int:
    """Run one synthetic DP3 inference and training step on CPU or CUDA."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            print("cuda requested but torch.cuda.is_available() is false", file=sys.stderr)
            return 2
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    torch.manual_seed(0)
    policy = make_tiny_policy().to(device)
    batch = make_synthetic_batch(device=device)

    with torch.no_grad():
        result = policy.predict_action(batch["obs"])

    loss, loss_dict = policy.compute_loss(batch)
    loss.backward()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    optimizer.step()

    print(f"device: {device}")
    print(f"action shape: {tuple(result['action'].shape)}")
    print(f"action_pred shape: {tuple(result['action_pred'].shape)}")
    print(f"loss: {float(loss.detach().cpu()):.6f}")
    print(f"loss_dict: {loss_dict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
