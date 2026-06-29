from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from pg3d.policies.dp3 import SimpleDP3
from pg3d.policies.dp3.normalizer import LinearNormalizer
from pg3d.utils.serialization import jsonable


def checkpoint_path_for_step(checkpoint_dir: Path, step: int, *, final: bool = False) -> Path:
    """Return the step-named checkpoint path under a checkpoint directory."""
    prefix = "final_step" if final else "step"
    return checkpoint_dir / f"{prefix}_{step:08d}.pt"


def latest_reach_checkpoint(checkpoint_dir: Path) -> Path:
    """Return the latest step-named reach checkpoint in a directory.

    Final checkpoints win ties at the same step because they include the final trainer state.
    """
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"checkpoint directory does not exist: {checkpoint_dir}")
    candidates: list[tuple[int, int, Path]] = []
    for path in checkpoint_dir.glob("*.pt"):
        parsed = _parse_step_checkpoint_name(path)
        if parsed is not None:
            step, is_final = parsed
            candidates.append((step, 1 if is_final else 0, path))
    if not candidates:
        raise FileNotFoundError(f"no step_*.pt or final_step_*.pt files in {checkpoint_dir}")
    return max(candidates)[2]


def should_save_checkpoint(step: int, checkpoint_every: int) -> bool:
    """Return whether this training step should write a periodic checkpoint."""
    return checkpoint_every > 0 and step % checkpoint_every == 0


def save_reach_policy_checkpoint(
    path: Path,
    *,
    policy: SimpleDP3,
    ema_policy: SimpleDP3 | None,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None,
    policy_kwargs: dict[str, Any],
    args: Any,
    step: int,
    best_val_loss: float | None,
) -> None:
    """Save a DP3 reach checkpoint using the shared trainer/eval payload format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_version": "pg3d.dp3_reach.v2",
            "model": model_state(policy),
            "ema_model": model_state(ema_policy) if ema_policy is not None else None,
            "normalizer": {
                key: value.detach().cpu() for key, value in policy.normalizer.state_dict().items()
            },
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "policy_kwargs": jsonable(policy_kwargs),
            "args": jsonable(vars(args)),
            "step": step,
            "best_val_loss": best_val_loss,
        },
        path,
    )


def load_reach_policy_from_checkpoint(
    path: Path,
    *,
    device: torch.device,
    prefer_ema: bool = True,
) -> SimpleDP3:
    """Load a DP3 reach checkpoint written by `save_reach_policy_checkpoint`."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    policy = SimpleDP3(**checkpoint["policy_kwargs"])
    policy.set_normalizer(LinearNormalizer.from_state_dict(checkpoint["normalizer"]))
    checkpoint_model = (
        checkpoint.get("ema_model")
        if prefer_ema and checkpoint.get("ema_model") is not None
        else checkpoint["model"]
    )
    policy.load_state_dict(checkpoint_model, strict=False)
    policy.to(device)
    policy.eval()
    return policy


def model_state(policy: SimpleDP3 | None) -> dict[str, torch.Tensor]:
    """Return model weights without the normalizer buffers stored separately."""
    if policy is None:
        return {}
    return {
        key: value.detach().cpu()
        for key, value in policy.state_dict().items()
        if not key.startswith("normalizer.")
    }


def _parse_step_checkpoint_name(path: Path) -> tuple[int, bool] | None:
    stem = path.stem
    if stem.startswith("final_step_"):
        step_text = stem.removeprefix("final_step_")
        is_final = True
    elif stem.startswith("step_"):
        step_text = stem.removeprefix("step_")
        is_final = False
    else:
        return None
    if not step_text.isdigit():
        return None
    return int(step_text), is_final
