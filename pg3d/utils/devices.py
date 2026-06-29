from __future__ import annotations

import torch


def select_device(value: str) -> torch.device:
    """Select a PyTorch device from the common pg3d CLI device argument."""
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but torch.cuda.is_available() is false")
    return torch.device(value)
