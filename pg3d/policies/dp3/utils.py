from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TypeVar

import torch

T = TypeVar("T")


def dict_apply(x: Mapping[str, T], func: Callable[[T], T]) -> dict[str, T]:
    """Apply ``func`` to each leaf value in a nested mapping."""
    result: dict[str, T] = {}
    for key, value in x.items():
        if isinstance(value, Mapping):
            result[key] = dict_apply(value, func)  # type: ignore[assignment]
        else:
            result[key] = func(value)
    return result


def optimizer_to(
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
) -> torch.optim.Optimizer:
    """Move tensor-valued optimizer state to ``device`` and return the optimizer."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device)
    return optimizer
