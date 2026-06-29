from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def to_numpy(value: Any) -> np.ndarray:
    """Convert tensor-like ManiSkill/SAPIEN values to NumPy without importing simulators."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        try:
            value = value.numpy()
        except TypeError:
            # Some tensor-like wrappers expose numpy() but reject it; np.asarray handles them.
            pass
    return np.asarray(value)


def bool_info(info: Mapping[str, Any], key: str) -> bool:
    """Read a scalar boolean from a ManiSkill info dict."""
    return bool(np.asarray(to_numpy(info[key])).reshape(-1)[0]) if key in info else False


def float_info(info: Mapping[str, Any], key: str, *, default: float) -> float:
    """Read a scalar float from a ManiSkill info dict."""
    if key not in info:
        return float(default)
    return float(np.asarray(to_numpy(info[key])).reshape(-1)[0])


def float_value(value: Any) -> float:
    """Read the first scalar float from a tensor-like value."""
    return float(np.asarray(to_numpy(value)).reshape(-1)[0])


def bool_any(value: Any) -> bool:
    """Return true when any element in a tensor-like value is truthy."""
    return bool(np.any(to_numpy(value)))


def frame_to_numpy(frame: Any) -> np.ndarray:
    """Convert a renderer frame to a uint8 image, unbatching single-env frames."""
    array = to_numpy(frame)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    return array.astype(np.uint8, copy=False)
