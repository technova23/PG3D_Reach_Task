from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import nn


class SingleFieldLinearNormalizer(nn.Module):
    """Affine normalizer for one tensor field.

    The default identity parameters have shape ``[1]`` and broadcast over the
    trailing feature dimension. Real datasets can replace them with fitted
    per-dimension statistics later.
    """

    def __init__(
        self,
        scale: torch.Tensor | None = None,
        offset: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if scale is None:
            scale = torch.ones(1, dtype=torch.float32)
        if offset is None:
            offset = torch.zeros_like(scale)
        self.register_buffer("scale", scale.detach().clone().float())
        self.register_buffer("offset", offset.detach().clone().float())

    @classmethod
    def identity(cls) -> SingleFieldLinearNormalizer:
        """Create a no-op normalizer."""
        return cls()

    @classmethod
    def create_manual(
        cls,
        scale: torch.Tensor | np.ndarray,
        offset: torch.Tensor | np.ndarray,
    ) -> SingleFieldLinearNormalizer:
        """Create a normalizer from explicit affine parameters."""
        return cls(_as_tensor(scale), _as_tensor(offset))

    def normalize(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Apply the stored affine normalization to a tensor-like value."""
        x_tensor = _as_tensor(x).to(device=self.scale.device, dtype=self.scale.dtype)
        return x_tensor * self.scale + self.offset

    def unnormalize(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Invert the stored affine normalization."""
        x_tensor = _as_tensor(x).to(device=self.scale.device, dtype=self.scale.dtype)
        return (x_tensor - self.offset) / self.scale

    def forward(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Alias for :meth:`normalize` so the normalizer can be used as a module."""
        return self.normalize(x)


class LinearNormalizer(nn.Module):
    """Dictionary normalizer used by DP3 policies."""

    def __init__(self, fields: Mapping[str, SingleFieldLinearNormalizer] | None = None) -> None:
        super().__init__()
        self.fields = nn.ModuleDict(fields or {})

    @classmethod
    def identity_for_keys(cls, keys: list[str]) -> LinearNormalizer:
        """Create identity normalizers for each named tensor field."""
        return cls({key: SingleFieldLinearNormalizer.identity() for key in keys})

    @classmethod
    def standardize_from_data(
        cls,
        data: Mapping[str, np.ndarray | torch.Tensor],
        *,
        eps: float = 1e-6,
    ) -> LinearNormalizer:
        """Fit per-feature standardization statistics from arrays.

        Each array is flattened over leading dimensions and fit over the final
        feature dimension, matching DP3 tensors such as ``[T, N, 3]`` point
        clouds or ``[T, 7]`` action labels.
        """
        return cls.from_data(data, eps=eps)

    @classmethod
    def from_data(
        cls,
        data: Mapping[str, np.ndarray | torch.Tensor],
        *,
        modes: Mapping[str, str] | None = None,
        eps: float = 1e-6,
    ) -> LinearNormalizer:
        """Fit per-field affine normalizers with a selectable mode per field.

        ``modes`` maps a field name to ``"standardize"`` (zero-mean/unit-std,
        the default for any field not listed) or ``"minmax"`` (scale the
        observed range to ``[-1, 1]``). Use ``"minmax"`` for the ``action``
        field so the diffusion scheduler's ``clip_sample=True`` clamp to
        ``[-1, 1]`` does not truncate valid action tails.
        """
        modes = dict(modes or {})
        fields: dict[str, SingleFieldLinearNormalizer] = {}
        for key, value in data.items():
            tensor = _as_tensor(value).float()
            if tensor.ndim == 0:
                raise ValueError(f"cannot fit normalizer for scalar field {key!r}")
            flattened = tensor.reshape(-1, tensor.shape[-1])
            mode = modes.get(key, "standardize")
            if mode == "standardize":
                mean = flattened.mean(dim=0)
                std = flattened.std(dim=0, unbiased=False).clamp_min(eps)
                scale = 1.0 / std
                offset = -mean / std
            elif mode == "minmax":
                minimum = flattened.min(dim=0).values
                maximum = flattened.max(dim=0).values
                span = (maximum - minimum).clamp_min(eps)
                scale = 2.0 / span
                offset = -1.0 - minimum * scale
            else:
                raise ValueError(f"unsupported normalize mode {mode!r} for field {key!r}")
            fields[key] = SingleFieldLinearNormalizer.create_manual(scale=scale, offset=offset)
        return cls(fields)

    @classmethod
    def from_state_dict(cls, state_dict: Mapping[str, Any]) -> LinearNormalizer:
        """Rebuild a normalizer from a ``LinearNormalizer.state_dict()`` mapping."""
        grouped: dict[str, dict[str, torch.Tensor]] = {}
        for key, value in state_dict.items():
            parts = key.split(".")
            if len(parts) != 3 or parts[0] != "fields" or parts[2] not in {"scale", "offset"}:
                continue
            grouped.setdefault(parts[1], {})[parts[2]] = _as_tensor(value).float()
        missing = {
            field
            for field, values in grouped.items()
            if "scale" not in values or "offset" not in values
        }
        if missing:
            raise ValueError(f"incomplete normalizer fields: {sorted(missing)}")
        return cls(
            {
                field: SingleFieldLinearNormalizer.create_manual(
                    scale=values["scale"],
                    offset=values["offset"],
                )
                for field, values in grouped.items()
            }
        )

    def __getitem__(self, key: str) -> SingleFieldLinearNormalizer:
        """Return a field normalizer, lazily creating an identity normalizer."""
        if key not in self.fields:
            self.fields[key] = SingleFieldLinearNormalizer.identity()
        return self.fields[key]

    def __setitem__(self, key: str, value: SingleFieldLinearNormalizer) -> None:
        """Set the normalizer for one field."""
        self.fields[key] = value

    def normalize(
        self,
        x: Mapping[str, torch.Tensor] | torch.Tensor,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """Normalize either a tensor field mapping or a single tensor."""
        if isinstance(x, Mapping):
            return {key: self[key].normalize(value) for key, value in x.items()}
        return self["_default"].normalize(x)

    def unnormalize(
        self, x: Mapping[str, torch.Tensor] | torch.Tensor
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """Unnormalize either a tensor field mapping or a single tensor."""
        if isinstance(x, Mapping):
            return {key: self[key].unnormalize(value) for key, value in x.items()}
        return self["_default"].unnormalize(x)

    def forward(
        self,
        x: Mapping[str, torch.Tensor] | torch.Tensor,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """Alias for :meth:`normalize`."""
        return self.normalize(x)


def _as_tensor(x: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Convert numpy arrays to tensors while preserving existing tensors."""
    if isinstance(x, torch.Tensor):
        return x
    return torch.from_numpy(x)
