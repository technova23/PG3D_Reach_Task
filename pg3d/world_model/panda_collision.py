from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from pg3d.world_model.panda_fk import (
    PANDA_MOVABLE_COLLISION_LINKS,
    panda_link_transforms,
)


@dataclass(frozen=True)
class PandaCollisionPointTemplate:
    """Deterministic collision-surface points expressed in their owning link frames."""

    local_points: np.ndarray
    link_indices: np.ndarray
    link_counts: tuple[int, ...]
    sample_seed: int

    def __post_init__(self) -> None:
        points = np.asarray(self.local_points, dtype=np.float32)
        indices = np.asarray(self.link_indices, dtype=np.int64)
        link_count = len(PANDA_MOVABLE_COLLISION_LINKS)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"local_points must have shape [N, 3], got {points.shape}")
        if indices.shape != (points.shape[0],):
            raise ValueError(
                f"link_indices must have shape [{points.shape[0]}], got {indices.shape}"
            )
        if len(self.link_counts) != link_count:
            raise ValueError(f"link_counts must contain {link_count} entries")
        if any(count <= 0 for count in self.link_counts):
            raise ValueError("every movable collision link must have at least one point")
        if sum(self.link_counts) != points.shape[0]:
            raise ValueError("link_counts must sum to the number of local points")
        if indices.size and (int(indices.min()) < 0 or int(indices.max()) >= link_count):
            raise ValueError("link_indices contain an unknown Panda collision link")
        actual_counts = tuple(int(value) for value in np.bincount(indices, minlength=link_count))
        if actual_counts != tuple(self.link_counts):
            raise ValueError("link_counts must match link_indices")
        if not np.isfinite(points).all():
            raise ValueError("local_points must be finite")
        if self.sample_seed < 0:
            raise ValueError("sample_seed must be non-negative")
        object.__setattr__(self, "local_points", points.copy())
        object.__setattr__(self, "link_indices", indices.copy())
        object.__setattr__(self, "link_counts", tuple(int(value) for value in self.link_counts))

    @property
    def point_count(self) -> int:
        return int(self.local_points.shape[0])

    def allocation(self) -> dict[str, int]:
        return dict(zip(PANDA_MOVABLE_COLLISION_LINKS, self.link_counts, strict=True))


class DifferentiablePandaCollisionPoints(nn.Module):
    """Transform a fixed Panda collision template with differentiable batched FK."""

    def __init__(
        self,
        template: PandaCollisionPointTemplate,
        *,
        gripper_open: float = 0.04,
    ) -> None:
        super().__init__()
        if not np.isfinite(gripper_open) or not 0.0 <= gripper_open <= 0.04:
            raise ValueError("gripper_open must be finite and within [0, 0.04]")
        self.gripper_open = float(gripper_open)
        self.link_counts = template.link_counts
        self.sample_seed = template.sample_seed
        self.register_buffer(
            "local_points",
            torch.as_tensor(template.local_points, dtype=torch.float32),
        )
        self.register_buffer(
            "link_indices",
            torch.as_tensor(template.link_indices, dtype=torch.long),
        )

    @property
    def point_count(self) -> int:
        return int(self.local_points.shape[0])

    def allocation(self) -> dict[str, int]:
        return dict(zip(PANDA_MOVABLE_COLLISION_LINKS, self.link_counts, strict=True))

    def forward(
        self,
        q: torch.Tensor,
        world_from_base: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return world-frame collision points with shape ``[..., N, 3]``."""
        transforms = panda_link_transforms(
            q,
            world_from_base,
            gripper_open=self.gripper_open,
        )
        indices = self.link_indices.to(device=q.device)
        point_transforms = torch.index_select(transforms, dim=-3, index=indices)
        local_points = self.local_points.to(device=q.device, dtype=q.dtype)
        ones = torch.ones((local_points.shape[0], 1), device=q.device, dtype=q.dtype)
        homogeneous = torch.cat((local_points, ones), dim=-1)
        world = torch.matmul(point_transforms, homogeneous[..., None]).squeeze(-1)
        return world[..., :3]
