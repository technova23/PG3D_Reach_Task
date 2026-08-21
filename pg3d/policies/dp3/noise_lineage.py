from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


def stable_lineage_seed(*identity: object) -> int:
    """Derive a portable positive torch seed from a semantic identity."""
    payload = "\x1f".join(str(item) for item in identity).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value & ((1 << 63) - 1)


@dataclass(frozen=True)
class ITPSNoiseDraw:
    purpose: str
    seed: int
    diffusion_index: int | None = None
    diffusion_timestep: int | None = None
    inner_index: int | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "purpose": self.purpose,
            "seed": int(self.seed),
            "diffusion_index": self.diffusion_index,
            "diffusion_timestep": self.diffusion_timestep,
            "inner_index": self.inner_index,
        }


@dataclass(frozen=True)
class ITPSNoiseLineage:
    """Every stochastic draw needed to regenerate one guided ITPS proposal."""

    candidate_seed: int
    root_identity: str
    initial_noise: ITPSNoiseDraw
    inner_renoising: tuple[ITPSNoiseDraw, ...]

    @classmethod
    def derive(
        cls,
        *,
        candidate_seed: int,
        diffusion_timesteps: Iterable[int],
        inner_steps: int,
        root_identity: str = "guided_proposal",
    ) -> ITPSNoiseLineage:
        if candidate_seed < 0:
            raise ValueError("candidate_seed must be non-negative")
        if inner_steps <= 0:
            raise ValueError("inner_steps must be positive")
        timesteps = tuple(int(value) for value in diffusion_timesteps)
        initial = ITPSNoiseDraw(
            purpose="initial_diffusion_noise",
            seed=stable_lineage_seed(root_identity, candidate_seed, "initial"),
        )
        draws = tuple(
            ITPSNoiseDraw(
                purpose="inner_mcmc_renoising",
                seed=stable_lineage_seed(
                    root_identity,
                    candidate_seed,
                    "renoise",
                    diffusion_index,
                    timestep,
                    inner_index,
                ),
                diffusion_index=diffusion_index,
                diffusion_timestep=timestep,
                inner_index=inner_index,
            )
            for diffusion_index, timestep in enumerate(timesteps)
            for inner_index in range(inner_steps - 1)
        )
        return cls(
            candidate_seed=int(candidate_seed),
            root_identity=root_identity,
            initial_noise=initial,
            inner_renoising=draws,
        )

    def inner_draw(
        self,
        *,
        diffusion_index: int,
        diffusion_timestep: int,
        inner_index: int,
    ) -> ITPSNoiseDraw:
        for draw in self.inner_renoising:
            if (
                draw.diffusion_index == diffusion_index
                and draw.diffusion_timestep == diffusion_timestep
                and draw.inner_index == inner_index
            ):
                return draw
        raise ValueError(
            "noise lineage has no inner draw for "
            f"diffusion_index={diffusion_index}, timestep={diffusion_timestep}, "
            f"inner_index={inner_index}"
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": "pg3d.itps_noise_lineage.v1",
            "candidate_seed": int(self.candidate_seed),
            "root_identity": self.root_identity,
            "initial_noise": self.initial_noise.to_json(),
            "inner_renoising": [draw.to_json() for draw in self.inner_renoising],
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> ITPSNoiseLineage:
        if payload.get("schema_version") != "pg3d.itps_noise_lineage.v1":
            raise ValueError("unsupported ITPS noise-lineage schema")

        def draw(raw: dict[str, Any]) -> ITPSNoiseDraw:
            return ITPSNoiseDraw(
                purpose=str(raw["purpose"]),
                seed=int(raw["seed"]),
                diffusion_index=(
                    None if raw.get("diffusion_index") is None else int(raw["diffusion_index"])
                ),
                diffusion_timestep=(
                    None
                    if raw.get("diffusion_timestep") is None
                    else int(raw["diffusion_timestep"])
                ),
                inner_index=(
                    None if raw.get("inner_index") is None else int(raw["inner_index"])
                ),
            )

        return cls(
            candidate_seed=int(payload["candidate_seed"]),
            root_identity=str(payload["root_identity"]),
            initial_noise=draw(dict(payload["initial_noise"])),
            inner_renoising=tuple(draw(dict(raw)) for raw in payload["inner_renoising"]),
        )
