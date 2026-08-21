from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GuidedFixtureEpisode:
    output_index: int
    source_output_index: int
    dataset_episode_index: int
    simulator_seed: int
    policy_seed: int
    constraint_file: Path
    constraint_sha256: str


@dataclass(frozen=True)
class GuidedFixtureManifest:
    path: Path
    name: str
    classification: str
    dataset: Path
    checkpoint: Path
    episodes: tuple[GuidedFixtureEpisode, ...]

    def episode(self, output_index: int) -> GuidedFixtureEpisode:
        for episode in self.episodes:
            if episode.output_index == output_index:
                return episode
        raise KeyError(f"fixture has no output episode {output_index}")

    def to_json(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "name": self.name,
            "classification": self.classification,
            "dataset": str(self.dataset),
            "checkpoint": str(self.checkpoint),
            "episodes": [
                {
                    "output_index": item.output_index,
                    "source_output_index": item.source_output_index,
                    "dataset_episode_index": item.dataset_episode_index,
                    "simulator_seed": item.simulator_seed,
                    "policy_seed": item.policy_seed,
                    "constraint_file": str(item.constraint_file),
                    "constraint_sha256": item.constraint_sha256,
                }
                for item in self.episodes
            ],
        }


def load_guided_fixture_manifest(path: Path, *, repo_root: Path) -> GuidedFixtureManifest:
    resolved = path.resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "pg3d.itps_beam_fixture.v1":
        raise ValueError("unsupported ITPS-beam fixture schema")
    if payload.get("classification") != "development_only":
        raise ValueError("the current ITPS-beam MVP accepts only development fixtures")
    raw_episodes = list(payload.get("episodes", []))
    if not raw_episodes:
        raise ValueError("fixture manifest must contain episodes")
    episodes = []
    for expected_output, raw_value in enumerate(raw_episodes):
        raw = dict(raw_value)
        if int(raw["output_index"]) != expected_output:
            raise ValueError("fixture output indices must be contiguous and ordered")
        constraint_path = _resolve_path(str(raw["constraint_file"]), repo_root=repo_root)
        expected_hash = str(raw["constraint_sha256"])
        actual_hash = hashlib.sha256(constraint_path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(
                f"fixture constraint hash mismatch for output {expected_output}: "
                f"{actual_hash} != {expected_hash}"
            )
        episodes.append(
            GuidedFixtureEpisode(
                output_index=expected_output,
                source_output_index=int(raw["source_output_index"]),
                dataset_episode_index=int(raw["dataset_episode_index"]),
                simulator_seed=int(raw["simulator_seed"]),
                policy_seed=int(raw["policy_seed"]),
                constraint_file=constraint_path,
                constraint_sha256=expected_hash,
            )
        )
    return GuidedFixtureManifest(
        path=resolved,
        name=str(payload["name"]),
        classification=str(payload["classification"]),
        dataset=_resolve_path(str(payload["dataset"]), repo_root=repo_root),
        checkpoint=_resolve_path(str(payload["checkpoint"]), repo_root=repo_root),
        episodes=tuple(episodes),
    )


def _resolve_path(value: str, *, repo_root: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else repo_root / path).resolve()
