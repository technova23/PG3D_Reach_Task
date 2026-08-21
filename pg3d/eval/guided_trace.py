from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pg3d.world_model import ActionChunk

ConditioningEntry = Mapping[str, np.ndarray | bool | float]


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def action_sha256(action_chunk: ActionChunk | np.ndarray) -> str:
    actions = action_chunk.actions if isinstance(action_chunk, ActionChunk) else action_chunk
    return array_sha256(np.asarray(actions))


def conditioning_window_sha256(window: Iterable[ConditioningEntry]) -> str:
    digest = hashlib.sha256()
    for observation_index, entry in enumerate(window):
        for key in sorted(entry):
            value = np.asarray(entry[key])
            digest.update(f"{observation_index}:{key}:".encode())
            digest.update(array_sha256(value).encode("ascii"))
    return digest.hexdigest()


@dataclass(frozen=True)
class ConditioningBundleStore:
    root: Path

    def store(self, window: list[ConditioningEntry]) -> tuple[str, Path]:
        content_hash = conditioning_window_sha256(window)
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self.root / f"{content_hash}.npz"
        if destination.exists():
            loaded = self.load(content_hash)
            if conditioning_window_sha256(loaded) != content_hash:
                raise ValueError(f"corrupt conditioning bundle: {destination}")
            return content_hash, destination
        arrays: dict[str, np.ndarray] = {}
        manifest: list[list[dict[str, str]]] = []
        for observation_index, entry in enumerate(window):
            fields = []
            for field_index, key in enumerate(sorted(entry)):
                storage_key = f"o{observation_index:03d}_f{field_index:03d}"
                arrays[storage_key] = np.asarray(entry[key])
                fields.append({"name": key, "storage_key": storage_key})
            manifest.append(fields)
        arrays["__manifest__"] = np.frombuffer(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8"),
            dtype=np.uint8,
        )
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.npz")
        np.savez_compressed(temporary, **arrays)
        temporary.replace(destination)
        return content_hash, destination

    def load(self, content_hash: str) -> list[dict[str, np.ndarray]]:
        if len(content_hash) != 64:
            raise ValueError("conditioning hash must be a SHA-256 hex digest")
        path = self.root / f"{content_hash}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"missing conditioning bundle: {path}")
        try:
            with np.load(path, allow_pickle=False) as bundle:
                manifest = json.loads(bytes(bundle["__manifest__"]).decode("utf-8"))
                window = [
                    {
                        str(field["name"]): np.asarray(bundle[str(field["storage_key"])]).copy()
                        for field in fields
                    }
                    for fields in manifest
                ]
        except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"corrupt conditioning bundle: {path}") from exc
        actual_hash = conditioning_window_sha256(window)
        if actual_hash != content_hash:
            raise ValueError(
                f"conditioning bundle hash mismatch: {actual_hash} != {content_hash}"
            )
        return window


class GuidedProposalTraceWriter:
    """Append-only proposal trace with content-addressed DP3 conditioning windows."""

    def __init__(
        self,
        root: Path,
        *,
        run_metadata: Mapping[str, Any],
        allow_existing: bool = False,
    ) -> None:
        self.root = root
        self.records_path = root / "proposals.jsonl"
        self.conditioning = ConditioningBundleStore(root / "conditioning")
        if self.records_path.exists() and not allow_existing:
            raise FileExistsError(f"proposal trace already exists: {self.records_path}")
        root.mkdir(parents=True, exist_ok=True)
        self.run_metadata = dict(run_metadata)
        if not self.records_path.exists():
            self.records_path.touch()

    def store_conditioning(self, window: list[ConditioningEntry]) -> tuple[str, Path]:
        return self.conditioning.store(window)

    def store_guidance_geometry(self, arrays: Mapping[str, np.ndarray]) -> tuple[str, Path]:
        root = self.root / "guidance_geometry"
        root.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        normalized = {}
        for key in sorted(arrays):
            value = np.asarray(arrays[key])
            normalized[key] = value
            digest.update(key.encode("utf-8"))
            digest.update(array_sha256(value).encode("ascii"))
        content_hash = digest.hexdigest()
        destination = root / f"{content_hash}.npz"
        if not destination.exists():
            temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.npz")
            np.savez_compressed(temporary, **normalized)
            temporary.replace(destination)
        return content_hash, destination

    def record(
        self,
        *,
        proposal_id: str,
        purpose: str,
        replan: int,
        depth: int,
        parent_id: str | None,
        ancestry: Iterable[str],
        branch_index: int,
        action_chunk: ActionChunk,
        conditioning_hash: str,
        itps_config: Mapping[str, Any],
        noise_lineage: Mapping[str, Any],
        score_data: Mapping[str, Any] | None,
        retained: bool | None,
        executed_lineage: bool = False,
        guidance_geometry_hash: str | None = None,
        guidance_geometry_bundle: str | None = None,
        guidance_target: str | None = None,
    ) -> dict[str, Any]:
        record = {
            "schema_version": "pg3d.guided_proposal.v1",
            "proposal_id": proposal_id,
            "purpose": purpose,
            "replan": int(replan),
            "depth": int(depth),
            "parent_id": parent_id,
            "ancestry": list(ancestry),
            "branch_index": int(branch_index),
            "conditioning_hash": conditioning_hash,
            "conditioning_bundle": str(self.conditioning.root / f"{conditioning_hash}.npz"),
            "raw_action_chunk": np.asarray(action_chunk.actions).tolist(),
            "action_dtype": np.asarray(action_chunk.actions).dtype.str,
            "action_shape": list(np.asarray(action_chunk.actions).shape),
            "action_sha256": action_sha256(action_chunk),
            "action_mode": action_chunk.action_mode,
            "action_dt": float(action_chunk.dt),
            "itps_config": dict(itps_config),
            "noise_lineage": dict(noise_lineage),
            "score_data": None if score_data is None else dict(score_data),
            "retained": retained,
            "executed_lineage": bool(executed_lineage),
            "guidance_geometry_hash": guidance_geometry_hash,
            "guidance_geometry_bundle": guidance_geometry_bundle,
            "guidance_target": guidance_target,
            **self.run_metadata,
        }
        with self.records_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        return record


def load_proposal_record(path: Path, proposal_id: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    matches = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("proposal_id") == proposal_id:
            matches.append(record)
    if not matches:
        raise KeyError(f"proposal not found: {proposal_id}")
    if len(matches) != 1:
        raise ValueError(f"proposal id is not unique: {proposal_id}")
    return matches[0]


def verify_replayed_action(
    record: Mapping[str, Any],
    replayed: np.ndarray,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> dict[str, Any]:
    expected = np.asarray(record["raw_action_chunk"], dtype=np.dtype(record["action_dtype"]))
    actual = np.asarray(replayed, dtype=expected.dtype)
    exact_hash = action_sha256(actual) == record["action_sha256"]
    return {
        "exact_action_hash": exact_hash,
        "allclose": bool(
            expected.shape == actual.shape and np.allclose(expected, actual, rtol=rtol, atol=atol)
        ),
        "rtol": float(rtol),
        "atol": float(atol),
        "maximum_absolute_error": (
            float(np.max(np.abs(expected - actual)))
            if expected.shape == actual.shape and expected.size
            else None
        ),
    }
