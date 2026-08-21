#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pg3d.constraints import AvoidProjection, AvoidRegion, constraints_from_json
from pg3d.constraints.torch_geometry import avoidance_energy
from pg3d.eval import ConditioningBundleStore, load_proposal_record, verify_replayed_action
from pg3d.policies.dp3 import ITPSNoiseLineage
from pg3d.policies.dp3.checkpoint import load_reach_policy_from_checkpoint
from pg3d.world_model.panda_collision import (
    DifferentiablePandaCollisionPoints,
    PandaCollisionPointTemplate,
)
from scripts.eval_constrained_reach import (
    _itps_eef_path,
    _itps_robot_points,
    _repeat_obs_window_to_torch,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay one stored guided ITPS proposal.")
    parser.add_argument("trace", type=Path, help="proposals.jsonl")
    parser.add_argument("proposal_id")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args(argv)

    record = load_proposal_record(args.trace, args.proposal_id)
    checkpoint = Path(record["checkpoint_id"])
    actual_checkpoint_hash = _file_sha256(checkpoint)
    if actual_checkpoint_hash != record["checkpoint_sha256"]:
        raise ValueError("checkpoint hash does not match the proposal trace")
    conditioning_store = ConditioningBundleStore(Path(record["conditioning_bundle"]).parent)
    obs_window = conditioning_store.load(str(record["conditioning_hash"]))
    geometry = _load_geometry(record)
    device = torch.device(args.device)
    policy = load_reach_policy_from_checkpoint(
        checkpoint,
        device=device,
        prefer_ema=record.get("checkpoint_model", "ema") == "ema",
    )
    config = dict(record["itps_config"])
    constraints = [
        constraint
        for constraint in constraints_from_json(record["constraints"])
        if isinstance(constraint, (AvoidRegion, AvoidProjection))
    ]
    target = str(record["guidance_target"])
    world_from_base = torch.as_tensor(
        geometry["world_from_base"], device=device, dtype=policy.dtype
    )
    collision_model = (
        _collision_model(geometry, device=device, dtype=policy.dtype)
        if target == "robot"
        else None
    )

    def guidance_fn(trajectory: torch.Tensor) -> torch.Tensor:
        if target == "robot":
            if collision_model is None:
                raise RuntimeError("robot-target replay is missing collision geometry")
            points = _itps_robot_points(policy, trajectory, world_from_base, collision_model)
        else:
            points = _itps_eef_path(policy, trajectory, world_from_base)
        return avoidance_energy(
            points,
            constraints,
            target=target,  # type: ignore[arg-type]
            mode=str(config["energy"]),  # type: ignore[arg-type]
            temperature=float(config["barrier_temperature"]),
        )

    obs_batch = _repeat_obs_window_to_torch(
        obs_window,  # type: ignore[arg-type]
        k=1,
        device=device,
        goal_marker_points=int(getattr(policy, "goal_marker_points", 0)),
        goal_marker_radius=float(getattr(policy, "goal_marker_radius", 0.01)),
    )
    output = policy.predict_action_itps(
        obs_batch,
        noise_lineage=ITPSNoiseLineage.from_json(dict(record["noise_lineage"])),
        guidance_fn=guidance_fn,
        guide_ratio=float(config["guide_ratio"]),
        mcmc_steps=int(config["mcmc_steps"]),
    )
    replayed = output["action"][0].detach().cpu().numpy().astype(np.float32, copy=True)
    result = verify_replayed_action(record, replayed, rtol=1e-5, atol=1e-6)
    same_stack = (
        dict(record.get("dependency_versions", {})) == _dependency_versions()
        and record.get("git_revision") == _git_revision()
    )
    result.update({"proposal_id": args.proposal_id, "same_stack": same_stack})
    print(json.dumps(result, indent=2, sort_keys=True))
    if same_stack:
        return 0 if result["exact_action_hash"] else 1
    return 0 if result["allclose"] else 1


def _load_geometry(record: dict[str, Any]) -> dict[str, np.ndarray]:
    path = Path(record["guidance_geometry_bundle"])
    if not path.is_file():
        raise FileNotFoundError(f"missing guidance geometry bundle: {path}")
    with np.load(path, allow_pickle=False) as bundle:
        geometry = {key: np.asarray(bundle[key]).copy() for key in bundle.files}
    digest = hashlib.sha256()
    from pg3d.eval.guided_trace import array_sha256

    for key in sorted(geometry):
        digest.update(key.encode("utf-8"))
        digest.update(array_sha256(geometry[key]).encode("ascii"))
    if digest.hexdigest() != record["guidance_geometry_hash"]:
        raise ValueError("guidance geometry hash does not match the proposal trace")
    return geometry


def _collision_model(
    geometry: dict[str, np.ndarray],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> DifferentiablePandaCollisionPoints:
    if "local_points" not in geometry:
        raise ValueError("robot-target replay requires stored collision-point geometry")
    template = PandaCollisionPointTemplate(
        local_points=geometry["local_points"],
        link_indices=geometry["link_indices"],
        link_counts=tuple(int(value) for value in geometry["link_counts"]),
        sample_seed=int(geometry["sample_seed"]),
    )
    return DifferentiablePandaCollisionPoints(
        template,
        gripper_open=float(geometry["gripper_open"]),
    ).to(device=device, dtype=dtype)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dependency_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0], "torch": torch.__version__}
    for distribution in ("numpy", "diffusers", "zarr", "mani-skill"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def _git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


if __name__ == "__main__":
    raise SystemExit(main())
