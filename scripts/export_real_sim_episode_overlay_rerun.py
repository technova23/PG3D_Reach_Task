#!/usr/bin/env python
"""Overlay full real and sim episodes from a mixed reach Zarr in one Rerun file.

The exporter reads stored arrays directly; it does not replay simulation, infer
masks, transform coordinates, or apply an additional crop. Episodes of unequal
length are synchronized by normalized progress from 0% to 100%.
"""

from __future__ import annotations

import argparse
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import zarr

REAL_COLOR = [30, 144, 255]
SIM_COLOR = [255, 70, 190]
REAL_GOAL_COLOR = [0, 210, 255]
SIM_GOAL_COLOR = [255, 170, 0]


def episode_ranges(episode_ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ends = np.asarray(episode_ends, dtype=np.int64)
    if ends.ndim != 1 or ends.size == 0 or np.any(np.diff(ends) <= 0):
        raise ValueError("episode_ends must be a non-empty strictly increasing 1D array")
    starts = np.concatenate([np.asarray([0], dtype=np.int64), ends[:-1]])
    return starts, ends


def synchronized_indices(real_length: int, sim_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Map unequal episodes onto a shared normalized-progress timeline."""
    if real_length <= 0 or sim_length <= 0:
        raise ValueError("episode lengths must be positive")
    count = max(real_length, sim_length)
    progress = np.linspace(0.0, 1.0, count, dtype=np.float64)
    real = np.rint(progress * (real_length - 1)).astype(np.int64)
    sim = np.rint(progress * (sim_length - 1)).astype(np.int64)
    return real, sim


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metadata = json.loads((args.dataset / "metadata.json").read_text(encoding="utf-8"))
    root = zarr.open_group(str(args.dataset), mode="r")
    data = root["data"]
    _validate_schema(data)
    starts, ends = episode_ranges(np.asarray(root["meta"]["episode_ends"][:]))
    real_range, sim_range = _provenance_episode_ranges(metadata)
    _validate_episode(args.real_episode, real_range, "real")
    _validate_episode(args.sim_episode, sim_range, "sim")

    real = _load_episode(data, starts, ends, args.real_episode)
    sim = _load_episode(data, starts, ends, args.sim_episode)
    real_indices, sim_indices = synchronized_indices(len(real["point_cloud"]), len(sim["point_cloud"]))

    _write_overlay(
        args.output,
        real=real,
        sim=sim,
        real_indices=real_indices,
        sim_indices=sim_indices,
        point_radius=args.point_radius,
    )
    summary = {
        "dataset": str(args.dataset),
        "output": str(args.output),
        "coordinate_handling": "raw stored coordinates; no transform or additional crop",
        "real_episode": args.real_episode,
        "real_rows": [int(starts[args.real_episode]), int(ends[args.real_episode])],
        "real_frames": int(len(real["point_cloud"])),
        "sim_episode": args.sim_episode,
        "sim_rows": [int(starts[args.sim_episode]), int(ends[args.sim_episode])],
        "sim_frames": int(len(sim["point_cloud"])),
        "synchronized_frames": int(len(real_indices)),
        "synchronization": "normalized episode progress with nearest frame",
        "writer_rerun_sdk": _rerun_sdk_version(),
        "colors": {
            "real_point_cloud_and_tcp": REAL_COLOR,
            "sim_point_cloud_and_tcp": SIM_COLOR,
            "real_goal": REAL_GOAL_COLOR,
            "sim_goal": SIM_GOAL_COLOR,
        },
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--real-episode", type=int, default=689)
    parser.add_argument("--sim-episode", type=int, default=3702)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--point-radius", type=float, default=0.003)
    args = parser.parse_args(argv)
    if args.point_radius <= 0:
        parser.error("--point-radius must be positive")
    return args


def _validate_schema(data: Any) -> None:
    required = ("point_cloud", "target_position", "tcp_pose")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"mixed dataset is missing required arrays: {missing}")


def _rerun_sdk_version() -> str:
    try:
        return version("rerun-sdk")
    except PackageNotFoundError:
        return "unknown"


def _provenance_episode_ranges(
    metadata: dict[str, Any],
) -> tuple[tuple[int, int], tuple[int, int]]:
    sources = {source.get("name"): source for source in metadata.get("sources", [])}
    if "real" not in sources or "sim" not in sources:
        raise ValueError("metadata.json must contain real and sim provenance sources")
    return tuple(sources["real"]["episode_range"]), tuple(sources["sim"]["episode_range"])


def _validate_episode(episode: int, bounds: tuple[int, int], label: str) -> None:
    if not bounds[0] <= episode < bounds[1]:
        raise ValueError(f"{label} episode {episode} is outside provenance range {bounds}")


def _load_episode(
    data: Any,
    starts: np.ndarray,
    ends: np.ndarray,
    episode: int,
) -> dict[str, np.ndarray]:
    start, end = int(starts[episode]), int(ends[episode])
    point_cloud = np.asarray(data["point_cloud"][start:end], dtype=np.float32)
    target_position = np.asarray(data["target_position"][start:end], dtype=np.float32)
    tcp_pose = np.asarray(data["tcp_pose"][start:end], dtype=np.float32)
    if point_cloud.ndim != 3 or point_cloud.shape[-1] != 3:
        raise ValueError(f"point_cloud must have shape [T,N,3], got {point_cloud.shape}")
    if not np.all(np.isfinite(point_cloud)):
        raise ValueError(f"episode {episode} contains non-finite point-cloud values")
    return {
        "point_cloud": point_cloud,
        "target_position": target_position,
        "tcp_position": tcp_pose[:, :3],
    }


def _write_overlay(
    output: Path,
    *,
    real: dict[str, np.ndarray],
    sim: dict[str, np.ndarray],
    real_indices: np.ndarray,
    sim_indices: np.ndarray,
    point_radius: float,
) -> None:
    try:
        import rerun as rr
    except Exception as exc:
        raise RuntimeError("Rerun export requires the viz dependencies") from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    rr.init("pg3d_real_sim_episode_overlay", spawn=False)
    rr.save(str(output))
    rr.log(
        "comparison/real/goal",
        rr.Points3D(real["target_position"][0:1], colors=REAL_GOAL_COLOR, radii=0.014),
        timeless=True,
    )
    rr.log(
        "comparison/sim/goal",
        rr.Points3D(sim["target_position"][0:1], colors=SIM_GOAL_COLOR, radii=0.014),
        timeless=True,
    )

    denominator = max(len(real_indices) - 1, 1)
    for timeline_step, (real_index, sim_index) in enumerate(zip(real_indices, sim_indices)):
        rr.set_time_sequence("frame", timeline_step)
        rr.set_time_seconds("progress", timeline_step / denominator)
        rr.log(
            "comparison/real/point_cloud",
            rr.Points3D(
                real["point_cloud"][real_index], colors=REAL_COLOR, radii=point_radius
            ),
        )
        rr.log(
            "comparison/sim/point_cloud",
            rr.Points3D(sim["point_cloud"][sim_index], colors=SIM_COLOR, radii=point_radius),
        )
        rr.log(
            "comparison/real/tcp/current",
            rr.Points3D(
                real["tcp_position"][real_index : real_index + 1],
                colors=REAL_COLOR,
                radii=0.009,
            ),
        )
        rr.log(
            "comparison/sim/tcp/current",
            rr.Points3D(
                sim["tcp_position"][sim_index : sim_index + 1],
                colors=SIM_COLOR,
                radii=0.009,
            ),
        )
        _log_trail(rr, "comparison/real/tcp/trail", real["tcp_position"][: real_index + 1], REAL_COLOR)
        _log_trail(rr, "comparison/sim/tcp/trail", sim["tcp_position"][: sim_index + 1], SIM_COLOR)
        rr.log("comparison/frame_indices/real", rr.Scalar(int(real_index)))
        rr.log("comparison/frame_indices/sim", rr.Scalar(int(sim_index)))
    rr.disconnect()


def _log_trail(rr: Any, entity: str, points: np.ndarray, color: list[int]) -> None:
    if len(points) >= 2:
        rr.log(entity, rr.LineStrips3D([points], colors=color, radii=0.0025))


if __name__ == "__main__":
    raise SystemExit(main())
