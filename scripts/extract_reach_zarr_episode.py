#!/usr/bin/env python
"""Extract one trajectory from a reach Zarr by episode index.

This reads ``meta/episode_ends`` to resolve the episode slice, then exports the
selected episode as either a compressed ``.npz`` file or a tiny one-episode
Zarr. The default output is ``episode_<index>.npz``.

Example:
    python scripts/extract_reach_zarr_episode.py \
        --dataset artifacts/pg3d_reach_balanced.zarr \
        --episode-index 12 \
        --out artifacts/episode_12.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import zarr


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = zarr.open_group(str(args.dataset), mode="r")
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    if args.episode_index < 0 or args.episode_index >= len(episode_ends):
        raise SystemExit(
            f"--episode-index {args.episode_index} is outside dataset range "
            f"[0, {len(episode_ends) - 1}]"
        )

    start = 0 if args.episode_index == 0 else int(episode_ends[args.episode_index - 1])
    end = int(episode_ends[args.episode_index])
    if end <= start:
        raise SystemExit(f"episode {args.episode_index} is empty")

    episode = _read_episode(root, start=start, end=end)
    out_path = args.out or args.dataset.parent / f"episode_{args.episode_index:03d}.npz"
    if args.format == "npz":
        np.savez_compressed(out_path, **episode)
    else:
        _write_episode_zarr(out_path, episode, overwrite=args.overwrite)

    print(
        f"saved episode {args.episode_index} rows={end - start} "
        f"start={start} end={end} out={out_path}"
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract one reach trajectory from a Zarr.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument(
        "--format",
        choices=["npz", "zarr"],
        default="npz",
        help="export format for the extracted episode",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite --out when exporting to Zarr",
    )
    args = parser.parse_args(argv)
    if args.episode_index < 0:
        raise ValueError("--episode-index must be non-negative")
    return args


def _read_episode(root: Any, *, start: int, end: int) -> dict[str, np.ndarray]:
    data = root["data"]
    keys = [
        "state",
        "action",
        "sim_action",
        "point_cloud",
        "robot_mask",
        "point_valid_mask",
        "target_position",
        "tcp_pose",
        "success",
    ]
    episode: dict[str, np.ndarray] = {"episode_start": np.asarray(start), "episode_end": np.asarray(end)}
    for key in keys:
        if key not in data:
            continue
        episode[key] = np.asarray(data[key][start:end])
    return episode


def _write_episode_zarr(out_path: Path, episode: dict[str, np.ndarray], *, overwrite: bool) -> None:
    if out_path.exists() and not overwrite:
        raise SystemExit(f"{out_path} already exists; pass --overwrite to replace it")
    out_root = zarr.open_group(str(out_path), mode="w" if overwrite else "w-")
    data = out_root.create_group("data")
    meta = out_root.create_group("meta")
    for key, value in episode.items():
        if key in {"episode_start", "episode_end"}:
            continue
        chunks = (min(max(1, value.shape[0]), 1024),) + value.shape[1:]
        data.array(name=key, data=value, chunks=chunks)
    meta.array(name="episode_ends", data=np.asarray([episode["episode_end"]], dtype=np.int64))
    out_root.attrs["source"] = "extract_reach_zarr_episode"


if __name__ == "__main__":
    raise SystemExit(main())
