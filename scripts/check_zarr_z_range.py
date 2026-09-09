#!/usr/bin/env python
"""Measure the ACTUAL Z range present in a reach zarr dataset's arrays --
ground truth from the real generated data, not just what metadata.json
records as the configured sampling bounds (config vs. what was actually
produced can differ: rejection sampling, an older/edited config, etc.).

Reads directly via the `zarr` package -- no pg3d imports needed, so this
runs even in a minimal environment.

Usage:
    python scripts/check_zarr_z_range.py /path/to/dataset.zarr
"""

from __future__ import annotations

import sys

import numpy as np
import zarr


def _report(name: str, arr: np.ndarray, z_col: int) -> None:
    if arr.ndim < 2 or arr.shape[-1] <= z_col:
        print(f"  {name}: shape {arr.shape} -- no column index {z_col}, skipping")
        return
    z = arr[..., z_col].reshape(-1)
    print(
        f"  {name} Z (col {z_col}): "
        f"min={z.min():.4f}  max={z.max():.4f}  mean={z.mean():.4f}  std={z.std():.4f}  "
        f"n={z.size}"
    )
    # A handful of representative points (min, 25/50/75th pct, max) -- cheap
    # sanity check for whether the range is genuinely spread or one outlier.
    pct = np.percentile(z, [0, 25, 50, 75, 100])
    print(f"    percentiles [0,25,50,75,100]: {np.round(pct, 4).tolist()}")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python scripts/check_zarr_z_range.py /path/to/dataset.zarr", file=sys.stderr)
        return 2

    root = zarr.open_group(argv[0], mode="r")
    data = root["data"]
    keys = list(data.keys())
    print(f"dataset: {argv[0]}")
    print(f"arrays in data/: {keys}\n")

    episode_ends = root["meta"]["episode_ends"][:]
    print(f"num_episodes: {len(episode_ends)}  num_steps: {int(episode_ends[-1]) if len(episode_ends) else 0}\n")

    # target_position: [N, 3] (x, y, z) -- the actual per-step goal-conditioning
    # value the policy was trained against. This is the one that matters most
    # for "what Z range did the checkpoint see as a goal".
    if "target_position" in data:
        print("target_position (goal the policy conditions on):")
        _report("target_position", data["target_position"][:], z_col=2)
        print()

    # tcp_pose: [N, 7] (x,y,z, qw,qx,qy,qz) -- where the ARM ACTUALLY WAS across
    # the recorded trajectories. Different question from target_position: this
    # is "what heights did the robot itself occupy", useful if you also want to
    # know the trajectory envelope, not just the goal-sampling envelope.
    if "tcp_pose" in data:
        print("tcp_pose (robot's actual recorded trajectory):")
        _report("tcp_pose", data["tcp_pose"][:], z_col=2)
        print()

    # state: often [N, 7] joint positions, not Cartesian -- Z isn't meaningful
    # here directly, so skip it unless explicitly asked; listed for visibility.
    if "target_position" not in data and "tcp_pose" not in data:
        print(
            "Neither target_position nor tcp_pose found in data/ -- this "
            "dataset may use a different schema. Full key list printed above; "
            "inspect one of those arrays' shape/columns directly instead."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
