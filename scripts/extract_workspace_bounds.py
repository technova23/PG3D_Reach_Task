#!/usr/bin/env python
"""Extract the workspace bounds a reach dataset (and its checkpoint) was
generated/trained against, straight from metadata.json.

Field names here match dataset_generation/write_maniskill_reach_dataset.py's
own metadata dict exactly (metadata["start_sampling"]["reach_workspace_bounds"],
etc.) -- not guessed. Every lookup is defensive (.get with a None/"missing"
fallback) since not every dataset in this repo was necessarily written by
that exact function (older/other writers may have a different shape), and a
silently-wrong bound is worse than an honest "not present in this file".

Usage:
    python scripts/extract_workspace_bounds.py /path/to/dataset.zarr
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _fmt_bounds(bounds) -> str:
    """[[xlo,xhi],[ylo,yhi],[zlo,zhi]] -> one readable line, or a not-present note."""
    if bounds is None:
        return "(not present in metadata.json)"
    try:
        axes = "XYZ"
        return "  ".join(
            f"{axis}=[{float(lo):+.3f}, {float(hi):+.3f}]"
            for axis, (lo, hi) in zip(axes, bounds, strict=True)
        )
    except Exception as exc:  # noqa: BLE001 -- malformed value, show it raw instead of crashing
        return f"(couldn't parse as a [3,2] box: {bounds!r} -- {exc})"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python scripts/extract_workspace_bounds.py /path/to/dataset.zarr", file=sys.stderr)
        return 2

    dataset_path = Path(argv[0])
    metadata_path = dataset_path / "metadata.json"
    if not metadata_path.exists():
        print(f"no metadata.json found at {metadata_path}", file=sys.stderr)
        return 1

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    print(f"dataset       : {dataset_path}")
    print(f"env_id        : {metadata.get('env_id', '(missing)')}")
    print(f"action_mode   : {metadata.get('action_mode', '(missing)')}")

    env_kwargs = metadata.get("env_kwargs", {})
    print("\nenv_kwargs (env construction args -- goal_center/goal_half_extents live here):")
    print(json.dumps(env_kwargs, indent=2, sort_keys=True))
    goal_center = env_kwargs.get("goal_center")
    goal_half_extents = env_kwargs.get("goal_half_extents")
    if goal_center is not None and goal_half_extents is not None:
        goal_box = [
            [c - h, c + h] for c, h in zip(goal_center, goal_half_extents, strict=True)
        ]
        print("\n  -> goal sampling box (goal_center +/- goal_half_extents):")
        print(f"     {_fmt_bounds(goal_box)}")

    sampling = metadata.get("start_sampling", {})
    if not sampling:
        print(
            "\nNo 'start_sampling' block in metadata.json -- this dataset likely wasn't "
            "written by dataset_generation/write_maniskill_reach_dataset.py's own "
            "_collect_multimodal_episodes path, or is an older schema. Check env_kwargs "
            "above, or the raw metadata dump below, instead.",
            file=sys.stderr,
        )
    else:
        print("\nstart_sampling (governs where START poses were sampled from):")
        print(f"  randomize_start              : {sampling.get('randomize_start')}")
        print(f"  reach_workspace_bounds        : {_fmt_bounds(sampling.get('reach_workspace_bounds'))}")
        print(f"  start_bounds                  : {_fmt_bounds(sampling.get('start_bounds'))}")
        print(f"  waypoint_reach_workspace_bounds: {_fmt_bounds(sampling.get('waypoint_reach_workspace_bounds'))}")
        print(f"  min_height_override            : {sampling.get('min_height_override')}")
        print(f"  max_height_override            : {sampling.get('max_height_override')}")
        print(f"  waypoint_min_height_override   : {sampling.get('waypoint_min_height_override')}")
        print(f"  waypoint_max_height_override   : {sampling.get('waypoint_max_height_override')}")

    print(f"\nFull metadata.json top-level keys: {sorted(metadata.keys())}")
    print("(re-run with the dataset path piped through `python -m json.tool` for the full dump)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
