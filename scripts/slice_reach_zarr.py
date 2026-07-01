#!/usr/bin/env python
"""Slice the first N episodes of a reach zarr into a NEW zarr, leaving the source untouched.

The reach dataset stores one episode (= one ``(seed, family)`` trajectory) per entry
in ``meta/episode_ends``, and the writer lays the ~11-12 family variants of a single
start-goal *seed* down as one contiguous block. Taking a contiguous *prefix* of
episodes therefore keeps whole seeds (all their family branches) intact -- which is
exactly what a data-scaling / multimodality ablation needs. A random subsample
(e.g. ``--max-train-episodes``) would instead shred those per-seed blocks.

This tool copies ``data/*`` up to the cut row and ``meta/episode_ends`` up to the cut
episode, snapping the requested episode count DOWN to the nearest seed boundary so the
slice never ends in the middle of a seed (which would leave a mode with only a few of
its branches). The input is opened read-only; nothing is written back to it.

Example:
    python scripts/slice_reach_zarr.py \
        --in  /scratch2/skills/pg3d_reach_regen_abcd.zarr \
        --episodes 2998 \
        --out /scratch2/skills/pg3d_reach_regen_abcd_262seeds.zarr
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr

# Per-episode start/goal used only to detect seed boundaries (a seed == one
# start-goal reset, replayed as several contiguous family variants).
_SEED_KEY_FIELDS = ("eef_pos", "goal_pos")
_COPY_BLOCK_ROWS = 8192  # rows copied per chunk so large arrays never load whole


def _episode_start_rows(episode_ends: np.ndarray) -> np.ndarray:
    return np.concatenate([[0], episode_ends[:-1]]).astype(np.int64)


def _seed_block_boundaries(root: zarr.Group, episode_ends: np.ndarray) -> np.ndarray:
    """Return the episode indices at which a new seed (start-goal reset) begins.

    Boundaries are derived from the start (eef_pos) and goal (goal_pos) at each
    episode's first timestep; consecutive episodes sharing both belong to one seed.
    """
    starts = _episode_start_rows(episode_ends)
    keys = []
    for field in _SEED_KEY_FIELDS:
        if field not in root["data"]:
            raise KeyError(f"data/{field} missing; cannot detect seed boundaries")
        keys.append(np.asarray(root["data"][field][:])[starts])
    key = np.concatenate(keys, axis=1).round(4)
    same_as_prev = np.all(key[1:] == key[:-1], axis=1)
    # Episode 0 always starts a seed; thereafter a boundary is where the key changes.
    return np.concatenate([[0], np.nonzero(~same_as_prev)[0] + 1])


def _snap_to_seed_boundary(
    requested: int, boundaries: np.ndarray, n_episodes: int, *, snap: bool
) -> int:
    """Snap the requested episode count down to a clean seed boundary (end of a seed)."""
    # The "end" episode counts are the boundary starts (excluding 0) plus the total.
    seed_end_counts = np.concatenate([boundaries[1:], [n_episodes]])
    if requested in set(seed_end_counts.tolist()):
        return requested
    if not snap:
        raise SystemExit(
            f"--episodes {requested} splits a seed; nearest clean boundaries are "
            f"{seed_end_counts[seed_end_counts <= requested][-1:].tolist()} (below) / "
            f"{seed_end_counts[seed_end_counts >= requested][:1].tolist()} (above). "
            f"Re-run with one of those or drop --no-snap."
        )
    below = seed_end_counts[seed_end_counts <= requested]
    if below.size == 0:
        raise SystemExit(f"--episodes {requested} is smaller than the first seed")
    return int(below[-1])


def _copy_sliced_metadata_json(
    in_path: Path, out_path: Path, n_episodes_out: int, n_seeds_out: int
) -> None:
    """Copy the metadata.json sidecar, slicing its per-episode ``episodes`` list.

    metadata.json is a plain file at the zarr root (not a zarr group), so the
    group copy above misses it. The trainer reads ``point_cloud_saliency`` from
    it for goal-marker alignment and the per-episode ``episodes`` list for
    checkpoint-rollout seed selection -- without it, training falls back to
    default goal-marker geometry, which can corrupt the PointNet scene branch.
    The ``episodes`` list is in episode order, so it is truncated to the kept
    episodes; global config (saliency, env, stats) is copied unchanged.
    """
    src_json = in_path / "metadata.json"
    if not src_json.exists():
        print("  (no metadata.json in source -- skipping)")
        return
    md = json.loads(src_json.read_text(encoding="utf-8"))
    episodes = md.get("episodes")
    if isinstance(episodes, list):
        if len(episodes) < n_episodes_out:
            raise SystemExit(
                f"metadata.json has {len(episodes)} episodes < slice {n_episodes_out}; "
                f"refusing to write a misaligned sidecar"
            )
        md["episodes"] = episodes[:n_episodes_out]
    # Leave all other fields (including data-generation stats like
    # num_collected_demos) untouched -- they describe the original collection run,
    # not the slice. Slice provenance goes in a dedicated key instead.
    md["slice"] = {"source": str(in_path), "episodes": n_episodes_out, "seeds": n_seeds_out}
    (out_path / "metadata.json").write_text(json.dumps(md), encoding="utf-8")
    print(f"  metadata.json (episodes {len(episodes) if isinstance(episodes, list) else 'n/a'} "
          f"-> {n_episodes_out})")


def _copy_array_prefix(src: zarr.Array, dst_group: zarr.Group, name: str, cut_rows: int) -> None:
    """Create dst_group[name] with the same chunks/dtype/compressor and copy [:cut_rows]."""
    out_shape = (cut_rows,) + tuple(src.shape[1:])
    dst = dst_group.create_dataset(
        name,
        shape=out_shape,
        chunks=src.chunks,
        dtype=src.dtype,
        compressor=src.compressor,
        overwrite=True,
    )
    for lo in range(0, cut_rows, _COPY_BLOCK_ROWS):
        hi = min(lo + _COPY_BLOCK_ROWS, cut_rows)
        dst[lo:hi] = src[lo:hi]
    dst.attrs.update(dict(src.attrs))


def slice_zarr(in_path: Path, out_path: Path, requested_episodes: int, *, snap: bool, force: bool) -> None:
    if out_path.resolve() == in_path.resolve():
        raise SystemExit("--out must differ from --in (refusing to overwrite the source)")
    if out_path.exists() and not force:
        raise SystemExit(f"{out_path} already exists; pass --force to overwrite")

    root = zarr.open_group(str(in_path), mode="r")  # read-only: source is never modified
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    n_episodes = int(episode_ends.size)
    total_rows = int(episode_ends[-1]) if n_episodes else 0

    if not 0 < requested_episodes <= n_episodes:
        raise SystemExit(f"--episodes must be in [1, {n_episodes}], got {requested_episodes}")

    boundaries = _seed_block_boundaries(root, episode_ends)
    n_episodes_out = _snap_to_seed_boundary(requested_episodes, boundaries, n_episodes, snap=snap)
    cut_rows = int(episode_ends[n_episodes_out - 1])
    n_seeds_out = int(np.sum(boundaries < n_episodes_out))

    if n_episodes_out != requested_episodes:
        print(f"[snap] requested {requested_episodes} episodes -> {n_episodes_out} "
              f"(clean seed boundary, {n_seeds_out} whole seeds)")

    out_root = zarr.open_group(str(out_path), mode="w")
    data_out = out_root.create_group("data")
    meta_out = out_root.create_group("meta")

    for name in root["data"].array_keys():
        src = root["data"][name]
        if src.shape[0] != total_rows:
            raise SystemExit(
                f"data/{name} has {src.shape[0]} rows, expected {total_rows}; "
                f"unexpected layout, aborting to avoid a corrupt slice"
            )
        _copy_array_prefix(src, data_out, name, cut_rows)
        print(f"  data/{name:<20} {src.shape} -> {data_out[name].shape}")

    # meta/episode_ends are absolute row offsets and already start at 0, so the first
    # n_episodes_out entries are self-consistent with the sliced data (no re-basing).
    meta_out.create_dataset(
        "episode_ends",
        data=episode_ends[:n_episodes_out],
        chunks=root["meta"]["episode_ends"].chunks,
        dtype=episode_ends.dtype,
        overwrite=True,
    )
    # Copy any other per-episode / per-row meta arrays defensively.
    for name in root["meta"].array_keys():
        if name == "episode_ends":
            continue
        src = root["meta"][name]
        if src.shape[0] == n_episodes:
            meta_out.create_dataset(name, data=np.asarray(src[:n_episodes_out]), overwrite=True)
        elif src.shape[0] == total_rows:
            _copy_array_prefix(src, meta_out, name, cut_rows)
        else:
            meta_out.create_dataset(name, data=np.asarray(src[:]), overwrite=True)
        print(f"  meta/{name}")

    out_root.attrs.update(dict(root.attrs))
    data_out.attrs.update(dict(root["data"].attrs))
    meta_out.attrs.update(dict(root["meta"].attrs))

    # Copy the metadata.json sidecar (goal-marker config + per-episode list),
    # truncating the episode list to match the slice.
    _copy_sliced_metadata_json(in_path, out_path, n_episodes_out, n_seeds_out)

    # Verify the slice re-opens and reports the expected episode/seed/row counts.
    check = zarr.open_group(str(out_path), mode="r")
    ee_out = np.asarray(check["meta"]["episode_ends"][:], dtype=np.int64)
    print(
        "\n[done] wrote slice:"
        f"\n  episodes : {n_episodes} -> {ee_out.size}"
        f"\n  seeds    : {int(boundaries.size)} -> {n_seeds_out}"
        f"\n  rows     : {total_rows} -> {int(ee_out[-1])}"
        f"\n  out      : {out_path}"
        f"\n  source   : {in_path} (unchanged)"
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="in_path", type=Path, required=True, help="source reach .zarr (read-only)")
    p.add_argument("--out", dest="out_path", type=Path, required=True, help="destination .zarr (new)")
    p.add_argument("--episodes", type=int, default=2998, help="episodes to keep (snapped to a seed boundary)")
    p.add_argument("--no-snap", dest="snap", action="store_false", help="error instead of snapping if N splits a seed")
    p.add_argument("--force", action="store_true", help="overwrite --out if it exists")
    args = p.parse_args(argv)
    slice_zarr(args.in_path, args.out_path, args.episodes, snap=args.snap, force=args.force)


if __name__ == "__main__":
    main()
