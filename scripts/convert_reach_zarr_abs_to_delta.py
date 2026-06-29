from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import zarr


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # ---- plot-only mode: skip conversion entirely -------------------------
    if args.plot_only:
        plot_out = _resolve_plot_output(args.plot_output, args.input)
        _plot_delta_magnitudes(
            args.input,
            plot_out,
            n_episodes=args.plot_episodes,
            start_index=args.plot_start_index,
        )
        return 0

    # ---- normal conversion -----------------------------------------------
    if args.output is None:
        raise ValueError("--output is required unless --plot-only is given")

    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output} already exists; pass --overwrite")
        shutil.rmtree(args.output)

    print(f"copying dataset: {args.input} -> {args.output}", flush=True)
    shutil.copytree(args.input, args.output)

    metadata_path = args.output / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_action_mode = str(metadata.get("action_mode", "abs_joint"))
    if source_action_mode == "delta_joint" and not args.force:
        raise ValueError(
            f"{args.input} metadata already says action_mode=delta_joint; "
            "pass --force only if /data/action is actually still absolute"
        )
    if source_action_mode != "abs_joint" and not args.force:
        raise ValueError(
            f"expected metadata action_mode=abs_joint, got {source_action_mode!r}; "
            "pass --force to convert anyway"
        )

    root = zarr.open_group(str(args.output), mode="a")
    data = root["data"]
    for key in ("action", "state"):
        if key not in data:
            raise KeyError(f"missing /data/{key}")
    action = data["action"]
    state = data["state"]
    if action.ndim != 2 or action.shape[1] < 7:
        raise ValueError(f"/data/action must have shape [T, >=7], got {action.shape}")
    if state.ndim != 2 or state.shape[1] < 7:
        raise ValueError(f"/data/state must have shape [T, >=7], got {state.shape}")
    if action.shape[0] != state.shape[0]:
        raise ValueError(f"action length {action.shape[0]} != state length {state.shape[0]}")

    old_stats = _action_stats(action)
    converted_rows = 0
    for start in range(0, action.shape[0], args.chunk_size):
        end = min(start + args.chunk_size, action.shape[0])
        action_chunk = np.asarray(action[start:end], dtype=np.float32)
        state_chunk = np.asarray(state[start:end, :7], dtype=np.float32)
        action_chunk[:, :7] = action_chunk[:, :7] - state_chunk
        action[start:end] = action_chunk
        converted_rows += end - start
        if converted_rows == action.shape[0] or converted_rows % max(args.chunk_size * 10, 1) == 0:
            print(f"converted {converted_rows}/{action.shape[0]} rows", flush=True)

    new_stats = _action_stats(action)
    metadata["action_mode"] = "delta_joint"
    metadata["converted_from_action_mode"] = source_action_mode
    metadata["conversion"] = {
        "type": "abs_joint_to_delta_joint",
        "input": str(args.input),
        "output": str(args.output),
        "formula": "new_action[:, :7] = old_action[:, :7] - state[:, :7]",
        "preserved_arrays": ["sim_action", "state", "point_cloud", "target_position", "tcp_pose"],
        "old_action_stats": old_stats,
        "new_action_stats": new_stats,
    }
    if "summary" in metadata and "arrays" in metadata["summary"]:
        metadata["summary"]["arrays"]["action"] = {
            "shape": list(action.shape),
            "dtype": str(action.dtype),
        }
    metadata_path.write_text(
        json.dumps(_jsonable(metadata), indent=2, sort_keys=True), encoding="utf-8"
    )

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "rows": int(action.shape[0]),
        "source_action_mode": source_action_mode,
        "output_action_mode": "delta_joint",
        "old_action_stats": old_stats,
        "new_action_stats": new_stats,
    }
    summary_path = args.output / "conversion_summary.json"
    summary_path.write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"saved converted dataset: {args.output}", flush=True)
    print(f"saved summary: {summary_path}", flush=True)

    # ---- optional: plot after conversion ----------------------------------
    if args.plot:
        plot_out = _resolve_plot_output(args.plot_output, args.output)
        _plot_delta_magnitudes(
            args.output,
            plot_out,
            n_episodes=args.plot_episodes,
            start_index=args.plot_start_index,
            action_mode="delta_joint",
        )

    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a pg3d reach zarr dataset and convert /data/action from abs_joint to"
            " delta_joint. Pass --plot-only to skip conversion and only plot delta magnitudes."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="source .zarr directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="new converted .zarr directory (required unless --plot-only)",
    )
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="convert even if metadata action_mode is not abs_joint",
    )

    # -- plot flags ----------------------------------------------------------
    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument(
        "--plot",
        action="store_true",
        help="after conversion, plot delta action magnitudes and save a PNG next to output zarr",
    )
    plot_group.add_argument(
        "--plot-only",
        action="store_true",
        help=(
            "skip conversion entirely; read --input and plot delta magnitudes. "
            "Works for both abs_joint and delta_joint zarrs (auto-detects from metadata.json)."
        ),
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help="path for the output PNG (default: <zarr_dir>_delta_magnitudes.png)",
    )
    parser.add_argument(
        "--plot-episodes",
        type=int,
        default=366,
        help="number of unique-seed episodes to plot (default: 366)",
    )
    parser.add_argument(
        "--plot-start-index",
        type=int,
        default=0,
        help="skip this many unique seeds before selecting episodes (default: 0)",
    )

    args = parser.parse_args(argv)
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    if not args.input.is_dir():
        raise ValueError(f"--input must be a zarr directory, got {args.input}")
    if not args.plot_only and args.output is None:
        parser.error("--output is required unless --plot-only is given")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if args.plot_episodes <= 0:
        raise ValueError("--plot-episodes must be positive")
    if args.plot_start_index < 0:
        raise ValueError("--plot-start-index must be non-negative")
    return args


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _resolve_plot_output(explicit: Path | None, zarr_path: Path) -> Path:
    if explicit is not None:
        return explicit
    return zarr_path.parent / f"{zarr_path.name}_delta_magnitudes.png"


def _plot_delta_magnitudes(
    zarr_path: Path,
    output_png: Path,
    *,
    n_episodes: int = 366,
    start_index: int = 0,
    action_mode: str | None = None,
) -> None:
    """Plot per-step L2 norm of delta actions for unique-seed episodes and save a PNG.

    If action_mode is None, auto-detects from zarr metadata.json.
    For abs_joint zarrs, computes delta = action[:7] - state[:7] on the fly.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print(f"opening zarr for plotting: {zarr_path}", flush=True)
    root = zarr.open_group(str(zarr_path), mode="r")
    data = root["data"]
    for key in ("action", "state", "episode_ends"):
        src = data if key != "episode_ends" else root["meta"]
        if key not in src:
            raise KeyError(f"missing {'meta' if key == 'episode_ends' else 'data'}/{key} in zarr")

    action = np.asarray(data["action"][:, :7], dtype=np.float32)
    state = np.asarray(data["state"][:, :7], dtype=np.float32)
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)

    # Auto-detect action mode
    if action_mode is None:
        meta_path = zarr_path / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            action_mode = str(meta.get("action_mode", "abs_joint"))
        else:
            action_mode = "abs_joint"
            print("warning: no metadata.json found, assuming abs_joint", flush=True)

    if action_mode == "abs_joint":
        print("computing delta = action - state (abs_joint mode)", flush=True)
        delta = action - state
    else:
        delta = action  # already delta_joint

    # Load per-episode seeds for unique-seed selection
    meta_path = zarr_path / "metadata.json"
    episode_seeds: list[int] = []
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        episode_seeds = [int(ep["seed"]) for ep in meta.get("episodes", []) if "seed" in ep]

    n_total_episodes = len(episode_ends)

    # Select unique-seed episode indices (same logic as eval_reach_checkpoint_unique_seeds.py)
    if episode_seeds and len(episode_seeds) == n_total_episodes:
        seen: set[int] = set()
        unique_ep_indices: list[int] = []
        for ep_idx, seed in enumerate(episode_seeds):
            if seed in seen:
                continue
            seen.add(seed)
            unique_ep_indices.append(ep_idx)
        selected = unique_ep_indices[start_index : start_index + n_episodes]
        print(
            f"unique seeds: {len(unique_ep_indices)} total, "
            f"selected {len(selected)} (start_index={start_index})",
            flush=True,
        )
    else:
        if episode_seeds:
            print(
                f"warning: episode_seeds count ({len(episode_seeds)}) != "
                f"n_total_episodes ({n_total_episodes}); selecting sequentially",
                flush=True,
            )
        selected = list(range(start_index, min(start_index + n_episodes, n_total_episodes)))
        print(f"selected {len(selected)} episodes sequentially", flush=True)

    if not selected:
        raise RuntimeError("no episodes selected for plotting")

    # Collect per-step delta norms per episode
    norms_list: list[np.ndarray] = []
    for ep_idx in selected:
        ep_start = int(episode_ends[ep_idx - 1]) if ep_idx > 0 else 0
        ep_end = int(episode_ends[ep_idx])
        ep_norms = np.linalg.norm(delta[ep_start:ep_end], axis=1).astype(np.float32)
        norms_list.append(ep_norms)

    episode_lengths = np.array([len(n) for n in norms_list])
    print(
        f"episode lengths: min={episode_lengths.min()} max={episode_lengths.max()} "
        f"median={np.median(episode_lengths):.0f} mean={episode_lengths.mean():.1f}",
        flush=True,
    )

    max_len = int(episode_lengths.max())
    steps = np.arange(max_len)

    # Build padded matrix for aggregate statistics
    padded = np.full((len(norms_list), max_len), np.nan, dtype=np.float32)
    for i, norms in enumerate(norms_list):
        padded[i, : len(norms)] = norms

    with np.errstate(all="ignore"):
        mean_norm = np.nanmean(padded, axis=0)
        p10 = np.nanpercentile(padded, 10, axis=0)
        p25 = np.nanpercentile(padded, 25, axis=0)
        p75 = np.nanpercentile(padded, 75, axis=0)

    # Low-activity threshold at 30% of overall mean — marks waypoint pauses
    overall_mean = float(np.nanmean(mean_norm))
    low_thresh = 0.30 * overall_mean

    # ---- figure: two rows --------------------------------------------------
    fig, (ax_main, ax_count) = plt.subplots(
        2, 1, figsize=(16, 9), gridspec_kw={"height_ratios": [4, 1]}, sharex=True
    )

    # Individual episode traces (very thin, translucent)
    for norms in norms_list:
        ax_main.plot(
            range(len(norms)), norms,
            color="steelblue", alpha=0.08, linewidth=0.5, rasterized=True,
        )

    # IQR band + mean + 10th percentile
    ax_main.fill_between(steps, p25, p75, alpha=0.25, color="steelblue", label="IQR (25–75 %)")
    ax_main.plot(steps, mean_norm, color="navy", linewidth=2.0, label="Mean")
    ax_main.plot(
        steps, p10, color="firebrick", linewidth=1.2, linestyle="--", label="10th percentile"
    )

    # Highlight low-activity regions (candidate waypoint-pause zones)
    dip_mask = mean_norm < low_thresh
    if np.any(dip_mask):
        ax_main.axhline(
            low_thresh, color="darkorange", linewidth=1.0, linestyle=":",
            label=f"30 % of mean ({low_thresh:.4f} rad)",
        )
        # Merge consecutive dip steps into spans
        in_span = False
        span_start = 0
        for s in range(max_len):
            if dip_mask[s] and not in_span:
                span_start = s
                in_span = True
            elif not dip_mask[s] and in_span:
                ax_main.axvspan(span_start - 0.5, s - 0.5, alpha=0.15, color="darkorange", zorder=0)
                in_span = False
        if in_span:
            ax_main.axvspan(
                span_start - 0.5, max_len - 0.5, alpha=0.15, color="darkorange", zorder=0
            )

    ax_main.set_ylabel("Delta action L2 norm (rad)", fontsize=11)
    ax_main.set_title(
        f"Per-step delta action magnitude — {len(selected)} unique-seed episodes "
        f"| action_mode={action_mode}",
        fontsize=12,
    )
    ax_main.legend(loc="upper right", fontsize=9)
    ax_main.grid(True, alpha=0.3)
    ax_main.set_xlim(0, max_len - 1)

    # Bottom row: active episode count per step
    count_per_step = np.sum(~np.isnan(padded), axis=0).astype(np.float32)
    ax_count.fill_between(steps, 0, count_per_step, color="steelblue", alpha=0.5)
    ax_count.set_xlabel("Step within episode", fontsize=11)
    ax_count.set_ylabel("Active episodes", fontsize=9)
    ax_count.set_xlim(0, max_len - 1)
    ax_count.grid(True, alpha=0.3)

    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_png), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved plot: {output_png}", flush=True)


# ---------------------------------------------------------------------------
# Helpers (unchanged from original)
# ---------------------------------------------------------------------------


def _action_stats(action: Any) -> dict[str, Any]:
    mins: np.ndarray | None = None
    maxs: np.ndarray | None = None
    total = np.zeros((int(action.shape[1]),), dtype=np.float64)
    total_sq = np.zeros((int(action.shape[1]),), dtype=np.float64)
    count = 0
    for start in range(0, int(action.shape[0]), 8192):
        end = min(start + 8192, int(action.shape[0]))
        chunk = np.asarray(action[start:end], dtype=np.float64)
        chunk_min = np.min(chunk, axis=0)
        chunk_max = np.max(chunk, axis=0)
        mins = chunk_min if mins is None else np.minimum(mins, chunk_min)
        maxs = chunk_max if maxs is None else np.maximum(maxs, chunk_max)
        total += np.sum(chunk, axis=0)
        total_sq += np.sum(chunk * chunk, axis=0)
        count += chunk.shape[0]
    if count == 0:
        return {"count": 0}
    assert mins is not None and maxs is not None
    mean = total / count
    variance = np.maximum(total_sq / count - mean * mean, 0.0)
    return {
        "count": int(count),
        "min": mins.astype(float).tolist(),
        "max": maxs.astype(float).tolist(),
        "mean": mean.astype(float).tolist(),
        "std": np.sqrt(variance).astype(float).tolist(),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
