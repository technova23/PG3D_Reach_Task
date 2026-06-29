"""Dataset diversity audit for pg3d reach zarr files.

Usage:
    uv run python -m pg3d.multimodality --dataset artifacts/pg3d_reach_regen_abcd.zarr
    uv run python -m pg3d.multimodality --dataset artifacts/pg3d_reach_regen_abcd.zarr --plot
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from scipy.spatial.distance import pdist


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Diversity audit for a pg3d reach zarr dataset.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to .zarr directory")
    parser.add_argument("--plot", action="store_true", help="Save PNG plots to <dataset>/diversity_plots/")
    parser.add_argument("--pairwise-cap", type=int, default=4000,
                        help="Max EEF points for pairwise distance (default 4000)")
    args = parser.parse_args(argv)

    root = zarr.open_group(str(args.dataset), mode="r")
    meta = _load_metadata(args.dataset)
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    episode_metas = meta.get("episodes", [])

    data = root["data"]
    eef_pos = np.asarray(data["eef_pos"][:], dtype=np.float32)         # (N,3)
    tcp_pose = np.asarray(data["tcp_pose"][:], dtype=np.float32)       # (N,7)
    target_pos = np.asarray(data["target_position"][:], dtype=np.float32)  # (N,3)
    action = np.asarray(data["action"][:], dtype=np.float32)           # (N,7)
    state = np.asarray(data["state"][:], dtype=np.float32)             # (N,9)

    valid_mask = None
    if "point_valid_mask" in data:
        valid_mask = np.asarray(data["point_valid_mask"][:], dtype=bool)  # (N,1024)

    sep = "=" * 72

    # ── 1. Overview ──────────────────────────────────────────────────────────
    n_episodes = len(episode_ends)
    n_rows = int(episode_ends[-1])
    print(sep)
    print("DATASET OVERVIEW")
    print(sep)
    print(f"  Path            : {args.dataset}")
    print(f"  Env             : {meta.get('env_id', 'unknown')}")
    print(f"  Action mode     : {meta.get('action_mode', 'unknown')}")
    print(f"  Episodes        : {n_episodes}")
    print(f"  Total rows      : {n_rows}")
    tg = meta.get("trajectory_generation", {})
    print(f"  Variants/reset  : {tg.get('variants_per_reset', '?')}")
    print(f"  Lateral-Z offset: {tg.get('lateral_z_offset', '?')} m")
    print(f"  Waypoint attempts: {tg.get('waypoint_attempts', '?')}")

    # ── 2. Episode length distribution ───────────────────────────────────────
    ep_starts = np.concatenate([[0], episode_ends[:-1]])
    ep_lengths = episode_ends - ep_starts
    print(f"\n{sep}")
    print("EPISODE LENGTH DISTRIBUTION")
    print(sep)
    _print_stats("Length (steps)", ep_lengths)
    short = int(np.sum(ep_lengths < 30))
    long = int(np.sum(ep_lengths > 120))
    print(f"  Episodes < 30 steps  : {short}")
    print(f"  Episodes > 120 steps : {long}")

    # ── 3. Success rates ─────────────────────────────────────────────────────
    successes = [bool(e.get("success", False)) for e in episode_metas]
    families_all = [e.get("trajectory_family", "unknown") for e in episode_metas]
    seeds_all = [e.get("seed") for e in episode_metas]
    final_dists = [e.get("final_distance") for e in episode_metas if e.get("final_distance") is not None]

    print(f"\n{sep}")
    print("SUCCESS RATES")
    print(sep)
    total_succ = sum(successes)
    print(f"  Overall: {total_succ}/{n_episodes} = {100*total_succ/max(n_episodes,1):.1f}%")
    by_family: dict[str, list[bool]] = defaultdict(list)
    for fam, succ in zip(families_all, successes):
        by_family[fam].append(succ)
    for fam in sorted(by_family):
        vals = by_family[fam]
        pct = 100 * sum(vals) / max(len(vals), 1)
        print(f"    {fam:<22}: {sum(vals):4d}/{len(vals):4d} = {pct:.1f}%")
    if final_dists:
        _print_stats("Final distance (m)", np.array(final_dists))

    # ── 4. Trajectory family balance ─────────────────────────────────────────
    family_counts = Counter(families_all)
    print(f"\n{sep}")
    print("TRAJECTORY FAMILY BALANCE")
    print(sep)
    total_ep = n_episodes
    for fam, cnt in sorted(family_counts.items()):
        bar = "█" * int(40 * cnt / max(max(family_counts.values()), 1))
        print(f"  {fam:<22}: {cnt:5d} ({100*cnt/total_ep:5.1f}%) {bar}")
    counts_arr = np.array(list(family_counts.values()), dtype=float)
    imbalance = counts_arr.max() / counts_arr.min() if counts_arr.min() > 0 else float("inf")
    print(f"\n  Imbalance ratio (max/min) : {imbalance:.2f}  (ideal = 1.0, <1.5 is good)")

    # ── 5. Seed coverage ─────────────────────────────────────────────────────
    unique_seeds = sorted(set(s for s in seeds_all if s is not None))
    n_seeds = len(unique_seeds)
    variants_per_seed = Counter(seeds_all)
    vps_vals = np.array(list(variants_per_seed.values()), dtype=int)
    print(f"\n{sep}")
    print("SEED COVERAGE")
    print(sep)
    print(f"  Unique seeds            : {n_seeds}")
    print(f"  Episodes per seed — min : {vps_vals.min()}")
    print(f"  Episodes per seed — mean: {vps_vals.mean():.1f}")
    print(f"  Episodes per seed — max : {vps_vals.max()}")
    # seeds with < 4 variants (under-represented)
    thin_seeds = sum(1 for v in vps_vals if v < 4)
    print(f"  Seeds with < 4 variants : {thin_seeds}")
    full_seeds = sum(1 for v in vps_vals if v >= len(family_counts))
    print(f"  Seeds with all families : {full_seeds}  ({100*full_seeds/max(n_seeds,1):.0f}%)")

    # ── 6. EEF spatial coverage ──────────────────────────────────────────────
    print(f"\n{sep}")
    print("EEF POSITION COVERAGE  (all rows)")
    print(sep)
    for i, ax in enumerate(["X", "Y", "Z"]):
        lo, hi = float(eef_pos[:, i].min()), float(eef_pos[:, i].max())
        print(f"  {ax}: [{lo:+.3f}, {hi:+.3f}]  range = {hi-lo:.3f} m")

    voxel_size = 0.02
    voxels = np.floor(eef_pos / voxel_size).astype(np.int32)
    unique_voxels = len(np.unique(voxels, axis=0))
    workspace_vol = float(np.prod(eef_pos.max(axis=0) - eef_pos.min(axis=0)))
    occupied_vol = unique_voxels * voxel_size ** 3
    print(f"\n  Voxel size       : {voxel_size*100:.0f} cm")
    print(f"  Occupied voxels  : {unique_voxels}")
    print(f"  Occupied volume  : {occupied_vol:.4f} m³")
    print(f"  Workspace volume : {workspace_vol:.4f} m³")
    print(f"  Coverage ratio   : {occupied_vol/max(workspace_vol,1e-9):.3f}")

    # ── 7. Z-direction diversity (specifically) ───────────────────────────────
    print(f"\n{sep}")
    print("Z-DIRECTION DIVERSITY  (important for constrained reach)")
    print(sep)
    # EEF Z per family
    ep_start_idx = np.concatenate([[0], episode_ends[:-1]]).astype(int)
    ep_end_idx = episode_ends.astype(int)
    family_z: dict[str, list[float]] = defaultdict(list)
    for i, (s, e, fam) in enumerate(zip(ep_start_idx, ep_end_idx, families_all)):
        z_vals = eef_pos[s:e, 2]
        family_z[fam].append(float(z_vals.max() - z_vals.min()))  # Z excursion per episode

    for fam in sorted(family_z):
        zexc = np.array(family_z[fam])
        print(f"  {fam:<22} Z-excursion: mean={zexc.mean():.3f}m  max={zexc.max():.3f}m")

    # Z histogram across all EEF positions
    z_all = eef_pos[:, 2]
    z_bins = np.linspace(z_all.min(), z_all.max(), 11)
    z_hist, _ = np.histogram(z_all, bins=z_bins)
    print(f"\n  Z histogram ({z_all.min():.2f} → {z_all.max():.2f} m, 10 bins):")
    max_bar = z_hist.max()
    for b_lo, b_hi, cnt in zip(z_bins, z_bins[1:], z_hist):
        bar = "█" * int(30 * cnt / max(max_bar, 1))
        print(f"    [{b_lo:+.3f}, {b_hi:+.3f}): {cnt:6d}  {bar}")

    # ── 8. Goal position diversity ───────────────────────────────────────────
    # sample one row per episode (first row = initial obs for that episode)
    ep_goal = target_pos[ep_start_idx]  # (n_episodes, 3)
    print(f"\n{sep}")
    print("GOAL POSITION COVERAGE  (one per episode)")
    print(sep)
    for i, ax in enumerate(["X", "Y", "Z"]):
        lo, hi = float(ep_goal[:, i].min()), float(ep_goal[:, i].max())
        print(f"  {ax}: [{lo:+.3f}, {hi:+.3f}]  range = {hi-lo:.3f} m")
    goal_voxels = np.floor(ep_goal / voxel_size).astype(np.int32)
    unique_goal_voxels = len(np.unique(goal_voxels, axis=0))
    print(f"  Unique goal voxels (2cm): {unique_goal_voxels}")

    # ── 9. Action/joint distribution ─────────────────────────────────────────
    print(f"\n{sep}")
    print("ACTION DISTRIBUTION  (all rows, 7 joints)")
    print(sep)
    print(f"  {'Joint':<8} {'min':>8} {'mean':>8} {'max':>8} {'std':>8}")
    for j in range(action.shape[1]):
        col = action[:, j]
        print(f"  J{j:<7} {col.min():>8.4f} {col.mean():>8.4f} {col.max():>8.4f} {col.std():>8.4f}")

    # ── 10. State (joint angle) diversity ────────────────────────────────────
    joint_state = state[:, :7]  # first 7 = arm joints
    print(f"\n{sep}")
    print("JOINT STATE DIVERSITY  (7 arm joints, all rows)")
    print(sep)
    print(f"  {'Joint':<8} {'range':>8} {'std':>8}")
    for j in range(7):
        col = joint_state[:, j]
        print(f"  J{j:<7} {col.max()-col.min():>8.4f} {col.std():>8.4f}")

    # ── 11. Family centroid separation ───────────────────────────────────────
    print(f"\n{sep}")
    print("FAMILY CENTROID SEPARATION  (mean EEF position per family)")
    print(sep)
    fam_centroids: dict[str, np.ndarray] = {}
    for fam in sorted(family_z.keys()):
        mask_rows = np.concatenate([
            np.arange(s, e)
            for s, e, f in zip(ep_start_idx, ep_end_idx, families_all)
            if f == fam
        ])
        fam_centroids[fam] = eef_pos[mask_rows].mean(axis=0)
        print(f"  {fam:<22} centroid: {fam_centroids[fam]}")

    if len(fam_centroids) >= 2:
        centroids_arr = np.stack(list(fam_centroids.values()))
        sep_dists = pdist(centroids_arr)
        print(f"\n  Centroid pairwise distances: min={sep_dists.min():.4f}  mean={sep_dists.mean():.4f}  max={sep_dists.max():.4f} m")

    # ── 12. Pairwise EEF distance (subsampled) ───────────────────────────────
    print(f"\n{sep}")
    print(f"PAIRWISE EEF DISTANCE  (subsampled to {args.pairwise_cap} rows)")
    print(sep)
    cap = min(args.pairwise_cap, len(eef_pos))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(eef_pos), cap, replace=False)
    dists = pdist(eef_pos[idx])
    _print_stats("Pairwise distance (m)", dists)

    # ── 13. Point cloud validity ─────────────────────────────────────────────
    if valid_mask is not None:
        print(f"\n{sep}")
        print("POINT CLOUD VALIDITY")
        print(sep)
        valid_frac = valid_mask.mean()
        rows_fully_invalid = int(np.sum(valid_mask.sum(axis=1) == 0))
        print(f"  Mean valid fraction     : {valid_frac:.4f}")
        print(f"  Rows with zero valid pts: {rows_fully_invalid}")

    # ── 14. Waypoint Z diversity (from metadata) ─────────────────────────────
    z_waypoints_all: list[float] = []
    for ep in episode_metas:
        wps = ep.get("trajectory_waypoints") or []
        for wp in wps:
            if isinstance(wp, (list, tuple)) and len(wp) >= 3:
                z_waypoints_all.append(float(wp[2]))

    if z_waypoints_all:
        wz = np.array(z_waypoints_all)
        print(f"\n{sep}")
        print("WAYPOINT Z DIVERSITY  (from episode metadata)")
        print(sep)
        _print_stats("Waypoint Z (m)", wz)

    # ── 15. Overall diversity score ───────────────────────────────────────────
    xyz_vol = float(np.prod(eef_pos.max(axis=0) - eef_pos.min(axis=0)))
    coverage_score = min(occupied_vol / max(xyz_vol, 1e-9), 1.0)
    spread_score = float(dists.mean())
    separation_score = float(sep_dists.mean()) if len(fam_centroids) >= 2 else 0.0
    balance_score = max(0.0, 1.0 - (imbalance - 1.0) / 5.0)

    # normalise spread & separation to [0,1] using rough workspace scale
    spread_norm = min(spread_score / 0.5, 1.0)
    sep_norm = min(separation_score / 0.1, 1.0)

    overall = 0.35 * coverage_score + 0.25 * spread_norm + 0.25 * sep_norm + 0.15 * balance_score

    print(f"\n{sep}")
    print("DIVERSITY SUMMARY")
    print(sep)
    print(f"  Coverage score (voxel/workspace)  : {coverage_score:.4f}  (weight 35%)")
    print(f"  Spread score   (pairwise EEF dist): {spread_norm:.4f}  (weight 25%)")
    print(f"  Separation     (family centroids) : {sep_norm:.4f}  (weight 25%)")
    print(f"  Balance        (family imbalance) : {balance_score:.4f}  (weight 15%)")
    print(f"\n  *** OVERALL DIVERSITY SCORE : {overall:.4f} ***")
    print(f"      (0.0 = no diversity, 1.0 = perfect)")

    # ── 16. Optional plots ───────────────────────────────────────────────────
    if args.plot:
        _make_plots(args.dataset, eef_pos, ep_goal, families_all, ep_start_idx, ep_end_idx, ep_lengths)
        _plot_workspace_coverage(args.dataset, eef_pos, ep_goal, families_all, ep_start_idx, seeds_all, meta)

    print()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_metadata(dataset_path: Path) -> dict[str, Any]:
    p = Path(dataset_path) / "metadata.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _print_stats(label: str, arr: np.ndarray) -> None:
    a = arr.astype(float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        print(f"  {label}: no finite values")
        return
    print(f"  {label}:")
    print(f"    min={a.min():.4f}  p5={np.percentile(a,5):.4f}  "
          f"mean={a.mean():.4f}  median={np.median(a):.4f}  "
          f"p95={np.percentile(a,95):.4f}  max={a.max():.4f}  std={a.std():.4f}")


def _make_plots(
    dataset_path: Path,
    eef_pos: np.ndarray,
    ep_goal: np.ndarray,
    families_all: list[str],
    ep_starts: np.ndarray,
    ep_ends: np.ndarray,
    ep_lengths: np.ndarray,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plots")
        return

    out_dir = Path(dataset_path) / "diversity_plots"
    out_dir.mkdir(exist_ok=True)

    unique_fams = sorted(set(families_all))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_fams)))
    fam_color = {f: c for f, c in zip(unique_fams, colors)}

    # --- Plot 1: EEF XY scatter by family ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("EEF Position Coverage by Family")
    pairs = [("X", "Y", 0, 1), ("X", "Z", 0, 2), ("Y", "Z", 1, 2)]
    for ax, (xl, yl, xi, yi) in zip(axes, pairs):
        for fam in unique_fams:
            mask_rows = np.concatenate([
                np.arange(s, e)
                for s, e, f in zip(ep_starts, ep_ends, families_all)
                if f == fam
            ])
            sample = mask_rows[::max(1, len(mask_rows)//500)]
            ax.scatter(eef_pos[sample, xi], eef_pos[sample, yi],
                       s=1, alpha=0.3, color=fam_color[fam], label=fam)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_aspect("equal")
    axes[0].legend(markerscale=8, fontsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / "eef_coverage.png", dpi=120)
    plt.close()

    # --- Plot 2: Goal position coverage ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Goal Position Coverage")
    for ax, (xl, yl, xi, yi) in zip(axes, pairs):
        for fam in unique_fams:
            ep_mask = [i for i, f in enumerate(families_all) if f == fam]
            if ep_mask:
                pts = ep_goal[ep_mask]
                ax.scatter(pts[:, xi], pts[:, yi], s=5, alpha=0.5,
                           color=fam_color[fam], label=fam)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_aspect("equal")
    axes[0].legend(markerscale=3, fontsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / "goal_coverage.png", dpi=120)
    plt.close()

    # --- Plot 3: Z-histogram ---
    fig, ax = plt.subplots(figsize=(8, 4))
    z_all = eef_pos[:, 2]
    ax.hist(z_all, bins=40, color="steelblue", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("EEF Z position (m)")
    ax.set_ylabel("Row count")
    ax.set_title("Z-direction distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "z_histogram.png", dpi=120)
    plt.close()

    # --- Plot 4: Episode length histogram ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ep_lengths, bins=30, color="darkorange", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Episode length (steps)")
    ax.set_ylabel("Count")
    ax.set_title("Episode length distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "episode_lengths.png", dpi=120)
    plt.close()

    print(f"\n  Plots saved to {out_dir}/")


def _plot_workspace_coverage(
    dataset_path: Path,
    eef_pos: np.ndarray,
    ep_goal: np.ndarray,
    families_all: list[str],
    ep_start_idx: np.ndarray,
    seeds_all: list[Any],
    meta: dict[str, Any] | None = None,
) -> None:
    """Plot one start/end pair per UNIQUE seed in ManiSkill world coords.

    Only the first episode encountered for each seed contributes a start and an
    end point, so per-seed variants are collapsed to a single representative.

    Args:
        dataset_path: Path to zarr dataset
        eef_pos: All EEF positions (N, 3) in ManiSkill world coordinates
        ep_goal: Goal positions (n_episodes, 3) in ManiSkill world coordinates
        families_all: Trajectory family per episode
        ep_start_idx: Start row index for each episode
        seeds_all: Seed per episode (used to dedupe to unique seeds)
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("matplotlib or 3D plotting not available — skipping workspace coverage plot")
        return

    out_dir = Path(dataset_path) / "diversity_plots"
    out_dir.mkdir(exist_ok=True)

    # Pick the first episode index for each unique seed
    seen_seeds: set = set()
    unique_ep_idx: list[int] = []
    for i, s in enumerate(seeds_all):
        if s is None:
            continue
        if s not in seen_seeds:
            seen_seeds.add(s)
            unique_ep_idx.append(i)
    # Fallback: if no seeds available, use every episode
    if not unique_ep_idx:
        unique_ep_idx = list(range(len(ep_goal)))
    unique_ep_idx_arr = np.array(unique_ep_idx, dtype=int)

    # One start (first EEF pos) and one end (goal) per unique seed
    start_positions = eef_pos[ep_start_idx.astype(int)[unique_ep_idx_arr]]  # (n_seeds, 3)
    end_positions = ep_goal[unique_ep_idx_arr]                              # (n_seeds, 3)
    all_goal_positions = end_positions  # unique-seed goals only

    print(f"\n  Unique seeds plotted: {len(unique_ep_idx)} (of {len(seeds_all)} episodes)")

    # Compute bounding box around the unique-seed start and end points
    all_points = np.vstack([start_positions, end_positions])
    bb_min = all_points.min(axis=0)
    bb_max = all_points.max(axis=0)
    bb_center = (bb_min + bb_max) / 2
    bb_size = bb_max - bb_min

    # Goal sampling region — read from the dataset's own metadata (the env that
    # generated it), NOT a hardcoded default. This is where goals are uniformly
    # drawn: goal_center +/- goal_half_extents (or the union of goal_regions).
    task = (meta or {}).get("task", {}) if isinstance(meta, dict) else {}
    goal_bounds = task.get("goal_bounds")
    goal_env_id = task.get("env_id", (meta or {}).get("env_id", "unknown"))
    if goal_bounds and len(goal_bounds) == 3:
        goal_region_min = np.array([b[0] for b in goal_bounds], dtype=float)
        goal_region_max = np.array([b[1] for b in goal_bounds], dtype=float)
    else:
        # Fallback to goal_center/half_extents, else the Narrow default
        gc = task.get("goal_center", [0.0, 0.0, 0.35])
        gh = task.get("goal_half_extents", [0.08, 0.08, 0.08])
        goal_region_min = np.array(gc, dtype=float) - np.array(gh, dtype=float)
        goal_region_max = np.array(gc, dtype=float) + np.array(gh, dtype=float)
    goal_region_half_extents = (goal_region_max - goal_region_min) / 2.0
    goal_region_center = (goal_region_max + goal_region_min) / 2.0

    # ManiSkill TableSceneBuilder geometry (world coords, from the saved AABB in
    # mani_skill/utils/scene_builder/table/scene_builder.py).
    # The table TOP SURFACE is at Z = 0; the slab extends downward to Z ~= -0.92.
    table_min = np.array([-0.7402168, -1.2148621, -0.91964257])
    table_max = np.array([0.4688596, 1.2030163, 0.0])
    table_surface_z = 0.0

    # Panda robot base sits on the table surface
    robot_base = np.array([-0.615, 0.0, 0.0])

    # Print coverage stats
    print(f"\n{'='*72}")
    print("WORKSPACE COVERAGE ANALYSIS  (ManiSkill world coordinates)")
    print(f"{'='*72}")
    print(f"  Start positions: {len(start_positions)} points")
    print(f"    Min: X={start_positions[:, 0].min():+.3f}, Y={start_positions[:, 1].min():+.3f}, Z={start_positions[:, 2].min():+.3f}")
    print(f"    Max: X={start_positions[:, 0].max():+.3f}, Y={start_positions[:, 1].max():+.3f}, Z={start_positions[:, 2].max():+.3f}")
    print(f"\n  End (goal) positions: {len(end_positions)} points")
    print(f"    Min: X={end_positions[:, 0].min():+.3f}, Y={end_positions[:, 1].min():+.3f}, Z={end_positions[:, 2].min():+.3f}")
    print(f"    Max: X={end_positions[:, 0].max():+.3f}, Y={end_positions[:, 1].max():+.3f}, Z={end_positions[:, 2].max():+.3f}")
    print(f"\n  Combined bounding box (start + end):")
    print(f"    Min: X={bb_min[0]:+.3f}, Y={bb_min[1]:+.3f}, Z={bb_min[2]:+.3f}")
    print(f"    Max: X={bb_max[0]:+.3f}, Y={bb_max[1]:+.3f}, Z={bb_max[2]:+.3f}")
    print(f"    Size: {bb_size[0]:.3f} x {bb_size[1]:.3f} x {bb_size[2]:.3f} m")
    print(f"\n  Goal sampling region (env '{goal_env_id}', from dataset metadata):")
    print(f"    Center: {goal_region_center}")
    print(f"    Half-extents: {goal_region_half_extents}")
    print(f"    Min: {goal_region_min}")
    print(f"    Max: {goal_region_max}")
    print(f"\n  ManiSkill table (surface at Z=0, slab extends down to Z~=-0.92):")
    print(f"    Min: [{table_min[0]:+.3f}, {table_min[1]:+.3f}, {table_min[2]:+.3f}]")
    print(f"    Max: [{table_max[0]:+.3f}, {table_max[1]:+.3f}, {table_max[2]:+.3f}]")
    print(f"    Surface (Z): {table_surface_z:.3f} m  (robot base + start/goal points sit above this)")
    print(f"  Robot base: {robot_base}")
    print(f"\n  Coverage vs goal sampling region:")
    for i, ax in enumerate(["X", "Y", "Z"]):
        denom = 2 * goal_region_half_extents[i]
        pct = 100 * bb_size[i] / denom if denom > 1e-9 else float("nan")
        print(f"    {ax}: {pct:5.1f}% of axis range")
    print(f"\n  Unique goal positions: {len(all_goal_positions)}")

    # Create 3D plot
    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(111, projection="3d")

    # Plot unique-seed start positions (blue X) and end/goal positions (green o)
    ax.scatter(
        start_positions[:, 0], start_positions[:, 1], start_positions[:, 2],
        s=35, alpha=0.8, color="tab:blue", marker="x", linewidths=2,
        label=f"Starts ({len(start_positions)} seeds)"
    )
    ax.scatter(
        end_positions[:, 0], end_positions[:, 1], end_positions[:, 2],
        s=25, alpha=0.7, color="tab:green", marker="o", edgecolors="none",
        label=f"Goals ({len(end_positions)} seeds)"
    )

    # Draw the table top surface as a filled plane at Z=0
    xx, yy = np.meshgrid(
        np.linspace(table_min[0], table_max[0], 2),
        np.linspace(table_min[1], table_max[1], 2),
    )
    zz = np.full_like(xx, table_surface_z)
    ax.plot_surface(xx, yy, zz, color="tan", alpha=0.35, edgecolor="brown", linewidth=1)
    # proxy handle for legend
    ax.plot([], [], [], color="tan", linewidth=6, label="Table surface (Z=0)")

    # Draw robot base marker on the table
    ax.scatter([robot_base[0]], [robot_base[1]], [robot_base[2]],
               s=160, color="black", marker="s", label="Robot base")

    # Draw bounding box around data
    _draw_bbox_3d(ax, bb_min, bb_max, color="red", linestyle="-", linewidth=2, label="Data bounding box")

    # Draw goal sampling region box
    _draw_bbox_3d(ax, goal_region_min, goal_region_max, color="green", linestyle="--", linewidth=2, label="Goal sampling region")

    ax.set_xlabel("X (m)", fontsize=10)
    ax.set_ylabel("Y (m)", fontsize=10)
    ax.set_zlabel("Z (m)", fontsize=10)
    ax.set_title("Reach Scene: Table + Robot + Start/Goal Points (3D)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Set axis limits: full table footprint, Z from table surface up through the points
    margin = 0.1
    z_top = max(bb_max[2], goal_region_max[2]) + margin
    ax.set_xlim([table_min[0] - margin, table_max[0] + margin])
    ax.set_ylim([table_min[1] - margin, table_max[1] + margin])
    ax.set_zlim([table_surface_z - 0.05, z_top])

    plt.tight_layout()
    plt.savefig(out_dir / "workspace_coverage_3d.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Top-down XY view
    fig, ax = plt.subplots(figsize=(8, 8))

    # Unique-seed starts (blue X) and goals (green o)
    ax.scatter(start_positions[:, 0], start_positions[:, 1], s=40, alpha=0.8,
              color="tab:blue", marker="x", linewidths=2, label=f"Starts ({len(start_positions)} seeds)")
    ax.scatter(end_positions[:, 0], end_positions[:, 1], s=25, alpha=0.7,
              color="tab:green", marker="o", label=f"Goals ({len(end_positions)} seeds)")

    # Draw table footprint (tan rectangle)
    rect_table = plt.Rectangle(
        (table_min[0], table_min[1]),
        table_max[0] - table_min[0], table_max[1] - table_min[1],
        fill=True, facecolor="tan", edgecolor="brown", linewidth=2.5, alpha=0.25, label="Table footprint"
    )
    ax.add_patch(rect_table)

    # Robot base marker
    ax.scatter([robot_base[0]], [robot_base[1]], s=160, color="black", marker="s", label="Robot base", zorder=5)

    # Draw bounding box (2D projection)
    rect_data = plt.Rectangle(
        (bb_min[0], bb_min[1]),
        bb_size[0], bb_size[1],
        fill=False, edgecolor="red", linewidth=2, label="Data bounding box", linestyle="-"
    )
    ax.add_patch(rect_data)

    rect_goal = plt.Rectangle(
        (goal_region_min[0], goal_region_min[1]),
        goal_region_half_extents[0]*2, goal_region_half_extents[1]*2,
        fill=False, edgecolor="green", linewidth=2, label="Goal sampling region", linestyle="--"
    )
    ax.add_patch(rect_goal)

    ax.set_xlabel("X (m)", fontsize=10)
    ax.set_ylabel("Y (m)", fontsize=10)
    ax.set_title("Top-Down XY View — Table + Robot + Start/Goal Points", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    # Show full table footprint with margin
    margin = 0.15
    ax.set_xlim([table_min[0] - margin, table_max[0] + margin])
    ax.set_ylim([table_min[1] - margin, table_max[1] + margin])
    plt.tight_layout()
    plt.savefig(out_dir / "workspace_coverage_xy.png", dpi=120)
    plt.close()

    # Side view (XZ)
    fig, ax = plt.subplots(figsize=(8, 6))

    # Unique-seed starts (blue X) and goals (green o)
    ax.scatter(start_positions[:, 0], start_positions[:, 2], s=40, alpha=0.8,
              color="tab:blue", marker="x", linewidths=2, label=f"Starts ({len(start_positions)} seeds)")
    ax.scatter(end_positions[:, 0], end_positions[:, 2], s=25, alpha=0.7,
              color="tab:green", marker="o", label=f"Goals ({len(end_positions)} seeds)")

    # Draw table slab (from underside up to surface at Z=0)
    rect_table = plt.Rectangle(
        (table_min[0], table_min[2]),
        table_max[0] - table_min[0], table_max[2] - table_min[2],
        fill=True, facecolor="tan", edgecolor="brown", linewidth=2.5, alpha=0.3, label="Table slab"
    )
    ax.add_patch(rect_table)

    rect_data = plt.Rectangle(
        (bb_min[0], bb_min[2]),
        bb_size[0], bb_size[2],
        fill=False, edgecolor="red", linewidth=2, label="Data bounding box", linestyle="-"
    )
    ax.add_patch(rect_data)

    rect_goal = plt.Rectangle(
        (goal_region_min[0], goal_region_min[2]),
        goal_region_half_extents[0]*2, goal_region_half_extents[2]*2,
        fill=False, edgecolor="green", linewidth=2, label="Goal sampling region", linestyle="--"
    )
    ax.add_patch(rect_goal)

    # Robot: vertical marker standing on the table surface at x = robot_base x
    ax.plot([robot_base[0], robot_base[0]], [table_surface_z, 0.9],
            color="black", linewidth=3, alpha=0.7, label="Robot (base->reach)")
    ax.scatter([robot_base[0]], [table_surface_z], s=160, color="black", marker="s", zorder=5)

    ax.set_xlabel("X (m)", fontsize=10)
    ax.set_ylabel("Z (m)", fontsize=10)
    ax.set_title("Side XZ View — Table (slab), Robot on it, Start/Goal Points above", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    # Show table slab down to its underside, robot, and points above
    margin = 0.15
    ax.set_xlim([table_min[0] - margin, table_max[0] + margin])
    ax.set_ylim([table_min[2] - 0.05, max(bb_max[2], 0.9) + margin])
    plt.tight_layout()
    plt.savefig(out_dir / "workspace_coverage_xz.png", dpi=120)
    plt.close()

    print(f"\n  Workspace coverage plots saved to {out_dir}/ (in ManiSkill world coordinates)")


def _draw_bbox_3d(ax, bb_min, bb_max, color="black", linestyle="-", linewidth=1, label=None):
    """Draw a 3D bounding box on a matplotlib 3D axis."""
    vertices = np.array([
        [bb_min[0], bb_min[1], bb_min[2]],
        [bb_max[0], bb_min[1], bb_min[2]],
        [bb_max[0], bb_max[1], bb_min[2]],
        [bb_min[0], bb_max[1], bb_min[2]],
        [bb_min[0], bb_min[1], bb_max[2]],
        [bb_max[0], bb_min[1], bb_max[2]],
        [bb_max[0], bb_max[1], bb_max[2]],
        [bb_min[0], bb_max[1], bb_max[2]],
    ])

    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom
        [4, 5], [5, 6], [6, 7], [7, 4],  # top
        [0, 4], [1, 5], [2, 6], [3, 7],  # vertical
    ]

    for i, edge in enumerate(edges):
        pts = vertices[edge]
        ax.plot3D(*pts.T, color=color, linestyle=linestyle, linewidth=linewidth,
                  label=label if i == 0 else "")


if __name__ == "__main__":
    main()
