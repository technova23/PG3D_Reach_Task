"""Visual + quantitative diagnostics for a multimodal reach dataset.

Usage:
    python dataset_generation/analyze_multimodal_dataset.py \
        --dataset artifacts/debug_multimodal_test.zarr \
        --output-dir artifacts/dataset_analysis

The quantitative "path spread" metric (see `_family_path_spread_metric`) exists
specifically to A/B two datasets (e.g. old vs. expanded xArm7 workspace) before
sinking time into a full retrain: it measures, for each unique (start, goal)
pair, how much the ~12 trajectory-family variants actually diverge from one
another geometrically. Low spread means rejection/reranking has little to
choose between at inference time, regardless of how many candidates it samples.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr


def _path_lateral_width(path: np.ndarray) -> float:
    """Peak perpendicular deviation of `path` from the straight chord joining
    its first and last points -- i.e. how far the trajectory "bows out".
    Mirrors `_path_lateral_width` in scripts/eval_constrained_reach.py; kept as
    an independent copy here so this script stays free of the eval scripts'
    heavy mani_skill/torch imports.
    """
    points = np.asarray(path, dtype=np.float32).reshape(-1, 3)
    if points.shape[0] < 2:
        return 0.0
    start = points[0]
    chord = points[-1] - start
    chord_length = float(np.linalg.norm(chord))
    rel = points - start.reshape(1, 3)
    if chord_length <= 1e-8:
        return float(np.max(np.linalg.norm(rel, axis=1)))
    unit = chord / chord_length
    projection = rel @ unit
    perpendicular = rel - np.outer(projection, unit)
    return float(np.max(np.linalg.norm(perpendicular, axis=1)))


def _family_path_spread_metric(
    *,
    tcp_pose: np.ndarray,
    goal_pos: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    decimals: int = 2,
) -> dict:
    """Quantify path multimodality: average, across unique (start, goal) pairs,
    the standard deviation of `_path_lateral_width` across that pair's family
    variants. Episodes are grouped by rounded start/goal position match.

    Family variants generated in the same reset share the same *nominal*
    start/goal position, but are no longer bit-identical: under
    --randomize-start-goal-orientation (write_maniskill_reach_dataset.py),
    each family independently IK/motion-plans its own start pose (same
    target xyz, different target orientation), so the actual achieved
    position can differ by small numerical amounts across families. Default
    decimals=2 (~1cm) tolerates that while still separating genuinely
    different start/goal pairs, which are normally resampled much farther
    apart than that.
    """
    start_xyz = tcp_pose[starts][:, :3]
    goal_xyz = goal_pos[starts]
    groups: dict[tuple, list[int]] = {}
    for episode_idx, (s, g) in enumerate(zip(start_xyz, goal_xyz, strict=False)):
        key = (
            tuple(np.round(s, decimals).tolist()),
            tuple(np.round(g, decimals).tolist()),
        )
        groups.setdefault(key, []).append(episode_idx)

    per_pair_spread: list[float] = []
    per_pair_family_count: list[int] = []
    for episode_indices in groups.values():
        if len(episode_indices) < 2:
            continue
        widths = []
        for episode_idx in episode_indices:
            s = starts[episode_idx]
            e = ends[episode_idx]
            widths.append(_path_lateral_width(tcp_pose[s:e, :3]))
        per_pair_spread.append(float(np.std(widths)))
        per_pair_family_count.append(len(episode_indices))

    if not per_pair_spread:
        return {
            "num_start_goal_pairs": 0,
            "mean_family_count_per_pair": None,
            "mean_lateral_width_std": None,
            "median_lateral_width_std": None,
        }
    return {
        "num_start_goal_pairs": len(per_pair_spread),
        "mean_family_count_per_pair": float(np.mean(per_pair_family_count)),
        "mean_lateral_width_std": float(np.mean(per_pair_spread)),
        "median_lateral_width_std": float(np.median(per_pair_spread)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visual + quantitative multimodality diagnostics for a reach dataset."
    )
    parser.add_argument("--dataset", type=Path, required=True, help="Path to the Zarr dataset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/dataset_analysis"),
        help="Directory to write plots and summary.json to.",
    )
    parser.add_argument(
        "--num-families",
        type=int,
        default=12,
        help="Number of trajectory families to expect (for the overlay plot).",
    )
    parser.add_argument(
        "--position-match-decimals",
        type=int,
        default=2,
        help=(
            "Decimal places used to group episodes by start/goal position when "
            "computing path_multimodality (default 2, ~1cm). Loosen (lower) if a dataset "
            "generated with --randomize-start-goal-orientation shows num_start_goal_pairs=0 "
            "-- independent per-family orientation IK solves make positions no longer "
            "bit-identical across family variants."
        ),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    root = zarr.open_group(str(args.dataset), mode="r")

    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    goal_pos = np.asarray(root["data"]["goal_pos"][:], dtype=np.float32)
    tcp_pose = np.asarray(root["data"]["tcp_pose"][:], dtype=np.float32)
    family_id = np.asarray(root["data"]["trajectory_family_id"][:], dtype=np.int64).squeeze()

    num_episodes = len(episode_ends)
    starts = np.concatenate([[0], episode_ends[:-1]])
    ends = episode_ends
    episode_lengths = ends - starts

    # --------------------------------------------------
    # 1. Episode length distribution
    # --------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.hist(episode_lengths, bins=40)
    plt.title("Episode Length Distribution")
    plt.xlabel("Episode Length")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(args.output_dir / "episode_length_distribution.png")
    plt.close()

    # --------------------------------------------------
    # 2. Episodes per family
    # --------------------------------------------------
    traj_family_counts: dict[int, int] = {}
    for s in starts:
        fid = int(family_id[s])
        traj_family_counts[fid] = traj_family_counts.get(fid, 0) + 1

    plt.figure(figsize=(8, 5))
    plt.bar(list(traj_family_counts.keys()), list(traj_family_counts.values()))
    plt.title("Episodes Per Family")
    plt.xlabel("Family ID")
    plt.ylabel("Number of Episodes")
    plt.tight_layout()
    plt.savefig(args.output_dir / "episodes_per_family.png")
    plt.close()

    # --------------------------------------------------
    # 3. Start Goal Diversity
    # --------------------------------------------------
    start_tcp = tcp_pose[starts][:, :3]
    goal_xyz = goal_pos[starts]

    plt.figure(figsize=(7, 7))
    plt.scatter(start_tcp[:, 0], start_tcp[:, 1], s=10, label="Start TCP")
    plt.scatter(goal_xyz[:, 0], goal_xyz[:, 1], s=10, label="Goal")
    plt.title("Start Goal Diversity")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(args.output_dir / "start_goal_diversity.png")
    plt.close()

    # --------------------------------------------------
    # 4. Random TCP Trajectories
    # --------------------------------------------------
    rng = np.random.default_rng(0)
    num_plot = min(40, num_episodes)
    selected = rng.choice(np.arange(num_episodes), size=num_plot, replace=False)

    plt.figure(figsize=(8, 8))
    for ep in selected:
        s = starts[ep]
        e = ends[ep]
        traj = tcp_pose[s:e, :3]
        plt.plot(traj[:, 0], traj[:, 1], alpha=0.5)
    plt.title("Random TCP Trajectory Diversity")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(args.output_dir / "random_tcp_diversity.png")
    plt.close()

    # --------------------------------------------------
    # 5. Family Diversity Overlay
    # --------------------------------------------------
    plt.figure(figsize=(8, 8))
    for fam in range(args.num_families):
        fam_eps = [ep for ep in range(num_episodes) if int(family_id[starts[ep]]) == fam]
        if not fam_eps:
            continue
        for ep in fam_eps[:10]:
            s = starts[ep]
            e = ends[ep]
            traj = tcp_pose[s:e, :3]
            plt.plot(traj[:, 0], traj[:, 1], alpha=0.6)
    plt.title("Trajectory Family Diversity")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(args.output_dir / "family_diversity_overlay.png")
    plt.close()

    # --------------------------------------------------
    # 6. Quantitative multimodality metric
    # --------------------------------------------------
    spread_metric = _family_path_spread_metric(
        tcp_pose=tcp_pose,
        goal_pos=goal_pos,
        starts=starts,
        ends=ends,
        decimals=args.position_match_decimals,
    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------
    summary = {
        "dataset": str(args.dataset),
        "num_episodes": int(num_episodes),
        "num_steps": int(episode_ends[-1]),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "min_episode_length": int(np.min(episode_lengths)),
        "max_episode_length": int(np.max(episode_lengths)),
        "family_counts": {str(k): int(v) for k, v in traj_family_counts.items()},
        "path_multimodality": spread_metric,
    }

    with (args.output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("Saved analysis to:", args.output_dir)
    print(
        "path_multimodality: "
        f"num_start_goal_pairs={spread_metric['num_start_goal_pairs']} "
        f"mean_family_count_per_pair={spread_metric['mean_family_count_per_pair']} "
        f"mean_lateral_width_std={spread_metric['mean_lateral_width_std']} "
        f"median_lateral_width_std={spread_metric['median_lateral_width_std']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
