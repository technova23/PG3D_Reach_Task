#!/usr/bin/env python
"""Visualize xArm7 gripper reach trajectories from a zarr dataset using matplotlib.

Usage:
    python scripts/visualize_xarm_trajectories.py \\
        --dataset artifacts/pg3d_xarm7_gripper_reach_viz.zarr \\
        --output artifacts/xarm_trajectory_analysis
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Visualize xArm7 gripper reach trajectories from a zarr dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="artifacts/pg3d_xarm7_gripper_reach_viz.zarr",
        help="Path to zarr dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/xarm_trajectory_analysis",
        help="Output directory for plots",
    )
    args = parser.parse_args(argv)

    dataset_path = args.dataset
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {dataset_path}")
    try:
        root = zarr.open_group(dataset_path, mode="r")
    except Exception as e:
        print(f"Error opening zarr: {e}")
        print("Data generation may still be running. Try again in a moment.")
        return 1

    episode_ends = root["meta"]["episode_ends"][:]
    goal_pos = root["data"]["goal_pos"][:]
    tcp_pose = root["data"]["tcp_pose"][:]
    family_id = root["data"]["trajectory_family_id"][:].squeeze()

    num_episodes = len(episode_ends)
    print(f"Loaded {num_episodes} episodes")

    # Episode boundaries
    starts = np.concatenate([[0], episode_ends[:-1]])
    ends = episode_ends
    episode_lengths = ends - starts

    # --------------------------------------------------
    # 1. Episode length distribution
    # --------------------------------------------------
    print("  → plotting episode length distribution...")
    plt.figure(figsize=(8, 5))
    plt.hist(episode_lengths, bins=40, edgecolor="black", alpha=0.7)
    plt.title("Episode Length Distribution (xArm7 Gripper Reach)")
    plt.xlabel("Episode Length (steps)")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "01_episode_length_distribution.png", dpi=100)
    plt.close()

    # --------------------------------------------------
    # 2. Episodes per family
    # --------------------------------------------------
    print("  → plotting episodes per family...")
    traj_family_counts = {}
    for s in starts:
        fid = int(family_id[s])
        traj_family_counts[fid] = traj_family_counts.get(fid, 0) + 1

    family_names = [
        "downward_arc",
        "downward_straight",
        "left_arc",
        "left_straight",
        "right_arc",
        "right_straight",
        "up_arc",
        "up_straight",
        "left_wide_arc",
        "left_wide_straight",
        "right_wide_arc",
        "right_wide_straight",
    ]

    plt.figure(figsize=(10, 5))
    fam_ids = sorted(traj_family_counts.keys())
    fam_labels = [
        family_names[fid] if fid < len(family_names) else f"family_{fid}"
        for fid in fam_ids
    ]
    plt.bar(fam_labels, [traj_family_counts[fid] for fid in fam_ids], edgecolor="black")
    plt.title("Episodes Per Trajectory Family (xArm7 Gripper Reach)")
    plt.xlabel("Trajectory Family")
    plt.ylabel("Number of Episodes")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "02_episodes_per_family.png", dpi=100)
    plt.close()

    # --------------------------------------------------
    # 3. Start Goal Diversity (XY scatter)
    # --------------------------------------------------
    print("  → plotting start/goal diversity...")
    start_tcp = tcp_pose[starts][:, :3]
    goal_xyz = goal_pos[starts]

    plt.figure(figsize=(8, 8))
    plt.scatter(
        start_tcp[:, 0],
        start_tcp[:, 1],
        s=20,
        alpha=0.6,
        label="Start TCP",
        edgecolors="blue",
    )
    plt.scatter(
        goal_xyz[:, 0],
        goal_xyz[:, 1],
        s=20,
        alpha=0.6,
        label="Goal",
        edgecolors="red",
        marker="x",
    )
    plt.title("Start Goal Diversity (XY Plane, xArm7 Gripper Reach)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(output_dir / "03_start_goal_diversity.png", dpi=100)
    plt.close()

    # --------------------------------------------------
    # 4. Random TCP Trajectories
    # --------------------------------------------------
    print("  → plotting random TCP trajectories...")
    rng = np.random.default_rng(0)
    num_plot = min(40, num_episodes)

    selected = rng.choice(
        np.arange(num_episodes),
        size=num_plot,
        replace=False,
    )

    plt.figure(figsize=(10, 10))
    for ep in selected:
        s = starts[ep]
        e = ends[ep]
        traj = tcp_pose[s:e, :3]
        plt.plot(traj[:, 0], traj[:, 1], alpha=0.5, linewidth=0.8)

    plt.title(f"Random TCP Trajectories ({num_plot} episodes, XY Plane)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(output_dir / "04_random_tcp_trajectories.png", dpi=100)
    plt.close()

    # --------------------------------------------------
    # 5. Family Diversity Overlay
    # --------------------------------------------------
    print("  → plotting family diversity overlay...")
    plt.figure(figsize=(10, 10))

    colors = plt.cm.tab20(np.linspace(0, 1, 12))

    for fam in range(12):
        fam_eps = []

        for ep in range(num_episodes):
            if int(family_id[starts[ep]]) == fam:
                fam_eps.append(ep)

        if not fam_eps:
            continue

        fam_eps = fam_eps[:10]  # Up to 10 per family

        for ep in fam_eps:
            s = starts[ep]
            e = ends[ep]
            traj = tcp_pose[s:e, :3]

            label = family_names[fam] if fam < len(family_names) else f"family_{fam}"
            if ep == fam_eps[0]:  # Label once per family
                plt.plot(
                    traj[:, 0],
                    traj[:, 1],
                    alpha=0.6,
                    linewidth=1.0,
                    color=colors[fam],
                    label=label,
                )
            else:
                plt.plot(
                    traj[:, 0],
                    traj[:, 1],
                    alpha=0.6,
                    linewidth=1.0,
                    color=colors[fam],
                )

    plt.title("Trajectory Family Diversity Overlay (XY Plane, up to 10 per family)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend(fontsize=8, loc="best")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(output_dir / "05_family_diversity_overlay.png", dpi=100)
    plt.close()

    # --------------------------------------------------
    # 6. Z-axis trajectory visualization (side view)
    # --------------------------------------------------
    print("  → plotting Z-axis trajectories...")
    plt.figure(figsize=(10, 6))

    for ep in selected[:20]:  # First 20 episodes
        s = starts[ep]
        e = ends[ep]
        traj = tcp_pose[s:e, :3]
        steps = np.arange(len(traj))
        plt.plot(steps, traj[:, 2], alpha=0.5, linewidth=0.8)

    plt.title("Z-axis Evolution Over Steps (20 random episodes)")
    plt.xlabel("Step")
    plt.ylabel("Z (m)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "06_z_axis_trajectories.png", dpi=100)
    plt.close()

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------
    print("  → writing summary...")
    summary = {
        "num_episodes": int(num_episodes),
        "num_steps": int(episode_ends[-1]),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "min_episode_length": int(np.min(episode_lengths)),
        "max_episode_length": int(np.max(episode_lengths)),
        "family_counts": {
            str(k): int(v) for k, v in sorted(traj_family_counts.items())
        },
        "mean_start_goal_distance": float(
            np.mean(np.linalg.norm(goal_xyz - start_tcp, axis=1))
        ),
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Saved analysis to: {output_dir}")
    print(f"  → {num_episodes} episodes across {len(traj_family_counts)} families")
    print(f"  → mean episode length: {summary['mean_episode_length']:.1f} steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
