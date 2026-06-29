import json
from pathlib import Path

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr

DATASET = "artifacts/debug_multimodal_test.zarr"
OUTDIR = Path("artifacts/dataset_analysis")
OUTDIR.mkdir(parents=True, exist_ok=True)

root = zarr.open_group(DATASET, mode="r")

episode_ends = root["meta"]["episode_ends"][:]

goal_pos = root["data"]["goal_pos"][:]
tcp_pose = root["data"]["tcp_pose"][:]
family_id = root["data"]["trajectory_family_id"][:].squeeze()

num_episodes = len(episode_ends)

# --------------------------------------------------
# Episode boundaries
# --------------------------------------------------

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
plt.savefig(OUTDIR / "episode_length_distribution.png")
plt.close()

# --------------------------------------------------
# 2. Episodes per family
# --------------------------------------------------

traj_family_counts = {}

for s in starts:
    fid = int(family_id[s])
    traj_family_counts[fid] = traj_family_counts.get(fid, 0) + 1

plt.figure(figsize=(8, 5))
plt.bar(
    list(traj_family_counts.keys()),
    list(traj_family_counts.values()),
)
plt.title("Episodes Per Family")
plt.xlabel("Family ID")
plt.ylabel("Number of Episodes")
plt.tight_layout()
plt.savefig(OUTDIR / "episodes_per_family.png")
plt.close()

# --------------------------------------------------
# 3. Start Goal Diversity
# --------------------------------------------------

start_tcp = tcp_pose[starts][:, :3]
goal_xyz = goal_pos[starts]

plt.figure(figsize=(7, 7))

plt.scatter(
    start_tcp[:, 0],
    start_tcp[:, 1],
    s=10,
    label="Start TCP",
)

plt.scatter(
    goal_xyz[:, 0],
    goal_xyz[:, 1],
    s=10,
    label="Goal",
)

plt.title("Start Goal Diversity")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.savefig(OUTDIR / "start_goal_diversity.png")
plt.close()

# --------------------------------------------------
# 4. Random TCP Trajectories
# --------------------------------------------------

rng = np.random.default_rng(0)

num_plot = min(40, num_episodes)

selected = rng.choice(
    np.arange(num_episodes),
    size=num_plot,
    replace=False,
)

plt.figure(figsize=(8, 8))

for ep in selected:
    s = starts[ep]
    e = ends[ep]

    traj = tcp_pose[s:e, :3]

    plt.plot(
        traj[:, 0],
        traj[:, 1],
        alpha=0.5,
    )

plt.title("Random TCP Trajectory Diversity")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis("equal")
plt.tight_layout()
plt.savefig(OUTDIR / "random_tcp_diversity.png")
plt.close()

# --------------------------------------------------
# 5. Family Diversity Overlay
# --------------------------------------------------

plt.figure(figsize=(8, 8))

for fam in range(12):

    fam_eps = []

    for ep in range(num_episodes):
        if int(family_id[starts[ep]]) == fam:
            fam_eps.append(ep)

    if not fam_eps:
        continue

    fam_eps = fam_eps[:10]

    for ep in fam_eps:

        s = starts[ep]
        e = ends[ep]

        traj = tcp_pose[s:e, :3]

        plt.plot(
            traj[:, 0],
            traj[:, 1],
            alpha=0.6,
        )

plt.title("Trajectory Family Diversity")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis("equal")
plt.tight_layout()
plt.savefig(OUTDIR / "family_diversity_overlay.png")
plt.close()

# --------------------------------------------------
# Summary
# --------------------------------------------------

summary = {
    "num_episodes": int(num_episodes),
    "num_steps": int(episode_ends[-1]),
    "mean_episode_length": float(np.mean(episode_lengths)),
    "min_episode_length": int(np.min(episode_lengths)),
    "max_episode_length": int(np.max(episode_lengths)),
    "family_counts": {
        str(k): int(v)
        for k, v in traj_family_counts.items()
    },
}

with open(OUTDIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("Saved analysis to:", OUTDIR)

