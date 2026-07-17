"""Build a mixed real+sim training zarr at /scratch2/skills/pg3d_real_sim_mixed.zarr.

Base = real_reach_golden_copy.zarr (real robot, world-frame-fixed + tcp_pose
backfilled via FK -- see fix_real_zarr_for_mixing.py). Appended = sim's
pg3d_xarm7_gripper_reach_final.zarr, restricted to the 5 fields the real copy
actually has (action, state, point_cloud, target_position, tcp_pose) -- sim's
extra fields (eef_pos, goal_pos, goal_relative, point_valid_mask, robot_mask,
sim_action, success, trajectory_family_id/onehot) have no real-data
counterpart and are dropped so both halves share one schema.

Written directly on /scratch2 (903GB free) -- root disk only has ~1.2GB free
and this needs ~3.8GB, almost entirely from sim's point_cloud.

Provenance: a `source` metadata.json records exactly which episode/step
ranges came from which original file, so the real portion can be identified
and sliced back out later (e.g. `episodes[:699]` / `steps[:34272]` is 100%
real, everything after is sim) without needing to diff file contents.
Neither original zarr, nor real_reach_golden_copy.zarr, is modified.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import zarr

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_COPY = REPO_ROOT / "Real-data-zarr-setup" / "real_reach_golden_copy.zarr"
SIM = Path("/scratch2/skills/pg3d_xarm7_gripper_reach_final.zarr")
DST = Path("/scratch2/skills/pg3d_real_sim_mixed.zarr")

FIELDS = ["action", "state", "point_cloud", "target_position", "tcp_pose"]


def main() -> None:
    if DST.exists():
        raise FileExistsError(f"refusing to overwrite existing {DST}")

    print(f"seeding {DST} from {REAL_COPY} (real, base)")
    shutil.copytree(REAL_COPY, DST)

    real_root = zarr.open_group(str(REAL_COPY), mode="r")  # read-only, untouched
    sim_root = zarr.open_group(str(SIM), mode="r")          # read-only, untouched
    dst_root = zarr.open_group(str(DST), mode="a")
    dst_data = dst_root["data"]

    real_episode_ends = np.asarray(real_root["meta"]["episode_ends"][:], dtype=np.int64)
    sim_episode_ends = np.asarray(sim_root["meta"]["episode_ends"][:], dtype=np.int64)
    real_steps = int(real_episode_ends[-1])
    sim_steps = int(sim_episode_ends[-1])
    real_episodes = len(real_episode_ends)
    sim_episodes = len(sim_episode_ends)

    print(f"real: {real_episodes} episodes, {real_steps} steps")
    print(f"sim : {sim_episodes} episodes, {sim_steps} steps")

    # --- append sim's 5 shared fields, one at a time to bound peak memory ---
    for key in FIELDS:
        print(f"appending sim/{key} ...")
        sim_arr = np.asarray(sim_root["data"][key][:])
        dst_data[key].append(sim_arr, axis=0)

    # --- extend episode_ends: sim's own ends offset by real's total step count ---
    combined_episode_ends = np.concatenate([real_episode_ends, sim_episode_ends + real_steps])
    dst_root["meta"]["episode_ends"].resize(combined_episode_ends.shape)
    dst_root["meta"]["episode_ends"][:] = combined_episode_ends

    total_steps = int(combined_episode_ends[-1])
    total_episodes = len(combined_episode_ends)
    print(f"combined: {total_episodes} episodes, {total_steps} steps")

    # --- provenance record, so the real portion can be found/removed later ---
    provenance = {
        "sources": [
            {
                "name": "real",
                "origin_file": str(REAL_COPY),
                "derived_from": str(REPO_ROOT / "Real-data-zarr-setup" / "real_reach_golden.zarr"),
                "episode_range": [0, real_episodes],
                "step_range": [0, real_steps],
                "notes": (
                    "Real robot recordings. point_cloud/target_position shifted by "
                    "[-0.615, 0, 0] (base-relative -> world frame); tcp_pose backfilled "
                    "via forward kinematics (was not recorded). See "
                    "fix_real_zarr_for_mixing.py."
                ),
            },
            {
                "name": "sim",
                "origin_file": str(SIM),
                "episode_range": [real_episodes, total_episodes],
                "step_range": [real_steps, total_steps],
                "notes": "ManiSkill-generated. Only the 5 fields shared with real data were kept.",
            },
        ],
        "fields": FIELDS,
        "how_to_remove_real_data": (
            f"episodes[{real_episodes}:] / steps[{real_steps}:] is 100% sim -- "
            f"slice both `data/*` arrays and `meta/episode_ends` (subtract {real_steps} "
            "from the kept episode_ends) to drop the real portion."
        ),
    }
    (DST / "metadata.json").write_text(json.dumps(provenance, indent=2))
    print(f"wrote provenance: {DST / 'metadata.json'}")


if __name__ == "__main__":
    main()
