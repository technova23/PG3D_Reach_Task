"""
This file constructs a zarr dataset from a set of real rosbag episodes.
Synchronization is nearest-neighbour (no interpolation): each 20 Hz grid
tick is assigned the joint state from the closest actual message.

python Real-data-zarr-setup/construct-zarr.py \
  --bag-root ../data-check/ \
  --target-json Real-data-zarr-setup/episode_target_mapping.json \
  --out-zarr real_reach_dataset.zarr \
  --hz 20
"""
  
import argparse
import json
import re
from pathlib import Path

import numpy as np
import zarr
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

def _parse_args():

    p = argparse.ArgumentParser(description="Construct zarr from real rosbag episodes")

    p.add_argument("--bag-root", type=Path, default=Path("../data-check/"))
    p.add_argument("--target-json", type=Path, default=Path("episode_target_mapping.json"))
    p.add_argument("--out-zarr", type=Path, default=Path("real_reach_dataset.zarr"))
    p.add_argument("--hz", type=float, default=20.0, help="Resample/sync frequency in Hz")
    return p.parse_args()

args = _parse_args()

BAG_ROOT = args.bag_root
TARGET_JSON = args.target_json
OUT_ZARR = args.out_zarr
RESAMPLE_HZ = float(args.hz)


def _parse_target(entry: dict) -> np.ndarray:
    """
    NOTE : Generate this json by running the `analyse-sim-data.ipynb` 

    Extract target_position from one episode_target_mapping entry.

    Expected entry shape (from episode_target_mapping.json):
      {
        "start_idx": int,
        "end_idx": int,
        "num_frames": int,
        "target_position": [x, y, z]
      }
    """
    pos = entry["target_position"]          # list of 3 floats
    return np.asarray(pos, dtype=np.float32)



def _sync_to_grid(
    times: np.ndarray,
    positions: np.ndarray,
    hz: float,
) -> np.ndarray:
    """
    Snap raw messages onto a uniform grid at `hz` Hz using nearest-neighbour
    lookup.  No values are fabricated — every returned row is a real message.

    Args:
        times:     shape (M,)   message timestamps in seconds, sorted.
        positions: shape (M, J) joint positions for each message.
        hz:        target grid frequency.

    Returns:
        state: shape (N, J) float32, where N = number of grid ticks.
    """
    t_start, t_end = float(times[0]), float(times[-1])
    if t_end <= t_start:
        return positions.astype(np.float32, copy=True)

    step = 1.0 / hz
    grid = np.arange(t_start, t_end + 1e-9, step, dtype=np.float64)

    # searchsorted gives the right-neighbour index for each grid tick
    right = np.clip(np.searchsorted(times, grid), 0, len(times) - 1)
    left  = np.clip(right - 1,                    0, len(times) - 1)

    # pick whichever neighbour is closer in time
    use_left = (grid - times[left]) < (times[right] - grid)
    idx = np.where(use_left, left, right)

    return positions[idx].astype(np.float32)


# episode_number -> [x, y, z]

typestore = get_typestore(Stores.ROS2_HUMBLE)

with open(TARGET_JSON, "r") as f:
    target_map = json.load(f)

all_states = []
all_actions = []
all_targets = []
episode_ends = []

running_end = 0

episode_dirs = sorted(BAG_ROOT.glob("real_episode_*"))

print(f"Found {len(episode_dirs)} episodes.\n")

for bag_dir in episode_dirs:

    m = re.match(r"real_episode_(\d+)_\d+", bag_dir.name)
    if m is None:
        print(f"Skipping {bag_dir.name}")
        continue

    episode_num = int(m.group(1))

    print(f"Episode {episode_num:04d}")

    states = []


    with AnyReader([bag_dir], default_typestore=typestore) as reader:

        lengths = {}
        msg_times = []
        msg_positions = []

        for conn, ts, raw in reader.messages():
            if conn.topic != "/xarm/joint_states":
                continue

            msg = reader.deserialize(raw, conn.msgtype)

            n = len(msg.position)
            lengths[n] = lengths.get(n, 0) + 1

            # print(f"{n} joints: {list(msg.name)}")

            # Skip malformed messages
            if n != 7:
                # print(f"Skipping malformed JointState with {n} joints.")
                continue

            # record timestamp (ns -> seconds) and position
            t = float(ts) / 1e9
            msg_times.append(t)
            msg_positions.append(np.asarray(msg.position, dtype=np.float64))

    print("\nJointState length histogram:")
    for k in sorted(lengths):
        print(f"  {k} joints : {lengths[k]} messages")

    if len(msg_positions) == 0:
        print("No valid joint states found.")
        continue

    times = np.asarray(msg_times, dtype=np.float64)
    positions = np.stack(msg_positions, axis=0)  # shape [M, 7]

    state = _sync_to_grid(times, positions, RESAMPLE_HZ)
    print(f"  Synced frames (nearest-neighbour): {state.shape[0]} @ {RESAMPLE_HZ}Hz")

    # -------------------------------------------------------------------------
    # action = next state
    # repeat last state for final action
    # -------------------------------------------------------------------------

    action = np.empty_like(state)
    action[:-1] = state[1:]
    action[-1] = state[-1]

    # -------------------------------------------------------------------------
    # target position
    # -------------------------------------------------------------------------

    key = str(episode_num)
    if key not in target_map:
        raise KeyError(f"Episode {episode_num} missing from target mapping.")

    target = _parse_target(target_map[key])  # shape (3,)
    target_position = np.tile(target, (len(state), 1))  # shape (N, 3)


    all_states.append(state)
    all_actions.append(action)
    all_targets.append(target_position)

    running_end += len(state)
    episode_ends.append(running_end)

    print(f"  Frames : {len(state)}")

# =============================================================================
# Concatenate
# =============================================================================

state = np.concatenate(all_states, axis=0)
action = np.concatenate(all_actions, axis=0)
target_position = np.concatenate(all_targets, axis=0)
episode_ends = np.asarray(episode_ends, dtype=np.int64)

print("\n==============================")
print("Final Dataset")
print("==============================")
print("State :", state.shape)
print("Action:", action.shape)
print("Target:", target_position.shape)
print("Episodes:", len(episode_ends))

# =============================================================================
# Write Zarr
# =============================================================================

root = zarr.open(str(OUT_ZARR), mode="w")

data = root.create_group("data")
meta = root.create_group("meta")

data.create_dataset(
    "state",
    data=state,
    chunks=(1024, state.shape[1]),
)

data.create_dataset(
    "action",
    data=action,
    chunks=(1024, action.shape[1]),
)

data.create_dataset(
    "target_position",
    data=target_position,
    chunks=(1024, 3),
)

meta.create_dataset(
    "episode_ends",
    data=episode_ends,
)

print("\nSaved to:", OUT_ZARR)
print(root.tree())