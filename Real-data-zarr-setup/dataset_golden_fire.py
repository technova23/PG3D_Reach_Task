#!/usr/bin/env python3
import os
import json
import re
from pathlib import Path
import sys
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

# Import the helpers from the newly created helper file
from dataset_creation_helper import (
    load_robot, check_reachability, extract_state_action, insert_goal_marker_points , 
    extract_depth, convert_to_zarr, MAX_DIFF, GOAL_MARKER_POINTS
)

# === GLOBAL CONFIGURATION ===
BAG_ROOT = Path("../data-check/")
TARGET_JSON = Path("episode_target_mapping.json")
OUT_ZARR = Path("real_reach_golden.zarr")
XARM7_URDF = Path(__file__).resolve().parents[1] / "pg3d/envs/xarm_adapter/assets/xarm7_with_gripper_colored.urdf"
# ============================

def main():
    print("=== Configuration ===")
    print(f"BAG_ROOT:    {BAG_ROOT}")
    print(f"TARGET_JSON: {TARGET_JSON}")
    print(f"OUT_ZARR:    {OUT_ZARR}")
    print(f"XARM7_URDF:  {XARM7_URDF}")
    print(f"MAX_DIFF:    {MAX_DIFF} m")
    print("=====================\n")

    with open(TARGET_JSON, "r") as f:
        target_map = json.load(f)

    fk_robot = load_robot(XARM7_URDF)
    episode_dirs = sorted(BAG_ROOT.glob("real_episode_*"))
    
    all_states = []
    all_actions = []
    all_targets = []
    all_pcds = []
    episode_ends = []
    
    running_end = 0
    n_success = 0
    n_skipped = 0

    for bag_dir in episode_dirs:
        m = re.match(r"real_episode_(\d+)_\d+", bag_dir.name)
        if not m: 
            continue
            
        episode_num = int(m.group(1))
        print(f"Processing Episode {episode_num:04d} ({bag_dir.name})")
        
        key = str(episode_num)
        if key not in target_map:
            print(f"  -> SKIP: Episode missing from target mapping.")
            n_skipped += 1
            continue
            
        target = np.asarray(target_map[key]["target_position"], dtype=np.float32)
        
        # 1. Check Reachability
        is_reachable, err = check_reachability(bag_dir, target, fk_robot)
        print(f"  Reachability Error: {err * 1000:.1f} mm")
        
        if not is_reachable:
            print(f"  -> SKIP: Not within {MAX_DIFF * 1000:.0f} mm.")
            n_skipped += 1
            continue
            
        # 2. Extract State and Action
        state, action, grid_times = extract_state_action(bag_dir)
        if state is None:
            print("  -> SKIP: Failed to extract states/actions.")
            n_skipped += 1
            continue
            
        # 3. Extract Depth (Point Clouds)
        print(f"  Extracting Point Clouds for {len(state)} frames...")
        pcds = extract_depth(bag_dir, grid_times)
        if pcds is None:
            print("  -> SKIP: Failed to extract point clouds.")
            n_skipped += 1
            continue
            
        # 4. Insert Goal Saliency Markers
        pcds = insert_goal_marker_points(
            pcds, 
            target, 
            num_points=GOAL_MARKER_POINTS
        )
        print(f"  Inserted {GOAL_MARKER_POINTS} goal markers into point cloud.")
        
        # Success!
        print(f"  -> SUCCESS! Added {len(state)} frames.")
        n_success += 1
        
        target_array = np.tile(target, (len(state), 1))
        
        all_states.append(state)
        all_actions.append(action)
        all_targets.append(target_array)
        all_pcds.append(pcds)
        
        running_end += len(state)
        episode_ends.append(running_end)

    print("\n==============================")
    print("Summary")
    print("==============================")
    print(f"Success : {n_success} episodes")
    print(f"Skipped : {n_skipped} episodes")

    if n_success > 0:
        print(f"\nConverting to Zarr at {OUT_ZARR}...")
        convert_to_zarr(OUT_ZARR, all_states, all_actions, all_targets, all_pcds, episode_ends)

if __name__ == "__main__":
    main()
