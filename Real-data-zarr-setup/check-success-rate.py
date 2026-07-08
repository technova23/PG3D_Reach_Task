#!/usr/bin/env python3
import json
import re
from pathlib import Path
import numpy as np

# Import the helpers
from dataset_creation_helper import (
    load_robot, check_reachability, MAX_DIFF
)

BAG_ROOT = Path("/scratch2/skills/reach_task_real_episodes_xarm7/july17-combined")
TARGET_JSON = Path("episode_target_mapping.json")
XARM7_URDF = Path(__file__).resolve().parents[1] / "pg3d/envs/xarm_adapter/assets/xarm7_with_gripper_colored.urdf"

def main():
    if not TARGET_JSON.exists():
        print(f"Error: {TARGET_JSON} not found.")
        return
        
    print(f"Loading target map from {TARGET_JSON}...")
    with open(TARGET_JSON, "r") as f:
        target_map = json.load(f)

    print(f"Loading robot URDF from {XARM7_URDF}...")
    fk_robot = load_robot(XARM7_URDF)
    
    episode_dirs = sorted(BAG_ROOT.glob("real_episode_*"))
    
    results = []
    
    print("\nEvaluating Reachability Success Rate...\n")
    print(f"{'Episode':<10} | {'Status':<10} | {'Error (mm)':<12}")
    print("-" * 38)
    
    for bag_dir in episode_dirs:
        m = re.match(r"real_episode_(\d+)_\d+", bag_dir.name)
        if not m:
            continue
            
        episode_num = int(m.group(1))
        key = str(episode_num)
        
        if key not in target_map:
            status = "MISSING JSON"
            print(f"{episode_num:<10d} | {status:<10} | {'N/A':<12}")
            results.append((episode_num, status, None))
            continue
            
        target = np.asarray(target_map[key]["target_position"], dtype=np.float32)
        is_reachable, err = check_reachability(bag_dir, target, fk_robot)
        
        status = "SUCCESS" if is_reachable else "FAILED"
        err_mm = err * 1000.0
        
        print(f"{episode_num:<10d} | {status:<10} | {err_mm:<12.1f}")
        results.append((episode_num, status, err_mm))
        
    # Tabulate Final Results
    total = len(results)
    success = sum(1 for r in results if r[1] == "SUCCESS")
    failed = sum(1 for r in results if r[1] == "FAILED")
    skipped = sum(1 for r in results if r[1] == "MISSING JSON")
    
    print("-" * 38)
    print("\n=== Final Summary ===")
    print(f"Total Episodes Processed: {total}")
    print(f"Successes: {success}")
    print(f"Failures:  {failed}")
    print(f"Skipped:   {skipped}")
    
    valid_eps = total - skipped
    if valid_eps > 0:
        rate = (success / valid_eps) * 100.0
        print(f"\nSuccess Rate: {rate:.1f}% (Threshold: {MAX_DIFF * 1000.0:.1f} mm)")
    
if __name__ == "__main__":
    main()
