import zarr
import numpy as np
import rerun as rr
from pathlib import Path

ZARR_PATH = "real_reach_golden.zarr"

def main():
    if not Path(ZARR_PATH).exists():
        print(f"Dataset {ZARR_PATH} not found. Please run dataset_golden_fire.py first.")
        return

    root = zarr.open(ZARR_PATH, mode="r")
    print(f"Loaded {ZARR_PATH}")
    
    print("\n=== Dataset Shapes ===")
    print(f"State:       {root['data/state'].shape}")
    print(f"Action:      {root['data/action'].shape}")
    print(f"Target:      {root['data/target_position'].shape}")
    print(f"Point Cloud: {root['data/point_cloud'].shape}")
    print(f"Ep Ends:     {root['meta/episode_ends'].shape}")
    
    ep_ends = root['meta/episode_ends'][:]
    if len(ep_ends) == 0:
        print("\nNo episodes found in the dataset.")
        return
        
    first_ep_end = ep_ends[0]
    
    print(f"\nLoading {first_ep_end} point cloud frames for Episode 1...")
    first_ep_pcds = root['data/point_cloud'][:first_ep_end]
    
    print("Visualizing across time in Rerun...")
    rr.init("verify_dataset", spawn=True)
    
    for i, pcd in enumerate(first_ep_pcds):
        rr.set_time_sequence("frame_idx", i)
        
        rr.log(
            "world/point_cloud", 
            rr.Points3D(
                positions=pcd, 
                radii=0.005
            )
        )
        
    print("\nDone! Rerun viewer should be open.")

if __name__ == "__main__":
    main()


