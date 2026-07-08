#!/usr/bin/env python3
"""
offset-correction.py

This script corrects the base offset between the Simulator (Sim) and the Real World (Real) distributions.
The Simulator has the robot base shifted by [-0.615, 0.0, 0.0] relative to the world origin.
The Real World has the robot base at the world origin [0.0, 0.0, 0.0].

To combine their data, we must align their coordinate frames.

Choice M1 (Correct Real Data):
- Shifts Real data to match the Sim world frame.
- Adds the offset [-0.615, 0.0, 0.0] to point_cloud, target_position, and tcp_pose.
- Use this if you want to train and execute policies in the Simulator world frame.

Choice M2 (Correct Sim Data):
- Shifts Sim data to match the Real world frame (Robot Base Frame).
- Removes (subtracts) the offset [-0.615, 0.0, 0.0] from point_cloud, target_position, and tcp_pose.
- Use this if you want to train and execute policies purely in the robot's base frame.
"""

import os
import shutil
import zarr
import numpy as np
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
CHOICE = "M1"  # "M1" or "M2"

INPUT_ZARR = "real_reach_golden_with_tcp_pose.zarr"
OUTPUT_ZARR = f"real_reach_golden_{CHOICE}_corrected.zarr"

OFFSET = np.array([-0.615, 0.0, 0.0], dtype=np.float32)
# ==========================================

def process_zarr(input_path, output_path, choice):
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        print(f"Error: Input zarr '{input_path}' does not exist.")
        return
        
    print(f"Copying {input_path} to {output_path}...")
    if output_path.exists():
        shutil.rmtree(output_path)
    shutil.copytree(input_path, output_path)
    
    print(f"Opening {output_path} for in-place modification...")
    root = zarr.open(str(output_path), mode="r+")
    data = root["data"]
    
    # We load everything into memory for fast operations (safe for < 50k frames)
    # If the dataset is too large, this should be done in chunks.
    print("Loading datasets into memory...")
    pc = np.array(data["point_cloud"][:])
    tp = np.array(data["target_position"][:])
    
    has_tcp = "tcp_pose" in data
    if has_tcp:
        tcp = np.array(data["tcp_pose"][:])
    
    if choice == "M1":
        print(f"Applying M1: ADDING offset {OFFSET} to match Sim World Frame")
        pc_shifted = pc + OFFSET
        tp_shifted = tp + OFFSET
        if has_tcp:
            tcp_shifted = tcp.copy()
            tcp_shifted[:, :3] += OFFSET
            
    elif choice == "M2":
        print(f"Applying M2: SUBTRACTING offset {OFFSET} to match Real Robot Base Frame")
        pc_shifted = pc - OFFSET
        tp_shifted = tp - OFFSET
        if has_tcp:
            tcp_shifted = tcp.copy()
            tcp_shifted[:, :3] -= OFFSET
    else:
        print(f"Error: Invalid choice {choice}")
        return

    print("Writing modified data back to Zarr...")
    data["point_cloud"][:] = pc_shifted
    data["target_position"][:] = tp_shifted
    if has_tcp:
        data["tcp_pose"][:] = tcp_shifted
        
    print(f"Successfully applied {choice} correction!")
    print(f"Output saved at: {output_path}")

if __name__ == "__main__":
    print(f"--- Offset Correction Script ---")
    print(f"Mode: {CHOICE}")
    process_zarr(INPUT_ZARR, OUTPUT_ZARR, CHOICE)
