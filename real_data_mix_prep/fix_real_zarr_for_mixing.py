"""Prepare a mix-ready copy of the real xArm7 reach dataset.

Two fixes applied ONLY to the copy (the source zarr is opened read-only and is
never written to):

1. Coordinate frame: real data's point_cloud/target_position are recorded
   robot-base-relative; sim data (pg3d_xarm7_gripper_reach_final.zarr) is
   recorded in world frame with the base offset baked in
   (ROBOT_BASE_POSITION = [-0.615, 0, 0], see pg3d/envs/xarm_adapter/reach_config.py).
   Confirmed empirically: shifting real's target_position by [-0.615, 0, 0]
   reproduces sim's range almost exactly (offset matched to within ~2mm on X,
   ~7mm on Y/Z -- consistent with real measurement noise, not a residual
   frame error). Applying the same fixed translation to point_cloud and the
   newly-computed tcp_pose below makes all three spatially consistent with
   sim's convention.

2. Missing tcp_pose: real data has no recorded end-effector pose at all, only
   `state` (7 joint angles, radians -- same convention/order as sim's `state`).
   tcp_pose is required by ReachDatasetConfig when --use-goal-encoder is set
   (pg3d/policies/dp3/reach_dataset.py:201). Computed here via forward
   kinematics through the same xarm7_with_gripper URDF used by simulation
   (pg3d/envs/xarm_adapter/assets/xarm7_with_gripper_colored.urdf), out to
   `link_tcp` -- a *fixed* joint 172mm off xarm_gripper_base_link (itself
   fixed off link7), so it depends only on the 7 arm joint angles, not on
   gripper finger state. This is exactly the same link ManiSkill's own
   agent.tcp_pose tracks for this robot (see agents.py: ee_link_name =
   "link_tcp" for the gripper variant), so the two are directly comparable.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytorch_kinematics as pk
import torch
import zarr
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "Real-data-zarr-setup" / "real_reach_golden.zarr"
DST = REPO_ROOT / "Real-data-zarr-setup" / "real_reach_golden_copy.zarr"
URDF_PATH = REPO_ROOT / "pg3d" / "envs" / "xarm_adapter" / "assets" / "xarm7_with_gripper_colored.urdf"

# From pg3d/envs/xarm_adapter/reach_config.py -- base-relative -> world frame.
BASE_RELATIVE_TO_WORLD = np.array([-0.615, 0.0, 0.0], dtype=np.float32)


def main() -> None:
    if DST.exists():
        raise FileExistsError(f"refusing to overwrite existing {DST}")

    print(f"copying {SRC} -> {DST}")
    shutil.copytree(SRC, DST)

    src_root = zarr.open_group(str(SRC), mode="r")  # read-only, never written
    dst_root = zarr.open_group(str(DST), mode="a")
    dst_data = dst_root["data"]

    state = np.asarray(src_root["data"]["state"][:], dtype=np.float32)
    point_cloud = np.asarray(src_root["data"]["point_cloud"][:], dtype=np.float32)
    target_position = np.asarray(src_root["data"]["target_position"][:], dtype=np.float32)
    n_steps = state.shape[0]
    print(f"loaded {n_steps} steps from source (read-only)")

    # --- 1. forward kinematics: state (7 joint angles) -> link_tcp pose ---
    urdf_data = URDF_PATH.read_bytes()
    chain = pk.build_serial_chain_from_urdf(urdf_data, "link_tcp", "link_base")
    joint_names = chain.get_joint_parameter_names()
    print(f"kinematic chain joints ({len(joint_names)}): {joint_names}")
    if len(joint_names) != state.shape[1]:
        raise ValueError(
            f"chain has {len(joint_names)} joints but state has {state.shape[1]} columns"
        )

    with torch.no_grad():
        th = torch.from_numpy(state)
        tf = chain.forward_kinematics(th, end_only=True)
        matrices = tf.get_matrix().numpy()  # [N, 4, 4]

    tcp_position_base_relative = matrices[:, :3, 3].astype(np.float32)
    rot_matrices = matrices[:, :3, :3].astype(np.float64)
    quat_xyzw = Rotation.from_matrix(rot_matrices).as_quat()  # scipy: [x,y,z,w]
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]].astype(np.float32)  # SAPIEN/ManiSkill convention

    # --- 2. base-relative -> world frame (real's convention -> sim's convention) ---
    tcp_position_world = tcp_position_base_relative + BASE_RELATIVE_TO_WORLD
    point_cloud_world = point_cloud + BASE_RELATIVE_TO_WORLD.reshape(1, 1, 3)
    target_position_world = target_position + BASE_RELATIVE_TO_WORLD.reshape(1, 3)

    tcp_pose = np.concatenate([tcp_position_world, quat_wxyz], axis=1).astype(np.float32)
    assert tcp_pose.shape == (n_steps, 7)

    # --- 3. write into the COPY only ---
    dst_data["point_cloud"][:] = point_cloud_world
    dst_data["target_position"][:] = target_position_world
    dst_data.create_dataset(
        "tcp_pose", data=tcp_pose, chunks=dst_data["state"].chunks, dtype=np.float32
    )

    print("wrote point_cloud (world frame), target_position (world frame), tcp_pose (new, world frame) to copy")

    # --- sanity: source untouched ---
    src_target_check = np.asarray(src_root["data"]["target_position"][:5])
    print(f"source target_position[:5] (should still be base-relative, unshifted):\n{src_target_check}")
    print(f"copy   target_position[:5] (should be shifted by {BASE_RELATIVE_TO_WORLD}):\n{np.asarray(dst_data['target_position'][:5])}")


if __name__ == "__main__":
    main()
