import os
import json
import re
from pathlib import Path
import numpy as np
import cv2
import zarr
import torch
import pytorch3d.ops as torch3d_ops
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

MAX_DIFF = 0.03 # 3cm
RESAMPLE_HZ = 20.0
TOTAL_POINTS = 1024
GOAL_MARKER_POINTS = 192
SCENE_POINTS = TOTAL_POINTS - GOAL_MARKER_POINTS
TCP_MARKER_POINTS = 0
ROBOT_POINT_FRACTION = 1.0

_ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]

# The transformation betwwen depth and cam frame 
T_DEPTH_TO_COLOR = np.array([
    [ 1.0,      0.0,      0.0,     -0.059 ],
    [ 0.0,      1.0,      0.0,    0.0    ],
    [ 0.0,      0.0,      1.0,    0.0    ],
    [ 0.0,      0.0,      0.0,      1.0    ]
], dtype=np.float32)

# extrinsics from colour cam to base
BASE_TO_EXT_CAM = np.array([
    [0.0534 , 0.0868 , -0.9948 , 1.7103],
     [0.9985 , -0.0156 , 0.0522 , 0.0043],
     [-0.011 , -0.9961 , -0.0875 , 0.7097],
     [ 0.0,     0.,      0.,      1.    ],
], dtype=np.float32)


WORK_SPACE = [
    [-0.05, 0.8],  # X
    [-0.5 , 0.5],  # Y
    [0.08 , 0.7]   # Z
]


def load_robot(urdf_path):
    """
    Just a basic func to load the URDF 
    """
    if not Path(urdf_path).exists():
        raise SystemExit(f"URDF not found: {urdf_path}")
    import yourdfpy
    return yourdfpy.URDF.load(str(urdf_path), load_meshes=False, build_scene_graph=True)


def _eef_pos(robot, q7):
    """
    q7 is a vector of 7 joint angles
    This function returns the position of the end effector in the base frame
    """
    cfg = {name: float(angle) for name, angle in zip(_ARM_JOINTS, q7)}
    robot.update_cfg(cfg)
    T = robot.get_transform("link_tcp")
    return T[:3, 3].astype(np.float32)


def check_reachability(bag_dir, target, fk_robot):
    """
    Checks if the end effector is within MAX_DIFF distance from the target
    Takes in the last joint angle -> performs FK -> computes error 
    """
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    last_q = None
    with AnyReader([bag_dir], default_typestore=typestore) as reader:
        connections = [x for x in reader.connections if x.topic == "/xarm/joint_states"]
        for conn, ts, raw in reader.messages(connections=connections):
            msg = reader.deserialize(raw, conn.msgtype)
            if len(msg.position) == 7:
                last_q = np.asarray(msg.position, dtype=np.float64)
                    
    if last_q is None:
        return False, 999.0
        
    final_eef = _eef_pos(fk_robot, last_q)
    err = float(np.linalg.norm(final_eef - target))
    return (err <= MAX_DIFF), err


def extract_state_action(bag_dir):
    """
    Extracts state and action from the bag file and resamples them to RESAMPLE_HZ .
    """
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    msg_times, msg_positions = [], []
    with AnyReader([bag_dir], default_typestore=typestore) as reader:
        connections = [x for x in reader.connections if x.topic == "/xarm/joint_states"]
        for conn, ts, raw in reader.messages(connections=connections):
            msg = reader.deserialize(raw, conn.msgtype)
            if len(msg.position) == 7:
                msg_times.append(float(ts) / 1e9)
                msg_positions.append(np.asarray(msg.position, dtype=np.float64))
                    
    if not msg_times: 
        return None, None, None
        
    times = np.asarray(msg_times)
    positions = np.stack(msg_positions, axis=0)
    
    t_start, t_end = times[0], times[-1]
    if t_end <= t_start: 
        return None, None, None
    
    step = 1.0 / RESAMPLE_HZ
    grid = np.arange(t_start, t_end + 1e-9, step, dtype=np.float64)
    
    right = np.clip(np.searchsorted(times, grid), 0, len(times) - 1)
    left  = np.clip(right - 1, 0, len(times) - 1)
    use_left = (grid - times[left]) < (times[right] - grid)
    idx = np.where(use_left, left, right)
    
    state = positions[idx].astype(np.float32)
    action = np.empty_like(state)
    action[:-1] = state[1:]
    action[-1] = state[-1]
    
    return state, action, grid

"""
Most of the helper functions are from : https://github.com/darshil0805/Point-Cloud-Processing/blob/main/data_processing_pipeline/extract_point_clouds_one_cam.py
"""


def deproject(depth, params):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float32)
    x = (u - params['cx']) * z / params['fx']
    y = (v - params['cy']) * z / params['fy']
    return np.stack((x, y, z), axis=-1).reshape(-1, 3)


def align_colors(pts_cam, rgb, color_params, T_depth_to_color):
    points_homo = np.column_stack([pts_cam, np.ones(len(pts_cam))])
    pts_color_frame = (T_depth_to_color @ points_homo.T).T[:, :3]
    z = pts_color_frame[:, 2]
    u = (pts_color_frame[:, 0] * color_params['fx'] / np.maximum(z, 0.001)) + color_params['cx']
    v = (pts_color_frame[:, 1] * color_params['fy'] / np.maximum(z, 0.001)) + color_params['cy']
    h, w, _ = rgb.shape
    u_idx, v_idx = np.round(u).astype(int), np.round(v).astype(int)
    in_image = (u_idx >= 0) & (u_idx < w) & (v_idx >= 0) & (v_idx < h)
    mask = (z > 0.01) & in_image
    colors = np.zeros((len(pts_cam), 3), dtype=np.float32)
    colors[mask] = rgb[v_idx[mask], u_idx[mask]] / 255.0
    return colors, mask


def farthest_point_sampling(points, num_points=1024, use_cuda=True):
    if len(points) == 0:
        return np.zeros((num_points, 3), dtype=np.float32), np.zeros((num_points,), dtype=np.int64)
    K = [min(num_points, len(points))]
    pc = torch.from_numpy(points).float()
    if use_cuda and torch.cuda.is_available():
        pc = pc.cuda()
    sampled, idx = torch3d_ops.sample_farthest_points(points=pc.unsqueeze(0), K=K)
    return sampled.squeeze(0).cpu().numpy(), idx.squeeze(0).cpu().numpy()


def goal_marker_offsets(*, num_points: int = 192, radius: float = 0.055) -> np.ndarray:
    if num_points == 0: return np.zeros((0, 3), dtype=np.float32)
    r = np.float32(radius)
    if r == 0: return np.zeros((num_points, 3), dtype=np.float32)
    pattern = [np.zeros(3, dtype=np.float32)]
    cross_dirs = np.asarray([[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0], [0, -1.0, 0], [0, 0, 1.0], [0, 0, -1.0]], dtype=np.float32)
    for direction in cross_dirs: pattern.append(r * direction)
    ring_count = max(0, num_points - len(pattern))
    for idx in range(ring_count):
        angle = 2.0 * np.pi * idx / max(ring_count, 1)
        ring_radius = r * (0.70 if idx % 2 == 0 else 1.00)
        z_offset = r * 0.25 * (1.0 if idx % 4 in {0, 1} else -1.0)
        pattern.append(np.asarray([ring_radius * np.cos(angle), ring_radius * np.sin(angle), z_offset], dtype=np.float32))
    if num_points <= len(pattern): return np.asarray(pattern[:num_points], dtype=np.float32)
    repeats = int(np.ceil(num_points / len(pattern)))
    return np.tile(np.asarray(pattern, dtype=np.float32), (repeats, 1))[:num_points]

def goal_marker_points(target_position, *, num_points: int = 192, radius: float = 0.055) -> np.ndarray:
    target = np.asarray(target_position, dtype=np.float32)
    offsets = goal_marker_offsets(num_points=num_points, radius=radius)
    if num_points == 0: return np.zeros((*target.shape[:-1], 0, 3), dtype=np.float32)
    return target[..., None, :] + offsets.reshape((1,) * (target.ndim - 1) + offsets.shape)

def insert_goal_marker_points(point_cloud, target_position, *, num_points: int = 192, radius: float = 0.055) -> np.ndarray:
    points = np.asarray(point_cloud, dtype=np.float32)
    if num_points == 0: return points.astype(np.float32, copy=True)
    marker = goal_marker_points(target_position, num_points=num_points, radius=radius)
    expected_marker_shape = (*points.shape[:-2], num_points, 3)
    marker = np.broadcast_to(marker, expected_marker_shape)
    output = points.astype(np.float32, copy=True)
    output[..., -num_points:, :] = marker
    return output


def process_point_cloud(pc_xyz, pc_rgb):
    mask = (
        (pc_xyz[:, 0] > WORK_SPACE[0][0]) & (pc_xyz[:, 0] < WORK_SPACE[0][1]) &
        (pc_xyz[:, 1] > WORK_SPACE[1][0]) & (pc_xyz[:, 1] < WORK_SPACE[1][1]) &
        (pc_xyz[:, 2] > WORK_SPACE[2][0]) & (pc_xyz[:, 2] < WORK_SPACE[2][1])
    )
    pc_xyz_c = pc_xyz[mask]
    pc_rgb_c = pc_rgb[mask]
    
    if len(pc_xyz_c) == 0:
        return np.zeros((TOTAL_POINTS, 3), dtype=np.float32), np.zeros((TOTAL_POINTS, 3), dtype=np.float32)
        
    num_pts = min(SCENE_POINTS, len(pc_xyz_c))
    pc_xyz_fps, idx = farthest_point_sampling(pc_xyz_c, num_points=num_pts, use_cuda=True)
    pc_rgb_fps = pc_rgb_c[idx]
    
    if len(pc_xyz_fps) < TOTAL_POINTS:
        pad_size = TOTAL_POINTS - len(pc_xyz_fps)
        pad_xyz = np.zeros((pad_size, 3), dtype=np.float32)
        pad_rgb = np.zeros((pad_size, 3), dtype=np.float32)
        pc_xyz_fps = np.vstack([pc_xyz_fps, pad_xyz])
        pc_rgb_fps = np.vstack([pc_rgb_fps, pad_rgb])
        
    return pc_xyz_fps, pc_rgb_fps


def extract_depth(bag_dir, grid_times):
    """
    This function extracts depth from the bag file and converts & subsamples it to a point cloud . 
    """
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    depth_info, color_info = None, None
    with AnyReader([bag_dir], default_typestore=typestore) as reader:
        connections = [x for x in reader.connections if x.topic in ["/camera/camera/depth/camera_info", "/camera/camera/color/camera_info"]]
        for conn, ts, raw in reader.messages(connections=connections):
            if conn.topic == "/camera/camera/depth/camera_info" and depth_info is None:
                msg = reader.deserialize(raw, conn.msgtype)
                try:
                    K = np.array(msg.k).reshape(3, 3)
                except AttributeError:
                    print(f"Warning: depth msg missing 'k'. Using default.")
                    K = np.array([[600., 0., 320.], [0., 600., 240.], [0., 0., 1.]])
                depth_info = {'fx': K[0, 0], 'fy': K[1, 1], 'cx': K[0, 2], 'cy': K[1, 2]}
            elif conn.topic == "/camera/camera/color/camera_info" and color_info is None:
                msg = reader.deserialize(raw, conn.msgtype)
                try:
                    K = np.array(msg.k).reshape(3, 3)
                except AttributeError:
                    print(f"Warning: color msg missing 'k'. Using default.")
                    K = np.array([[600., 0., 320.], [0., 600., 240.], [0., 0., 1.]])
                color_info = {'fx': K[0, 0], 'fy': K[1, 1], 'cx': K[0, 2], 'cy': K[1, 2]}
            if depth_info and color_info:
                break
    
    if not depth_info or not color_info:
        print("Missing camera info.")
        return None
        
    depth_msgs, color_msgs = [], []
    with AnyReader([bag_dir], default_typestore=typestore) as reader:
        connections = [x for x in reader.connections if x.topic in ["/camera/camera/depth/image_rect_raw", "/camera/camera/color/image_raw"]]
        for conn, ts, raw in reader.messages(connections=connections):
            if conn.topic == "/camera/camera/depth/image_rect_raw":
                msg = reader.deserialize(raw, conn.msgtype)
                depth_msgs.append((float(ts)/1e9, msg))
            elif conn.topic == "/camera/camera/color/image_raw":
                msg = reader.deserialize(raw, conn.msgtype)
                color_msgs.append((float(ts)/1e9, msg))
                
    if not depth_msgs or not color_msgs:
        return None
    
    dt = np.array([x[0] for x in depth_msgs])
    ct = np.array([x[0] for x in color_msgs])
    
    pcds = []
    for t_grid in grid_times:
        d_idx = np.argmin(np.abs(dt - t_grid))
        c_idx = np.argmin(np.abs(ct - t_grid))
        
        depth_msg = depth_msgs[d_idx][1]
        color_msg = color_msgs[c_idx][1]
        
        if depth_msg.encoding == "16UC1":
            depth = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape(depth_msg.height, depth_msg.width).astype(np.float32) / 1000.0
        elif depth_msg.encoding == "32FC1":
            depth = np.frombuffer(depth_msg.data, dtype=np.float32).reshape(depth_msg.height, depth_msg.width)
        else:
            depth = np.zeros((depth_msg.height, depth_msg.width), dtype=np.float32)
            
        if color_msg.encoding == "rgb8":
            color = np.frombuffer(color_msg.data, dtype=np.uint8).reshape(color_msg.height, color_msg.width, 3)
        elif color_msg.encoding == "bgr8":
            color = np.frombuffer(color_msg.data, dtype=np.uint8).reshape(color_msg.height, color_msg.width, 3)
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        else:
            color = np.frombuffer(color_msg.data, dtype=np.uint8).reshape(color_msg.height, color_msg.width, -1)
            if color.shape[-1] == 3:
                color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            else:
                color = np.zeros((color_msg.height, color_msg.width, 3), dtype=np.uint8)
                
        pts_cam = deproject(depth, depth_info)
        colors, proj_mask = align_colors(pts_cam, color, color_info, T_DEPTH_TO_COLOR)
        
        mask = proj_mask & (depth.reshape(-1) > 0.01) & (depth.reshape(-1) < 5.0)
        pts_v, clrs_v = pts_cam[mask], colors[mask]
        
        pts_g = (BASE_TO_EXT_CAM @ np.column_stack([pts_v, np.ones(len(pts_v))]).T).T[:, :3]
        pts_g_fps, clrs_v_fps = process_point_cloud(pts_g, clrs_v)
        
        pcds.append(pts_g_fps)
        
    return np.stack(pcds, axis=0)


def convert_to_zarr(out_zarr_path, all_states, all_actions, all_targets, all_pcds, episode_ends):
    if len(episode_ends) == 0:
        print("No successful episodes to write.")
        return
        
    state = np.concatenate(all_states, axis=0)
    action = np.concatenate(all_actions, axis=0)
    target_position = np.concatenate(all_targets, axis=0)
    point_cloud = np.concatenate(all_pcds, axis=0)
    episode_ends_arr = np.asarray(episode_ends, dtype=np.int64)

    root = zarr.open(str(out_zarr_path), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")

    data.create_dataset("state", data=state, chunks=(1024, state.shape[1]))
    data.create_dataset("action", data=action, chunks=(1024, action.shape[1]))
    data.create_dataset("target_position", data=target_position, chunks=(1024, 3))
    data.create_dataset("point_cloud", data=point_cloud, chunks=(64, TOTAL_POINTS, 3))
    meta.create_dataset("episode_ends", data=episode_ends_arr)
    
    print(f"\nSaved to: {out_zarr_path}")
    print(root.tree())
