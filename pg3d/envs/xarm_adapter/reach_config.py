"""xArm7 reach workspace bounds — base-relative and table-agnostic.

All bounds are defined as offsets from the robot BASE, not from table edges, so the
same numbers hold whatever table the arm is bolted to (sim or a differently sized
real bench). World-frame bounds are produced by adding the base position; only the
table *surface height* (a z floor) is table-dependent, and it is applied separately.

Numbers are verified by ``scripts/verify_xarm7_reachability.py`` — an mplib-IK sweep
of ``xarm7_nogripper`` (TCP = ``link_eef``) at the reach task's downward tabletop
orientation (quaternion [0,1,0,0] wxyz), IK-seeded from the rest keyframe. Re-run it
after any change to the base pose, rest keyframe, URDF, or TCP link.

* Max envelope (extreme reach, reference only): dx∈[-0.25,0.725], dy∈[-0.675,0.675],
  dz∈[-0.05,0.775]; max reach ~0.83 m (real xArm7 working radius ~0.7 m). Only ~42%
  of its interior is IK-reachable — do NOT sample here; it is metadata.
* Reach box below: symmetric (left/right) sampling box, verified 8/8 corners and
  99.7% interior reachable (7^3 grid). The ceiling is dz=0.37, not 0.40: at full
  height the two high+wide+close corners (dx=0.10, dy=±0.45, dz=0.40) overshoot the
  lateral reach by ~0.16 mm past the 1 mm IK threshold, so the box is trimmed to keep
  every corner genuinely reachable.
"""

from __future__ import annotations

import numpy as np

# World position where the env bolts the xArm7 base (see xarm_adapter/reach_env.py).
ROBOT_BASE_POSITION = np.array([-0.615, 0.0, 0.0], dtype=np.float32)

# Base-relative sampling box [ [dx_lo,dx_hi], [dy_lo,dy_hi], [dz_lo,dz_hi] ].
# Symmetric in dy (left/right of the robot). Verified by verify_xarm7_reachability.py:
# 8/8 corners + 99.7% interior IK-reachable (dz_hi trimmed 0.40->0.37; see module docstring).
XARM7_REACH_BOX_BASE = np.array(
    [
        [0.10, 0.50],   # forward (dx)
        [-0.45, 0.45],  # lateral (dy) — symmetric
        [0.05, 0.37],   # height  (dz) above the base/table surface
    ],
    dtype=np.float32,
)

# Base-relative MAX reach envelope (reference / metadata only; do not sample to the
# edge — near-singular and leaves no room for waypoint detours).
XARM7_MAX_ENVELOPE_BASE = np.array(
    [
        [-0.25, 0.725],
        [-0.675, 0.675],
        [-0.05, 0.775],
    ],
    dtype=np.float32,
)

# Base-relative point-cloud crop box: covers the base + the reach box with margin.
# Table-agnostic in XY (anchored to the base, not the table footprint).
XARM7_CROP_BOX_BASE = np.array(
    [
        [-0.15, 0.60],
        [-0.55, 0.55],
        [-0.02, 0.60],
    ],
    dtype=np.float32,
)


def world_bounds(base_box: np.ndarray, base_position: np.ndarray = ROBOT_BASE_POSITION) -> np.ndarray:
    """Convert a base-relative [3,2] box to world-frame bounds via the base position."""
    box = np.asarray(base_box, dtype=np.float32).reshape(3, 2)
    return (box + np.asarray(base_position, dtype=np.float32).reshape(3, 1)).astype(np.float32)


# Convenience world-frame bounds for the default base placement.
XARM7_REACH_WORKSPACE_BOUNDS = world_bounds(XARM7_REACH_BOX_BASE)
XARM7_WORKSPACE_BOUNDS = world_bounds(XARM7_CROP_BOX_BASE)

# ──────────────────────────────────────────────────────────────────────────────
# Camera extrinsics — eye-to-hand calibration (AX = XB), 5 samples
# RMS reprojection: 12.5 mm translation, 1.12° rotation.
#
# "Base to Camera" output from the calibration script = camera pose expressed
# in robot base frame, OpenCV convention (z-forward, y-down, x-right).
# ──────────────────────────────────────────────────────────────────────────────

# Camera origin in robot base frame [m].
XARM7_CAM_T_BASE = np.array([1.7443, 0.0636, 0.6918], dtype=np.float64)

# Rotation matrix: each COLUMN is a camera-frame axis expressed in base frame.
#   col 0 = camera +x (right)  ≈ base +y
#   col 1 = camera +y (down)   ≈ base -z
#   col 2 = camera +z (optical) ≈ base -x  → camera looks toward robot
XARM7_CAM_R_BASE = np.array(
    [
        [ 0.0162,  0.0573, -0.9982],
        [ 0.9981, -0.0604,  0.0128],
        [-0.0596, -0.9965, -0.0582],
    ],
    dtype=np.float64,
)

# SAPIEN quaternion [w, x, y, z] derived from XARM7_CAM_R_BASE via Shepperd's
# method (trace = -0.1024; largest diagonal is R[0,0] = 0.0162).
XARM7_CAM_Q_WXYZ = np.array([-0.4737, 0.5327, 0.4954, -0.4965], dtype=np.float64)

# ──────────────────────────────────────────────────────────────────────────────
# Camera intrinsics — Intel RealSense D455 (depth sensor)
# Real resolution: 848 × 480 px.  Depth FOV: 87° (H) × 58° (V).
# Depth range: 0.4 m – 6 m.
#
# Sim uses 128 × 128 to match Panda data-gen speed; FOV is matched to real.
# Increase SIM_CAM_WIDTH/HEIGHT to 424 × 240 (half-res) or 848 × 480 (full)
# if higher point-cloud density is needed.
# ──────────────────────────────────────────────────────────────────────────────
XARM7_REAL_CAM_WIDTH = 848
XARM7_REAL_CAM_HEIGHT = 480
XARM7_CAM_HFOV_DEG = 87.0
XARM7_CAM_VFOV_DEG = 58.0
XARM7_CAM_VFOV_RAD: float = float(np.deg2rad(XARM7_CAM_VFOV_DEG))

# Sim render resolution (speed vs. density trade-off).
XARM7_SIM_CAM_WIDTH = 128
XARM7_SIM_CAM_HEIGHT = 128
