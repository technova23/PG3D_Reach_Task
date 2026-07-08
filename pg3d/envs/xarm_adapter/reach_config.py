"""xArm7 reach workspace bounds — base-relative and table-agnostic.

All bounds are defined as offsets from the robot BASE, not from table edges, so the
same numbers hold whatever table the arm is bolted to (sim or a differently sized
real bench). World-frame bounds are produced by adding the base position; only the
table *surface height* (a z floor) is table-dependent, and it is applied separately.

Numbers are verified by ``scripts/verify_xarm7_reachability.py`` — an mplib-IK sweep
at the reach task's downward tabletop orientation (quaternion [0,1,0,0] wxyz),
IK-seeded from the rest keyframe. mplib's ``planner.IK`` is collision-aware (rejects
self-colliding solutions), so this sweep is only a faithful proxy for "can data-gen
service a goal here" if it's run against the *same robot variant that generates the
data* — re-run with ``--variant gripper`` (not the default ``nogripper``) after any
change to the base pose, rest keyframe, URDF, TCP link, or gripper collision meshes.

* Max envelope (extreme reach, reference only): dx∈[-0.25,0.725], dy∈[-0.675,0.675],
  dz∈[-0.05,0.775]; max reach ~0.83 m (real xArm7 working radius ~0.7 m). Only ~42%
  of its interior is IK-reachable — do NOT sample here; it is metadata.
* Reach box below: symmetric (left/right) sampling box. **Re-verified against the
  gripper variant (2026-07-02)** — the box was originally tuned only against
  ``xarm7_nogripper`` (bare-arm IK, no gripper collision geometry at all), which
  missed that the actual data-gen robot (``xarm7_gripper``) has a self-collision
  blind spot: at points close to the base + max lateral + high (e.g. dx=0.10,
  dy=±0.45, dz≥0.29), reaching there forces a wrist fold that collides
  ``xarm_gripper_base_link`` into ``link6`` — a byproduct of that link's collision
  hull being a single convex hull ~2.73x the true mesh volume (see
  [[xarm-gripper-mplib-fix]]). Against ``--variant gripper`` the un-trimmed box
  (dx_lo=0.10, dy=±0.45) was only 6/8 corners + 95.2% interior reachable. Trimmed
  dx_lo 0.10->0.18 and dy ±0.45->±0.42 (dz unchanged) to clear that blind spot:
  now 8/8 corners + 100.0% interior reachable at a dense 9^3 grid for
  ``--variant gripper`` (also re-checked ``nogripper``: 8/8 corners, 99.86%
  interior — unaffected). Retains ~75% of the original box's volume.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as _Rotation

# World position where the env bolts the xArm7 base (see xarm_adapter/reach_env.py).
ROBOT_BASE_POSITION = np.array([-0.615, 0.0, 0.0], dtype=np.float32)

# Base-relative sampling box [ [dx_lo,dx_hi], [dy_lo,dy_hi], [dz_lo,dz_hi] ].
# Symmetric in dy (left/right of the robot). Verified by verify_xarm7_reachability.py
# --variant gripper: 8/8 corners + 100.0% interior IK-reachable (9^3 grid) against the
# actual gripper-equipped robot; see module docstring for the 2026-07-02 re-tune.
XARM7_REACH_BOX_BASE = np.array(
    [
        [0.18, 0.50],   # forward (dx)
        [-0.42, 0.42],  # lateral (dy) — symmetric
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
# Camera extrinsics — eye-to-hand calibration (AX = XB), 6 samples.
# RMS reprojection: 9.5284 mm translation, 1.4328° rotation (see
# eye_to_hand_custom.py "FINAL RESULT: Base to Camera Transformation" output).
#
# "Base to Camera" output from the calibration script = camera pose expressed
# in robot base frame, OpenCV/pinhole optical convention (z-forward, y-down,
# x-right — this is what cv2 hand-eye solvers produce).
# ──────────────────────────────────────────────────────────────────────────────

# Camera origin in robot base frame [m].
XARM7_CAM_T_BASE = np.array([1.7103, 0.0043, 0.7097], dtype=np.float64)

# Rotation matrix: each COLUMN is an OpenCV-optical-frame camera axis expressed
# in the robot base frame (this is the raw calibration script output).
#   col 0 = camera +x (right)   ≈ base +y
#   col 1 = camera +y (down)    ≈ base -z
#   col 2 = camera +z (optical/forward) ≈ base -x
XARM7_CAM_R_BASE_OPENCV = np.array(
    [
        [ 0.0534,  0.0868, -0.9948],
        [ 0.9985, -0.0156,  0.0522],
        [-0.0110, -0.9961, -0.0875],
    ],
    dtype=np.float64,
)

# Per-episode camera domain randomization: uniform +/-10cm position and +/-2deg
# orientation on each axis independently, for viewpoint diversity in training data
# (not a calibration-error model -- that's XARM7_CAM_CALIB_ERROR_* below). Fixed
# for the whole episode -- set once in _randomize_camera_pose, never re-sampled
# mid-episode.
XARM7_CAM_POSITION_JITTER_M = 0.10
XARM7_CAM_ROTATION_JITTER_DEG = 2.0

# Calibration-error model: even after calibrating, the estimated camera pose used
# to convert depth into world/robot-frame points is never exactly right. Modeled
# as a per-episode (not per-step) Gaussian offset applied only when interpreting
# points into world frame -- see _sample_camera_calibration_error and the
# get_obs override in reach_env.py -- so the physical/rendering camera pose
# (XARM7_CAM_POSITION_JITTER_M/ROTATION_JITTER_DEG above) is untouched; only the
# point cloud's belief about where the camera was is perturbed. Values are the
# real measured eye-to-hand calibration RMS error (see reach_config.py's
# "Camera extrinsics" section: RMS Translation Error 9.5284 mm, RMS Rotation
# Error 1.4328 deg, eye_to_hand_custom.py output, 6 samples).
XARM7_CAM_CALIB_ERROR_TRANSLATION_STD_M = 0.0095284
XARM7_CAM_CALIB_ERROR_ROTATION_STD_DEG = 1.4328


def _opencv_camera_rotation_to_sapien(r_opencv: np.ndarray) -> np.ndarray:
    """Convert an OpenCV/pinhole-optical camera rotation to SAPIEN's convention.

    OpenCV optical frame: columns = [right (+x), down (+y), forward/optical (+z)].
    SAPIEN camera frame:  (forward, right, up) = (+x, -y, +z) — see
    ``mani_skill.utils.sapien_utils.look_at`` docstring. So SAPIEN's forward
    column is OpenCV's forward column, SAPIEN's "left" (+y) column is the
    negated OpenCV right column, and SAPIEN's up (+z) column is the negated
    OpenCV down column. Passing an OpenCV-convention matrix straight into
    SAPIEN silently points the camera ~90° off (confirmed empirically: it
    pointed down the base +y axis instead of at the robot, so the point
    cloud crop always fell outside the workspace bounds).
    """
    right, down, forward = r_opencv[:, 0], r_opencv[:, 1], r_opencv[:, 2]
    return np.stack([forward, -right, -down], axis=1)


XARM7_CAM_R_BASE = _opencv_camera_rotation_to_sapien(XARM7_CAM_R_BASE_OPENCV)

# SAPIEN quaternion [w, x, y, z] derived from XARM7_CAM_R_BASE.
XARM7_CAM_Q_WXYZ = np.asarray(
    _Rotation.from_matrix(XARM7_CAM_R_BASE).as_quat()[[3, 0, 1, 2]], dtype=np.float64
)

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
