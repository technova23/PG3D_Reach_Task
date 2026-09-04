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
* **Expanded 2026-08-08** (sim-only training data, no real-hardware transfer
  constraint) — dx_hi 0.50->0.65 and dz_hi 0.37->0.60 to substantially close the
  forward-reach/height gap versus Panda's sampling box (Panda x-width 0.84 /
  z-width 0.52 vs xArm7's old dx-width 0.32 / dz-width 0.32; dy was already
  Panda-comparable and is unchanged). dx_lo/dy left untouched since the
  documented blind spot above is anchored there and re-approaching it is
  higher-risk than expanding from the already-validated dx_lo=0.18 corner.
  Re-verified against ``--variant gripper`` at dx_hi=0.65/dz_hi=0.60: only
  4/8 corners + 88.3% interior reachable (9^3 grid) -- a real regression from
  the pre-expansion 8/8/100.0%, from the same high-dz wrist-fold failure mode.
  Trimmed dz_hi 0.60->0.50 to recover corners: 6/8 + 95.9% interior.
* **Benchmarked against Panda 2026-08-1x** — before trimming further toward
  8/8, swept Panda's own "Panda-comparable" reference box
  (``scripts/verify_panda_reachability.py``, same mplib-IK methodology) to
  check what bar Panda itself actually clears: only 4/8 corners + 93.1%
  interior -- Panda's own reference box is *not* 8/8/100% either. Since the
  goal is workspace parity with Panda (not an arbitrary 100% target), settled
  on dx_hi=0.65 (kept large, dx wasn't the regression driver) and dz_hi=0.55
  (split between the 0.60 regression point and the 0.50 recovery point) as
  the final box -- comparable to Panda's own achieved reachability rather
  than stricter than it. This exact combination (dx_hi=0.65, dz_hi=0.55)
  has not itself been independently re-swept; re-run
  ``scripts/verify_xarm7_reachability.py --variant gripper --grid 9`` to
  confirm before trusting it for a full data-gen run, and update this note
  with the actual corner/interior result once run.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as _Rotation

# World position where the env bolts the xArm7 base (see xarm_adapter/reach_env.py).
#
# Origin-shifted 2026-09-04: was [-0.615, 0.0, 0.0]. Robot base now sits at the
# world origin -- base-frame coordinates and world-frame coordinates are the
# same thing, matching how the real hand-eye calibration (XARM7_CAM_T_BASE/
# XARM7_CAM_Q_WXYZ below) already expresses camera pose in the robot's own base
# frame. This is a pure relabeling of the whole scene, not a physical change --
# ManiSkill's TableSceneBuilder places the table at a FIXED absolute world pose
# independent of the robot (see
# https://github.com/haosulab/ManiSkill/blob/main/mani_skill/utils/scene_builder/table/scene_builder.py),
# so PG3DReachXArm7Env._initialize_episode (xarm_adapter/reach_env.py) now
# explicitly re-offsets the table by the same +0.615m in x every episode to
# keep the actual physical arrangement identical to before this shift -- see
# _TABLE_ORIGIN_SHIFT_X / _offset_table_for_origin_shift there. This constant
# and ROBOT_BASE_POSE in xarm_adapter/reach_env.py are two INDEPENDENT copies
# (not one derived from the other) and must be kept in sync by hand.
ROBOT_BASE_POSITION = np.array([0.0, 0.0, 0.0], dtype=np.float32)

# Base-relative sampling box [ [dx_lo,dx_hi], [dy_lo,dy_hi], [dz_lo,dz_hi] ].
# Symmetric in dy (left/right of the robot). dx_lo=0.18 is UNCHANGED since the
# 2026-07-02 re-tune (see module docstring) -- kept deliberately below every
# subsequent update, including this one, below the box's own documented
# self-collision blind spot (xarm_gripper_base_link folding into link6 at low
# dx + high lateral/height), not because it was itself re-measured.
#
# 2026-09-04 (superseded same-day, see next note): first pass at the PHYSICALLY
# MEASURED real-robot workspace (0.90m forward, +/-0.345m lateral, 0.05-0.53m
# height) -- previously these were simulation-only estimates (see the
# 2026-08-08/2026-08-1x history below).
#
# 2026-09-04 (revised): forward/lateral re-measured -- dx_hi corrected
# 0.90->0.69m, dy widened +/-0.345->+/-0.45m. dz_hi (0.53m) unchanged from the
# same-day first pass. NOT YET RE-VERIFIED against this exact combination --
# re-run `scripts/verify_xarm7_reachability.py --variant gripper --grid 9`
# before trusting this box for a full data-gen run; the prior corner/interior
# reachability numbers in this docstring predate both 2026-09-04 passes and
# don't apply to either.
XARM7_REACH_BOX_BASE = np.array(
    [
        [0.18, 0.69],   # forward (dx) — measured real workspace, 2026-09-04 revision (was 0.90, before that 0.65)
        [-0.45, 0.45],  # lateral (dy) — measured real workspace, 2026-09-04 revision (was +/-0.345, before that +/-0.42)
        [0.05, 0.53],   # height  (dz) above the base/table surface — measured, 2026-09-04 (unchanged this revision)
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
        [-0.15, 0.79],  # re-tuned 2026-09-04 revision to keep ~0.10m margin over reach dx_hi=0.69 (was 1.00/dx_hi=0.90)
        [-0.55, 0.55],  # unchanged -- still ~0.10m margin over the wider reach dy=+/-0.45 (2026-09-04 revision)
        [-0.02, 0.70],  # expanded 2026-08-08; still >0.15m margin over reach dz_hi=0.53 (2026-09-04, unchanged)
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
# Camera extrinsics — eye-on-base calibration, `xarm_rs_on_base_calibration`
# (`calibration_type: eye_on_base`, `tracking_base_frame: camera_color_optical_frame`).
# Source calibration gives translation + a ROS/geometry_msgs quaternion
# (x, y, z, w — scalar LAST) for base -> camera_color_optical_frame, i.e. the
# same "camera pose in robot base frame, OpenCV/pinhole optical convention"
# semantics as the calibration this replaced, just quaternion-encoded instead
# of matrix-encoded. Converted to `XARM7_CAM_R_BASE_OPENCV` below via
# `Rotation.from_quat([x, y, z, w])` (scipy's own default input order already
# matches ROS's scalar-last order, so no reindex is needed on the way in --
# only the SAPIEN-side wxyz reindex further down is needed). Raw quaternion
# norm was 1.0000471 (not exactly unit) -- normalized before conversion.
#
# NOTE: this calibration did not come with an RMS reprojection error report
# (unlike the previous one, which had 9.5284mm / 1.4328deg from
# eye_to_hand_custom.py) -- XARM7_CAM_CALIB_ERROR_TRANSLATION_STD_M/
# ROTATION_STD_DEG below are still the OLD calibration's measured RMS values
# and should be re-measured/updated once this camera has its own error report.
# ──────────────────────────────────────────────────────────────────────────────

# Camera origin in robot base frame [m].
XARM7_CAM_T_BASE = np.array(
    [1.0795605013716922, -0.6623893074061149, 0.42449609765716245], dtype=np.float64
)

# Rotation matrix: each COLUMN is an OpenCV-optical-frame camera axis expressed
# in the robot base frame. Derived from the calibration's raw quaternion
# (x, y, z, w) = (-0.7159689816418976, -0.2618685692246198, 0.3499625258496742,
# 0.5444572899362736) via `Rotation.from_quat(...).as_matrix()` (normalized first).
XARM7_CAM_R_BASE_OPENCV = np.array(
    [
        [ 0.617938, -0.006099, -0.786203],
        [ 0.755988, -0.270051,  0.596284],
        [-0.215952, -0.962827, -0.162264],
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
# ManiSkill's CameraConfig.fov is passed straight through as `fovy` (VERTICAL
# FOV only) -- confirmed against mani_skill/sensors/camera.py. A square sim
# render (previously 128x128) therefore gets hfov == vfov == 58 deg, 29 deg
# narrower than the real 87 deg horizontal -- harmless with the old camera
# (1.34m back, on-axis, 100% of the reach workspace inside frustum either way)
# but NOT with the current eye_on_base calibration, which sits 0.93m out and
# well off-axis: the narrow horizontal field measurably clips the workspace
# (verified 2026-09-04: 97.70% of XARM7_REACH_BOX_BASE inside frustum at
# 128x128 vs 99.95% at the real camera's true 87x58 FOV -- a gap that exists
# purely from the aspect mismatch, not the calibration itself).
#
# Sim resolution below (164x96) is chosen to reproduce the real camera's
# aspect ratio at fovy=58deg: aspect = tan(87/2)/tan(58/2) = 1.712, and
# 164/96 = 1.708 lands within 0.2 deg of the true 87 deg horizontal FOV
# (86.88 deg) at essentially the old 128x128 point budget (15744 vs 16384
# raw points/frame) -- i.e. same render cost, sim frustum now matches real.
# Increase further (e.g. 212x120, true 848x480 aspect) if higher point-cloud
# density is needed.
# ──────────────────────────────────────────────────────────────────────────────
XARM7_REAL_CAM_WIDTH = 848
XARM7_REAL_CAM_HEIGHT = 480
XARM7_CAM_HFOV_DEG = 87.0
XARM7_CAM_VFOV_DEG = 58.0
XARM7_CAM_VFOV_RAD: float = float(np.deg2rad(XARM7_CAM_VFOV_DEG))

# Sim render resolution (speed vs. density trade-off). Aspect matched to the
# real camera's 87x58 FOV at fovy=58deg -- see note above. Was 128x128
# (square, hfov=58deg only) before the 2026-09-04 aspect fix.
XARM7_SIM_CAM_WIDTH = 164
XARM7_SIM_CAM_HEIGHT = 96
