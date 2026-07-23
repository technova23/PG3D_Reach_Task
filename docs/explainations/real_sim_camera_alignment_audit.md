# Real/Sim xArm Camera and Point-Cloud Alignment Audit

Date: 2026-07-23

## Purpose

This note records the camera-calibration and point-cloud alignment audit performed after a
visual overlay showed that the real and simulated xArm point clouds did not overlap at the robot
base. It distinguishes:

1. the nominal camera pose copied from the real calibration;
2. per-episode simulated camera and calibration randomization;
3. the persistent real/sim point-cloud offset;
4. the episode-dependent part of that offset.

The numerical registration results below are point-cloud-domain measurements. They should not be
interpreted as a direct measurement of physical camera displacement because robot-model
differences, visibility, sampling, and registration error also contribute.

## Dataset audited

The exact mixed dataset audited was:

```text
/scratch2/skills/pg3d_real_sim_mixed.zarr
```

From this repository, the user originally referred to it as:

```text
../../pg3d_real_sim_mixed.zarr
```

Its provenance metadata records:

| Source | Episode range | Step range |
|---|---:|---:|
| Real xArm | `[0, 699)` | `[0, 34,272)` |
| ManiSkill sim | `[699, 5,699)` | `[34,272, 341,838)` |

The mixed Zarr was produced by
[`real_data_mix_prep/build_real_sim_mixed_zarr.py`](../../real_data_mix_prep/build_real_sim_mixed_zarr.py).
The simulation source appended to it was:

```text
/scratch2/skills/pg3d_xarm7_gripper_reach_final.zarr
```

The M2/base-frame conversion produced from the audited mixed Zarr on 2026-07-23 is:

```text
/scratch2/skills/pg3d_real_sim_mixed_m2.zarr
```

The original M1 mixed Zarr was preserved. The M2 copy shifts all stored
`point_cloud`, `target_position`, and `tcp_pose[:3]` coordinates by
`[+0.615, 0, 0]` metres. It also shifts the spatial JSON metadata: crop bounds, environment goal
center, per-episode targets, start TCP poses, goal poses, sampled start positions, and trajectory
waypoints. Joint states, actions, pose orientations, episode boundaries, and other non-spatial
values are unchanged.

The M2 conversion was validated across every target and TCP row, every episode metadata record,
all state/action rows, and 1,028 point-cloud frames spanning the real/sim boundary and full dataset.
The resulting real and sim target X ranges are approximately `0.18–0.50 m`, as expected for the
M2 robot-base frame.

The real source was converted into the old M1 simulator world frame before mixing by adding
`[-0.615, 0, 0]` metres to its point clouds, targets, and FK-derived TCP positions. That conversion
is implemented in
[`real_data_mix_prep/fix_real_zarr_for_mixing.py`](../../real_data_mix_prep/fix_real_zarr_for_mixing.py).
Consequently, both halves of this particular mixed Zarr use the M1 world convention, where the
xArm base is at `[-0.615, 0, 0]`.

## Real calibration recovered from the image

The calibration image used for this audit is:

```text
/scratch2/skills/puru/pg3d/real_camera_calibration.jpeg
```

It reports this base-to-camera transform, with translation in metres:

```text
R_base_camera_opencv =
[[ 0.0534,  0.0868, -0.9948],
 [ 0.9985, -0.0156,  0.0522],
 [-0.0110, -0.9961, -0.0875]]

t_base_camera = [1.7103, 0.0043, 0.7097]
```

The image also reports final RMS reprojection errors of:

```text
translation: 9.5284 mm
rotation:    1.4328 degrees
```

The exact translation and rotation matrix are present in
[`pg3d/envs/xarm_adapter/reach_config.py`](../../pg3d/envs/xarm_adapter/reach_config.py).
The OpenCV optical-axis convention is explicitly converted to SAPIEN's camera-axis convention
before constructing the simulated camera pose.

For the M1 dataset, the nominal simulated camera position in the simulator world frame was:

```text
[-0.615, 0, 0] + [1.7103, 0.0043, 0.7097]
    = [1.0953, 0.0043, 0.7097] metres
```

Because the real clouds were shifted into the same M1 frame, the nominal real and simulated camera
poses are consistent in the stored coordinates.

## Simulated episode randomization

The nominal camera pose was not used unchanged for every simulated episode. The data-generation
environment sampled, once per episode:

```text
camera translation:
    delta_x, delta_y, delta_z ~ Uniform(-0.10 m, +0.10 m)

camera orientation:
    delta_roll, delta_pitch, delta_yaw ~ Uniform(-2 deg, +2 deg)
```

The sampled physical/rendering camera pose was held fixed within the episode.

After rendering, a second per-episode perturbation modeled calibration error when interpreting
points in world coordinates:

```text
translation component standard deviation: 0.0095284 m
rotation component standard deviation:    1.4328 deg
```

This second transform was also held fixed within the episode and was applied to the point-cloud
coordinates, not to the rendered camera.

The RMS numbers from the real calibration were used as the standard deviation of every axis.
Therefore, the approximate RMS magnitude of the sampled vector is larger than the measured scalar
RMS:

```text
translation magnitude RMS ~= sqrt(3) * 9.5284 mm = 16.5 mm
rotation magnitude RMS    ~= sqrt(3) * 1.4328 deg = 2.48 deg
```

At a camera-to-base distance of approximately 1.85 m, angular calibration error can create
centimetre-scale point displacement near the robot. The exact behavior is implemented in
[`pg3d/envs/xarm_adapter/reach_env.py`](../../pg3d/envs/xarm_adapter/reach_env.py).

The source simulation Zarr records pg3d commit `af30578614e4253eedef3c04ade388230d7ee816`
with a dirty worktree. The dataset metadata was written approximately 32 minutes before commit
`b7a0b3dbf94395e8ba618b1c095ad480de50ae87`, which committed the exact calibration and
randomization code described above. This is strong provenance evidence, but the realized camera
pose and calibration transform were not stored per episode.

## Registration method

The audit used the following simulator-free procedure:

1. Read the mixed Zarr without modifying it.
2. Match real and simulated frames by nearest-neighbor distance between their seven joint angles.
3. Register the corresponding robot point clouds with trimmed rigid ICP.
4. Represent the result as a real-to-sim rigid transform.
5. Group transforms by simulated episode.
6. Compare between-episode variation with within-episode variation.
7. Run real-to-real and sim-to-sim controls using the same joint matching and ICP procedure.

Pairs with joint-space distance greater than `0.10` radians or post-registration symmetric Chamfer
distance greater than `0.075` metres were excluded from the grouped analysis. The final real/sim
grouped analysis retained 129 simulated episodes with at least three accepted frame pairs each.

ICP estimates a point-cloud alignment transform, not camera pose itself. In particular, translation
and rotation estimates can absorb consistent URDF/mesh differences, partial visibility, different
point sampling, real depth artifacts, and local registration error.

## Results

### Persistent average real/sim difference

The median episode-level transform required to map real clouds onto simulated clouds was:

```text
translation = [-0.079, -0.046, -0.039] metres
translation magnitude = 0.099 metres
rotation magnitude = 4.7 degrees
```

The approximately 10 cm value is the length of one directional translation vector. It does not
mean that the clouds differ by 10 cm independently along every axis.

The episode-level translation distribution was:

| Component | 5th percentile | Median | 95th percentile |
|---|---:|---:|---:|
| X | -11.0 cm | -7.9 cm | -3.9 cm |
| Y | -10.1 cm | -4.6 cm | +1.9 cm |
| Z | -7.9 cm | -3.9 cm | +0.8 cm |

If the real clouds were at the center of the simulated point-cloud distribution, the median
real-to-sim transform would be close to `[0, 0, 0]`, with roughly balanced positive and negative
values. Instead, X remained negative through the 95th percentile, while zero was near the upper
tail for Y and Z. The real data may be covered by some randomized simulated episodes, but it was
not at the measured center of the simulated point-cloud distribution.

### Episode dependence

The point-cloud difference was not constant across episodes:

| Measurement | Translation variation | Rotation variation |
|---|---:|---:|
| Between sim episodes, median distance from global transform | 4.5 cm | 3.5 deg |
| Between sim episodes, 90th percentile | 7.8 cm | 5.6 deg |
| Within one sim episode, median | 1.8 cm | 2.2 deg |
| Real-to-real control, between episodes | 1.2 cm | 1.4 deg |
| Sim-to-sim control, between episodes | 4.1 cm | 2.9 deg |

Of the 129 accepted simulated episodes:

- 100 had an episode translation more than 3 cm from the global median transform.
- 77 had an episode rotation more than 3 degrees from the global median rotation.

The larger sim-to-sim variation relative to the real-to-real control confirms that the simulated
episode randomization is visible in the stored point clouds. The smaller within-episode variation
is consistent with the implementation holding each sampled perturbation fixed for the entire
episode.

The median symmetric Chamfer distance over accepted real/sim pairs changed from approximately
`75 mm` before rigid registration to `43 mm` after registration. The remaining error shows that a
single rigid transform does not explain all real/sim differences.

## Interpretation for robustness

The simulated point clouds do provide episode-dependent camera/calibration diversity. However,
the measured simulation distribution varies around a point-cloud-domain center that is itself
offset from the real distribution by approximately 10 cm and 4.7 degrees.

Symmetric camera randomization alone guarantees that samples are centered on the configured
nominal simulated camera pose. It does not guarantee that real point clouds are centered in the
resulting point-cloud distribution. That requires the nominal camera transform, robot geometry,
coordinate conversion, depth model, and point sampling to be jointly unbiased.

For future datasets, store these fields in per-episode metadata:

- nominal camera pose;
- sampled physical camera translation and rotation;
- sampled calibration-error transform;
- effective camera pose used for world-coordinate reconstruction;
- random seed and relevant camera/configuration version.

Those fields would permit a direct correlation between sampled perturbations and measured
point-cloud displacement, avoiding the need to infer episode transforms through ICP.
