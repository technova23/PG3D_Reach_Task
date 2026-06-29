# ManiSkill setup runbook

Status: primary simulator stack for pg3d.

Known constraints:

- ManiSkill/SAPIEN is optional so base imports and CPU-only tests stay simulator-free.
- `pyproject.toml` pins `mani_skill==3.0.1`; do not change simulator versions without updating
  the lockfile, this runbook, and the simulator ADR/status notes.
- The default smoke path uses `obs_mode="state"` and does not require rendering.
- Point-cloud/RGB-D/segmentation paths may require Vulkan and asset setup.
- The first built-in smoke task is `PickCube-v1`; custom `PG3DReach-Narrow-v0`,
  `PG3DReach-Medium-v0`, and `PG3DReach-Workspace-v0` tasks are available for reach dataset
  smoke/training.

## Install pg3d with ManiSkill

RTX 5090 / CUDA 12.9 path:

```bash
uv sync --extra cu129 --extra maniskill --group dev --group notebooks
```

CPU/debug path:

```bash
uv sync --extra cpu --extra maniskill --group dev
```

Optional Rerun visualization path:

```bash
uv sync --extra cu129 --extra maniskill --extra viz --group dev --group notebooks
```

The `viz` extra currently pins `rerun-sdk==0.22.1` because newer Rerun releases tested during P04
require NumPy 2, while pg3d still constrains NumPy to `<2`.

Keep the `maniskill` extra out of default docs/test environments. Base `pg3d` imports must not
require ManiSkill, SAPIEN, rendering, Vulkan, or a GPU.

## Assets

Set an asset directory if the default cache location is not desired:

```bash
export MS_ASSET_DIR=/path/to/maniskill_assets
```

For non-interactive smoke runs, avoid asset download prompts:

```bash
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
```

The basic `PickCube-v1` state smoke should not require large visual assets. Custom tasks or
visual-observation scripts may need additional assets later.

## Non-rendering smoke

```bash
uv run python scripts/check_maniskill.py
make maniskill-check
```

The script imports `gymnasium` and `mani_skill.envs`, creates `PickCube-v1` with
`obs_mode="state"` and `num_envs=1`, resets with `seed=0`, prints observation/action spaces, steps
one sampled action, and closes the environment.

## Observation adapter smoke

Save a structured state observation:

```bash
uv run python scripts/save_maniskill_observation.py \
  --obs-mode state_dict \
  --output-dir artifacts/maniskill_state_observation
```

Save a point-cloud observation with segmentation-derived Panda/cube/goal masks:

```bash
uv run python scripts/save_maniskill_observation.py \
  --obs-mode pointcloud \
  --output-dir artifacts/maniskill_pointcloud_observation
```

Save a short MP4 render alongside the observation:

```bash
uv run python scripts/save_maniskill_observation.py \
  --obs-mode pointcloud \
  --video-frames 8 \
  --output-dir artifacts/maniskill_pointcloud_video
```

Save a Rerun point-cloud artifact after installing the optional `viz` extra:

```bash
uv run python scripts/save_maniskill_observation.py \
  --obs-mode pointcloud \
  --rerun-path artifacts/maniskill_pointcloud.rrd \
  --output-dir artifacts/maniskill_pointcloud_rerun
```

## Reach dataset smoke

Generate a 3-demo smoke dataset from the custom reach task:

```bash
uv run python scripts/write_maniskill_reach_dataset.py \
  --num-demos 3 \
  --hold-steps 8 \
  --output /tmp/pg3d-reach-smoke.zarr \
  --overwrite
```

Replay the stored simulator actions:

```bash
uv run python scripts/replay_maniskill_reach_dataset.py \
  --dataset /tmp/pg3d-reach-smoke.zarr \
  --episodes 3
```

Replay with MP4 videos and Rerun timeline artifacts:

```bash
uv run python scripts/replay_maniskill_reach_dataset.py \
  --dataset /tmp/pg3d-reach-smoke.zarr \
  --episodes 3 \
  --video-dir artifacts/reach-dataset-smoke/videos \
  --rerun-dir artifacts/reach-dataset-smoke/rerun
```

Open one timeline artifact in the Rerun viewer:

```bash
uv run rerun artifacts/reach-dataset-smoke/rerun/episode_000.rrd
```

Use the `step` timeline in the Rerun viewer and press play.

The observation-save Rerun command above writes a single static point-cloud snapshot. The replay
command writes one `.rrd` per episode with a `step` timeline.

The writer registers pg3d reach tasks lazily, defaults to `PG3DReach-BalancedWorkspace-v0`, uses
the Panda robot with `control_mode="pd_joint_pos"`, saves 7D arm-only DP3 action labels, keeps full
simulator actions in `/data/sim_action`, records one DP3 action chunk of post-success hold-pose
rows by default (`--hold-steps 8`), and crops point clouds to the default workspace AABB:

```text
x: [-0.9, 0.7], y: [-0.6, 0.6], z: [0.0, 1.1]
```

The writer runs without a human viewer unless `--viewer` is passed. If the viewer opens as a black
window, make sure the command includes `--viewer`; the script only pumps viewer frames in that
mode. Add `--viewer-step-delay 0.03` to make live motion visible and `--viewer-hold-seconds 5` to
keep the window open briefly before cleanup.

Reach demos now show the target with a green marker and the sampled TCP start with a red marker.
The writer samples Cartesian starts from the selected task bounds by default, rejects starts too
close to the target, and only accepts starts that the Panda motion planner can reach from reset.
Pass `--no-randomize-start` to reproduce old fixed-start behavior.

Use `PG3DReach-Workspace-v0` for the diverse pre-constraints policy. Its goal distribution is
uniform over:

```text
x: [-0.30, 0.40], y: [-0.35, 0.35], z: [0.15, 0.75]
```

## Optional rendering/point-cloud checks

Do not make rendering part of the default smoke path. Once the adapter needs visual observations,
add a separate script for `obs_mode="pointcloud"` or RGB-D/segmentation and document:

- Vulkan driver/runtime status,
- required assets,
- `MS_ASSET_DIR`,
- whether the script can run headless on the workstation.

## Troubleshooting log

Append specific failures/fixes here rather than burying them in chat history.

### Missing optional dependency

If `scripts/check_maniskill.py` reports `Failed to import ManiSkill/Gymnasium`, run:

```bash
uv sync --extra cu129 --extra maniskill --group dev --group notebooks
```

### Vulkan or rendering failure

The default state smoke should not require rendering. If a point-cloud or RGB-D script fails in
renderer setup, first confirm the NVIDIA driver and Vulkan runtime before changing Python or torch
versions.
