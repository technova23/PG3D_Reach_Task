# Commands runbook

Update this file whenever setup, run, test, or eval commands change.

## Create environment

RTX 5090 / CUDA 12.9 path:

```bash
uv sync --extra cu129 --group dev
```

RTX 5090 / CUDA 12.9 path with ManiSkill:

```bash
uv sync --extra cu129 --extra maniskill --group dev --group notebooks
```

CPU/debug path:

```bash
uv sync --extra cpu --group dev
```

CPU/debug path with ManiSkill:

```bash
uv sync --extra cpu --extra maniskill --group dev
```

## Basic checks

```bash
make test
make lint
make smoke
make gpu-check
make maniskill-check
```

Equivalent direct commands:

```bash
uv run pytest
uv run ruff check .
uv run python scripts/smoke_imports.py
uv run python scripts/check_gpu.py
uv run python scripts/check_maniskill.py
```

## DP3 policy smoke

The pg3d-native DP3 slice is tested with synthetic point-cloud/state/action data:

```bash
uv run python scripts/smoke_dp3_policy.py --device cpu
uv run python scripts/smoke_dp3_policy.py --device cuda
```

Use the CPU smoke in sandbox/CI contexts. Use the CUDA smoke on the RTX 5090 workstation after
`make gpu-check` succeeds.

## Submodules

```bash
git submodule update --init --recursive
```

`external/dp3` is a private reference submodule. Runtime code should import
`pg3d.policies.dp3`, not `external/dp3`.

When cloning a new workstation:

```bash
git clone --recurse-submodules git@github.com:YOUR_ORG/pg3d.git
cd pg3d
uv sync --extra cu129 --group dev
```

## ManiSkill smoke

First install the optional extra as described in `docs/runbooks/maniskill_setup.md`, then run:

```bash
uv run python scripts/check_maniskill.py
make maniskill-check
```

The default smoke uses `PickCube-v1` with `obs_mode="state"` and no rendering.

## ManiSkill observation artifact

```bash
uv run python scripts/save_maniskill_observation.py --obs-mode state_dict \
  --output-dir artifacts/maniskill_state_observation
uv run python scripts/save_maniskill_observation.py --obs-mode pointcloud \
  --output-dir artifacts/maniskill_pointcloud_observation
```

For optional Rerun export, first sync the `viz` extra:

```bash
uv sync --extra cu129 --extra maniskill --extra viz --group dev --group notebooks
```

The optional `viz` extra uses `rerun-sdk==0.22.1` while pg3d remains on NumPy 1.x.

## ManiSkill reach dataset

Generate a small reach dataset:

```bash
uv run python scripts/write_maniskill_reach_dataset.py \
  --num-demos 5 \
  --hold-steps 8 \
  --trajectory-variants-per-reset 3 \
  --output artifacts/pg3d_reach_balanced.zarr \
  --overwrite
```

By default the writer uses `PG3DReach-BalancedWorkspace-v0` and planner-validated randomized TCP
starts. Each random start/goal setup tries `--trajectory-variants-per-reset` trajectory families
before moving to the next seed. Incomplete seed/start groups are skipped by default, so datasets do
not silently miss one requested family such as `downward_arc`. Use
`--allow-partial-variant-sets` only for debugging or old compatibility runs, and use
`--show-planner-output` only when diagnosing ManiSkill planner internals because expected failed
dry-run retries can print many `screw plan failed` lines.

The dataset writer is headless by default. To watch collection live in the ManiSkill viewer, add
`--viewer`; use `--viewer-step-delay` to slow the loop and `--viewer-hold-seconds` to keep the
window open briefly after collection:

```bash
uv run python scripts/write_maniskill_reach_dataset.py \
  --num-demos 1 \
  --max-attempts 3 \
  --viewer \
  --viewer-step-delay 0.03 \
  --viewer-hold-seconds 5 \
  --output artifacts/pg3d_reach_viewer_smoke.zarr \
  --overwrite
```

Replay the stored simulator actions:

```bash
uv run python scripts/replay_maniskill_reach_dataset.py \
  --dataset artifacts/pg3d_reach_narrow.zarr \
  --episodes 5
```

Replay with MP4 videos and per-episode Rerun timeline artifacts:

```bash
uv run python scripts/replay_maniskill_reach_dataset.py \
  --dataset artifacts/pg3d_reach_narrow.zarr \
  --episodes 5 \
  --video-dir artifacts/reach_replay/videos \
  --rerun-dir artifacts/reach_replay/rerun
```

Open a replay `.rrd` in the Rerun viewer:

```bash
uv run rerun artifacts/reach_replay/rerun/episode_000.rrd
```

Use the `step` timeline in the Rerun viewer and press play.

The dataset writer uses `PG3DReach-Narrow-v0`, `obs_mode="pointcloud"`, `pd_joint_pos`,
Panda arm-only 7D DP3 action labels, one extra action chunk of post-success hold-pose data, and a
fixed-size cropped point cloud by default. `PG3DReach-BalancedWorkspace-v0` is the P11 base-reach
reliability distribution.

Generate a small P11 balanced diagnostic dataset:

```bash
uv run python scripts/write_maniskill_reach_dataset.py \
  --env-id PG3DReach-BalancedWorkspace-v0 \
  --num-demos 100 \
  --max-attempts 200 \
  --max-steps-per-demo 100 \
  --hold-steps 8 \
  --num-points 512 \
  --output artifacts/reach-datasets/pg3d-reach-balanced-100.zarr \
  --overwrite
```

Inspect target distribution, raw goal visibility, ordered marker correctness, and train/validation
region balance:

```bash
uv run python scripts/diagnose_reach_dataset.py \
  --dataset artifacts/reach-datasets/pg3d-reach-balanced-100.zarr \
  --goal-marker-points 16 \
  --goal-marker-radius 0.015 \
  --val-ratio 0.1 \
  --split-seed 42
```

Pilot before launching the 500-episode dataset:

```bash
uv run python scripts/write_maniskill_reach_dataset.py \
  --env-id PG3DReach-Narrow-v0 \
  --num-demos 100 \
  --max-attempts 150 \
  --hold-steps 8 \
  --num-points 512 \
  --output artifacts/reach-datasets/pg3d-reach-narrow-100.zarr \
  --overwrite
```

Scale to the first nominal 500-episode narrow dataset after pilot replay inspection:

```bash
uv run python scripts/write_maniskill_reach_dataset.py \
  --env-id PG3DReach-Narrow-v0 \
  --num-demos 500 \
  --max-attempts 700 \
  --hold-steps 8 \
  --num-points 512 \
  --output artifacts/reach-datasets/pg3d-reach-narrow-500.zarr \
  --overwrite
```

Generate a broad workspace-uniform dataset for constraint/reranking policy pretraining:

```bash
export ART=/home/krishna/code/pg3d/artifacts/reach-datasets
export DATASET="$ART/pg3d-reach-workspace-1000.zarr"
export REPLAY="$ART/pg3d-reach-workspace-1000-replay"
export CKPTS="$ART/dp3-reach-workspace-1000-checkpoints"
export WANDB_DIR="$ART/wandb"
export WANDB_CACHE_DIR="$ART/wandb-cache"
export WANDB_CONFIG_DIR="$ART/wandb-config"
export UV_CACHE_DIR=/tmp/pg3d-uv-cache
export MPLCONFIGDIR=/tmp/pg3d-mpl
```

```bash
uv run python scripts/write_maniskill_reach_dataset.py \
  --env-id PG3DReach-Workspace-v0 \
  --num-demos 1000 \
  --max-attempts 1600 \
  --max-steps-per-demo 100 \
  --hold-steps 8 \
  --num-points 512 \
  --seed-start 0 \
  --output "$DATASET" \
  --overwrite
```

Replay a deterministic inspection subset with MP4 and Rerun artifacts:

```bash
uv run python scripts/replay_maniskill_reach_dataset.py \
  --dataset "$DATASET" \
  --episodes 50 \
  --video-dir "$REPLAY/videos" \
  --rerun-dir "$REPLAY/rerun" \
  --allow-failure
```

Open one inspection replay:

```bash
uv run rerun "$REPLAY/rerun/episode_000.rrd"
```

Create a held-out 50-episode validation dataset from solved workspace seeds before comparing
constrained methods. The writer skips planner failures and unsuccessful replays unless
`--keep-failures` is passed, so `--source dataset` evals on this Zarr file use fixed solvable
episodes instead of arbitrary fresh seeds:

```bash
export VAL_DATASET="$ART/pg3d-reach-workspace-val-50.zarr"
export VAL_REPLAY="$ART/pg3d-reach-workspace-val-50-replay"
export EVAL_OUT="$ART/constrained-reach-val-50"
```

```bash
uv run python scripts/write_maniskill_reach_dataset.py \
  --env-id PG3DReach-Workspace-v0 \
  --num-demos 50 \
  --max-attempts 250 \
  --max-steps-per-demo 100 \
  --hold-steps 8 \
  --num-points 512 \
  --seed-start 20000 \
  --output "$VAL_DATASET" \
  --overwrite
```

Verify the validation metadata:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

p = Path("/home/krishna/code/pg3d/artifacts/reach-datasets/pg3d-reach-workspace-val-50.zarr/metadata.json")
m = json.loads(p.read_text())
print(json.dumps({
    "env_id": m["env_id"],
    "num_requested_demos": m["num_requested_demos"],
    "num_collected_demos": m["num_collected_demos"],
    "num_attempts": m["num_attempts"],
    "success_rate": m["dataset_stats"]["success_rate"],
    "hold_coverage": m["dataset_stats"]["hold_coverage"],
    "final_distance": m["dataset_stats"]["final_distance"],
    "seed_start": m["seed_start"],
    "first_seed": m["episodes"][0]["seed"],
    "last_seed": m["episodes"][-1]["seed"],
}, indent=2))
PY
```

Strictly replay all validation episodes without visualization:

```bash
uv run python scripts/replay_maniskill_reach_dataset.py \
  --dataset "$VAL_DATASET" \
  --episodes 50
```

Generate validation replay MP4 and Rerun artifacts:

```bash
uv run python scripts/replay_maniskill_reach_dataset.py \
  --dataset "$VAL_DATASET" \
  --episodes 50 \
  --video-dir "$VAL_REPLAY/videos" \
  --rerun-dir "$VAL_REPLAY/rerun" \
  --allow-failure
```

## DP3 reach training smoke

Run a short dataset-loading and training smoke:

```bash
uv run python scripts/train_dp3_reach.py \
  --dataset artifacts/reach-dataset-smoke/pg3d-reach-smoke.zarr \
  --device cpu \
  --max-steps 1 \
  --batch-size 2 \
  --num-workers 0 \
  --val-ratio 0 \
  --checkpoint-dir artifacts/reach-dataset-smoke/checkpoints \
  --checkpoint-every 1 \
  --no-checkpoint-rollout-videos
```

Run dataset-only inference/eval against that checkpoint:

```bash
uv run python scripts/eval_dp3_reach_dataset.py \
  --dataset artifacts/reach-dataset-smoke/pg3d-reach-smoke.zarr \
  --checkpoint artifacts/reach-dataset-smoke/checkpoints/final_step_00000001.pt \
  --device cpu \
  --max-batches 1 \
  --batch-size 2
```

Enable W&B metric and histogram logging when the local W&B service can start:

```bash
uv run python scripts/train_dp3_reach.py \
  --dataset artifacts/reach-dataset-smoke/pg3d-reach-smoke.zarr \
  --device cpu \
  --max-steps 1 \
  --wandb-mode offline \
  --log-histograms \
  --checkpoint-dir artifacts/reach-dataset-smoke/checkpoints \
  --no-checkpoint-rollout-videos
```

In restricted sandboxes, W&B may fail to create its local cache/socket. The trainer logs a warning
and continues unless `--wandb-required` is set.

Moderate 5090 pilot training recipe:

```bash
uv run python scripts/train_dp3_reach.py \
  --dataset artifacts/reach-datasets/pg3d-reach-narrow-100.zarr \
  --device cuda \
  --max-steps 20000 \
  --batch-size 64 \
  --num-workers 4 \
  --val-ratio 0.1 \
  --val-every 500 \
  --lr 1e-4 \
  --warmup-steps 500 \
  --grad-clip-norm 1.0 \
  --use-ema \
  --wandb-mode online \
  --wandb-project pg3d \
  --wandb-name dp3-reach-narrow-100-stable \
  --checkpoint-dir artifacts/reach-datasets/dp3-reach-narrow-100-stable-checkpoints \
  --checkpoint-every 5000 \
  --checkpoint-rollout-count 5
```

Workspace-uniform 1000-episode training recipe for constraint/reranking pretraining:

```bash
uv run python scripts/train_dp3_reach.py \
  --dataset "$DATASET" \
  --device cuda \
  --max-steps 50000 \
  --batch-size 64 \
  --num-workers 4 \
  --val-ratio 0.1 \
  --val-every 500 \
  --max-val-batches 8 \
  --lr 1e-4 \
  --warmup-steps 1000 \
  --grad-clip-norm 1.0 \
  --use-ema \
  --wandb-mode online \
  --wandb-project pg3d \
  --wandb-name dp3-reach-workspace-1000-50k \
  --log-histograms \
  --histogram-every 1000 \
  --checkpoint-dir "$CKPTS" \
  --checkpoint-every 5000 \
  --checkpoint-rollout-dataset "$VAL_DATASET" \
  --checkpoint-rollout-count 5 \
  --checkpoint-rollout-selection-seed 0 \
  --checkpoint-rollout-max-steps 80 \
  --checkpoint-rollout-post-success-steps 8
```

The trainer defaults to `pad_after=n_action_steps-1`, cosine LR with warmup, AdamW
`betas=(0.95, 0.999)`, gradient clipping, EMA checkpoint state, and W&B validation/action-error
metrics. P11 training also defaults to `--goal-marker-points 16 --goal-marker-radius 0.015`,
which overwrites the final K policy-visible point slots with ordered target markers. Pass
`--goal-marker-points 0` only for old-checkpoint compatibility or ablations. DP3 normalizers are
fit from up to `--normalizer-max-steps 4096` evenly spaced Zarr timesteps by default, which keeps
startup reasonable for 1024-point datasets; pass `--normalizer-max-steps 0` for exact full-dataset
normalizer stats. The trainer writes
periodic `step_XXXXXXXX.pt` checkpoints and a final
`final_step_XXXXXXXX.pt` checkpoint under `--checkpoint-dir`. When W&B is active,
it attempts to log checkpoint-time rollout MP4s. Pass `--checkpoint-rollout-dataset "$VAL_DATASET"`
to use a deterministic random subset of five held-out validation episodes at every checkpoint
instead of mixed train/fresh seeds. Use `--no-checkpoint-rollout-videos` to skip
simulator/rendering rollouts during training.

Balanced workspace P11 pilot recipe:

```bash
export BAL_DATASET="$ART/pg3d-reach-balanced-1000.zarr"
export BAL_VAL_DATASET="$ART/pg3d-reach-balanced-val-100.zarr"
export BAL_CKPTS="$ART/dp3-reach-balanced-1000-checkpoints"
```

```bash
uv run python scripts/write_maniskill_reach_dataset.py \
  --env-id PG3DReach-BalancedWorkspace-v0 \
  --num-demos 1000 \
  --max-attempts 1800 \
  --max-steps-per-demo 100 \
  --hold-steps 8 \
  --num-points 512 \
  --seed-start 0 \
  --output "$BAL_DATASET" \
  --overwrite
```

```bash
uv run python scripts/write_maniskill_reach_dataset.py \
  --env-id PG3DReach-BalancedWorkspace-v0 \
  --num-demos 100 \
  --max-attempts 300 \
  --max-steps-per-demo 100 \
  --hold-steps 8 \
  --num-points 512 \
  --seed-start 20000 \
  --output "$BAL_VAL_DATASET" \
  --overwrite
```

```bash
uv run python scripts/train_dp3_reach.py \
  --dataset "$BAL_DATASET" \
  --device cuda \
  --max-steps 50000 \
  --batch-size 128 \
  --num-workers 4 \
  --val-ratio 0.1 \
  --val-every 500 \
  --max-val-batches 8 \
  --goal-marker-points 16 \
  --goal-marker-radius 0.015 \
  --normalizer-max-steps 4096 \
  --lr 1e-4 \
  --warmup-steps 1000 \
  --grad-clip-norm 1.0 \
  --use-ema \
  --wandb-mode online \
  --wandb-project pg3d \
  --wandb-name dp3-reach-balanced-1000-goal-tokens \
  --checkpoint-dir "$BAL_CKPTS" \
  --checkpoint-every 5000 \
  --checkpoint-rollout-dataset "$BAL_VAL_DATASET" \
  --checkpoint-rollout-count 5 \
  --checkpoint-rollout-selection-seed 0 \
  --checkpoint-rollout-max-steps 80 \
  --checkpoint-rollout-post-success-steps 8
```

Run closed-loop policy rollouts in ManiSkill and save MP4/Rerun artifacts:

```bash
uv sync --extra cu129 --extra maniskill --extra viz --group dev --group notebooks
```

```bash
uv run python scripts/rollout_dp3_reach_policy.py \
  --dataset artifacts/reach-dataset-smoke/pg3d-reach-smoke.zarr \
  --checkpoint artifacts/reach-dataset-smoke/checkpoints/final_step_00000001.pt \
  --source dataset \
  --episodes 3 \
  --device cuda \
  --output-dir artifacts/reach-dataset-smoke/policy-rollouts-dataset
```

Evaluate fresh seeds from the same reach distribution:

```bash
uv run python scripts/rollout_dp3_reach_policy.py \
  --dataset artifacts/reach-dataset-smoke/pg3d-reach-smoke.zarr \
  --checkpoint artifacts/reach-dataset-smoke/checkpoints/final_step_00000001.pt \
  --source fresh \
  --episodes 3 \
  --seed-start 10000 \
  --device cuda \
  --output-dir artifacts/reach-dataset-smoke/policy-rollouts-fresh
```

The rollout script re-observes after each configurable `--replan-stride` chunk, uses EMA checkpoint
weights by default when present, records one post-success hold window by default, and always logs
the goal marker in the Rerun timeline.

## World-model versus simulator rollout comparison

Compare a stable DP3 reach checkpoint against the P07 world model. The policy is queried from the
world-model branch, and ManiSkill executes the same action chunks for ground-truth comparison:

```bash
uv sync --extra cu129 --extra maniskill --extra viz --group dev --group notebooks
```

```bash
uv run python scripts/compare_world_model_rollout.py \
  --dataset artifacts/reach-datasets/pg3d-reach-narrow-100.zarr \
  --checkpoint-dir artifacts/reach-datasets/dp3-reach-narrow-100-stable-checkpoints \
  --source dataset \
  --episodes 3 \
  --device cuda \
  --output-dir artifacts/reach-datasets/world-model-vs-sim \
  --rerun \
  --video \
  --allow-failure
```

Open the overlay:

```bash
uv run rerun artifacts/reach-datasets/world-model-vs-sim/episode_000_comparison.rrd
```

The comparison script selects the latest step-named checkpoint from `--checkpoint-dir`, preferring
`final_step_*.pt` at the latest step. It uses a second ManiSkill ghost env to render robot-segmented
point clouds at imagined Panda qpos states, then overlays those clouds with the live simulator
rollout in Rerun. It writes one `episode_XXX_comparison.rrd` per compared episode. Use
`--source fresh --episodes 50 --seed-start 10000` for fresh-seed comparison after dataset-seed
overlays look sane.

## Constrained reach evaluation

The first MVP eval scaffold compares base DP3, candidate rejection, and world-model reranking on
the same fixed seeds and the same saved direct-path avoid-region constraints. Code-only waypoint
planning is a strong reach baseline and is not implemented in this scaffold, so do not over-claim
reach-only results.

Tiny fixed-seed smoke:

```bash
uv run python scripts/eval_constrained_reach.py \
  --dataset artifacts/reach-datasets/pg3d-reach-workspace-1000.zarr \
  --checkpoint-dir artifacts/reach-datasets/dp3-reach-workspace-1000-checkpoints \
  --methods base rejection reranking \
  --source fresh \
  --episodes 3 \
  --seed-start 10000 \
  --device cuda \
  --planning-horizon-chunks 1 \
  --execution-horizon-chunks 1 \
  --geometry-mode fast \
  --k-schedule 16 32 64 \
  --video \
  --video-every-episodes 10 \
  --rerun \
  --rerun-every-episodes 10 \
  --artifact-selection random \
  --artifact-episode-count 5 \
  --artifact-selection-seed 0 \
  --plots \
  --plot-every-episodes 10 \
  --profile \
  --profile-every-episodes 10 \
  --wandb-mode offline \
  --output-dir artifacts/constrained-reach-eval-smoke \
  --allow-failure
```

Longer multi-chunk planning smoke:

```bash
uv run python scripts/eval_constrained_reach.py \
  --dataset artifacts/reach-datasets/pg3d-reach-workspace-1000.zarr \
  --checkpoint-dir artifacts/reach-datasets/dp3-reach-workspace-1000-checkpoints \
  --methods base rejection reranking \
  --source fresh \
  --episodes 10 \
  --seed-start 10100 \
  --device cuda \
  --planning-horizon-chunks 2 \
  --execution-horizon-chunks 1 \
  --geometry-mode fast \
  --k-schedule 16 32 64 \
  --policy-batch-size 64 \
  --video \
  --video-every-episodes 10 \
  --constraint-overlay-video \
  --constraint-overlay-alpha 0.25 \
  --rerun \
  --rerun-every-episodes 10 \
  --artifact-selection random \
  --artifact-episode-count 5 \
  --artifact-selection-seed 0 \
  --plots \
  --plot-every-episodes 10 \
  --profile \
  --wandb-mode online \
  --wandb-project pg3d \
  --wandb-name constrained-reach-p10-smoke \
  --output-dir artifacts/constrained-reach-eval-multichunk \
  --allow-failure
```

Validation-set comparison on fixed solved workspace episodes:

```bash
uv run python scripts/eval_constrained_reach.py \
  --dataset "$VAL_DATASET" \
  --checkpoint-dir "$CKPTS" \
  --methods base rejection reranking \
  --source dataset \
  --episodes 50 \
  --device cuda \
  --seed 0 \
  --planning-horizon-chunks 2 \
  --execution-horizon-chunks 1 \
  --geometry-mode fast \
  --k-schedule 16 32 64 \
  --policy-batch-size 64 \
  --video \
  --video-every-episodes 10 \
  --rerun \
  --rerun-every-episodes 10 \
  --plots \
  --plot-every-episodes 10 \
  --profile \
  --profile-every-episodes 10 \
  --wandb-mode online \
  --wandb-project pg3d \
  --wandb-name constrained-reach-val-50 \
  --output-dir "$EVAL_OUT" \
  --allow-failure
```

Dataset-derived Cartesian pose waypoint comparison:

```bash
uv run python scripts/eval_constrained_reach.py \
  --dataset "$VAL_DATASET" \
  --checkpoint "$CKPT" \
  --methods reranking \
  --source dataset \
  --episodes 10 \
  --unique-dataset-seeds \
  --device cuda \
  --seed 0 \
  --constraint-type cartesian_pose \
  --cartesian-pose-path-fraction 0.5 \
  --cartesian-pose-position-tolerance 0.02 \
  --cartesian-pose-rotation-tolerance 0.35 \
  --planning-horizon-chunks 2 \
  --execution-horizon-chunks 1 \
  --geometry-mode fast \
  --k-schedule 32 \
  --video \
  --rerun \
  --artifact-selection all \
  --constraint-overlay-video \
  --output-dir "$EVAL_OUT" \
  --allow-failure
```

With `--constraint-type cartesian_pose`, constrained eval samples the target EEF pose from the
selected dataset episode's saved `/data/tcp_pose` path at the requested arc-length fraction. The
generated `constraints/episode_XXX.json` stores the pose plus zarr provenance metadata including
dataset episode index, local frame index, global frame index, and path fraction. This mode requires
`--source dataset`; use the same generated constraints with `--constraints-dir` when comparing
multiple horizon or method settings exactly.

Balanced 20k nominal-path starter:

```bash
export BAL_VAL_DATASET="$ART/pg3d-reach-balanced-val-100.zarr"
export BAL_20K_CKPT="$ART/dp3-reach-balanced-1000-checkpoints/step_00020000.pt"
export BAL_20K_BASE_OUT="$ART/dp3-reach-balanced-1000-rollouts/val-25-at-20k-iters"
export BAL_NOMINAL_CONSTRAINTS="$ART/constrained-reach-balanced-20k-nominal-r003-val25"
export BAL_CONSTRAINED_OUT="$ART/constrained-reach-balanced-20k-r003-val25"
```

First gate the base checkpoint on the same 25 held-out balanced validation episodes. Stop here if
fewer than 15 episodes reach; reranking should not be interpreted until base reach is good enough:

```bash
uv run python scripts/rollout_dp3_reach_policy.py \
  --checkpoint "$BAL_20K_CKPT" \
  --checkpoint-model ema \
  --dataset "$BAL_VAL_DATASET" \
  --source dataset \
  --episodes 25 \
  --device cuda \
  --max-steps 100 \
  --post-success-steps 8 \
  --output-dir "$BAL_20K_BASE_OUT" \
  --allow-failure
```

Build fixed nominal-path constraints from the base-success subset. This writes
`constraints/episode_XXX.json`, `episode_indices.txt`, `paths/episode_XXX.npy`, and
`manifest.json`. The default sphere is centered at 50% arc length on the successful nominal TCP
path with radius `0.03m`. The first 2026-05-19 run with the 20k checkpoint selected only 7/25
base-success episodes, below the 15-success gate, so the main constrained eval should remain
blocked until base reach improves:

```bash
uv run python scripts/build_nominal_path_constraints.py \
  --checkpoint "$BAL_20K_CKPT" \
  --checkpoint-model ema \
  --dataset "$BAL_VAL_DATASET" \
  --episodes 25 \
  --device cuda \
  --max-steps 100 \
  --post-success-steps 8 \
  --avoid-radius 0.03 \
  --path-fraction 0.5 \
  --min-successes 15 \
  --output-dir "$BAL_NOMINAL_CONSTRAINTS"
```

When the base gate passes, evaluate all three methods on the exact same selected dataset episodes
and constraints:

```bash
uv run python scripts/eval_constrained_reach.py \
  --checkpoint "$BAL_20K_CKPT" \
  --checkpoint-model ema \
  --dataset "$BAL_VAL_DATASET" \
  --source dataset \
  --episode-indices-file "$BAL_NOMINAL_CONSTRAINTS/episode_indices.txt" \
  --constraints-dir "$BAL_NOMINAL_CONSTRAINTS/constraints" \
  --methods base rejection reranking \
  --episodes 25 \
  --device cuda \
  --seed 0 \
  --max-steps 100 \
  --planning-horizon-chunks 2 \
  --execution-horizon-chunks 1 \
  --geometry-mode fast \
  --k-schedule 16 32 64 \
  --policy-batch-size 128 \
  --video \
  --rerun \
  --artifact-selection random \
  --artifact-episode-count 5 \
  --artifact-selection-seed 0 \
  --plots \
  --plot-every-episodes 5 \
  --profile \
  --profile-every-episodes 5 \
  --wandb-mode online \
  --wandb-project pg3d \
  --wandb-name constrained-reach-balanced-20k-r003-val25 \
  --output-dir "$BAL_CONSTRAINED_OUT" \
  --allow-failure
```

Write a final method comparison plot with Wilson 95% confidence intervals:

```bash
uv run python scripts/plot_constrained_reach_summary.py \
  --summary "$EVAL_OUT/summary.json" \
  --output "$EVAL_OUT/plots/comparative_success_ci.png"
```

Print the main numeric comparison:

```bash
uv run python - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ["EVAL_OUT"])
summary = json.loads((out / "summary.json").read_text())["by_method"]
for method, stats in summary.items():
    print(f"\n{method}")
    for key in [
        "reach_success_rate",
        "constraint_satisfied_rate",
        "combined_success_rate",
        "final_target_distance_mean",
        "min_clearance_mean",
        "candidate_feasibility_fraction_mean",
    ]:
        print(f"  {key}: {stats.get(key)}")
PY
```

Outputs include `constraints/episode_XXX.json`, `metrics.jsonl`, `decisions.jsonl`,
`step_traces.jsonl`, `summary.json`, optional `timings.jsonl`, optional `plots/*.png`, optional
`videos/{method}/episode_XXX.mp4`, and optional `rerun/{method}/episode_XXX.rrd`.
When `--constraints-dir` is provided, eval copies the loaded precomputed constraints into the
output constraints directory and records the source in `summary.json`.
With `--artifact-selection random --artifact-episode-count 5 --artifact-selection-seed 0`,
metrics still cover all 50 validation episodes, while MP4/Rerun artifacts are written only for
one deterministic random subset of five validation episodes. The selected output indices, dataset
episode indices, and seeds are recorded in `summary.json`.

When `--video` is enabled, constrained-eval MP4s render the avoid region in a separate
visualization-only ManiSkill env by default. This keeps the policy/control env unchanged while
showing a translucent orange keep-out sphere or box in the saved video. Use
`--no-constraint-overlay-video` to fall back to plain simulator renders, and tune the visual with
`--constraint-overlay-alpha` and `--constraint-overlay-color R G B`. Rerun exports log the same
avoid region under `world/constraints/avoid_region_*` as persistent wireframe geometry.
For Cartesian pose constraints, `metrics.jsonl` stores executed best-frame pose summaries,
`step_traces.jsonl` stores per-frame TCP pose/action/pose-error rows, and `decisions.jsonl` stores
the selected world-model branch including the selected action chunk, imagined q/EEF rollout, and
compact candidate summaries.

One-episode overlay smoke for visual inspection:

```bash
export OVERLAY_OUT="$ART/constrained-reach-overlay-smoke"

uv run python scripts/eval_constrained_reach.py \
  --dataset "$VAL_DATASET" \
  --checkpoint-dir "$CKPTS" \
  --methods reranking \
  --source dataset \
  --episodes 1 \
  --device cuda \
  --seed 0 \
  --planning-horizon-chunks 2 \
  --execution-horizon-chunks 1 \
  --geometry-mode fast \
  --k-schedule 16 32 64 \
  --policy-batch-size 64 \
  --video \
  --rerun \
  --artifact-selection all \
  --constraint-overlay-video \
  --constraint-overlay-alpha 0.25 \
  --constraint-overlay-color 1.0 0.25 0.05 \
  --output-dir "$OVERLAY_OUT" \
  --allow-failure

uv run rerun "$OVERLAY_OUT/rerun/reranking/episode_000.rrd"
```

The default `--geometry-mode fast` avoids rendering ghost-env robot point clouds for every
candidate timestep. It scores candidates from q/EEF trajectories and only renders ghost point
clouds when a future imagined state must be fed back into DP3 for multi-chunk planning. Use
`--geometry-mode exact` on a 1-episode spot check when validating that the fast path agrees with
the original full-render path.

Exact-vs-fast spot check:

```bash
uv run python scripts/eval_constrained_reach.py \
  --dataset artifacts/reach-datasets/pg3d-reach-workspace-1000.zarr \
  --checkpoint-dir artifacts/reach-datasets/dp3-reach-workspace-1000-checkpoints \
  --methods reranking \
  --source fresh \
  --episodes 1 \
  --seed-start 10100 \
  --device cuda \
  --planning-horizon-chunks 2 \
  --execution-horizon-chunks 1 \
  --geometry-mode exact \
  --k-schedule 16 \
  --profile \
  --output-dir artifacts/constrained-reach-eval-exact-spot \
  --allow-failure
```

Then rerun with `--geometry-mode fast` and compare `summary.json`, `metrics.jsonl`, and
`timings.jsonl`. Keep `--seed` fixed between runs so DP3 candidate sampling is controlled. For
larger sweeps, prefer fast mode plus periodic artifacts.

How to read the printed episode metrics:

- `reach=True` means ManiSkill reported task success at least once during the rollout.
- `constraint=True` means the executed TCP path stayed outside the avoid-region sphere for the
  whole rollout.
- `combined=True` means both reach success and constraint satisfaction were achieved.
- `final` is the final TCP-to-goal distance in meters; lower is better.
- `clearance` is the minimum signed distance from the executed TCP path to the avoid region after
  margin. Positive is outside, near zero grazes the boundary, and negative means the TCP entered
  the forbidden region.

The `--k-schedule 16 32 64` setting is the controller fallback schedule. Controller methods first
score 16 sampled candidate chunks. If no feasible candidate is found, they try 32 more, then 64
more. If all candidates violate the constraint, the controller still returns the least-bad
candidate and records that fallback in `decisions.jsonl`.

Planning and execution horizons are separate chunk counts. With
`--planning-horizon-chunks 2 --execution-horizon-chunks 1`, the controller imagines two DP3 chunks
into the future, feeds the imagined point cloud back into the policy between chunks, scores the
concatenated imagined rollout, executes only the first selected chunk in ManiSkill, then re-observes
and repeats. Setting both values to 1 gives the one-chunk receding-horizon case.

### 2026-05-17 validation-set result

The first `pg3d-reach-workspace-val-50.zarr` run used the fixed solved-seed validation workflow
above, then evaluated `base`, `rejection`, and `reranking` with two planned chunks and one executed
chunk. The final comparison plot at
`artifacts/reach-datasets/constrained-reach-val-50/plots/comparative_success_ci.png` was checked
against both `summary.json` and the raw `metrics.jsonl` rows. It correctly plots the three boolean
rates with Wilson 95% confidence intervals.

Observed rates:

| method | reach success | constraint satisfied | combined success | mean final distance | mean min clearance |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 0.02 | 0.32 | 0.00 | 0.233 m | -0.0207 m |
| rejection | 0.02 | 0.36 | 0.00 | 0.233 m | -0.0215 m |
| reranking | 0.02 | 0.22 | 0.00 | 0.218 m | -0.0339 m |

This is not evidence that reranking helps yet. It means the current workspace checkpoint almost
never reaches the goal on this validation set: only one of 50 episodes reached for each method,
and that reached episode violated the avoid region, so combined success is zero. Constraint-only
success means the executed TCP path stayed outside the avoid sphere, but it can still fail the
task by ending far from the target. Before tuning constraint controllers, debug base policy reach
success on the same validation dataset with constraints treated as diagnostics.

## W&B

```bash
wandb login
# or for offline/debug:
export WANDB_MODE=offline
```

## Long-running training policy

Do not run long training jobs from Codex unless explicitly instructed. Codex should prepare commands/configs and run only smoke-scale jobs by default.
