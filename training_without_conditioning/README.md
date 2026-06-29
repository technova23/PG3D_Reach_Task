# pg3d

`pg3d` studies programmatic geometric guidance for 3D diffusion robot policies.
The current project is simulation-only and uses ManiSkill/SAPIEN as the primary
simulator stack.

The reach-first MVP is:

1. adapt a DP3-style point-cloud diffusion policy to ManiSkill reach data,
2. build a kinematic point-cloud world model from joint-action chunks,
3. score executable geometric constraints such as `avoid_region`,
4. use candidate rejection/reranking in receding horizon mode,
5. move to pick-and-place only after constrained reach works.

The full source-of-truth research plan is `docs/project_proposal.html`.

## Current Status

- Package name: `pg3d`.
- Python: 3.11.
- Dependency manager: `uv`.
- Workstation target: RTX 5090 with PyTorch CUDA 12.9.
- Simulator: ManiSkill/SAPIEN, installed through an optional `maniskill` extra.
- Base policy: pg3d-native DP3 policy core under `pg3d/policies/dp3`.
- Active simulator smoke: `scripts/check_maniskill.py` using `PickCube-v1` with
  `obs_mode="state"`.
- Active reach task/data path: custom `PG3DReach-Narrow-v0` /
  `PG3DReach-Medium-v0` tasks and a Zarr writer for smoke-scale DP3-compatible
  reach datasets.
- RLBench, PyRep, CoppeliaSim, and real-robot/xArm implementation work are not
  active backends in this repo.

## Setup

For the main workstation environment:

```bash
uv sync --extra cu129 --extra maniskill --group dev --group notebooks
```

For CPU-only docs/tests without the simulator:

```bash
uv sync --extra cpu --group dev
```

For CPU-only work with ManiSkill installed:

```bash
uv sync --extra cpu --extra maniskill --group dev
```

For optional Rerun visualization artifacts:

```bash
uv sync --extra cu129 --extra maniskill --extra viz --group dev --group notebooks
```

If this is a fresh clone, initialize submodules:

```bash
git submodule update --init --recursive
```

`external/dp3` is reference material during migration. Runtime imports should use
`pg3d.policies.dp3`, not `external/dp3`.

## Checks

```bash
make smoke
make test
make lint
make gpu-check
make maniskill-check
```

Equivalent direct commands:

```bash
uv run python scripts/smoke_imports.py
uv run pytest
uv run ruff check .
uv run python scripts/check_gpu.py
uv run python scripts/check_maniskill.py
```

The default ManiSkill check is non-rendering. Point-cloud/RGB-D/segmentation
checks should stay separate because they may require Vulkan and asset setup.

## Reach Dataset Smoke

Generate a small custom-reach dataset:

```bash
uv run python scripts/write_maniskill_reach_dataset.py \
  --num-demos 3 \
  --hold-steps 8 \
  --output artifacts/reach-dataset-smoke/pg3d-reach-smoke.zarr \
  --overwrite
```

Replay stored simulator actions and write videos plus Rerun timelines:

```bash
uv run python scripts/replay_maniskill_reach_dataset.py \
  --dataset artifacts/reach-dataset-smoke/pg3d-reach-smoke.zarr \
  --episodes 3 \
  --video-dir artifacts/reach-dataset-smoke/videos \
  --rerun-dir artifacts/reach-dataset-smoke/rerun
```

Open a replay artifact:

```bash
uv run rerun artifacts/reach-dataset-smoke/rerun/episode_000.rrd
```

Use the `step` timeline in the Rerun viewer and press play. The dataset writer
stores 7D Panda arm labels for DP3 and keeps full simulator actions separately
for replay. By default it records one extra DP3 action chunk of hold-pose data
after first success so terminal chunks learn to stay at the goal.

## Reach Dataset Pilot

Before generating the 500-episode dataset, create and inspect a 50-100 episode
pilot:

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

Replay a few pilot episodes with videos and Rerun timelines:

```bash
uv run python scripts/replay_maniskill_reach_dataset.py \
  --dataset artifacts/reach-datasets/pg3d-reach-narrow-100.zarr \
  --episodes 5 \
  --video-dir artifacts/reach-datasets/pg3d-reach-narrow-100-replay/videos \
  --rerun-dir artifacts/reach-datasets/pg3d-reach-narrow-100-replay/rerun
```

Once the pilot looks right, generate the 500-episode dataset:

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

## DP3 Training Smoke

Run a one-step behavior-cloning smoke on the reach dataset:

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

Check dataset-only inference against the checkpoint:

```bash
uv run python scripts/eval_dp3_reach_dataset.py \
  --dataset artifacts/reach-dataset-smoke/pg3d-reach-smoke.zarr \
  --checkpoint artifacts/reach-dataset-smoke/checkpoints/final_step_00000001.pt \
  --device cpu \
  --max-batches 1
```

Train the first moderate 5090 pilot checkpoint:

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

The trainer writes `step_XXXXXXXX.pt` checkpoints at `--checkpoint-every` intervals
and always writes `final_step_XXXXXXXX.pt` in the checkpoint directory. When W&B
is active, checkpoint-time rollout MP4s are logged best-effort; use
`--no-checkpoint-rollout-videos` to skip simulator/rendering rollouts during
training.

## DP3 Policy Rollout Smoke

Roll out a trained reach checkpoint in a live ManiSkill environment and save
local MP4, Rerun, and JSON artifacts:

```bash
uv run python scripts/rollout_dp3_reach_policy.py \
  --dataset artifacts/reach-dataset-smoke/pg3d-reach-smoke.zarr \
  --checkpoint artifacts/reach-dataset-smoke/checkpoints/final_step_00000001.pt \
  --source dataset \
  --episodes 3 \
  --device cuda \
  --output-dir artifacts/reach-dataset-smoke/policy-rollouts-dataset
```

For fresh seeds from the same reach distribution:

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

The rollout script loads env configuration from the dataset metadata, uses
closed-loop action chunks, uses EMA checkpoints by default when present, keeps
rolling briefly after success for stability diagnostics, and writes
`summary.json`, `metrics.jsonl`, `episode_*.mp4`, and `episode_*.rrd`.

## World-Model Comparison

Compare checkpoint-predicted action chunks rolled out through the P07 world
model against the same chunks executed in ManiSkill:

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

The comparison uses a second ManiSkill ghost env to render robot-segmented point
clouds at imagined Panda qpos states. Rerun overlays the world-model branch and
the live simulator branch with distinct robot-point colors, writing one
`episode_XXX_comparison.rrd` file per compared episode.

## Docs

- `AGENTS.md`: durable agent instructions.
- `docs/project_proposal.html`: source-of-truth research proposal.
- `docs/status.md`: current state and next steps.
- `docs/milestones.md`: staged implementation plan.
- `docs/runbooks/commands.md`: canonical commands.
- `docs/runbooks/maniskill_setup.md`: ManiSkill setup notes.
- `docs/adr/`: durable design decisions.
- `docs/prompts/`: Codex milestone prompts.
