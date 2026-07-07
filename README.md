# pg3d

`pg3d` studies **programmatic geometric guidance for 3D diffusion robot policies**.
The current project is simulation-only and uses ManiSkill/SAPIEN as the primary
simulator stack.

The reach-first MVP is:

1. adapt a DP3-style point-cloud diffusion policy to ManiSkill reach data,
2. build a kinematic point-cloud world model from joint-action chunks,
3. score executable geometric constraints such as `avoid_region`,
4. use candidate rejection/reranking in receding-horizon mode,
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
  `PG3DReach-Medium-v0` tasks and a Zarr writer for DP3-compatible reach datasets.
- RLBench, PyRep, CoppeliaSim, and real-robot/xArm work are not active backends.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `pg3d/` | Core library: policies (`policies/dp3`), env adapters (`envs/maniskill_adapter`), world model, constraints, candidate composition, eval, viz, logging, utils. |
| `scripts/` | Runnable CLI entry points for dataset generation, training, rollout, evaluation, world-model comparison, and visualization (see below). |
| `dataset_generation/` | Standalone multimodal reach dataset writer + analysis utilities. |
| `new_training/` | Self-contained DP3 training variant with its own `dp3/` package copy. |
| `training_without_conditioning/` | DP3 training variant without trajectory-family conditioning. |
| `tests/` | Pytest suite covering datasets, policy, training, constraints, world model, and eval. |
| `docs/` | Research proposal, status, milestones, ADRs, runbooks, and Codex prompts. |
| `external/dp3` | Reference DP3 implementation used during migration (not a runtime import). |
| `artifacts/` | Generated datasets, checkpoints, rollouts, and Rerun/video artifacts. |

## Setup

Main workstation (GPU + ManiSkill):

```bash
uv sync --extra cu129 --extra maniskill --group dev --group notebooks
```

CPU-only docs/tests without the simulator:

```bash
uv sync --extra cpu --group dev
```

CPU-only with ManiSkill installed:

```bash
uv sync --extra cpu --extra maniskill --group dev
```

With optional Rerun visualization:

```bash
uv sync --extra cu129 --extra maniskill --extra viz --group dev --group notebooks
```

Fresh clone — initialize submodules:

```bash
git submodule update --init --recursive
```

`external/dp3` is reference material during migration. Runtime imports should use
`pg3d.policies.dp3`, not `external/dp3`.

## Checks

```bash
make smoke          # uv run python scripts/smoke_imports.py
make test           # uv run pytest
make lint           # uv run ruff check .
make gpu-check      # uv run python scripts/check_gpu.py
make maniskill-check # uv run python scripts/check_maniskill.py
```

The default ManiSkill check is non-rendering. Point-cloud/RGB-D/segmentation
checks should stay separate because they may require Vulkan and asset setup.

## Scripts Reference

All scripts live in `scripts/` and run via `uv run python scripts/<name>.py`.
Pass `--help` for the full flag set.

### Environment & sanity checks

| Script | What it does |
| --- | --- |
| `smoke_imports.py` | Imports every `pg3d` subpackage and core dependency (numpy, scipy, zarr, trimesh, wandb, …) to verify the environment is wired up. |
| `check_gpu.py` | Prints `torch` version, `torch.version.cuda`, and CUDA availability. |
| `check_maniskill.py` | Non-rendering ManiSkill smoke test: builds `PickCube-v1` with `obs_mode="state"` and prints the observation space. |
| `smoke_dp3_policy.py` | Runs one synthetic DP3 inference and one training step on CPU or CUDA. |
| `save_maniskill_observation.py` | Saves a single adapted ManiSkill observation artifact for inspection. |

### Dataset generation & processing

| Script | What it does |
| --- | --- |
| `write_maniskill_reach_dataset.py` | Primary dataset writer — generates a structured, multimodal pg3d ManiSkill reach dataset in Zarr (7D Panda labels for DP3 + full sim actions for replay, optional hold-pose chunk after success). |
| `replay_maniskill_reach_dataset.py` | Replays stored simulator actions from a reach Zarr and writes videos + Rerun timelines. |
| `trajectory_curved_datageneration.py` | Generates a reach dataset by smoothing existing multimodal waypoint paths into curved trajectories. |
| `convert_reach_zarr_abs_to_delta.py` | Copies a reach Zarr converting `/data/action` from `abs_joint` to `delta_joint` (`--plot-only` just plots delta magnitudes). |
| `diagnose_reach_dataset.py` | Reports target distribution and P11 goal-marker diagnostics for a reach Zarr. |
| `audit_goal_consistency.py` | Audits consistency between stored goal markers, `target_position`, and TCP pose. |
| `multimodality.py` | Diversity / multimodality audit for a pg3d reach Zarr dataset. |

### Training

| Script | What it does |
| --- | --- |
| `train_dp3_reach.py` | pg3d-native DP3 training loop on a reach Zarr. Writes periodic `step_*.pt` and a `final_step_*.pt` checkpoint, supports EMA, warmup, grad clipping, W&B logging, and best-effort checkpoint-time rollout videos. |

### Evaluation & inference

| Script | What it does |
| --- | --- |
| `eval_dp3_reach_dataset.py` | Dataset-only DP3 inference sanity check against a checkpoint (no simulator). |
| `eval_reach_checkpoint_unique_seeds.py` | Evaluates a plain DP3 checkpoint on first-occurrence unique dataset seeds or fresh random start/goal pairs. Plain policy rollout only — no constraints/rejection/reranking/video/Rerun. |
| `eval_constrained_reach.py` | Evaluates base DP3, candidate **rejection**, and **reranking** on constrained reach (avoid-region constraints). |
| `accuracy_chk_nitin.py` | Accuracy-check evaluation variant (reuses the nitin rollout helpers). |

### Policy rollout

| Script | What it does |
| --- | --- |
| `rollout_dp3_reach_policy.py` | Rolls out a trained DP3 reach checkpoint in live ManiSkill (`--source dataset` or `fresh`), using closed-loop action chunks and EMA weights by default; writes `summary.json`, `metrics.jsonl`, `episode_*.mp4`, and `episode_*.rrd`. |
| `rollout_dp3_reach_policy_nitin.py` | Variant of the rollout driver (shared helpers used by the accuracy/eval scripts). |

### World model

| Script | What it does |
| --- | --- |
| `compare_world_model_rollout.py` | Compares checkpoint-predicted action chunks rolled out through the P07 world model against the same chunks executed in ManiSkill; overlays both branches in Rerun (`episode_*_comparison.rrd`). |
| `visualize_world_model_rollout.py` | Writes a synthetic robot-only world-model rollout artifact. |
| `trajectory_tree_world_model.py` | Constrained-reach evaluation (base DP3 / rejection / reranking) over a world-model trajectory tree. |

### Constraints & candidate visualization

| Script | What it does |
| --- | --- |
| `build_nominal_path_constraints.py` | Builds small `avoid_region` constraints along successful nominal DP3 reach paths. |
| `plot_constrained_reach_summary.py` | Plots constrained-reach success rates with Wilson confidence intervals. |
| `visualize.py` | Visualizes natural stochastic DP3 checkpoint rollout candidates (no trajectory-family conditioning); selects the most diverse dataset episode. |
| `viz_constrained_candidates_rerun.py` | Rerun-based constrained-candidate visualizer (same natural-candidate family as `visualize.py`). |
| `visualize_constrained_candidates_multimodality.py` | Constrained-candidate visualizer focused on multimodality. |
| `visualize_constrained_candidates_for_varun's_dp3py` | Runs the constrained-candidate Rerun visualizer using checkpoints produced by the external `3D-Diffusion-Policy` `TrainDP3Workspace`. |
| `visualize_training_dataset.py` | Visualizes recorded training-Zarr TCP trajectories with a virtual avoid sphere. |

## Common Workflows

### Reach dataset smoke

```bash
uv run python scripts/write_maniskill_reach_dataset.py \
  --num-demos 3 --hold-steps 8 \
  --output artifacts/reach-dataset-smoke/pg3d-reach-smoke.zarr --overwrite

uv run python scripts/replay_maniskill_reach_dataset.py \
  --dataset artifacts/reach-dataset-smoke/pg3d-reach-smoke.zarr \
  --episodes 3 \
  --video-dir artifacts/reach-dataset-smoke/videos \
  --rerun-dir artifacts/reach-dataset-smoke/rerun

uv run rerun artifacts/reach-dataset-smoke/rerun/episode_000.rrd
```

Use the `step` timeline in the Rerun viewer and press play.

### Reach dataset pilot → full

```bash
# 100-episode pilot
uv run python scripts/write_maniskill_reach_dataset.py \
  --env-id PG3DReach-Narrow-v0 --num-demos 100 --max-attempts 150 \
  --hold-steps 8 --num-points 512 \
  --output artifacts/reach-datasets/pg3d-reach-narrow-100.zarr --overwrite

# 500-episode full set
uv run python scripts/write_maniskill_reach_dataset.py \
  --env-id PG3DReach-Narrow-v0 --num-demos 500 --max-attempts 700 \
  --hold-steps 8 --num-points 512 \
  --output artifacts/reach-datasets/pg3d-reach-narrow-500.zarr --overwrite
```

### DP3 training

```bash
# one-step CPU smoke
uv run python scripts/train_dp3_reach.py \
  --dataset artifacts/reach-dataset-smoke/pg3d-reach-smoke.zarr \
  --device cpu --max-steps 1 --batch-size 2 --num-workers 0 --val-ratio 0 \
  --checkpoint-dir artifacts/reach-dataset-smoke/checkpoints \
  --checkpoint-every 1 --no-checkpoint-rollout-videos

# moderate 5090 pilot
uv run python scripts/train_dp3_reach.py \
  --dataset artifacts/reach-datasets/pg3d-reach-narrow-100.zarr \
  --device cuda --max-steps 20000 --batch-size 64 --num-workers 4 \
  --val-ratio 0.1 --val-every 500 --lr 1e-4 --warmup-steps 500 \
  --grad-clip-norm 1.0 --use-ema \
  --wandb-mode online --wandb-project pg3d --wandb-name dp3-reach-narrow-100-stable \
  --checkpoint-dir artifacts/reach-datasets/dp3-reach-narrow-100-stable-checkpoints \
  --checkpoint-every 5000 --checkpoint-rollout-count 5
```

### Inference & rollout

```bash
# dataset-only inference sanity check
uv run python scripts/eval_dp3_reach_dataset.py \
  --dataset artifacts/reach-dataset-smoke/pg3d-reach-smoke.zarr \
  --checkpoint artifacts/reach-dataset-smoke/checkpoints/final_step_00000001.pt \
  --device cpu --max-batches 1

# live ManiSkill rollout (dataset or fresh seeds)
uv run python scripts/rollout_dp3_reach_policy.py \
  --dataset artifacts/reach-dataset-smoke/pg3d-reach-smoke.zarr \
  --checkpoint artifacts/reach-dataset-smoke/checkpoints/final_step_00000001.pt \
  --source fresh --episodes 3 --seed-start 10000 --device cuda \
  --output-dir artifacts/reach-dataset-smoke/policy-rollouts-fresh
```

### World-model comparison

```bash
uv run python scripts/compare_world_model_rollout.py \
  --dataset artifacts/reach-datasets/pg3d-reach-narrow-100.zarr \
  --checkpoint-dir artifacts/reach-datasets/dp3-reach-narrow-100-stable-checkpoints \
  --source dataset --episodes 3 --device cuda \
  --output-dir artifacts/reach-datasets/world-model-vs-sim \
  --rerun --video --allow-failure

uv run rerun artifacts/reach-datasets/world-model-vs-sim/episode_000_comparison.rrd
```

Rerun overlays the world-model branch and the live-simulator branch with distinct
robot-point colors, one `episode_XXX_comparison.rrd` per episode.

### Real world lab data creation

```bash
cd Real-data-zarr-setup

# 1. Create the dataset (syncs rosbags, FK filtering, injects goal markers)
python3 dataset_golden_fire.py

# 2. Verify the output dataset and visualize the trajectory in Rerun
python3 verify-dataset.py
```

## Docs

- `AGENTS.md`: durable agent instructions.
- `docs/project_proposal.html`: source-of-truth research proposal.
- `docs/status.md`: current state and next steps.
- `docs/milestones.md`: staged implementation plan.
- `docs/runbooks/`: canonical commands and ManiSkill setup notes.
- `docs/adr/`: durable design decisions.
- `docs/prompts/`: Codex milestone prompts.
