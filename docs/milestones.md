# Milestones

`docs/project_proposal.html` is the source-of-truth research plan. This file breaks that proposal
into repo execution milestones, so P03/P04/P05 split the proposal's simulator scaffold and custom
reach/data-adapter work into smaller Codex-sized tasks.

## M0 — Repo, docs, dependency mirrors, and workstation setup

Goal: create a Codex-friendly repo with reproducible commands and private dependency mirrors.

Deliverables:

- `AGENTS.md`, docs, ADRs, prompts, runbooks.
- `uv` project with Python 3.11 and PyTorch CUDA 12.9 path.
- Private DP3 mirror as a submodule.
- ManiSkill install notes and non-rendering smoke script.
- Smoke checks for package import, pytest, lint, and RTX 5090 PyTorch.

Done when:

- `make test`, `make lint`, and `make gpu-check` pass on the workstation, or failures are documented.
- Submodules can be cloned on another workstation.

## M1 — ManiSkill smoke and observation adapter

Goal: launch a built-in ManiSkill task, inspect observations, and convert ManiSkill outputs into pg3d observation objects.

Deliverables:

- `pg3d.envs.maniskill_adapter` package.
- `Observation` dataclass/model with point cloud, robot state, masks, and sim GT.
- Script to launch/reset/step `PickCube-v1` or `PushCube-v1` without rendering.
- Adapter path for state first, then RGB-D/point-cloud/segmentation summaries when rendering is configured.

Done when:

- A local smoke script resets a built-in ManiSkill task and prints observation/action spaces.
- The runbook includes exact uv commands, asset notes, and Vulkan/rendering caveats.

## M2 — Reach demonstrations and dataset writer

Goal: generate/replay nominal ManiSkill reach demos and write DP3-compatible data.

Deliverables:

- Built-in task smoke path and, if needed, a narrow `PG3DReach` custom task.
- Demo generation script for Reach-Narrow and Reach-Medium variants.
- Action chunk extraction for absolute joint targets and delta joint targets.
- Dataset writer producing schema-compatible Zarr or DP3 training files.
- Replay sanity script that approximates demo trajectories.

Done when:

- 5 smoke demos can be generated and replayed.
- Dataset shape/schema is documented.

## M3 — Nominal DP3 reach policy

Goal: train/evaluate a base DP3-style policy on nominal reach.

Deliverables:

- DP3 integration for the pg3d ManiSkill reach dataset.
- Training configs for Reach-Narrow and Reach-Medium.
- Evaluation script with W&B logging.
- First nominal success metrics and videos.

Done when:

- A smoke-scale training run starts and loads data correctly.
- A real training run produces a checkpoint and nominal reach evaluation.

## M3.5 — P11 balanced reach reliability

Goal: make base DP3 reach reliable before interpreting constrained reach.

Deliverables:

- Ordered target-marker tokens in the final K point-cloud slots, default `K=16`, with checkpointed
  marker settings and `K=0` compatibility for old checkpoints/ablations.
- DP3 encoder branch that preserves ordered marker semantics separately from the PointNet scene
  branch.
- `PG3DReach-BalancedWorkspace-v0` target distribution: 70% core-practical and 30%
  bounded-practical workspace samples, avoiding the old workspace extremes.
- Dataset diagnostic command for target distribution, raw goal visibility, marker correctness, and
  train/validation region balance.
- Region-stratified validation reporting before P10 is rerun.

Done when:

- A 20-episode overfit check reaches near-perfect dataset-seed closed-loop success.
- A held-out balanced validation run reaches at least 80% overall success, at least 90% core-region
  success, and median final distance below 2.5 cm, or the blocker is documented.

## M4 — Kinematic point-cloud world model v0

Goal: imagine future robot geometry and end-effector paths from candidate joint-action chunks.

Deliverables:

- FK/chunk integration module.
- Robot link/mesh point sampler.
- Point-cloud compositor that deletes current robot points and inserts future robot points.
- `ImaginedRollout` object with q trajectory, EEF path, future robot clouds, masks, and composited scene clouds.
- Visualizer for candidate rollouts and future point-cloud overlays.

Done when:

- Recorded demo action chunks produce plausible imagined EEF/robot paths.
- A visualization shows current cloud, imagined robot cloud, and EEF trajectory.

## M5 — Constraint programs v0

Goal: handwritten `avoid_region` and smoothness constraints over imagined rollouts.

Deliverables:

- Constraint base class.
- Simple region geometries: sphere, box; cylinder optional.
- `AvoidRegion(target="eef")`.
- Smoothness and policy-sample-consensus/deviation costs.
- JSON serialization for constraint instances.
- Synthetic tests for geometry costs.

Done when:

- Constraint costs are correct on synthetic trajectories.
- Constraint configs can be saved/reloaded and visualized.

## M6 — Rejection/reranking controller for constrained reach

Goal: sample K candidate chunks, imagine, score, select, and execute in receding horizon mode.

Deliverables:

- `BaseController`, `RejectionController`, `RerankingController`.
- Configurable K: 16, 32, 64.
- Scoring terms: goal distance, clearance, smoothness, sample-consensus deviation.
- Candidate diagnostics and W&B logging.
- Visual comparison of base vs rejected vs selected candidates.

Done when:

- On constrained ManiSkill reach, reranker runs end-to-end for at least a few episodes.

## M7 — First constrained reach MVP

Goal: produce the first scientific go/no-go result.

Deliverables:

- Evaluation overlay that samples avoid regions near likely nominal paths.
- Fixed base-success subset workflow with precomputed nominal-path avoid-region JSON for the
  balanced-checkpoint rerun.
- Methods: base DP3, DP3+rejection, DP3+world-model reranking, code-only waypoint baseline, simple ITPS-style rank baseline.
- Metrics: reach success, constraint satisfaction, combined success, min clearance, final target distance, smoothness, candidate feasibility fraction.
- Confidence intervals and qualitative videos.

Done when:

- Results over fixed seeds are logged and summarized.
- There is a clear internal demo video.

## M8+ — Post-MVP expansion

After go/no-go:

- pick-and-place nominal policy,
- grasp/object proxy attachment,
- no-overflight / avoid-projection,
- stronger code-only baseline,
- energy-guided denoising,
- LLM-generated constraints,
- CodeDiffuser-style baseline,
- RISE adapter only if needed.
