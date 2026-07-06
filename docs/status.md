# pg3d status

Last updated: 2026-07-06

## Current objective

Bootstrap a sim-only research codebase for programmatic geometric guidance of 3D diffusion policies. The first MVP is constrained reaching in ManiSkill/SAPIEN:

- base policy: DP3-style point-cloud diffusion policy,
- simulator: ManiSkill/SAPIEN, with built-in task smoke plus custom narrow/medium/workspace reach tasks,
- action representation: start with absolute joint target chunks; keep delta joint chunks as fallback,
- world model: kinematic robot-geometry point-cloud imagination from joint-action chunks,
- first constraint: `avoid_region` over the end-effector path,
- first composition operator: candidate rejection/reranking, not energy guidance.

## Current phase

Simulator migration to ManiSkill/SAPIEN is complete in the active code path. The repo has a
pg3d-native simulation-free DP3 slice under `pg3d/policies/dp3` with synthetic import, inference,
and training-step smoke tests. ManiSkill is tracked as a pinned optional uv extra, while base `pg3d`
imports stay simulator-free. A small non-rendering ManiSkill smoke script validates a built-in
`PickCube-v1` environment. The first observation adapter now targets Franka/Panda `PickCube-v1`
state and point-cloud observations, including segmentation-derived robot/object masks when a live
ManiSkill env context is available. P05 adds custom `PG3DReach-Narrow-v0` /
`PG3DReach-Medium-v0` / `PG3DReach-Workspace-v0` tasks plus a smoke-scale Zarr dataset writer for
DP3-compatible reach data.
P06 adds a simulation-free reach Zarr sequence loader for pg3d-native DP3, plus CPU smoke
training/eval scripts with optional W&B metrics and histogram logging. P06 now also has a
closed-loop policy rollout script that loads a trained reach checkpoint, runs it in live
`PG3DReach-*` ManiSkill environments, and writes MP4 videos, Rerun timelines, and JSON metrics for
dataset-seed or fresh-seed rollouts. The current detour adds post-success hold-pose data to the
reach dataset writer and upgrades the trainer with validation, cosine warmup, gradient clipping,
EMA checkpoint state, directory-based periodic checkpoints, best-effort W&B checkpoint rollout
videos, and richer diagnostics for stable non-trivial training runs.
P07 now adds a pure NumPy robot-only kinematic point-cloud world model. It interprets absolute and
delta joint chunks, removes current robot points with `Observation.robot_mask`, inserts future
robot geometry from a simulator-free provider interface, and writes synthetic rollout artifacts for
visual inspection. The first comparison path now adds a lazy ManiSkill ghost-env Panda geometry
provider plus a checkpoint rollout comparison script that feeds imagined point clouds back into
the policy and writes per-episode Rerun overlays for world-model versus simulator rollouts. P08 now
adds the first handwritten constraint objects: sphere/box/cylinder regions,
`AvoidRegion(target="eef")`, Cartesian waypoint pose constraints, cylindrical passage
constraints, trajectory smoothness, an obstructing direct-path region helper, and JSON round-trip
helpers. P09
adds pure rejection and reranking controllers with K fallback, hard-then-score feasibility,
candidate diagnostics, and a policy-input seam for future DP3 rolling-window adapters. P10 now
adds the first constrained-reach evaluation scaffold connecting DP3, ManiSkill, the ghost-env world
model, direct-path avoid-region overlays, and base/rejection/reranking methods with fixed seeds,
JSONL metrics, per-episode constraint JSON, optional MP4/Rerun artifacts, W&B logging, and Wilson
interval summaries. The eval runner now defaults to a faster q/EEF scoring mode that avoids
per-timestep ghost point-cloud renders during candidate scoring, while preserving an exact
full-render mode for small validation spot checks. It also supports timing JSONL, periodic local
plots, deterministic validation-subset video/Rerun artifacts, and incremental W&B progress
logging. Training checkpoint rollout videos can now use a held-out validation Zarr instead of
mixed train/fresh seeds. Constrained-eval visualization artifacts now also show the sampled
avoid-region geometry: Rerun exports log persistent keep-out wireframes, and MP4s use a
best-effort separate render-only ManiSkill env so visual overlays do not alter policy observations
or simulator control. The eval runner can also consume precomputed per-episode constraints and a
fixed dataset episode-index file, so nominal-path avoid regions can be built once from base
rollouts and reused across base/rejection/reranking comparisons.
Cartesian pose constraint rollouts now keep the target pose visible in Rerun exports as a
static position marker plus orientation triad, so you can inspect the intended EEF pose next to
the executed rollout. Executed-episode Cartesian pose metrics now evaluate the full recorded
`EpisodePath` instead of dropping to position-only TCP arrays, so pose-only constraints no longer
default to satisfied when the EEF never reaches the requested pose. Constrained eval now also
writes per-step `step_traces.jsonl` rows with executed TCP pose, policy/simulator action,
task-goal distance, and Cartesian pose errors, plus richer `decisions.jsonl` selected-branch
records with chosen action chunks, imagined EEF/q rollouts, and compact candidate summaries for
world-model reproducibility checks. Cartesian pose constraints can now also be generated directly
from the selected dataset episode's saved `/data/tcp_pose` demonstration path at a configured
arc-length fraction, so waypoint-pose experiments can use known demonstrated EEF poses instead of
opaque precomputed targets.
P11 starts the base-reach reliability pass. DP3 reach policy inputs now reserve an ordered tail
slice of the XYZ point cloud for deterministic goal tokens by default
(`goal_marker_points=16`, `goal_marker_radius=0.015`), while keeping public policy keys limited to
`obs.point_cloud`, `obs.agent_pos`, and `action`. The encoder preserves those ordered tokens through
a small marker MLP branch instead of relying on PointNet's permutation-invariant scene branch.
`PG3DReach-BalancedWorkspace-v0` adds a 70/30 mixed practical/workspace target distribution that
avoids the previous workspace extremes but still tests spatial coverage. The current constrained
reach candidate is the 20k-step balanced checkpoint at
`artifacts/reach-datasets/dp3-reach-balanced-1000-checkpoints/step_00020000.pt`, but its first
25-episode held-out gate selected only 7 base-success episodes, below the 15-episode minimum for
interpreting constrained reranking.
The reach dataset writer is headless by default again and now has an explicit `--viewer` mode that
pumps ManiSkill human-render frames during collection, with optional step delay and post-run hold
time for live visual inspection.
The writer now defaults to the balanced workspace task, adds a red start marker beside the existing
green target marker, and samples planner-validated randomized TCP starts so three-variant setups
scatter across the table workspace instead of repeating a fixed start.
The writer now suppresses expected ManiSkill screw-planner retry chatter by default and requires a
complete requested trajectory-family set per seed/start group before writing those variants, so
large multimodal datasets no longer silently keep seeds that missed a family such as
`downward_arc`.
DP3 reach dataset loading now matches the generated Zarr schema with 1024-point clouds, 9D state,
7D arm actions, and `target_position`/`goal_pos` goal aliases; normalizer fitting uses a bounded
deterministic timestep subset by default so large Zarr datasets can begin training without reading
the full point-cloud tensor into memory.

## Immediate next steps

1. Diagnose why the 20k balanced checkpoint reached only 7/25 held-out balanced validation
   episodes in the first gate.
2. Inspect successful and failed base rollouts from the held-out balanced set, then decide whether
   to continue training, adjust inference settings, or revisit the training distribution.
3. Rerun the nominal-path constraint builder only after the 25-episode base gate reaches at least
   15 successes; target the original 25 selected successes for the starter constrained eval.
4. Run P10 base/rejection/reranking on the fixed base-success subset only after that gate passes.

## Active risks

- DP3 upstream was designed around older Python/CUDA assumptions; pg3d now ports only the
  simulation-free model core and avoids upstream benchmark dependencies.
- ManiSkill v3 is a fast-moving stack; keep the adapter isolated and commands pinned in runbooks.
- Rendering and point-cloud observation modes may require Vulkan/driver setup beyond the
  non-rendering `obs_mode="state"` smoke.
- Optional Rerun visualization is pinned to `rerun-sdk==0.22.1` while pg3d remains on NumPy 1.x.
- Reach is useful for mechanism validation, but code-only planners may be strong; avoid over-claiming from reach-only results.
- The kinematic point-cloud world model is the novel project pivot and should be validated visually early.
- New clones and fresh virtualenvs must sync the `maniskill` optional extra before running
  ManiSkill smoke checks.

## Decisions already made

- Project/repo/package name: `pg3d` for now.
- Sim-only for this phase; real robot hardware code is out of scope.
- ManiSkill/SAPIEN is the primary simulator.
- RLBench/PyRep/CoppeliaSim are deprecated and removed from active dependencies/backends.
- ManiSkill should be installed as an optional uv dependency, not carried as a submodule.
- DP3 is the only base policy for P0; RISE is deferred.
- DP3 runtime code should live in `pg3d/policies/dp3`; `external/dp3` is a temporary reference
  submodule during migration.
- Start with reach, then move to pick-and-place, then place-into-container.
- Start with handwritten constraints; LLM-generated constraints are later.
- Start with reranking/rejection; energy guidance is later.
- Use W&B from day one, but keep offline/debug modes available.
- ManiSkill observations use typed pg3d dataclasses and keep policy-visible point clouds/agent state
  separate from simulator ground truth and eval/debug masks.
- Robot masks are first-class observation metadata for the world model.
- Franka/Panda is the first robot target for built-in ManiSkill smoke and observation adaptation.
- Reach dataset DP3 action labels are 7D Panda arm joint targets/deltas; full simulator actions are
  stored separately for replay.
- Reach dataset replay can now save MP4 videos and per-episode Rerun timeline artifacts.
- DP3 reach training consumes only point cloud, agent position, and action arrays; simulator
  ground-truth/debug arrays stay out of policy batches. For P11 reach reliability, the loader and
  live policy-input adapters overwrite the final K point-cloud slots with deterministic ordered
  target markers derived from `/data/target_position`; no separate scalar target key is exposed to
  the policy by default.
- Standalone DP3 policy rollout visualization is local-first: MP4, Rerun `.rrd`, `metrics.jsonl`,
  and `summary.json`. The trainer can also upload a small configurable set of checkpoint-time MP4
  rollout videos to W&B when W&B and ManiSkill rendering are available.
- Reach datasets should include one DP3 action chunk of post-success hold-pose data by default so
  terminal policy chunks learn to stay at the goal.
- Stable DP3 reach checkpoints should prefer EMA weights for eval/rollout when present.
- DP3 reach training checkpoints are now directory-based: periodic files use `step_XXXXXXXX.pt`
  and final files use `final_step_XXXXXXXX.pt`.
- World model v0 is NumPy-first and simulator-free; ManiSkill/SAPIEN robot FK and mesh sampling
  must stay behind `RobotGeometryProvider`.
- The first real Panda geometry provider uses a second ManiSkill ghost env for rendered
  robot-segmented point clouds. Pure URDF/FK mesh sampling remains a later optimization.
- Pre-constraints reach policy training should use `PG3DReach-Workspace-v0`, which samples goals
  uniformly over `x[-0.30, 0.40]`, `y[-0.35, 0.35]`, and `z[0.15, 0.75]`.
- P11 nominal reach training should use `PG3DReach-BalancedWorkspace-v0` for the next reliability
  pass: 70% core-practical goals in `x[-0.14, 0.24]`, `y[-0.20, 0.20]`, `z[0.28, 0.56]`, plus
  30% bounded-practical goals in `x[-0.26, 0.34]`, `y[-0.30, 0.30]`, `z[0.20, 0.68]`.
- Constraint v0 is Python-object first with JSON config round-trips; full robot collision and IK are
  deferred.
- Composition v0 is policy-generic and simulator-free. The real DP3 adapter should wrap
  `SimpleDP3.predict_action` into `sample_action_chunks` instead of importing DP3 inside
  `pg3d.composition`.
- Constrained reach eval uses direct-path spherical avoid regions as the first repeatable overlay.
  Planning horizon and execution horizon are separate chunk counts; the default is one planned
  chunk and one executed chunk before re-observation.
- The P11 balanced-checkpoint constrained rerun uses precomputed nominal-path spherical avoid
  regions on a held-out base-success subset. This isolates steering behavior from base reach
  failure, and results must be labeled as base-success-subset constrained evals.
- Eval geometry mode defaults to `fast`; use `--geometry-mode exact` for one-episode reference
  comparisons when validating speedups.
- Constrained reach validation should use a held-out solved validation Zarr with `--source dataset`
  rather than arbitrary `--source fresh` seeds when comparing methods.
- Checkpoint rollout videos and eval MP4/Rerun artifacts should use a deterministic random
  5-episode subset of the held-out validation Zarr for comparable in-distribution visual checks.
- Avoid-region visualization is eval-only and visual-only: overlays are allowed in constrained
  eval MP4/Rerun artifacts, but they must not enter policy-visible point clouds, segmentation
  masks, or the control env.
- The first 50-episode workspace validation comparison showed 2% reach success for all three
  methods and 0% combined success, so the current bottleneck is base reach reliability rather than
  constraint selection.
- Code-only waypoint planning is a strong reach baseline and remains unimplemented in P10; any
  first constrained-reach results should document that limitation.
- Franka gripper / custom URDF / Robotiq work is deferred to a later non-critical manipulation
  milestone; it should not block the current base reach reliability pass.

## Latest work log

See `docs/worklog/`.

- Simulator choice is recorded in `docs/adr/0002-maniskill-primary-simulator.md`.
- Observation schema and mask policy are recorded in
  `docs/adr/0008-observation-schema-and-masks.md`.
- Current canonical setup command:
  `uv sync --extra cu129 --extra maniskill --group dev --group notebooks`.
- Optional visualization setup command:
  `uv sync --extra cu129 --extra maniskill --extra viz --group dev --group notebooks`.
- Current validation: `uv lock --check`, `make smoke`, `make test`, `make lint`,
  `make gpu-check`, `make maniskill-check`, and the state/point-cloud/MP4/Rerun observation
  artifact scripts pass on the RTX 5090 workstation environment. P05 reach dataset smoke/replay
  visualization plus P06 DP3 reach training/eval/rollout smoke validation are recorded in the
  worklog. The hold-tail dataset/training stability pass has also been validated with pure tests, a
  5-demo hold dataset smoke, short CPU training/eval, offline W&B outside the sandbox, and one
  dataset/fresh live rollout smoke. The intermediate-checkpoint pass adds pure tests for
  step-named checkpoint paths, periodic/final checkpoint writing, mixed rollout-video seed
  selection, lazy training imports, and non-fatal checkpoint-rollout failures. It also validates
  a two-step checkpoint-directory smoke and an outside-sandbox offline W&B checkpoint-video smoke.
  A focused cleanup pass then consolidated duplicate JSON/array/device/checkpoint helpers without
  changing scientific behavior, refreshed the custom reach setup notes, improved `make clean` for
  nested `__pycache__` directories, and passed ruff, 48 pytest tests, smoke imports, and
  `git diff --check`. A post-P07 cleanup audit found no active dead simulator code or stale
  checkpoint commands; the only current cleanup was doc alignment plus clarifying comments around
  intentional best-effort fallbacks. P07 world-model v0 adds pure synthetic tests and a simulator-free
  visualization artifact script. The next integration adds a lazy ManiSkill ghost-env geometry
  provider plus `scripts/compare_world_model_rollout.py`; workstation execution is still needed
  for full Rerun overlay validation because the sandbox cannot access a supported SAPIEN render
  device. `PG3DReach-Workspace-v0` is now available for the pre-constraints diverse reach policy,
  and a 5-demo workspace smoke plus MP4/Rerun replay passed outside the sandbox. P08 constraint v0
  adds pure tests for sphere/box signed distances, EEF avoid-region costs, smoothness costs,
  obstructing direct-path region generation, serialization, and lazy imports. P09 composition v0
  adds pure tests for rejection/reranking selection, fallback K schedules, least-bad fallback,
  diagnostics, future DP3 policy-input plumbing, and lazy imports. P10 constrained reach eval adds
  pure tests for overlay generation, Wilson intervals, metric aggregation, clearance, horizon
  validation, multi-chunk rollout concatenation, timing aggregation, periodic artifact selection,
  batched DP3 sampling, fast-mode render counts, and lazy eval imports.
  Avoid-region artifact visualization adds pure wireframe tests plus a constrained-eval MP4 overlay
  path that falls back to plain video if the separate render-only ManiSkill env cannot create
  visual actors. P11 ordered goal tokens and balanced workspace sampling add pure tests for marker
  insertion, encoder branching, rollout/eval input transforms, reach metadata, and checkpoint-aware
  training defaults; the dataset generation and retraining commands have been verified on the
  user's workstation. The 20k balanced constrained-reach starter adds a nominal-path constraint
  builder, precomputed constraint loading for constrained eval, and pure tests for the new fixed
  subset protocol. The first 25-episode held-out gate for the 20k checkpoint selected only 7
  base-success episodes, so the main constrained eval was intentionally not run.
  A small extraction helper now supports `--episode-index` for exporting a single reach trajectory
  from a saved Zarr to `.npz` or a one-episode Zarr.
