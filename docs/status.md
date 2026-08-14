# pg3d status

Last updated: 2026-08-14

## Current objective

Bootstrap a sim-only research codebase for programmatic geometric guidance of 3D diffusion policies. The first MVP is constrained reaching in ManiSkill/SAPIEN:

- base policy: DP3-style point-cloud diffusion policy,
- simulator: ManiSkill/SAPIEN, with built-in task smoke plus custom narrow/medium/workspace reach tasks,
- action representation: start with absolute joint target chunks; keep delta joint chunks as fallback,
- world model: kinematic robot-geometry point-cloud imagination from joint-action chunks,
- first constraint: `avoid_region` over the end-effector path,
- first composition operator: candidate rejection/reranking, not energy guidance.

## ITPS baseline branch

Work for the inference-time policy steering baseline is now isolated on the
`itps-stochastic-sampling-baseline` branch. The current branch adds:

- a faithful `SimpleDP3.stochastic_sample(...)` implementation of the released ITPS
  annealed-MCMC loop: four inner steps by default, guide ratio 60, clean-sample
  re-noising at the same diffusion level, and advancement only on the final inner step,
- an isolated deterministic DDIM scheduler and seeded initial/inner-loop forward noise,
  so ITPS cannot change the scheduler used by base DP3 or later evaluation methods,
- differentiable Panda FK matched to ManiSkill's `panda_v2.urdf`, including the TCP and
  live robot-base transforms; the live 11-configuration check passed with 0.61 micrometers
  maximum position error,
- EEF guidance over every configured sphere, box, or projected rectangle, with selectable
  smooth-barrier or exact-hinge energy and support for constraint margins and weights,
- ITPS as a method in `scripts/eval_constrained_reach.py`; the incomplete standalone sampler
  diagnostic was removed so there is one closed-loop evaluation entrypoint,
- per-replan JSONL logging of the raw diffusion action chunk and the executed
  simulator action in constrained-eval runs, so first-chunk versus later-chunk
  behavior can be inspected directly.

ITPS guides the full normalized diffusion horizon, differentiably unnormalizes actions to
physical joint radians for FK, and returns the same execution slice as normal DP3. It remains an
EEF-only baseline; differentiable whole-robot collision guidance and sketch input are out of scope.
The active DP3 architecture now has one canonical module implementation at
`pg3d/policies/dp3/modules.py`; two unreferenced legacy copies at the top of `pg3d/policies/`
were removed before writing line-referenced diffusion-policy documentation.
A code-anchored explanation of ordinary DP3 training and DDIM inference now lives at
`docs/explainations/diffusion_policy.md`; its ITPS extension is intentionally reserved for a
follow-up pass.
The first paper-comparison protocol now lives at
`docs/explainations/itps_reranking_experiment_plan.md`. It makes stable combined
task-and-constraint success the primary paper outcome, retains ordinary combined success for
interim pilots, defines
task/safety/motion/compute diagnostics on executed ManiSkill trajectories, and lays out paired
ITPS-versus-reranking experiments and statistical reporting. The plan now requires realistic,
camera-visible simulator obstacle actors before the primary comparison: their observed surface
points enter the ordinary policy point cloud, while their simulator-known pose and collision
primitives provide matched constraint geometry to reranking and ITPS. Point-cloud-only boundary
inference is deferred to a later perception ablation. The protocol also makes paired MP4 videos
and Rerun `.rrd` point-cloud timelines required outputs for a deterministic representative
episode subset; plots alone do not complete an experiment. The current evaluator covers the
pilot metrics and now logs stable-hold success, TCP/whole-robot violation
duration/integral/events, TCP/joint path length, physical-time acceleration/jerk,
maximum joint velocity, executed-action replan discontinuity, and robust continuous
summaries. E0 protocol validation now also assigns an order-independent shared
policy seed per episode, fingerprints serialized constraints, records source/checkpoint/run
identities, rejects mismatched method pairs, and reports CUDA-synchronized end-to-end
action-selection latency. A one-step live ManiSkill smoke passed for base, rejection,
reranking, and ITPS on the same dataset episode and constraint.
E1 now has an explicit no-constraint evaluator mode and a completed fixed 25-episode
nominal checkpoint gate. The initial 65k checkpoint reached 14/25 and stably held
12/25, missing the 15/25 gate by one. The existing 100k checkpoint was then evaluated
on the same nominal-validation episodes without using constrained outcomes: it
reached 15/25 (60%) and stably held 14/25 (56%), clearing the unchanged gate exactly.
It is now locked for E3, whose definitive constrained-test episodes must be disjoint
from these 25 checkpoint-selection episodes. A deterministic artifact-enabled
replication had identical endpoints and produced five validated MP4/`.rrd` pairs.
A separate three-episode `K=1`, zero-guidance regression produced bit-identical
base/rejection/reranking chunks and complete MP4/`.rrd` pairs for all four methods;
ITPS appropriately remained different because its zero-energy path still uses
four-step annealed-MCMC inference rather than ordinary single-pass DDIM. Those
historical artifacts were generated before the ITPS reverse scheduler changed from
DDPM to DDIM and must not be mixed with new DDIM-ITPS results.
E2 has its first end-to-end realistic-obstacle slice: an axis-aligned box is now a
collidable actor in the control environment rather than a render-only overlay. Live
camera segmentation tracks it through cropping and Rerun, and runtime validation
requires actor half-extents to match the serialized constraint. The first GPU smoke
retained 192 raw obstacle points but only 5 after cropping and in the final policy
tensor. A semantic-preserving minimum quota now coexists with the robot quota and
fixed tensor size; a repeated smoke retained 36 cropped/final points with a requested
minimum of 32.
Yaw-oriented boxes are now synchronized across the control actor, serialized
`BoxRegion`, NumPy metrics, differentiable ITPS energy, and Rerun visualization. A
35-degree live smoke retained 36 obstacle points in the final policy input and wrote
valid MP4/`.rrd` artifacts.
A named tall-carton family is also live with predeclared dimensions, distinct visual
material, and schema-complete obstacle ID/family/pose/collision fields. Its 20-degree
smoke observed 457 raw points and retained 46 in the final policy input.
The curved-object family now uses a finite vertical cylinder with matched collidable,
serialized, NumPy, Torch, and Rerun geometry. Visual inspection caught and corrected
ManiSkill's native X-axis cylinder orientation. The corrected smoke observed 218 raw
points and retained 36 in the final policy input.
The composite cabinet family now creates seven collidable actors and the same seven
serialized `BoxRegion` primitives for its sides, top/bottom, back, shelf, and open
door. Its smoke observed 768 raw points and retained 92 in the final policy input;
the decoded MP4 clearly shows the open structure.
E2 is complete: an eight-case collision probe covering intersecting and separated
placements for box, carton, cylinder, and cabinet produced zero disagreements between
raw PhysX contacts and serialized signed clearance. Every family has also passed the
camera-to-crop-to-policy point-count gate with a decoded MP4 and matching `.rrd`.
Artifact provenance is now enforced by `pg3d.artifact_manifest.v1`. Every selected
MP4/`.rrd` pair is linked to an exact metrics row, constraint file/fingerprint,
obstacle configuration, paired seeds, checkpoint, dataset, and git commit, with
non-empty size and SHA-256 validation. Video-only evaluation is rejected.
The experiment plan now makes paired MP4/Rerun outputs a completion gate for E3--E9:
aggregate graphs alone are insufficient, and every major experiment must retain a
deterministic qualitative suite linked to the exact numerical rows.
Rerun timelines now contain the exact final DP3 point tensor and separate semantic
robot/scene/obstacle/goal-marker entities, executed TCP path, target and collision
wireframes, per-step clearance/violation scalars, reranking candidate paths and
scores, and selected predicted paths for base/reranking/ITPS. A live three-method
recording was inspected with `rerun rrd print`.
Future policy-input recordings are now authored natively with Rerun 0.35 without
changing the policy/simulator dependency environment. The NumPy 1/Rerun 0.22 main
environment first writes a neutral `pg3d.policy_pointcloud_bundle.v1` `.npz` plus
JSON metadata; an ignored `.venv-rerun35` environment running NumPy 2 and
`rerun-sdk==0.35.0` converts and validates the `.rrd`. The neutral bundle retains the
exact fixed-size DP3 tensor at every step and its valid/robot/obstacle/scene/goal
masks, TCP, and target. Artifact manifests hash the bundle and metadata beside the
RRD and record the writer version. Do not upgrade NumPy or Rerun in the main
environment to solve viewer compatibility.
A four-family exact-input suite now lives under
`artifacts/e2-policy-input-rerun35`: rotated box, tall carton, vertical cylinder,
and open cabinet. Every neutral bundle has two `[1024, 3]` tensors (reset and one
executed step) from the same dataset episode and locked 100k checkpoint. Initial
obstacle counts in the exact policy tensors are respectively 32, 32, 32, and 64;
all four native 0.35 RRDs passed the isolated parser.
A companion moving simulator suite under `artifacts/e2-simulator-visuals-rerun35`
pairs each family with a 512×512 MP4 at the true 20 Hz control rate and a synchronized
exact-input bundle/native 0.35 RRD. The task horizon is now 150 steps, stopping after
a 16-step stable-success hold. On the shared episode, box/cylinder/cabinet stably
succeeded after 123/113/115 total steps; carton used all 150 steps without reaching.
All four videos decoded completely and all four RRDs passed the isolated parser.
Generated embodied obstacles are now tabletop-supported by default. For
dataset/direct-path placement, their collision height is resolved before simulator
construction to cover the selected path Z plus a 2 cm margin while keeping the
bottom at `z=0`; grounded placement is not shifted away afterward. Cabinets align
the nominal path through the back panel instead of the open interior. The regenerated
four-family suites span `z=0` to `z=0.4431` for the selected path point at
`z=0.4231`; signed clearance there is negative for every family.
Candidate-midpath pilots now have a clearance-safe replacement mode. Final grounded
obstacles retain their exact midpoint-derived XY, yaw, size, and grounded Z, but are
excluded before serialization or rollout when the stored initial whole-robot cloud
has less than 2 cm signed clearance. The committed 40-entry pool contains original
pilot unique-seed ranks 25--34 followed by unallocated ranks 85--114, leaving locked
test ranks 35--84 untouched. Valid attempts are remapped to contiguous outputs and
their source indices, clearances, and exclusions are recorded for auditability.
Candidate-path generation is seeded by source-pool index before each placement, so
the obstacle and exclusion decision do not depend on which evaluation methods ran
for earlier accepted episodes.
The accepted 75 cm clearance-safe population is now frozen as the tracked fixture
`configs/eval/e3_candidate_midpath_75cm_frozen_v1`. It fixes ten dataset episodes,
all paired seeds, exact grounded carton poses, and matched EEF/robot constraint
variants. Future candidate-midpath pilot, method, and horizon ablations must load
this fixture rather than regenerate or independently filter placements. It remains
tuning/ablation data and does not replace the locked 50-episode definitive E3 test.
Avoid-region and avoid-projection reranking now distinguish feasible candidates by
minimum obstacle clearance. Hard satisfaction remains the existing margin-violation
test, while the primary score adds a positive rational soft-clearance cost that falls
as clearance increases. The default decay scale is 5 cm and can be set to zero for
the historical hinge-only ablation. Rejection remains first-feasible, and ITPS keeps
its separate differentiable energy.
Future constrained evaluation now defaults to a 150-step task horizon instead of 80.
The stable-success hold remains separate, so successes stop after the configured
hold while failures receive all 150 task steps.
Embodied-obstacle evaluation now stops immediately after the first raw PhysX contact
or non-positive whole-robot signed clearance. The first-contact frame is kept,
collision metadata and a dedicated termination reason are logged, and constraint
plus combined success are forced false so post-impact motion cannot distort safety
or trajectory metrics.
Whole-robot safety is no longer flattened across time: enabling the metric retains one
robot cloud per executed timestep and reports primary violation duration, fraction,
integral, and event count. Executed joint targets now also report overall and
replan-boundary action discontinuity. A nine-step/two-replan live smoke populated all
new fields.
Compute work is now measured rather than inferred from candidate settings. Episode
rows contain actual denoiser calls and batch-item-equivalent evaluations, vectorized
ITPS FK calls and pose counts, reranking EEF/cloud query and render counts, per-replan
totals, and absolute/incremental peak PyTorch CUDA allocation during action selection.
The counters exclude Rerun and post-hoc whole-robot grading. A live `K=2` smoke
measured 100/100 denoiser calls/evaluations for base, 100/200 plus 16 EEF queries for
reranking, and 400/400 plus 396 FK calls over 6,336 poses for ITPS.
Paired statistical reporting is now serialized for every method pair. It records
explicit `method_a - method_b` differences, deterministic episode-paired bootstrap
intervals for binary and continuous endpoints, exact two-sided McNemar tests for
binary outcomes, excluded-pair counts, and path-length comparisons conditioned on
both methods achieving stable combined success. A live three-method smoke wrote all
three expected comparisons to `summary.json`.
The E3 data split is now frozen before constrained outcomes are examined. The first
25 unique seeds remain checkpoint-selection data, the next 10 are pilot/tuning data,
and the following 50 are the definitive test pool. The exact dataset indices and
simulator seeds are tracked in `configs/eval/e3_episode_split.json`; validation
confirmed the three partitions are disjoint and match the current dataset metadata.

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
adds the first handwritten constraint objects: sphere/box regions, `AvoidRegion(target="eef")`,
trajectory smoothness, an obstructing direct-path region helper, and JSON round-trip helpers. P09
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

1. Build nominal-path realistic-obstacle constraints on the locked pilot episodes
   without tuning on method outcomes.
2. Enable the evaluator to use those precomputed constraints with collidable,
   camera-visible actors.
3. Run the paired E3 pilot with required MP4/`.rrd` artifacts before scaling.

## Active risks

- DP3 upstream was designed around older Python/CUDA assumptions; pg3d now ports only the
  simulation-free model core and avoids upstream benchmark dependencies.
- ManiSkill v3 is a fast-moving stack; keep the adapter isolated and commands pinned in runbooks.
- Rendering and point-cloud observation modes may require Vulkan/driver setup beyond the
  non-rendering `obs_mode="state"` smoke.
- The main environment remains pinned to Rerun 0.22/NumPy 1; all new policy-input
  `.rrd` files require the isolated Rerun 0.35 exporter described in the runbook.
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
- The E3 realistic-obstacle pilot supersedes the sphere only for the paper comparison: its
  precomputed tall-carton `BoxRegion` instances retain nominal-path XY placement but are grounded
  on the world-`z=0` tabletop. The control actor, policy-visible camera points, and serialized
  collision geometry use the same pose.
- Semantic obstacle quotas are kept in the prefix of the fixed-size point tensor so ordered
  trailing goal tokens cannot erase the reserved camera-observed obstacle points.
- Qualitative artifact validation is fail-closed: hashes are recomputed, every MP4 frame is
  decoded, and every Rerun `.rrd` is parsed before a run manifest is accepted.
- The locked E3 nominal-base-success tuning pilot is complete on eight grounded-carton
  instances: stable combined success was base 5/8, rejection 5/8, reranking 8/8, and ITPS
  6/8. All 32 MP4/`.rrd` pairs validated. This is pilot evidence only; the 50 definitive
  test episodes remain untouched.
- Full-distribution constraint generation no longer drops base failures:
  `build_nominal_path_constraints.py --path-source dataset_demo` uses every selected
  episode's stored successful TCP path and resolves one shared grounded actor height.
  The original `policy_success` source remains the separately labelled mechanism subset.
- Definitive E3 settings are frozen in `configs/eval/e3_protocol.json`. The protocol
  launcher covers both populations, resolves actor geometry from the builder manifest,
  refuses accidental output overwrite, and requires a labeled MP4/native-Rerun pair
  for every method/episode in addition to aggregate plots.
- The first definitive v1 attempt was stopped and excluded after an initial-geometry
  audit found one tall box already intersecting the sampled robot by 4.26 cm. Protocol
  v2 deterministically searches only demonstration-path-intersecting placements with
  at least 2 cm initial robot clearance, then recomputes that gate before method execution.
- The partial v2 definitive run is also excluded: visual inspection showed that a
  shallow whole-robot penetration could continue when PhysX emitted no contact pair.
  Protocol v3 checks geometric contact online every step and moves MP4 identity text
  into an external header so it cannot obscure the camera scene.
- Constrained-evaluation MP4s now show method/episode/seed, obstacle family, outcome,
  clearance, and contact state above every camera frame. The identical payload is embedded in
  native Rerun recordings and verified against the metrics row by the artifact manifest.
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
