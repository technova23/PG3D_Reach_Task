# System architecture

## Package layout

```text
pg3d/
  envs/
    maniskill_adapter/      # simulator-specific wrappers and data collection
  policies/                 # policy interface and DP3 adapter
    dp3/                    # pg3d-native, simulation-free DP3 policy core
  world_model/              # kinematic point-cloud imagination
  constraints/              # executable geometric constraint objects
  composition/              # rejection, reranking, receding horizon, later guidance
  baselines/                # code-only and prior-style baselines
  eval/                     # metrics, experiment runners, confidence intervals
  viz/                      # point-cloud/trajectory/constraint visualization
  logging/                  # W&B and structured logs
  utils/
```

Keep simulator and policy dependencies lazy. Importing `pg3d` should not require ManiSkill,
SAPIEN, rendering/GPU simulator dependencies, or DP3. DP3 runtime imports should use
`pg3d.policies.dp3`; the private
`external/dp3` submodule is reference material during migration and should not be imported by
pg3d runtime code.

## DP3 policy slice

The pg3d-native DP3 slice keeps only the model and training primitives needed for the ManiSkill
reach MVP:

- point-cloud/state encoder,
- 1D diffusion action model,
- normalizer, mask generator, EMA/checkpoint utilities as needed,
- synthetic import/inference/loss smoke tests,
- ManiSkill reach Zarr sequence loader and smoke trainer/eval scripts.

It intentionally excludes upstream DP3 benchmark/simulation dependencies such as MuJoCo, Gym,
MetaWorld, DexArt, RRL, PyTorch3D, and task-generation scripts.

## Core objects

### Observation

Durable policy/evaluation boundary decisions for this schema are recorded in
`docs/adr/0008-observation-schema-and-masks.md`.

```python
@dataclass
class Observation:
    point_cloud: np.ndarray          # [N, 3], policy-visible
    point_features: dict[str, Any]   # optional aligned point features
    robot_mask: np.ndarray | None    # [N], optional but important for world model
    object_masks: dict[str, np.ndarray]
    robot_state: RobotState
    sim_gt: SimGroundTruth | None    # eval/debug only; not policy input
```

Current shape conventions:

- `point_cloud`: `float32 [N, 3]`, finite XYZ world points, DP3-visible.
- `point_features["rgb"]`: optional `uint8 [N, 3]`; DP3 color use is opt-in.
- `point_features["camera_index"]`: `int16 [N]`, camera provenance for debugging/artifacts.
- `point_features["segmentation"]` or `point_features["instance_id"]`: optional `int64 [N]`
  simulator segmentation ids; do not feed
  this to policies by default.
- `robot_mask`: optional `bool [N]` derived from simulator segmentation; required by the
  world model for robot-point removal.
- `object_masks`: optional named `bool [N]` masks for eval/debug, such as reach `target` and
  distractors. These are not policy inputs by default.
- `RobotState.as_agent_pos()`: joint positions only for the first DP3 reach adapter.
- `SimGroundTruth.target_position`: optional `float32 [3]` from ManiSkill task state/info
  state; eval/debug only.

P04 ManiSkill adapter conventions:

- Default conversion uses `obs_mode="state_dict"` so the adapter can read structured
  `agent.qpos`, `agent.qvel`, `extra.tcp_pose`, and `extra.goal_pos`.
- `obs_mode="pointcloud"` reads `pointcloud.xyzw[..., :3]` as world XYZ, `pointcloud.rgb` as
  optional RGB, and `pointcloud.segmentation` as raw simulator ids.
- Panda robot masks are derived from link `per_scene_id` values when the live ManiSkill env is
  passed as adapter context; raw observations alone are not enough to map ids to robot/object names.

P05 reach dataset conventions:

- Custom ManiSkill reach tasks register lazily as `PG3DReach-Narrow-v0`,
  `PG3DReach-Medium-v0`, `PG3DReach-Workspace-v0`, and
  `PG3DReach-BalancedWorkspace-v0`.
- `PG3DReach-Workspace-v0` samples goals uniformly in a broad Cartesian cuboid with center
  `(0.05, 0.0, 0.45)` and half extents `(0.35, 0.35, 0.30)` for pre-constraints reach policy
  diversity.
- `PG3DReach-BalancedWorkspace-v0` is the P11 reliability distribution: 70% core-practical goals
  in `x[-0.14, 0.24]`, `y[-0.20, 0.20]`, `z[0.28, 0.56]`, and 30% bounded-practical goals in
  `x[-0.26, 0.34]`, `y[-0.30, 0.30]`, `z[0.20, 0.68]`.
- DP3 policy arrays use `/data/point_cloud` as `float32 [T, 512, 3]`, `/data/state` as
  Panda qpos `float32 [T, 9]`, and `/data/action` as arm-only `float32 [T, 7]`.
- Replay/debug arrays keep `/data/sim_action`, `/data/robot_mask`, `/data/point_valid_mask`,
  `/data/target_position`, `/data/tcp_pose`, `/data/success`, and `/meta/episode_ends`.
- Point clouds are cropped to a workspace AABB before deterministic downsample/pad so far
  outliers from ManiSkill point-cloud rendering do not dominate the DP3 input.
- The dataset writer records one post-success hold-pose action chunk by default so terminal action
  labels teach the policy to remain at the reached goal instead of immediately ending the episode.

P06 DP3 reach training conventions:

- `ReachSequenceDataset` samples fixed-horizon windows from the P05 Zarr schema and exposes only
  policy-visible fields: `obs.point_cloud`, `obs.agent_pos`, and `action`.
- Simulator/eval arrays such as target position, TCP pose, success, robot masks, and simulator
  actions remain out of policy batches.
- P11 ordered goal tokens are injected into `obs.point_cloud` from `/data/target_position` at
  dataset-load and live policy-input time. By default, the final 16 XYZ points are overwritten with
  a deterministic target-centered marker at radius 1.5 cm; `target_position` itself remains
  eval/debug metadata, not a separate policy key.
- `DP3Encoder` keeps the usual PointNet scene branch for the first `N-K` points and adds a small
  ordered marker MLP over the final K points. Set `goal_marker_points=0` for old checkpoint
  compatibility and ablations.
- Normalizers are fit per final feature dimension for `point_cloud`, `agent_pos`, and `action`.
- `scripts/train_dp3_reach.py` is a behavior-cloning loop with `pad_after=n_action_steps-1`,
  validation splits, cosine LR warmup, gradient clipping, EMA checkpoints, and optional W&B
  diagnostics. It writes step-named checkpoints under a checkpoint directory, always writes a
  final checkpoint when checkpointing is enabled, and can log best-effort checkpoint-time rollout
  MP4s to W&B without making simulator/rendering failures fatal. Failures to initialize W&B do not
  block local smoke training unless explicitly requested.
- `scripts/rollout_dp3_reach_policy.py` is the first closed-loop policy rollout path. It loads
  env/crop/action metadata from the Zarr dataset, keeps a rolling `n_obs_steps` observation window,
  converts 7D DP3 arm labels back into full Panda simulator actions, and saves local MP4/Rerun/JSON
  artifacts for dataset-seed or fresh-seed rollouts. Rollout loading prefers EMA checkpoint weights
  when they are available and records post-success drift/action metrics.

### ActionChunk

```python
@dataclass
class ActionChunk:
    actions: np.ndarray              # [H, action_dim]
    action_mode: Literal["abs_joint", "delta_joint", "ee_pose"]
    dt: float
    metadata: dict[str, Any]
```

Default P0 action representation: absolute joint target chunks. Delta joint chunks are fallback. EE-pose chunks are deferred.

### ImaginedRollout

```python
@dataclass
class ImaginedRollout:
    q: np.ndarray                    # [H, dof]
    eef_path: np.ndarray             # [H, 3]
    robot_point_clouds: list[np.ndarray]
    scene_point_clouds: list[np.ndarray]
    robot_masks: list[np.ndarray]
    action_chunk: ActionChunk
    metadata: dict[str, Any]
```

### Constraint

```python
class Constraint(Protocol):
    def cost(self, rollout: ImaginedRollout, scene: SceneContext) -> dict[str, float]: ...
    def satisfied(self, rollout: ImaginedRollout, scene: SceneContext) -> bool: ...
```

Start with Python objects, but every constraint instance should be JSON-serializable for replay.

## P0 data flow

```text
ManiSkill observation
  -> ObservationAdapter
  -> DP3 policy samples K ActionChunks
  -> GeometricWorldModel imagines each chunk
  -> ConstraintProgram scores each imagined rollout
  -> RerankingController selects chunk
  -> ManiSkill executes first chunk / first action horizon
  -> repeat
```

## World model v0

The first world model is kinematic/geometric:

1. Read current robot joint state.
2. Integrate/interpret candidate joint-action chunk into future joint states.
3. Run FK for future end-effector path.
4. Sample robot link geometry at future joint states.
5. Remove current robot points from current point cloud using `robot_mask`.
6. Insert future robot points into the static scene cloud.
7. Return imagined future clouds and trajectories.

No learned dynamics. No contact dynamics. No object attachment until pick-and-place.

P07 implementation conventions:

- `pg3d.world_model` is NumPy-first so observations, masks, Zarr data, `.npz` artifacts, and
  visualization stay on the same simple array boundary. Torch/GPU batching is deferred until
  reranking scale or energy guidance requires it.
- `ActionChunk` supports `abs_joint` and `delta_joint`; `ee_pose` is explicitly deferred.
- Joint chunks use prefix semantics: a 7D Panda arm chunk updates the first seven entries of a 9D
  qpos, while trailing joints hold their previous values. Delta chunks are cumulative from the
  previously imagined joint state.
- `RobotGeometryProvider` is the only FK/mesh boundary. Providers return world-frame EEF positions
  and world-frame robot point clouds; ManiSkill/SAPIEN robot loading belongs behind that interface.
- `GeometricWorldModel` requires `Observation.robot_mask`, removes current robot points with it,
  and inserts provider-generated future robot points into the static scene cloud with aligned
  future robot masks.
- The first live Panda provider is a ManiSkill ghost-env adapter outside `pg3d.world_model`. It
  resets a second reach env with the same seed, sets Panda qpos for imagined states, and renders
  robot-segmented point clouds so world-model feedback matches the policy's observation domain.
- `scripts/compare_world_model_rollout.py` compares a checkpoint-driven world-model closed loop
  against a live ManiSkill rollout executing the same policy action chunks. The policy is queried
  from the world-model branch, while the simulator branch is ground-truth comparison.

## Constraint v0

P08 implements a Python-first, JSON-serializable constraint API:

- `SphereRegion` and `BoxRegion` expose signed distances where positive is outside, zero is on the
  boundary, and negative is inside.
- `SceneContext` carries optional eval/debug context such as target position, named regions, and
  metadata.
- `AvoidRegion(target="eef")` scores the imagined EEF path against a sphere or box keep-out region
  with optional clearance margin and weight.
- `SmoothnessCost(target="q"|"eef", order=1|2)` penalizes first- or second-order trajectory
  differences.
- Constraint configs round-trip through JSON-safe dictionaries and a small registry.
- `make_obstructing_avoid_region(start, goal)` creates a small keep-out sphere centered on the
  direct EEF path between start and goal for constrained reach eval setup.

Full robot-body collision, IK, policy sample-consensus costs, and multi-constraint reranking remain
M6+ work.

## Composition v0

P09 adds simulator-free candidate rejection and reranking controllers:

- `ControllerInput` carries the current `Observation`, `SceneContext`, and optional policy-specific
  input. Future DP3 adapters can pass rolling observation windows through `policy_input` without
  changing controller logic.
- `Policy.sample_action_chunks(policy_input, k, rng)` returns candidate `ActionChunk` objects.
  Optional `score_surrogate` values are lower-is-better soft costs.
- `RejectionController` keeps policy order and selects the first feasible candidate from K samples.
- `RerankingController` scores all sampled candidates and selects the best feasible candidate.
- The default K fallback schedule is 16, 32, then 64. If no candidate is feasible, controllers
  return the least-bad fallback with explicit diagnostics.
- Candidate diagnostics record constraint costs/satisfaction, final goal distance, trajectory
  smoothness, sample-consensus deviation, optional policy surrogate, total score, attempted K, and
  selection reason.

The real DP3/ManiSkill adapter remains separate: it should wrap `SimpleDP3.predict_action` into
`sample_action_chunks` and feed the controller rolling-window policy inputs.

## Constrained reach eval scaffold

P10 connects the current DP3, ManiSkill, world-model, constraint, and controller pieces into the
first MVP evaluation runner:

- `scripts/eval_constrained_reach.py` compares `base`, `rejection`, and `reranking` on fixed
  reach seeds with the same checkpoint and the same saved episode constraint JSON.
- The first constrained overlay uses one direct-path sphere between the initial TCP and goal. It
  is intentionally simple and repeatable; nominal-path obstacle placement is later work.
- Planning and execution horizons are separate. `planning_horizon_chunks=1` is the current
  receding-chunk case; larger values imagine multiple DP3 chunks through the P07 world model while
  executing only `execution_horizon_chunks` before re-observing in ManiSkill.
- Controller methods use a DP3 adapter that batches repeated rolling observation windows through
  `SimpleDP3.predict_action` to sample K stochastic action chunks.
- Metrics include reach success, constraint satisfaction from the executed simulator TCP path,
  combined success, final/min target distance, min clearance, smoothness, candidate feasibility
  fraction, fallback counts, Wilson intervals for boolean rates, and JSONL controller diagnostics.
- Code-only waypoint planning remains a strong reach baseline and is not implemented in this
  scaffold; reach-only results should not be over-claimed without that comparison.

## Logging

Every experiment should emit:

- config YAML/JSON,
- constraint instance JSON,
- per-episode metrics JSONL,
- W&B logs where available,
- qualitative videos/plots,
- git commit hashes for main repo and submodules.
