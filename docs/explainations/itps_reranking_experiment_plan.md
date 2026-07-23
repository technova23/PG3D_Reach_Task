# ITPS versus reranking: experiment and metric plan

Status: metrics protocol v1; experiment list is the initial paper plan
Scope: Franka/Panda constrained reach using the same trained DP3 checkpoint

## 1. Comparison question

The main comparison is whether inference-time policy steering (ITPS) or geometric
candidate reranking better preserves the nominal reaching behavior while satisfying a
new keep-out constraint.

Both methods must be evaluated from the same initial simulator states, with the same
goal and constraint instances, base checkpoint, observation history, action
representation, control frequency, execution horizon, episode limit, and success
tolerance. Metrics must be computed from the **executed ManiSkill trajectory**, not
from an ITPS energy or a reranker's imagined trajectory.

The primary result is not constraint satisfaction alone: a robot can satisfy a
keep-out constraint by never moving. The paper's primary endpoint is stable combined
task-and-constraint success. Ordinary combined success remains the interim endpoint
for pilots until stable holding is added to the evaluator.

## 2. Finalized metrics

### 2.1 Primary metrics

These metrics belong in the main comparison table.

| Metric | Episode-level definition | Aggregate/reporting |
| --- | --- | --- |
| **Goal reached** | `True` if the executed TCP enters the task success tolerance at least once. Use ManiSkill's `info["success"]` when it is exactly equivalent to the configured TCP-to-goal threshold; otherwise compute `min_t ||x_t - g|| <= epsilon_goal` directly and record the threshold. | Rate with 95% confidence interval. |
| **Constraint satisfied** | `True` if the minimum signed clearance over the complete executed trajectory is at least `-epsilon_constraint`. The scored geometry target (TCP, gripper proxy, or whole robot) must be fixed for every compared method. | Rate with 95% confidence interval. |
| **Combined success** | `goal_reached AND constraint_satisfied` in the same episode. | Interim pilot endpoint: rate with 95% confidence interval. |
| **Stable combined success** | Constraint is satisfied and, after first reaching the goal, TCP-to-goal distance remains within `epsilon_goal` for a fixed hold window of `H_hold` simulator steps. An episode that terminates before completing the required hold is a failure unless termination itself guarantees the hold condition. | **Primary paper endpoint**: rate with 95% confidence interval. |

`goal_reached` is retained because it answers the simple question "did the robot ever
reach the goal?" `stable_combined_success` prevents a transient goal contact followed
by drift from being counted as a fully successful deployment.

### 2.2 Task-quality metrics

| Metric | Definition | Direction |
| --- | --- | --- |
| Final target distance | `||x_T - g||` in metres. | Lower |
| Minimum target distance | `min_t ||x_t - g||` in metres. | Lower |
| Time to first goal | Simulator steps and seconds until the first successful step; report only for goal-reaching episodes and also treat failures as censored in plots. | Lower |
| TCP path length | `sum_t ||x_t - x_(t-1)||` in metres. | Lower, conditional on success |
| Joint path length | `sum_t ||q_t - q_(t-1)||_2` in radians. | Lower, conditional on success |
| Episode length | Executed simulator steps until success-hold completion, termination, or timeout. | Lower, conditional on success |
| Post-success drift | Maximum TCP-to-goal distance during the fixed hold window after first success. | Lower |

Path length and episode length must not be interpreted without success: a stationary
failure has excellent-looking efficiency.

### 2.3 Constraint-quality metrics

Let `c_t` be signed clearance after the configured margin, where positive is safe and
negative is penetration.

| Metric | Definition | Direction |
| --- | --- | --- |
| Minimum clearance | `min_t c_t` in metres. | Higher |
| Maximum violation depth | `max(0, -min_t c_t)` in metres. | Lower |
| Violation duration | Number and fraction of executed steps for which `c_t < -epsilon_constraint`. | Lower |
| Integrated violation | `sum_t max(0, -c_t) * dt` in metre-seconds. | Lower |
| Violation-event count | Number of safe-to-unsafe transitions, not the number of violating samples. | Lower |

Minimum clearance distinguishes a near miss from deep penetration, while duration and
integrated violation distinguish a momentary boundary crossing from sustained unsafe
motion. The primary constraint metric should eventually use whole-robot geometry.
TCP-only clearance remains a required diagnostic because ITPS currently provides
EEF-only guidance.

### 2.4 Motion-quality metrics

| Metric | Definition | Direction |
| --- | --- | --- |
| TCP acceleration/jerk | Mean squared second/third finite difference of TCP position, divided by the appropriate power of `dt`. | Lower |
| Joint acceleration/jerk | Mean squared second/third finite difference of arm joint position, divided by the appropriate power of `dt`. | Lower |
| Maximum joint velocity | Maximum absolute finite-difference joint velocity. | Lower / within limits |
| Action discontinuity | Mean and maximum norm between consecutive executed joint targets, including replan boundaries. | Lower |

These must use physical units and a shared control timestep. The current unscaled
second-difference `smoothness` value may remain for backward compatibility, but it
should not be the only paper metric.

### 2.5 Compute metrics

| Metric | Definition |
| --- | --- |
| Action-selection latency | Wall-clock time from receiving an observation window to returning the selected action chunk; report median, p90, and p95 per replan. Synchronize CUDA around timing. |
| Episode planning time | Sum of action-selection time across replans, excluding rendering and artifact writing. |
| Policy-network evaluations | Total denoiser forward passes per replan and per episode. |
| FK / geometry evaluations | Number of differentiable FK calls for ITPS and imagined rollout/geometry evaluations for reranking. |
| Peak GPU memory | Maximum allocated GPU memory during action selection. |
| Throughput | Executed control steps per wall-clock second without video, Rerun, W&B upload, or plot generation. |

Latency should be reported both at the methods' standard settings and under a
compute-matched comparison. Candidate count alone is not a fair compute budget because
ITPS and reranking use the denoiser differently.

### 2.6 Method-specific diagnostics

These explain behavior but must not be used as direct cross-method wins.

- Reranking/rejection: candidate feasibility fraction, candidates attempted, fallback
  count, selected rank, selected predicted clearance, and prediction-versus-execution
  clearance error.
- ITPS: guide ratio, MCMC inner steps, energy before/after guidance, gradient norm,
  denoising steps, and NaN/divergence count.
- Both: raw diffusion chunk, executed simulator action, replans, timeouts, simulator
  errors, and deterministic seed metadata.

Candidate feasibility fraction has no natural ITPS equivalent, and ITPS energy has no
natural reranking equivalent. They should appear in diagnostic plots, not the headline
table.

## 3. Required qualitative artifacts

Graphs and aggregate tables are not sufficient evidence for these experiments. Every
experiment run must write inspectable trajectory artifacts alongside its numerical
results.

### 3.1 MP4 videos

Save simulator-rendered MP4s for a deterministic, predeclared episode subset shared by
all methods. The subset must include, when available:

- at least one stable combined success,
- one goal-reaching constraint violation,
- one constraint-satisfying task failure,
- one complete failure or fallback, and
- examples from every realistic obstacle family used in that experiment.

Each video should show the same camera view and playback rate across methods, with the
goal, realistic obstacle actor, method name, episode/seed, and success/constraint state
visible. Constraint overlays may be included for clarity, but the actual camera-visible
actor must also be present; an overlay must never substitute for the obstacle observed
by the policy.

### 3.2 Rerun point-cloud timelines

Write a corresponding `.rrd` file for every saved MP4 episode. At each simulator step
or replan, the Rerun timeline should contain:

- the exact cropped point cloud supplied to DP3, with scene, robot, obstacle, and goal
  marker points distinguishable;
- the uncropped camera point cloud when artifact size permits;
- executed robot/TCP trajectory and target;
- obstacle pose plus simulator-known collision primitives or mesh;
- signed-clearance values and collision/violation state;
- for reranking, all sampled candidate paths, feasibility/cost, and the selected path;
- for ITPS, the final guided trajectory and available before/after guidance path or
  energy diagnostics; and
- in world-model studies, imagined robot/scene point clouds alongside the subsequent
  executed simulator point cloud.

The `.rrd` and MP4 must use the same method, episode, simulator seed, policy seed,
obstacle ID, and constraint ID as the corresponding metrics row.

### 3.3 Artifact selection and organization

- Use one deterministic artifact-selection manifest for paired method comparisons so
  base, rejection, reranking, and ITPS visualize the exact same episodes.
- Generate artifacts for a compact representative subset during large sweeps, not
  necessarily every sweep episode. Definitive E3 runs must include all predeclared
  qualitative categories above.
- E2 obstacle validation must produce at least one MP4/`.rrd` pair per obstacle family
  showing that camera-visible obstacle points survive into the final policy tensor.
- Store artifacts under the run directory using stable paths such as
  `videos/<method>/episode_NNN.mp4` and `rerun/<method>/episode_NNN.rrd`.
- Write `artifact_manifest.json` linking each MP4 and `.rrd` to its metrics row,
  constraint JSON, obstacle config, checkpoint, and git commit.
- Treat a missing or unreadable required artifact as an incomplete experiment, even
  when plots and summary metrics were produced.

## 4. Statistical reporting

- Use a paired episode design: every method receives the exact same episode index,
  simulator seed, initial state, goal, and serialized constraint.
- Report the number of attempted episodes and every failure. Do not silently drop
  timeouts, numerical errors, or simulator failures. Report infrastructure failures
  separately and rerun the entire paired episode for all methods when justified.
- For each binary rate, report the count, rate, and 95% Wilson interval.
- For the primary method difference, report paired differences in stable combined
  success with a paired bootstrap confidence interval; McNemar's exact test may be
  included as a secondary test.
- For continuous metrics, report median and interquartile range in addition to
  mean and standard deviation. Use paired bootstrap intervals for method differences.
- Predeclare stable combined success as the paper's primary endpoint. Combined success
  is the pilot endpoint only while the stable-success logger is unavailable. All other
  metrics are secondary or diagnostic.
- Report full-distribution evaluation separately from the nominal-base-success subset.
  The latter is a steering-mechanism evaluation and must not be described as general
  task success.

## 5. Experiment list

### E0 — Metric and implementation validation

Run synthetic trajectories and a tiny fixed-seed simulator smoke to verify goal
thresholds, signed-clearance convention, hold-window behavior, trajectory sampling
frequency, timing boundaries, and identical episode/constraint loading across methods.

### E1 — Nominal checkpoint gate

Evaluate the unmodified DP3 checkpoint without active constraints on the held-out
Franka/Panda reach distribution. Report task metrics and require the predeclared base
success gate before interpreting steering results. Also run ITPS and reranking with
zero guidance/constraint weight to catch unintended changes to ordinary inference.

### E2 — Realistic obstacle embodiment and observation validation

Replace visual-only floating keep-out spheres with static simulator actors representing
plausible reach-scene obstacles. Start with:

1. axis-aligned and rotated boxes,
2. a tall carton or equipment enclosure,
3. a cupboard/cabinet represented by its body, shelves, and open-door geometry, and
4. one curved object such as a cup or cylinder.

Each obstacle has two synchronized representations:

- **policy observation:** the actor is created in the control environment before camera
  rendering, so its naturally visible and occluded surface points enter the ordinary
  scene point cloud with no synthetic obstacle-point injection;
- **constraint evaluation:** the simulator-known actor pose and collision primitives
  provide the clearance/collision representation consumed by reranking and ITPS.

The main comparison must not infer obstacle boundaries from the point cloud. The
simulator geometry is controlled ground truth and must be identical for both methods.
Point-cloud-only geometry estimation is a later perception ablation, not a prerequisite.

Before policy experiments, verify for every obstacle family that:

- the actor exists in the control environment, not only the render-only video
  environment;
- camera-visible obstacle points survive workspace cropping and fixed-count
  downsampling;
- obstacle points are labelled non-robot and remain in the world model's static scene
  when current robot points are replaced;
- the final DP3 tensor still contains obstacle points after goal-marker slots are
  overwritten;
- the serialized actor pose used for rendering exactly matches the collision geometry
  supplied to both steering methods; and
- simulator contact and geometric signed-clearance checks agree on synthetic
  intersecting and non-intersecting robot configurations.

Log obstacle points before cropping, after cropping/downsampling, and in the final DP3
input. If small or occluded obstacles are systematically lost, add an obstacle-point
sampling quota or semantic-preserving sampler before running the primary comparison.

### E3 — Primary constrained-reach comparison

Compare:

1. base DP3,
2. rejection/filtering,
3. world-model reranking, and
4. ITPS.

Use fixed nominal-path constraints and the same paired episode set. Report all primary,
constraint-quality, task-quality, motion-quality, and compute metrics. Run both:

- the full held-out distribution, and
- the explicitly labelled nominal-base-success subset.

### E4 — Constraint difficulty sweep

Sweep obstacle size/clearance margin, pose, orientation, placement along the nominal
path, and visible fraction. Use at least easy, medium, and hard levels chosen before
looking at method results. Report results by obstacle family as well as pooled, and
plot stable combined success and minimum clearance against difficulty.

### E5 — Compute/performance frontier

For reranking, sweep candidate count/fallback schedule. For ITPS, sweep MCMC inner
steps and guide ratio. Present:

- each method's recommended/default setting,
- approximately compute-matched settings based on denoiser evaluations or measured
  action-selection latency, and
- success-versus-latency Pareto curves.

### E6 — Guidance-strength and scoring ablations

- ITPS: guide ratio, MCMC inner steps, smooth barrier versus hinge energy, and barrier
  temperature.
- Reranking: rejection versus weighted reranking, prior/smoothness proxy on/off, and
  one-chunk versus multi-chunk imagination.

Do not tune on the final test episodes; use a disjoint validation set.

### E7 — Geometry-target ablation

Evaluate TCP/EEF-only, gripper proxy, and whole-robot constraint measurement. Until
ITPS supports differentiable whole-robot guidance, clearly separate:

- the matched EEF-guidance comparison, and
- whole-robot executed safety evaluation, where both methods are measured but only
  reranking may use that geometry internally.

### E8 — Robustness and repeatability

Repeat the primary comparison across multiple policy-sampling seeds and, if available,
multiple independently trained checkpoint seeds. Test at least small perturbations to
initial joint state, target position, and point-cloud observation. Report both
episode-paired and checkpoint-level variation.

### E9 — World-model prediction study

For reranking candidates, compare predicted versus executed TCP path, minimum
clearance, and feasibility classification. Report path error, clearance error,
precision/recall for predicted violations, and selected-versus-unselected candidate
calibration. This isolates whether failures arise from sampling, scoring, or model
error.

## 6. Required logging schema

Every episode row should contain, at minimum:

```text
run_id, git_commit, checkpoint_id, method, method_config,
episode_index, simulator_seed, policy_seed, constraint_id,
obstacle_id, obstacle_family, obstacle_pose, obstacle_collision_geometry,
obstacle_points_raw, obstacle_points_cropped, obstacle_points_policy_input,
goal_threshold_m, constraint_tolerance_m, control_dt_s,
goal_reached, stable_goal_reached, constraint_satisfied,
combined_success, stable_combined_success,
first_success_step, final_target_distance_m, min_target_distance_m,
min_clearance_m, max_violation_depth_m, violation_steps,
violation_fraction, integrated_violation_m_s,
tcp_path_length_m, joint_path_length_rad, post_success_drift_m,
action_selection_time_s, episode_planning_time_s,
denoiser_evaluations, geometry_evaluations, peak_gpu_memory_bytes,
steps, replans, timeout, error_type,
video_path, rerun_path, artifact_manifest_path
```

Store per-step TCP positions, joint positions, target distances, clearances, and timing
events in a trajectory/debug artifact rather than expanding the episode summary row.

## 7. Current implementation coverage and gaps

The evaluator now logs goal reached (as `reach_success`), stable goal and stable
combined success over the configured post-success window, constraint satisfaction,
first-success step, final/minimum target distance, minimum clearance, TCP violation
depth/duration/fraction/integral/event count, TCP and joint path length, physical-time
TCP/joint acceleration and jerk, maximum joint velocity, steps, replans, candidate
feasibility fraction, fallback count, and optional timing events. Continuous summaries
include mean, standard deviation, median, and quartiles. Paired episode rows now carry
the run, checkpoint, dataset/source episode, simulator seed, shared policy seed, and
SHA-256 constraint identity; the evaluator rejects incomplete or mismatched method
pairs. CUDA-synchronized action-selection timing records total, median, p90, and p95
latency.

Before the paper-scale comparison, add:

- time-indexed whole-robot violation metrics rather than only flattened robot clearance,
- executed-action replan-boundary discontinuity,
- denoiser/FK/geometry operation counts,
- peak GPU memory, and
- paired statistical comparison utilities,
- obstacle/goal-marker point categories in Rerun, and
- richer candidate/guidance overlays linked through an artifact manifest.

No experiment should be blocked on every diagnostic metric. E0 and E1 can run with the
current schema. Complete realistic-obstacle observation validation in E2 before a
pilot E3, and complete the missing primary/compute fields and required qualitative
artifact outputs before the definitive E3 run.
