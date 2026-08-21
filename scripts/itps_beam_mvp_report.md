# ITPS-Guided Beam Search in PG3D: Current System and MVP Specification

## 1. Purpose of this report

This report fixes the scope of the ITPS-guided beam-search work. It documents:

1. what the current `itps_beam` controller actually does;
2. how it differs from standalone ITPS, guided reranking, and unguided beam search;
3. what the corrected pilot results establish and do not establish;
4. the exact minimum viable product (MVP) to complete next;
5. the experiments and acceptance criteria required to judge that MVP; and
6. the extensions that are explicitly deferred.

The purpose is to prevent further conceptual expansion before the core mechanism is implemented, validated, and evaluated.

---

## 2. Executive decision

The project will complete one staged method family:

> **A frozen goal-conditioned DP3 policy generates ITPS-guided action chunks at branch-specific imagined observations. A deterministic kinematic world model propagates each branch, an exact whole-robot geometric program verifies the complete prefix, and a bounded beam ranks branches using task cost, clearance, motion quality, and conservative feasible-continuation mass. Score weights and search allocation may adapt under fixed rules and compute caps. Only the first action chunk of the winning branch is executed before replanning.**

The first MVP stage tests whether physical three-chunk lookahead provides value beyond spending the same number of ITPS-guided samples at the current observation. Later stages add and isolate mass-aware scoring, adaptive weights, adaptive computation, and diversity.

The primary comparison is therefore:

- **H1 — ITPS-guided reranking:** ten guided candidates at the current observation, physical depth one;
- **H3 — ITPS-guided beam:** ten guided expansions distributed over a depth-three beam.

Both methods use exactly 4,000 denoiser evaluations per real replanning decision. This matched comparison isolates the allocation of guided proposals across imagined future states.

The diagnostic H1-versus-H3 comparison remains the first controlled stage, but the publishable MVP includes the complete search mechanism: feasible-policy-mass estimation, task/mass/cost combination, adaptive score weighting, uncertainty-aware sampling, dynamic beam allocation, and route-diversity preservation. These additions must be introduced sequentially and ablated rather than switched on simultaneously.

Only three research expansions are outside scope:

1. new constraint families beyond the three already selected;
2. carried-object manipulation; and
3. perception or inference of unknown object geometry.

---

## 3. Shared policy and evaluation system

### 3.1 Frozen DP3 policy

All methods use the same frozen EMA checkpoint:

```text
checkpoint: /scratch2/skills/train_final_Arya/step_00100000.pt
checkpoint step: 100,000
weights: EMA
action representation: absolute joint targets
```

The policy is a goal-conditioned conditional 1D U-Net with cross-attention. It predicts 16 seven-dimensional actions from three observation steps. The controller executes eight actions before obtaining a new real observation and replanning.

Important fixed diffusion settings are:

| Setting | Value |
|---|---:|
| Training diffusion steps | 350 |
| Inference denoising steps | 100 |
| Prediction target | Epsilon |
| Beta schedule | Scaled linear, 0.0001 to 0.05 |
| Point-cloud size | 1,024 |
| Observation steps | 3 |
| Action horizon | 16 |
| Executed actions per replan | 8 |

An ordinary DP3 proposal requires 100 denoiser evaluations. An ITPS proposal performs four inner MCMC updates at every diffusion level and therefore requires 400 denoiser evaluations.

### 3.2 Fixed evaluation protocol

| Setting | Value |
|---|---:|
| Maximum episode length | 300 simulator steps |
| Control timestep | 0.05 s |
| Execution horizon | 8 simulator steps |
| Constraint target | Whole robot |
| Geometric verifier | Exact robot geometry |
| Required clearance | 3 cm |
| Collision termination threshold | 3 cm |
| Goal threshold | 2.5 cm TCP-to-goal distance |
| Stable hold | Goal entry plus 16 consecutive in-threshold samples |
| Obstacle family | U-shaped obstacles |
| Executed clearance checking | Every simulator step |
| Contact termination | Enabled |

The stable-success implementation has been corrected. The hold counter must reset whenever the TCP leaves the 2.5 cm goal region. `stable_goal_reached` and `stable_combined_success` are the authoritative stable outcomes; merely entering the goal region once is a secondary outcome.

---

## 4. What standalone ITPS does

At each real replanning decision, standalone ITPS:

1. receives the current real DP3 observation window;
2. begins reverse diffusion from one sampled noisy action trajectory;
3. performs four inner MCMC guidance steps at each of 100 diffusion levels;
4. evaluates a differentiable smooth-barrier geometric energy on the predicted robot motion;
5. returns one guided 16-action trajectory;
6. executes the first eight actions; and
7. replans from the next real observation.

Its fixed settings are:

| Setting | Value |
|---|---:|
| Guided candidates per replan | 1 |
| Effective physical depth | 1 chunk |
| DDIM eta | 0 |
| Inner MCMC steps | 4 |
| Guide ratio | 60 |
| Barrier temperature | 0.01 |
| Differentiable robot points | 1,024 |
| Denoiser evaluations per replan | 400 |
| Geometry evaluations per replan | 6,336 |

ITPS supplies a local gradient toward lower geometric cost. It does not explicitly preserve multiple alternatives or query the policy again from several imagined future observations.

---

## 5. What ITPS-guided reranking H1 does

H1 separates proposal generation from exact verification:

1. receive the current real observation window;
2. independently sample ten ITPS-guided action chunks from that window;
3. propagate each chunk through the deterministic kinematic world model;
4. verify every imagined step with exact whole-robot geometry;
5. rank the candidates using the fixed selection rule;
6. execute the first eight actions of the selected candidate; and
7. discard the remaining prediction and replan from reality.

H1 has physical depth one. The earlier shared CLI metadata recorded `planning_horizon_chunks=3`, but the H1 implementation evaluated only one physical chunk. The corrected configuration must record depth one explicitly.

H1 uses ten ITPS candidates, or 4,000 denoiser evaluations per replanning decision. It answers:

> Is exact selection among several independently guided current-state proposals sufficient?

---

## 6. What ITPS-guided beam H3 does

### 6.1 Node state

Each beam node contains at least:

- its parent and ancestry;
- the complete imagined action prefix;
- the complete imagined robot-state prefix;
- a rolling three-observation DP3 window specific to that branch;
- exact prefix-feasibility results;
- decomposed constraint and task scores;
- the seed lineage of its diffusion and inner-loop samples; and
- deterministic tie-breaking information.

The branch-specific observation window is essential. H3 is not merely concatenating independently generated open-loop chunks from the original observation. It queries the policy again from each retained imagined state.

### 6.2 Root expansion

At the current real observation:

1. create one root node;
2. generate two independent ITPS-guided chunks;
3. imagine each chunk step-by-step;
4. reconstruct its robot point cloud and observation fields at the imagined joint states;
5. update its rolling DP3 observation window;
6. verify the complete prefix with exact whole-body geometry; and
7. retain at most two nodes.

### 6.3 Deeper expansion

At depths two and three:

1. take every retained parent;
2. repeat that parent's branch-specific observation window twice;
3. generate two different ITPS-guided diffusion samples;
4. propagate every child with the world model;
5. append its motion to the complete parent prefix;
6. update the child's rolling observation window using its newly imagined states;
7. verify and score the complete root-to-child prefix; and
8. globally prune the resulting children back to beam width two.

Sibling children share a conditioning observation window but receive different diffusion noise. Children of different parents receive different imagined observation histories.

### 6.4 Branch-specific imagined observations

For a parent chunk producing imagined states

\[
\hat{s}_1,\hat{s}_2,\ldots,\hat{s}_H,
\]

the next policy call receives the latest three imagined observations:

\[
\hat{o}_{H-2:H}.
\]

Each imagined observation contains:

- a cropped point cloud with the robot at the imagined joint state;
- imagined joint positions as `agent_pos`;
- the unchanged episode target;
- imagined end-effector position; and
- the corresponding TCP-pose fields used by the policy interface.

In exact mode, every imagined step is converted into an observation while the rolling window is updated. No simulator observation is inserted between search depths. The simulator is consulted again only after executing the leading action segment of the selected lineage.

### 6.5 Expansion budget

The H3 search uses width two, branch factor two, and depth three:

```text
depth 1: 1 parent  x 2 children = 2 expansions
depth 2: 2 parents x 2 children = 4 expansions
depth 3: 2 parents x 2 children = 4 expansions
total:                              10 expansions
```

Every expansion is an ITPS-guided proposal requiring 400 denoiser evaluations. H3 therefore uses 4,000 denoiser evaluations per replanning decision, exactly matching H1.

### 6.6 Receding-horizon execution

After depth three:

1. select the best final branch;
2. trace its ancestry back to its first action chunk;
3. execute only the first eight actions of that chunk;
4. discard the imagined tree; and
5. rebuild a new search tree from the next real observation.

This limits world-model error accumulation because only one chunk is executed before reality corrects the state estimate.

---

## 7. Guidance and verification have different roles

ITPS guidance and beam verification must not be conflated.

### ITPS guidance

- operates inside diffusion sampling;
- uses differentiable smooth-barrier geometry;
- supplies local gradients toward safer action trajectories;
- improves the probability that generated proposals are usable; and
- cannot guarantee exact feasibility.

### Exact programmatic verification

- operates on completed imagined chunks and prefixes;
- evaluates exact whole-robot geometry at every imagined step;
- treats feasibility as a hard decision;
- can reject a guided proposal that remains unsafe; and
- supplies auditable violation and clearance measurements.

### Beam search

- retains several guided alternatives;
- queries DP3 again from their different imagined outcomes;
- allocates later samples conditionally on earlier imagined decisions; and
- prevents later motion from erasing an earlier prefix violation.

The intended mechanism is therefore:

```text
local differentiable steering
        +
exact executable verification
        +
branch-conditioned physical lookahead
```

---

## 8. Feasibility and ranking

### 8.1 Hard feasibility

A prefix is feasible only when every configured geometric constraint is satisfied at every imagined step in the complete prefix. Feasible nodes always outrank infeasible nodes.

An earlier violation is permanent for that prefix. Appending later safe chunks cannot make the complete prefix feasible again.

### 8.2 Hard-soft hierarchy

Hard feasibility remains outside every weighted score. No amount of task progress, smoothness, policy likelihood, or estimated future mass may allow an infeasible prefix to displace a feasible prefix.

For infeasible prefixes, the fixed fallback ordering is:

1. minimum maximum violation;
2. minimum accumulated violation;
3. minimum terminal TCP-to-goal distance;
4. deterministic ancestry order.

Weighted and adaptive scoring is used only to rank exactly feasible prefixes. This prevents weight tuning from weakening the 3 cm geometric requirement.

At every intermediate depth, all competing nodes have the same physical prefix length. At the final depth, all final nodes also have the same length. A node's score must be recomputed over its complete prefix; a complete-prefix score must not be added to the parent's complete-prefix score, which would count early motion repeatedly.

### 8.3 Current safety-only ablation

The completed pilot used the following frozen weights:

```text
constraint:       1.0
goal_distance:    0.0
smoothness:       0.0
consensus:        0.0
policy_surrogate: 0.0
```

Consequently, feasible candidates were ranked only by constraint/soft-clearance cost. Goal distance affected the all-infeasible fallback but not ordinary feasible-node ranking.

These results remain useful and will be retained as a **safety-only scoring ablation**. They do not constitute the final mass-aware adaptive MVP.

### 8.4 Fixed task-aware baseline

Before mass estimation or adaptive weighting is introduced, implement a controlled task-aware baseline. For a feasible prefix \(n\), define physically normalized terms:

\[
\tilde J_g(n)=\operatorname{clip}\!\left(\frac{d_{\mathrm{goal}}(n)}{d_{\mathrm{ref}}},0,1\right),
\]

\[
\tilde J_c(n)=\operatorname{clip}\!\left(\frac{c_{\mathrm{buffer}}-c_{\min}(n)}{c_{\mathrm{buffer}}-c_{\mathrm{hard}}},0,1\right),
\]

and an equivalently fixed-scale smoothness term \(\tilde J_s(n)\). Here, \(c_{\mathrm{hard}}=3\) cm is the hard feasibility margin and \(c_{\mathrm{buffer}}>c_{\mathrm{hard}}\) is a conservative soft buffer. All normalizers must be declared before evaluation.

The fixed feasible-prefix score is:

\[
S_{\mathrm{fixed}}(n)=w_g\tilde J_g(n)+w_c\tilde J_c(n)+w_s\tilde J_s(n),
\]

with

\[
w_g,w_c,w_s\ge 0,\qquad w_g+w_c+w_s=1.
\]

The weights may be modified from the current constraint-only configuration. They must be selected on a development split or through a declared cross-validation procedure, never on the locked evaluation outcomes. The current vector \((0,1,0)\) is one ablation, not a protected default.

### 8.5 Feasible-policy-mass estimation

For every feasible node \(n\), estimate how much guided-policy continuation mass remains feasible from its branch-specific imagined state.

Draw \(K_n\) independent ITPS-guided continuation probes from the node observation window. A probe counts as viable only if its complete extended prefix remains exactly feasible over the configured mass-estimation horizon. If \(m_n\) of \(K_n\) probes are viable, maintain a beta-binomial posterior:

\[
p_v(n)\mid m_n,K_n\sim
\operatorname{Beta}(\alpha+m_n,\beta+K_n-m_n).
\]

Use a conservative lower credible bound rather than the raw fraction:

\[
p_{\mathrm{LCB}}(n)=Q_{\delta}\!\left[
\operatorname{Beta}(\alpha+m_n,\beta+K_n-m_n)
\right].
\]

The default prior, confidence level, probe horizon, and maximum probe budget must be frozen in configuration. The mass-risk term is:

\[
\tilde J_m(n)=
\frac{-\log(\epsilon+p_{\mathrm{LCB}}(n))}{-\log\epsilon}.
\]

This quantity is high when few safe guided continuations remain or when the estimate is uncertain. It distinguishes a narrowly feasible branch from a robust branch surrounded by feasible policy support.

Continuation probes must be conditioned on the node's imagined observation window. Root-conditioned samples reused at every depth are not valid mass estimates.

### 8.6 Mass-aware feasible-prefix score

The fixed mass-aware score is:

\[
S_{\mathrm{mass}}(n)=
w_g\tilde J_g(n)+
w_c\tilde J_c(n)+
w_s\tilde J_s(n)+
w_m\tilde J_m(n),
\]

where all weights are non-negative and sum to one. This score is applied only after exact prefix feasibility.

The required ablations are:

1. current avoidance-only score;
2. fixed goal/clearance/smoothness score;
3. fixed mass-aware score; and
4. adaptive mass-aware score.

### 8.7 Adaptive score weighting

The final MVP may change the convex score weights online according to observed search risk. The adaptive rule must be deterministic, bounded, and shared by H1/H3 variants where its inputs exist.

Maintain non-negative logits or dual variables for goal progress, clearance risk, smoothness, and viable mass. At each real replan, update them from declared deficits such as:

- insufficient goal progress;
- clearance approaching the hard margin;
- low feasible-mass lower bound;
- excessive action discontinuity; and
- repeated fallback to infeasible candidates.

One canonical projected update is:

\[
\lambda_{i,t+1}=\Pi_{[0,\lambda_i^{\max}]}
\left(\lambda_{i,t}+\rho_i e_{i,t}\right),
\]

followed by convex normalization:

\[
w_{i,t}=\frac{\exp(\lambda_{i,t}/T_w)}
{\sum_j\exp(\lambda_{j,t}/T_w)}.
\]

The error signals \(e_{i,t}\), update rates \(\rho_i\), bounds, and temperature must be chosen on development data and frozen. Log the complete weight vector at every replan.

Adaptive weighting must be compared with the best fixed convex combination at matched sampling budget. Otherwise, an improvement cannot be attributed to adaptation.

### 8.8 Uncertainty-aware probe allocation

A fixed number of mass probes per node wastes compute on obviously good or bad nodes. Begin with a small equal probe count, compute posterior intervals, and allocate additional probes only to nodes whose uncertainty can change the beam-retention decision.

Stop probing a node when:

- its optimistic score cannot enter the retained beam;
- its pessimistic score is already safely inside the retained beam; or
- the global guided-expansion budget is exhausted.

This implements adaptive computation while preserving a fixed maximum budget per replan. Report both the cap and actual work.

### 8.9 Dynamic beam allocation and route diversity

The final system may vary active beam width between declared minimum and maximum bounds. Increase width when several feasible nodes have overlapping score/mass intervals or represent distinct predicted routes. Contract it when one branch is decisively better.

Route diversity must be geometry-agnostic. Do not hard-code U-shape left/right labels. Represent each prefix by its TCP or robot-point trajectory and apply deterministic farthest-first or diversity-regularized retention after feasibility. Diversity may preserve a slightly worse feasible node only within a declared score tolerance; it may never preserve an infeasible node over a feasible one.

Sibling-consensus scoring must be evaluated as an ablation rather than assumed beneficial. Consensus can collapse distinct route alternatives and is therefore potentially harmful in non-convex scenes.

---

## 9. Difference from unguided beam search

The code structure of `beam` and `itps_beam` is the same at a conceptual level:

```text
beam:       ordinary DP3 proposals -> imagine -> verify/score -> prune
itps_beam:  ITPS-guided proposals  -> imagine -> verify/score -> prune
```

However, the completed configurations are not a controlled candidate-generation ablation:

| Setting | Unguided beam | ITPS-beam H3 |
|---|---:|---:|
| DDIM eta | 1.0 | 0 |
| Width | 8 | 2 |
| Branch factor | 32 | 2 |
| Depth | 3 | 3 |
| Expansions per replan | 544 | 10 |
| Denoiser evaluations per replan | 54,400 | 4,000 |

Therefore, unguided beam provides contextual evidence but does not isolate the effect of ITPS guidance. The causal MVP comparison remains H1 versus H3.

---

## 10. Corrected pilot results

The corrected safety-only H3 pilot produced the following minimum TCP-to-goal distances:

| Fixture | ITPS | H1 guided reranking | H3 guided beam | Closest method |
|---|---:|---:|---:|---|
| 000 | 45.80 cm | 37.56 cm | **17.70 cm** | H3 |
| 001 | 70.73 cm | 53.61 cm | **2.87 cm** | H3 |
| 006 | 50.52 cm | 50.15 cm | **13.71 cm** | H3 |
| 007 | 31.44 cm | 27.23 cm | **7.72 cm** | H3 |
| 008 | 22.32 cm | 46.10 cm | **18.76 cm** | H3 |
| 009 | 1.95 cm | 49.76 cm | **1.03 cm** | H3 |

Aggregate corrected outcomes were:

| Method | Mean minimum | Reached once | Safe episodes | Stable successes | Best fixtures |
|---|---:|---:|---:|---:|---:|
| H3 ITPS-beam | **10.30 cm** | 1/6 | **2/6** | 0/6 | **6/6** |
| ITPS | 37.13 cm | 1/6 | 1/6 | 0/6 | 0/6 |
| H1 ITPS-reranking | 44.07 cm | 0/6 | 0/6 | 0/6 | 0/6 |

H3 reduced mean closest distance by approximately 72% relative to standalone ITPS and was closest on every paired fixture. It nevertheless achieved no stable success and collided in several episodes.

The honest interpretation is:

> Under safety-only feasible-node scoring, distributing guided samples across branch-specific imagined future states substantially improved goal approach, but it did not produce reliable or stable constrained task completion.

The result is mechanism evidence, not a completed publication result.

---

## 11. What the pilot suggests

The frozen DP3 policy is goal-conditioned. Even with zero explicit goal-distance weight, its proposal distribution tends to produce goal-directed motion. H3 appears to preserve goal-directed continuations that remain safer over several imagined chunks better than flat H1 selection.

At the same time, safety-only ranking can prefer a high-clearance branch that stalls or drifts instead of entering and holding the goal region. Exact imagined feasibility also does not automatically guarantee executed feasibility because the kinematic world model may differ from the realized low-level controller transition.

The pilot therefore motivates the full algorithmic program in a controlled order:

1. introduce an explicit task-aware convex score instead of avoidance-only ranking;
2. estimate the future viable mass below every retained branch;
3. penalize branches with low or uncertain viable continuation mass;
4. adapt score weights when progress, clearance, or viability becomes deficient;
5. allocate probes and beam capacity to ambiguous or diverse branches; and
6. validate that imagined absolute-joint-target transitions adequately predict executed joint states and clearance.

The individual mechanisms must be added and ablated sequentially. A single run that changes scoring, mass estimation, beam allocation, and diversity simultaneously cannot support a causal novelty claim.

---

## 12. Full MVP research questions

The completed MVP answers a dependency-ordered sequence of questions:

1. **Physical lookahead:** Under a matched guided-proposal budget, does H3 outperform flat H1?
2. **Task scoring:** Does including normalized goal progress and smoothness improve stable completion over avoidance-only scoring without reducing exact safety?
3. **Viable mass:** Does a conservative estimate of future feasible policy mass identify robust branches that ordinary prefix cost misses?
4. **Adaptive weighting:** Does online reweighting outperform the best fixed convex score under matched search compute?
5. **Adaptive search:** Can uncertainty-aware probes and dynamic beam allocation achieve equal or better outcomes with less average compute?
6. **Diversity:** Does geometry-agnostic route diversity prevent premature beam collapse in non-convex scenes?

The final thesis is:

> **ITPS improves local proposal feasibility, while a world-model search that explicitly reasons about exact prefix feasibility, task cost, and remaining viable guided-policy mass selects robust long-horizon actions that neither flat guidance nor ordinary cost-based beam search reliably finds.**

The MVP is not intended to prove that beam search universally beats ITPS or that gradients are unnecessary.

---

## 13. Full MVP implementation specification

### 13.1 Freeze all shared components

Do not change:

- DP3 checkpoint or EMA weights;
- observation and action horizons;
- absolute-joint-target action representation;
- ITPS eta, MCMC steps, guide ratio, energy, temperature, or robot-point count;
- 3 cm whole-body clearance protocol;
- 2.5 cm goal threshold;
- 17-sample stable-goal requirement;
- execution horizon of eight simulator steps;
- policy and simulator evaluation seed protocol.

Existing fixtures remain the development starting point. Additional U-shaped instances may be created for locked evaluation if required for statistical validity; they must use the same constraint family and generation protocol.

### 13.2 H1 configuration

```yaml
method: itps_reranking
planning_horizon_chunks: 1
execution_horizon_chunks: 1
guided_candidates: 10
constraint_target: robot
geometry_mode: exact
selection: configurable_fixed_or_adaptive_convex
itps:
  eta: 0
  mcmc_steps: 4
  guide_ratio: 60
  energy: smooth
  barrier_temperature: 0.01
  robot_points: 1024
```

### 13.3 H3 configuration

```yaml
method: itps_beam
planning_horizon_chunks: 3
execution_horizon_chunks: 1
beam_width: 2
beam_branch_factor: 2
constraint_target: robot
geometry_mode: exact
selection: configurable_fixed_or_adaptive_convex
mass_estimator:
  enabled: true
  prior_alpha: <frozen development value>
  prior_beta: <frozen development value>
  lower_bound_quantile: <frozen development value>
  initial_probes_per_node: <frozen development value>
  max_guided_expansions_per_replan: <fixed cap>
adaptive_search:
  min_beam_width: <frozen development value>
  max_beam_width: <frozen development value>
  uncertainty_allocation: true
  route_diversity: true
itps:
  eta: 0
  mcmc_steps: 4
  guide_ratio: 60
  energy: smooth
  barrier_temperature: 0.01
  robot_points: 1024
```

### 13.4 Required deterministic behavior

For every proposal, record:

- simulator seed;
- shared policy seed;
- root replanning index;
- parent ancestry;
- initial diffusion-noise seed;
- every inner-MCMC noise seed, if stochastic;
- candidate index; and
- selected action hash.

Additionally record the exact scoring configuration, normalizers, score weights, adaptive state, mass posterior, confidence interval, probe-allocation decisions, active beam width, and diversity signature.

Replaying the same artifact must reproduce the same candidate actions, selections, executed actions, and outcome metrics up to documented numerical tolerance.

### 13.5 Required node telemetry

For every expanded node, record:

- depth and parent identity;
- prefix maximum violation;
- prefix accumulated violation;
- minimum prefix clearance;
- exact feasibility Boolean;
- terminal TCP-to-goal distance;
- soft-clearance cost;
- normalized goal, clearance, smoothness, and mass-risk terms;
- fixed or adaptive weight vector;
- mass probe count and viable count;
- posterior mean and conservative lower bound;
- final feasible-node score and fallback key;
- route-diversity descriptor;
- active beam width and probe-allocation reason;
- whether the node was retained;
- whether it became the executed lineage; and
- denoiser and geometry work.

This telemetry must make every pruning decision reconstructable.

---

## 14. MVP validation before full experiments

### 14.1 Selection tests

1. A feasible candidate always outranks an infeasible candidate.
2. Feasible score terms are normalized to declared physical scales.
3. Fixed convex weights are non-negative and sum to one.
4. Adaptive weights remain within bounds and sum to one.
5. Infeasible candidates follow maximum violation, integral, then distance regardless of weights.
6. Exact ties follow deterministic ancestry order.
7. H1 and depth-one H3 choose the same candidate when given the same candidates and scoring mode.

### 14.2 Prefix tests

1. A violation in an early chunk remains present at later depths.
2. Complete-prefix scores are not double-counted through parent accumulation.
3. Chunk-boundary motion terms include the parent-to-child boundary when logged.
4. Only the first chunk of the winning lineage is executed.

### 14.3 Observation-window tests

1. Siblings receive identical parent windows and different proposal seeds.
2. Children of different parents receive different branch-specific windows.
3. The depth-two window contains the last three observations of chunk one.
4. The depth-three window contains the last three observations of the complete retained lineage.
5. Exact-mode observation generation is deterministic.

### 14.4 World-model transition validation

For a representative set of executed chunks, compare:

- imagined versus executed joint states;
- imagined versus executed TCP positions;
- imagined versus executed minimum robot-obstacle clearance; and
- predicted feasible versus executed safe outcomes.

This is a validation of the existing kinematic rollout, not a commitment to learn a new dynamics model. If the mismatch is large enough to invalidate 3 cm verification, document it and use a fixed conservative verification margin rather than starting a new modeling project.

### 14.5 Stable-hold tests

1. Entry begins the consecutive counter at one.
2. Leaving the threshold resets the counter to zero.
3. Re-entry begins a new streak.
4. Termination occurs only after 17 consecutive in-threshold samples.
5. `stable_goal_reached`, `stable_combined_success`, maximum streak, and termination reason agree.

### 14.6 Feasible-mass tests

1. Mass probes are conditioned on the correct child-specific observation window.
2. A probe is viable only if the complete extended prefix is exactly feasible.
3. Posterior parameters equal prior plus observed viable/non-viable counts.
4. The conservative lower bound increases monotonically with additional viable probes.
5. Equal empirical fractions with fewer probes receive a more conservative lower bound.
6. Fixed seeds reproduce probe outcomes and posterior values.
7. A low-mass node can lose to a slightly more costly high-mass node only through the declared feasible score.

### 14.7 Adaptive-search tests

1. All nodes receive the configured minimum initial probes.
2. Extra probes are allocated only when uncertainty can change retention.
3. The global guided-expansion cap is never exceeded.
4. Active width remains within configured bounds.
5. Diversity never allows an infeasible node to replace a feasible node.
6. Disabling adaptation reproduces the fixed-budget beam.

### 14.8 Weight-calibration tests

1. The current avoidance-only vector is exactly reproducible.
2. Fixed weights are selected only from development data.
3. Adaptive-weight state resets at the declared episode boundary.
4. Every adaptive update can be reconstructed from logged errors.
5. Evaluation fixtures cannot change normalizers, learning rates, bounds, or weight initialization.

---

## 15. MVP experiment sequence

### Stage 1 — deterministic smoke

Run one fixture for H1 and H3. Confirm:

- ten guided proposals or expansions per replan;
- 4,000 denoiser evaluations per replan;
- valid node telemetry;
- correct branch-specific windows;
- correct fixed convex scoring and hard-feasibility ordering; and
- deterministic replay.

Do not proceed if any invariant fails.

### Stage 2 — controlled physical-lookahead baseline

Run the same six existing U-shaped fixtures with:

1. standalone ITPS;
2. H1 with fixed task-aware scoring; and
3. H3 with the same fixed task-aware scoring.

Retain the existing safety-only H3 results as an ablation. This stage isolates physical lookahead before mass estimation changes the search.

### Stage 3 — score calibration

On a declared development split, evaluate a small predeclared simplex of convex goal/clearance/smoothness weights. Select one fixed vector using stable combined success first, safety second, then final distance and compute. Freeze all normalizers and the selected vector.

Do not use the locked evaluation fixtures to choose weights.

### Stage 4 — feasible-mass implementation

Add mass probes and the beta-binomial conservative estimator. Compare at matched maximum and matched actual guided-expansion budgets:

1. fixed task-aware H3 without mass;
2. fixed task-aware H3 with posterior-mean mass;
3. fixed task-aware H3 with conservative lower-bound mass.

This stage establishes whether remaining feasible policy support is useful beyond immediate prefix cost.

### Stage 5 — adaptive weights

Compare the best fixed mass-aware convex score with the adaptive mass-aware score. Freeze update rules on development data and report the complete weight trajectories.

### Stage 6 — adaptive compute and diversity

Add uncertainty-aware probe allocation, bounded dynamic beam width, and route-diversity retention in separate ablations before combining them. Compare outcome quality at:

- equal maximum guided-expansion cap;
- equal average denoiser work; and
- equal wall-clock latency where practical.

### Stage 7 — repeated seeds and locked evaluation

Repeat every retained primary method with at least five fixed diffusion-seed lineages per fixture. Use a locked evaluation set distinct from weight and adaptation development. Existing fixtures may be retained as development fixtures; additional U-shaped fixtures may be constructed only when needed to obtain a credible locked test set.

Every paired trial must use the same root-level seed schedule wherever method structure permits. Differences caused by branch ancestry must remain reproducible and logged.

### Stage 8 — decision

Apply the predeclared acceptance rules below. Do not introduce new score terms or change mass/adaptation rules after inspecting locked outcomes.

---

## 16. Outcomes and acceptance criteria

### 16.1 Primary outcome

The primary outcome is:

> **Stable combined success:** the robot satisfies the 3 cm whole-body constraint throughout execution and remains within 2.5 cm of the goal for 17 consecutive evaluation samples.

### 16.2 Secondary outcomes

- entered the 2.5 cm goal region at least once;
- whole-episode constraint satisfaction;
- geometric collision termination;
- minimum and final TCP-to-goal distance;
- minimum clearance and violation integral;
- maximum consecutive goal-hold streak;
- action-selection latency per real replan;
- denoiser evaluations and forward calls;
- exact geometry evaluations;
- peak GPU memory; and
- predicted-versus-executed world-model error.

Minimum goal distance is a mechanism metric. It is not a substitute for stable task success, especially in a U-shaped environment where Euclidean proximity may occur on the wrong side of an obstacle.

### 16.3 Component gates

Retain physical H3 lookahead if:

1. it achieves at least two clean stable combined successes across the six fixtures;
2. at least two of those successes occur on fixtures where neither H1 nor standalone ITPS achieves stable combined success under the paired protocol;
3. it improves safety or stable combined success over H1 at the same guided-sample budget; and
4. its behavior is deterministically reproducible.

Retain the mass estimator in the final method only if it improves stable combined success, safety, or robust worst-seed performance over fixed task-aware H3 at matched compute. A change only in the estimated mass statistic is not sufficient.

Retain adaptive weighting only if it outperforms the best fixed convex weights on locked evaluation. Comparing it only with avoidance-only scoring is insufficient.

Retain adaptive allocation or diversity only if it improves outcome at matched compute or reduces average compute without reducing stable combined success and safety.

If a component fails its gate, remove that component while continuing with the strongest validated preceding stage. The full project does not require every proposed component to survive ablation.

---

## 17. What can be claimed if the MVP succeeds

If the acceptance gate is met, the defensible contribution is:

> A frozen diffusion policy can be searched closed-loop through branch-specific world-model observations. ITPS increases local proposal feasibility; exact programs verify complete prefixes; and conservative viable-mass estimation identifies branches with robust future policy support. Adaptive scoring and search allocation trade goal progress, clearance, motion quality, and continuation viability under bounded computation.

This is an algorithmic systems contribution centered on the interaction between:

- inference-time differentiable proposal guidance;
- a deterministic point-cloud/kinematic world model;
- exact executable whole-body verification;
- receding-horizon branch-conditioned policy search;
- conservative feasible-policy-mass estimation; and
- adaptive score and search-budget allocation.

---

## 18. Claims that remain prohibited

Even if the MVP succeeds, do not claim:

- that ITPS-beam universally outperforms ITPS;
- that beam search alone solves U-shaped motion planning;
- that finite-probe mass estimates equal the exact probability mass;
- that it handles arbitrary non-differentiable programs without additional experiments;
- that exact geometry is available from raw real-world perception;
- that the kinematic world model is an exact simulator;
- that the method has been validated for carried-object manipulation;
- that the six-fixture pilot is a statistically broad benchmark; or
- that unguided beam versus ITPS-beam isolates guidance under the existing unmatched configurations.

---

## 19. Explicit non-goals

Only the following research expansions are excluded from this project scope:

1. adding new constraint families beyond the three already selected;
2. carried-object manipulation and carried-object collision constraints; and
3. perception or inference of unknown obstacle geometry, including point-cloud-only boundary reconstruction.

All search, scoring, mass-estimation, adaptive-weighting, dynamic-allocation, diversity, reproducibility, world-model validation, baseline, and statistical-evaluation work described above remains in scope.

---

## 20. Immediate work order

1. Freeze this document as the canonical MVP specification.
2. Preserve the current avoidance-only score as an ablation.
3. Implement normalized goal, clearance, and smoothness terms with configurable convex weights.
4. Verify H1 records physical planning depth one.
5. Add candidate-level diffusion and MCMC seed lineage.
6. Add reconstructable per-node score and pruning telemetry.
7. Validate world-model one-chunk joint and clearance prediction error.
8. Run deterministic fixed-score H1 and H3 smoke tests.
9. Run the controlled physical-lookahead comparison.
10. Calibrate and freeze fixed convex weights on development data.
11. Implement conservative child-specific feasible-mass estimation.
12. Add mass-aware scoring and matched-compute ablations.
13. Implement deterministic adaptive score weighting.
14. Implement uncertainty-aware probe allocation.
15. Implement bounded dynamic beam width and geometry-agnostic route diversity.
16. Ablate every added component individually.
17. Run repeated seeds and locked evaluation.
18. Retain only components that pass their predeclared gates.

---

## 21. Final MVP boundary

The MVP is complete when the project has:

- one frozen task-aware H1 implementation;
- one frozen fixed-score H3 implementation;
- one conservative feasible-mass estimator;
- one frozen mass-aware H3 implementation;
- one deterministic adaptive-weight controller;
- bounded uncertainty-aware probe and beam allocation;
- geometry-agnostic route-diversity retention;
- matched guided-proposal budgets;
- branch-specific exact-mode imagined observation windows;
- exact whole-body complete-prefix verification;
- corrected stable-hold evaluation;
- deterministic seed and node telemetry;
- validated one-chunk world-model accuracy;
- development-only weight and adaptation calibration;
- paired repeated-seed results on locked U-shaped evaluation fixtures;
- component-wise ablations; and
- evidence-based keep-or-remove decisions using the predeclared gates.

The final reported method is the strongest sequence of components that survives these gates. Failed additions remain documented ablations rather than being protected as part of the final system.
