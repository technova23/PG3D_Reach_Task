# Non-convex and delayed-consequence scene experiment plan

Status: proposed follow-on benchmark plan

Date: 2026-08-14

Scope: constrained Franka/Panda reach after the box-obstacle ITPS baseline

Related protocol: `docs/explainations/itps_reranking_experiment_plan.md`

## 1. Purpose

The existing box environments mainly test local collision avoidance. They do not fully exercise
the main reason to perform policy-conditioned, multi-chunk imagination: an action can look safe
and goal-directed now while committing the robot to a dead end or an expensive future state.

The next benchmark suite should therefore contain scenes where success requires at least one of:

- leaving the direct goal path before a collision is imminent;
- committing early to one of several route classes;
- temporarily moving laterally or away from the goal;
- maintaining a consistent decision across several action chunks; or
- selecting an early whole-arm posture that preserves future clearance.

The principal hypothesis is:

> Obstacle-free, policy-native point-cloud imagination with multi-chunk beam search can preserve
> task-directed DP3 proposals while identifying delayed geometric consequences that local
> inference-time guidance or one-chunk reranking cannot reliably resolve.

This is a hypothesis to test, not a conclusion to assume.

## 2. Proposed method separation

The follow-on method should separate nominal policy observations from constraint evaluation.

### Policy/proposal environment

- Use an obstacle-free ghost environment for every DP3 query, including the root query and every
  imagined continuation.
- Match the obstacle-free training observation as closely as possible: robot, nominal scene, goal
  marker, point count, crop, sampling, and observation history.
- Do not add constraint-obstacle points to the policy input.
- At each real replan, synchronize the ghost root joint state from the executed robot state. The
  control environment's exteroceptive point cloud is not supplied to DP3.

### Constraint evaluator

- Keep the actual obstacle geometry outside the policy.
- Use the same mesh, signed-distance, or exact primitive representation for every compared method.
- Evaluate whole-robot clearance along every imagined trajectory.
- Permit obstacle-free ghost rollouts to pass through obstacles counterfactually; reject or penalize
  them only through the external constraint evaluator.

### Control environment

- Execute only the selected action prefix.
- Retain authoritative collision, clearance, task-success, and stable-hold grading.
- Return the executed joint state for the next receding-horizon root.

This factorization treats DP3 as a nominal task-motion prior and beam search as the mechanism that
composes it with new geometric constraints.

## 3. Required comparison and current starting point

Use one locked checkpoint, paired episodes, shared policy seeds, identical constraint geometry,
and identical execution horizons.

| ID | Method/input condition | Purpose | State on 2026-08-14 |
| --- | --- | --- | --- |
| A | Base DP3 | Nominal task baseline | Existing evaluation completed |
| B | ITPS | Inference-time steering baseline | Existing evaluation completed; whole-body box run is currently in progress |
| C | Reranking with control-environment point clouds containing obstacle points | Existing obstacle-visible composition baseline | Completed |
| D | Reranking with obstacle-free ghost observations and the same one-chunk cost/search | Isolate the effect of keeping policy input nominal | Not yet run |
| E | Multi-chunk continuation with obstacle-free ghost observations but no branching after the first chunk | Isolate longer conditional rollout from beam search | Current simple horizon mechanism exists; matched new-scene ablation not yet run |
| F | Multi-chunk beam search with obstacle-free ghost observations | Proposed full method | Not yet implemented/run |

The decisive comparison is E versus F for branching, and C versus D for obstacle-free policy
conditioning. A versus B versus F is the final method comparison. Do not attribute gains to point
cloud feedback if they appear only after changing the cost geometry, compute budget, or execution
horizon.

## 4. Scene-design rules

Every scene must satisfy these rules before method outcomes are inspected:

1. The initial and goal configurations are collision-free.
2. At least one whole-robot collision-free path exists with a recorded minimum clearance.
3. The obstacle actor and the geometry supplied to all cost evaluators are identical.
4. The key wrong decision occurs before its failure becomes obvious within one policy chunk.
5. Difficulty parameters and levels are declared before the definitive run.
6. Start/goal pairs and policy seeds are paired across all methods.
7. A motion-planner or carefully validated scripted path provides a feasibility witness.
8. An offline candidate-support check establishes whether DP3 samples any useful lateral/postural
   diversity. Beam search cannot recover a route outside the proposal distribution.
9. The scene must diagnose a stated failure mechanism, rather than merely being visually complex.
10. Dynamic obstacles and manipulation are excluded from this reach benchmark because the current
    world model does not predict their dynamics.

Obstacle dimensions should be scaled so the delayed consequence lies beyond the ordinary ITPS
action horizon but within the proposed multi-chunk search horizon. Each family should sweep one or
two interpretable dimensions instead of relying on a single hand-picked instance.

## 5. Prioritized scene families

### P1 — U-shaped cul-de-sac

Construct the U from three matched box primitives. Place the goal so the nominal direct path enters
the closed portion of the U. The successful route must move laterally toward the opening before
resuming goal progress.

Expected ITPS pressure: local clearance gradients can push the trajectory deeper into the cavity or
fail to provide the early escape commitment.

Sweep:

- cavity depth relative to one action chunk's travel;
- mouth width and whole-robot clearance;
- left/right opening offset; and
- goal depth/alignment behind the closed wall.

Important control: include an easy shallow U that ITPS should solve. If every level is an extreme
trap, the suite will not reveal a meaningful performance frontier.

Initial visualization smoke (2026-08-14): `configs/eval/e10_u_shape_smoke_v1` places a
`0.28 x 0.30 x 0.60 m` three-box U in dataset episode 305. Its opening faces the start, the nominal
start-goal line enters the cavity and crosses the closed back, and the stored initial whole-robot
clearance is 3.23 cm. The simulator actor/serialized-SDF collision probe agrees in intersecting and
separated cases. This is a geometry and observation smoke only, not a frozen difficulty level or
method result.

The preferred visualization was then moved to selected output episode 004 / dataset episode 1010
in `configs/eval/e10_u_shape_smoke_episode004_v1`, because the same envelope dominates the episode
000 workspace view. Episode 004 provides 7.91 cm initial whole-robot clearance and a longer, clearer
start-to-goal approach through the cavity toward the closed back. The original episode-000 artifact
is retained only as provenance for the first smoke.

### P2 — Long wall with a distant opening

Place a long wall across the direct path with its only opening far to the left or right. The direct
signed-distance gradient is predominantly normal to the wall and supplies little information about
which tangential direction reaches the opening.

Expected ITPS pressure: retreat from the wall, oscillation, or slow local sliding without committing
to the distant opening.

Sweep:

- opening side;
- lateral opening distance;
- wall length;
- opening width; and
- whether one or both ends are reachable within the episode horizon.

This is the simplest scene for demonstrating that local collision gradients can lack useful route
information.

### P3 — False-passage fork

Create two initially feasible corridors. Make the wider or more goal-aligned corridor terminate in a
dead end beyond one chunk, while the less direct corridor reaches the goal.

Expected ITPS pressure: the wrong corridor initially has better goal progress and clearance, and the
failure becomes visible only after an early route commitment.

Sweep:

- dead-end depth;
- width advantage of the false corridor;
- decision-point distance from the start; and
- lateral cost of entering the valid corridor.

This is the most direct test of policy-conditioned future observation: future continuations should
expose the dead end before its first action prefix is executed.

### P4 — Staggered gates or chicane

Use two or three walls whose openings alternate left and right, forming an S-shaped route.

Expected ITPS pressure: successive local corrections can undo one another, and receding-horizon
guidance may continually return toward the nominal direct path.

Sweep:

- number of gates;
- spacing in units of action chunks;
- alternating lateral offset; and
- gate width.

This scene tests whether the search preserves a consistent multi-step decision rather than merely
finding one good avoidance action.

### P5 — T-wall or asymmetric hook/C obstacle

A T-wall combines an early stem with a later cap. A hook or C shape provides one open route and one
closed route around a concavity.

Expected ITPS pressure: the first local deflection appears safe but leads toward the cap or closed
side. Symmetric variants can also produce ambiguous or cancelling gradients.

Sweep:

- stem/cavity depth;
- cap length or hook angle;
- asymmetry of the two apparent routes; and
- target offset.

Use these only after the U and distant-opening wall; they exercise similar local-minimum mechanisms
with more geometric complexity.

### P6 — Whole-body elbow trap

Keep the TCP's direct path clear while placing a post or side wall where the elbow or forearm will
collide later. Arrange the start so multiple early joint postures have similar TCP progress but only
one preserves future link clearance.

Expected ITPS pressure: simultaneous link gradients may choose a locally safe posture that becomes
infeasible later, especially when the decisive link is not yet near the obstacle.

Sweep:

- post position and radius;
- whether the threatened link is elbow, forearm, wrist, or hand;
- timing of the future collision; and
- clearance difference between alternative postures.

This is the first scene that specifically tests the scientific value of whole-robot point-cloud
imagination rather than EEF path planning.

### P7 — Doorway, shelf, or overhang reach

Place the target beyond a portal or inside an open shelf. The TCP fits through the opening, but the
arm must select an appropriate posture before entering so the elbow, wrist, and hand remain clear.

Expected ITPS pressure: gradients from opposing surfaces can conflict, and late posture correction
may be impossible after entering the narrow region.

Sweep:

- opening width and height;
- shelf depth;
- target depth;
- overhang height; and
- required entry posture.

This family is valuable but should follow the elbow trap because it combines route choice, narrow
passage, and whole-body posture in one harder diagnosis.

## 6. Recommended execution order

1. Finish and archive the running whole-body ITPS box experiment as the convex baseline.
2. Implement reusable composite-scene construction from matched box/cylinder primitives.
3. Build and validate one fixed U-shaped scene.
4. Add the distant-opening wall and false-passage fork.
5. Run small paired pilots for A, B, D, E, and F; use these only for integration and parameter
   selection.
6. Add the staggered-gate family after beam search works on the fork.
7. Add the elbow trap as the first whole-body-specific scene.
8. Freeze difficulty levels, start/goal instances, seeds, compute budgets, and artifact selection.
9. Run a held-out definitive suite without tuning on its outcomes.
10. Add doorway/shelf and additional C/T variants only after the core claims are interpretable.

## 7. Beam-search requirements

At depth `d`, expand each retained node with `K` stochastic DP3 continuations, evaluate the resulting
`B * K` children, and retain the top `B`. Do not call a collection of independent first-chunk
branches with one continuation each a beam search.

Each node should retain:

- ghost joint state and policy observation history;
- complete action prefix;
- cumulative constraint, goal-progress, and motion costs;
- minimum predicted whole-robot clearance;
- parent identity and route/homotopy diagnostic when available; and
- terminal-value or reachability estimate.

The score must prevent safe inaction from winning. Use hard feasibility first, followed by a
predeclared combination of terminal goal distance or progress, clearance preference, path length,
and smoothness. Record every component. Compare methods under both their standard settings and a
measured compute-matched budget.

## 8. Measurements and diagnostics

Use the main protocol's stable combined success as the primary endpoint. Also report:

- goal reach and stable hold;
- whole-robot and TCP constraint satisfaction;
- minimum clearance, violation depth/duration/integral, and first collision link;
- first incorrect route commitment and whether recovery remained possible;
- beam survival by route class at every depth;
- candidate feasibility fraction and fallback count;
- predicted-versus-executed clearance and TCP/robot path error;
- terminal goal distance and time to first goal;
- action-selection latency, denoiser evaluations, geometry evaluations, and peak memory; and
- task failure conditioned on safety, so stationary safe failures are explicit.

For the obstacle-free-input ablation, additionally record policy-input semantics and compare action
sample diversity between obstacle-visible control observations and obstacle-free ghost observations.
At minimum, report lateral endpoint dispersion, pairwise action-chunk distance, and the fraction of
samples belonging to each feasible route class.

## 9. Interpretation matrix

| Observation | Interpretation |
| --- | --- |
| D outperforms C | Removing unseen obstacle points from DP3 input helps preserve the nominal proposal distribution |
| E outperforms D | Policy-conditioned multi-chunk imagination adds value beyond one-chunk scoring |
| F outperforms E | Branching across later decisions, rather than horizon alone, is necessary |
| ITPS matches F on shallow U but falls behind on fork/deep U | Evidence supports the delayed-decision/local-minimum hypothesis |
| F is safe but rarely reaches | The objective may reward inaction, or DP3 lacks detour support |
| All methods fail and offline samples contain no valid route | The scene tests proposal support, not search quality |
| Predicted success fails in execution | World-model or controller-tracking error is the bottleneck |
| F wins only with much greater compute | Report a success/latency frontier rather than an unqualified method win |

The paper claim should be narrowed if the evidence supports only obstacle-free input, only longer
horizon, or only beam branching. These mechanisms must not be bundled into one unexplained gain.

## 10. Completion gate

A scene family is ready for definitive evaluation only when:

- actor geometry and evaluator geometry agree in intersecting and separated probes;
- a whole-robot collision-free witness path is saved;
- the intended delayed consequence occurs beyond one ordinary policy chunk;
- difficulty settings and seeds are frozen;
- all compared methods use paired roots, constraints, policy seeds, and budgets;
- MP4 and Rerun artifacts expose the policy input, imagined candidates/beam, selected trajectory,
  executed trajectory, and clearance timeline; and
- the artifact manifest links every retained qualitative example to its exact metrics row and
  configuration.

This suite extends the existing box protocol. It does not retroactively change the frozen E3 test;
any change to policy input or primary method settings requires a new, versioned experiment protocol.
