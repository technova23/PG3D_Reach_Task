# Non-convex U-scene experiment plan

Status: active U-object evaluation plan; other scene families deferred

Originally proposed: 2026-08-14

Revised: 2026-08-19

Related protocol: `docs/explainations/itps_reranking_experiment_plan.md`

## 1. Current scope

The immediate non-convex experiment is the fixed ten-object U-shape suite in
`configs/eval/e10_u_shape_box_derived_review_v1`. The next comparison is:

- whole-body ITPS with its existing one-chunk diffusion horizon; and
- whole-body world-model reranking with planning horizon 3 and execution horizon 1.

This replaces the earlier plan that made obstacle-free policy input, true beam search, and several
additional scene families part of the immediate experiment. Those remain possible extensions, but
they are not prerequisites for evaluating the finalized U objects.

The scientific question for this phase is deliberately narrower:

> Given the same fixed U geometry, start state, goal, constraint program, checkpoint, and policy
> seed, does three-chunk world-model continuation reranking improve stable task-and-constraint
> success over ITPS?

The result must be described as a comparison with continuation-based reranking. The current
implementation branches over first-chunk candidates and samples one continuation per surviving
branch at later chunks; it is not a full beam search that expands every retained node by `K` at
every depth.

## 2. Finalized U-object geometry

The ten U-object geometries are finalized and frozen as version 1. Their authoritative assets are:

- geometry guidance: `configs/eval/e10_u_shape_box_derived_guidance_v1.json`;
- fixture identities and dimensions: `configs/eval/e10_u_shape_box_derived_review_v1/fixture.json`;
- matched whole-robot constraints:
  `configs/eval/e10_u_shape_box_derived_review_v1/constraints/robot`; and
- matched EEF constraints:
  `configs/eval/e10_u_shape_box_derived_review_v1/constraints/eef`.

Every object is a three-box U with a fixed height of `0.75 m`. Its opening faces the recorded
episode start. The frozen XY envelopes are:

| Episodes | Full XY envelope |
| --- | --- |
| 000, 003, 005, 006, 008, 009 | `0.22 x 0.16 m` |
| 001, 002, 004 | `0.22 x 0.32 m` |
| 007 | `0.055 x 0.16 m` |

Episode 003 additionally retains its finalized `3 cm` center translation away from the recorded
start. The stored yaw values and all other centers are frozen exactly as serialized.

“Finalized” means these object dimensions, poses, component geometry, and constraint files are no
longer tuning variables for this experiment. Do not edit the v1 guidance or fixture in place. Any
later geometry change requires a new fixture ID and versioned guidance/config directory.

Object finalization does not assert that every start/goal pair is valid for a primary benchmark.
Environment validity and method outcomes remain separately measured facts.

## 3. Known environment-validity strata

Existing exact Panda surface-cloud and position-only RRTConnect checks establish the following
pre-evaluation strata without using ITPS or reranking outcomes:

- **Primary valid-start/witness set:** episodes 000, 001, 004, 006, 007, 008, and 009 have
  contact-free recorded starts and a found position-to-position path.
- **Invalid-start diagnostics:** episodes 002 and 005 start inside the finalized U geometry. They
  may be executed for failure diagnostics, but they must not enter the primary success-rate
  denominator.
- **Unresolved-route diagnostic:** episode 003 has a contact-free start and many collision-free IK
  goals, but no route was found in the completed 540-second bounded search. This is strong negative
  evidence, not a proof of infeasibility, so it also remains outside the primary witnessed-set
  aggregate.

Report results for all ten objects for transparency, then report the seven-episode primary
valid-start/witness stratum separately. Do not relabel an invalid start or unresolved route based
on a favorable policy outcome.

## 4. Locked method comparison

Use the locked 100k checkpoint and the dataset episodes, simulator seeds, and policy seeds stored
in the fixture. Both methods must receive the same episode-specific root, goal, U geometry, robot
constraint program, and seed.

### ITPS

- constraint target: whole robot;
- guide ratio: 60;
- MCMC inner steps: 4;
- energy: smooth barrier;
- barrier temperature: `0.01`;
- collision-surface points: 1024; and
- execution horizon: one chunk.

ITPS does not acquire a three-chunk world-model horizon merely because the evaluator is invoked
with `planning_horizon_chunks=3`. It continues to guide the normalized diffusion trajectory over
its ordinary action horizon.

### World-model reranking

- constraint target: whole robot;
- exact robot geometry mode;
- planning horizon: 3 chunks;
- execution horizon: 1 chunk;
- candidate fallback schedule: `16, 32, 64`; and
- receding-horizon replanning after each executed chunk.

The current policy input and continuation behavior should be reported exactly as implemented.
Obstacle-free policy conditioning, later-depth branching, DDIM-eta changes, and alternate candidate
diversity mechanisms are separate ablations and must not be silently bundled into this comparison.

## 5. Execution and artifact rules

The ten U objects have different sizes, so the control environment must be constructed with the
matching episode-specific envelope. Running one fixed-size actor across all ten constraints is
invalid. Preserve fixture output indices 000--009 so the order-independent policy seeds remain the
stored paired seeds.

Use:

- maximum task horizon: 150 simulator steps;
- stable-success hold: 16 steps;
- immediate termination on PhysX contact or non-positive whole-robot signed clearance;
- whole-robot clearance grading at every executed step;
- MP4 and native Rerun output for both methods on every object;
- exact policy-input bundles and constraint fingerprints;
- candidate and selected predicted paths in Rerun;
- action-selection timing, denoiser/geometry counts, and peak CUDA allocation; and
- one artifact manifest per episode-specific run.

Run each object in its own process with the matching envelope while preserving its fixture output
index. Because each process emits its own summary, combine the twenty method rows only after
validating paired identities and excluding no rows silently.

## 6. Outcomes and reporting

The primary endpoint is stable combined task-and-constraint success on the seven-episode witnessed
valid-start stratum. All-ten-object results are diagnostic and must identify episodes 002, 003, and
005 explicitly.

Also report:

- reach and stable-hold success;
- whole-robot and TCP constraint satisfaction;
- minimum clearance and violation depth, duration, integral, and event count;
- first collision source and step;
- final and minimum target distance;
- candidate feasibility fraction and fallback reason;
- executed TCP/joint path length, smoothness, and replan discontinuity;
- action-selection latency and measured denoiser/geometry work; and
- paired per-episode differences, without presenting a seven-episode pilot interval as a
  definitive population claim.

Before interpreting a reranking failure, inspect candidate support. If no sampled first-chunk route
leaves the U's failing route class, the result diagnoses the proposal distribution rather than the
ranking rule alone.

## 7. Interpretation boundaries

| Observation | Supported interpretation |
| --- | --- |
| H3/E1 reranking improves over ITPS on witnessed U scenes | Longer policy-conditioned geometric lookahead helps on these finalized U objects |
| Both methods fail and reranking samples contain no viable route | The current DP3 proposal support is the limiting factor |
| Reranking predicts safe candidates that collide in execution | World-model geometry or controller tracking is the limiting factor |
| ITPS matches reranking | Three-chunk continuation did not add measurable value at the tested budget |
| A gain appears only with substantially more compute | Report a success/latency frontier, not an unqualified method win |

This experiment cannot by itself establish that true beam search is necessary, that obstacle-free
policy input is beneficial, or that the result generalizes to other non-convex scene families.

## 8. Deferred scene-family backlog

The following families remain design directions only and will be implemented and finalized later:

1. long wall with a distant opening;
2. false-passage fork;
3. staggered gates or chicane;
4. T-wall, hook, or C-shaped obstacle;
5. whole-body elbow or forearm trap; and
6. doorway, shelf, or overhang reach.

No dimensions, placements, seeds, or completion order are frozen for these families. They should
receive new versioned fixtures and their own geometry/validity review before method outcomes are
examined. Their implementation is not part of the current U-object evaluation completion gate.

## 9. Completion gate for this phase

This phase is complete when:

1. all ten frozen object geometries are instantiated with actor/evaluator agreement;
2. ITPS and H3/E1 reranking produce paired rows for all ten fixture identities;
3. videos, Rerun timelines, exact-input bundles, and manifests validate for every method/object;
4. the primary seven-episode valid-start/witness result and the all-ten diagnostic result are both
   reported with their denominators; and
5. failures are attributed only after inspecting candidate support, predicted clearance, executed
   clearance, and compute diagnostics.

This plan does not modify the frozen E3 protocol. The U suite is a separate, versioned follow-on
experiment.
