# ADR 0009 — Nominal-path constrained reach eval

Date: 2026-05-19

## Status

Accepted

## Context

The first P10 constrained-reach validation used direct-path midpoint avoid spheres with a weak
workspace DP3 checkpoint. The result mostly measured base reach failure, not whether rejection or
world-model reranking can steer a working base policy around a small new keep-out region. P11 adds
a stronger balanced reach checkpoint and ordered goal tokens, so the next constrained evaluation
needs to isolate constraint steering from base-reach reliability.

## Decision

For the balanced-checkpoint constrained-reach rerun, build avoid regions from successful nominal
base-policy rollouts on held-out validation episodes. Each selected episode gets a precomputed
spherical `AvoidRegion` centered at a fixed arc-length fraction of the executed nominal TCP path,
with default radius `0.03m` and path fraction `0.5`. The exact constraint JSON and dataset episode
indices are saved before evaluating base, rejection, and reranking, and all methods consume the
same saved constraints.

## Consequences

This makes the first rerun a base-success-subset evaluation. It is better suited to testing whether
candidate rejection/reranking can nudge a trajectory around a small obstacle, but it must not be
reported as full-distribution constrained reach success. If too few of the 25 starter episodes are
base successes, the correct conclusion remains that base reach is still the blocker.

The older direct-path constraint generation stays available for historical comparison and quick
smokes, but precomputed nominal-path constraints are the preferred workflow for the P11 balanced
rerun.

## Alternatives considered

- Direct-path midpoint spheres: repeatable, but can create oversized or unnatural constraints and
  does not guarantee the nominal policy would interact with the region.
- Fresh random avoid regions: broader coverage, but too noisy for the first post-P11 steering check.
- Larger keep-out spheres: easier to visualize, but likely over-constrains reach and measures
  avoidance failure rather than small trajectory nudges.

## 2026-07-23 E3 amendment

The historical spherical protocol remains valid for the P11 diagnostic, but E3 uses
the realistic-obstacle requirement added later. The pilot's fixed nominal-path
instances are collidable tall cartons with half-extents `[0.055, 0.08, 0.16]` m and
20-degree yaw. Their XY position comes from the 0.5 arc-length point of the successful
nominal TCP path; their Z center is `0.16` m so the bottom face rests on ManiSkill's
world-`z=0` tabletop. The serialized `BoxRegion` and control actor share this exact
pose and geometry.

This grounding rule is explicit rather than inferred from camera points. The 10 locked
pilot episodes are used only for integration/tuning; the definitive 50-episode test
partition remains untouched.

The paper comparison reports two separately labelled populations:

- `policy_success`: run the locked base checkpoint without obstacles and retain only
  successful nominal paths. This isolates steering on episodes the policy can already
  solve.
- `dataset_demo`: use every selected episode's stored successful demonstration TCP
  path, regardless of whether the base checkpoint would solve that episode. This is
  the fixed-obstacle source for the full held-out distribution.

For `dataset_demo`, one shared actor height is resolved before evaluation from the
highest selected path anchor plus the predeclared top margin. Every serialized
constraint uses those same half-extents, so the single ManiSkill actor constructed
for the run exactly matches all episode constraints. Demonstration paths determine
obstacle placement only; no compared method outcome is inspected.

The first definitive full-distribution attempt exposed a validity defect before it
completed: a tall grounded midpoint box could overlap the robot at the initial
configuration, making whole-robot constraint success impossible at time zero even
without a PhysX contact pair. Those partial method rows are excluded.

Protocol v2 retains a path-intersecting carton but adds a geometry-only placement
gate. Candidate fractions are searched symmetrically around `0.5`, bounded to
`[0.2, 0.8]`. The center may move within 90% of the box footprint in deterministic
local-coordinate increments, which keeps the source path inside the box. The first
candidate with at least 2 cm signed clearance from the stored initial robot cloud is
serialized. An evaluator preflight recomputes that clearance and fails before any
compared method runs if the constraint is invalid. This repair uses only stored
demonstration and initial-geometry data, never compared-method outcomes.

## 2026-07-24 contact/artifact amendment

The v2 definitive attempt is also partial and excluded. PhysX contact-pair reporting
alone missed a visually apparent shallow robot/box penetration, so videos continued
after the intended terminal event. Protocol v3 checks whole-robot signed clearance
online after every simulator step and terminates when either that clearance is
non-positive or PhysX reports contact. It keeps the first contact frame and no later
frames.

MP4 identity remains mandatory, but it is rendered in a new canvas header above the
original camera frame. No simulator pixel may be covered by status text.

## 2026-08-10 feasible-candidate scoring amendment

Hard avoidance feasibility remains `max(margin - signed_distance, 0) <= tolerance`.
The primary avoidance cost used by reranking now adds a positive soft-clearance term

`clearance_scale^2 / (clearance_scale + max(min_signed_distance - margin, 0))`.

The term has units of meters, equals `clearance_scale` at the feasibility boundary,
and decreases toward zero as clearance grows. This removes the previous all-zero tie
between feasible candidates while leaving feasibility itself unchanged. The default
scale is 5 cm; scale zero is the historical hinge-only ablation. Rejection continues
to choose the first feasible policy sample, so the change affects its diagnostics but
not its selection rule. ITPS retains its separate guidance energy.
