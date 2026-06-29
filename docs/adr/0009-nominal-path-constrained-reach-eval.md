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
