# ADR 0012 — Staged ITPS-guided beam development MVP

Date: 2026-08-21

## Status

Accepted

## Context

The initial six-fixture ITPS-guided beam pilot showed substantially better goal
approach than flat guided reranking, but used avoidance-only scoring and did not
produce stable constrained success. The same fixtures and outcomes have already
informed method design, so they cannot serve as a locked evaluation set.

## Decision

Implement the staged method in `scripts/itps_beam_mvp_report.md`: normalized
task-aware feasible-prefix scoring, complete proposal replay and telemetry,
conservative feasible-continuation mass, deterministic adaptive weights and search,
and geometry-agnostic route diversity.

Freeze the six selected U-shape fixtures as development-only data under
`configs/eval/e10_itps_beam_mvp_v1`. H1 uses ten guided depth-one proposals. H3
uses depth three, width two, and branch factor two for ten physical expansions.
Mass-aware and adaptive H3 have a maximum of twenty total guided proposals per
real replan, including search expansions and probes.

## Consequences

Every component is introduced and ablated in dependency order. Failed components
are removed from the retained method. Five diffusion lineages quantify development
robustness, but no result on these six fixtures is described as locked-test or
statistically broad benchmark evidence. A distinct outcome-blind fixture suite is
required before publication claims.

All guided proposals, including pruned nodes and mass probes, must be independently
replayable from stored conditioning tensors and explicit diffusion/MCMC noise
lineages.

## Alternatives considered

- Treat the six fixtures as locked evaluation: rejected because their outcomes
  already influenced the methodology.
- Add new U-shape fixtures now: deferred by project-owner decision.
- Keep a single stateful generator seed per proposal: rejected because it does not
  provide independently auditable initial and inner-noise lineages.
