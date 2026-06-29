# ADR 0001 — Reach-first MVP

Date: 2026-05-16

## Status

Accepted

## Context

Pick-and-place is scientifically important but adds grasping, contact, object attachment, release, and object-pose evaluation. These would obscure whether the core mechanism works.

## Decision

The first MVP will use ManiSkill reach before pick-and-place. The initial path is a built-in ManiSkill smoke task followed by a narrowed/custom reach task if needed. The first constraint is `avoid_region` over the end-effector path.

## Consequences

Reach allows tighter control over start/goal distributions and clean validation of action chunks, FK, point-cloud world modeling, constraints, reranking, and visualization. It is not sufficient for the final paper claim because code-only planners may be strong; pick-and-place remains the first serious manipulation extension.

## Alternatives considered

- Pick-and-place first: more paper-relevant but too many confounders for the first sprint.
- Place-into-container first: even more brittle due to narrow final geometry and release behavior.
