# ADR 0004 — Start with joint action chunks

Date: 2026-05-16

## Status

Accepted, with empirical check required

## Context

The world model needs future robot joint states to render robot geometry and predict end-effector paths. EE-pose chunks would require IK before rendering every sampled candidate.

## Decision

Use absolute joint target chunks as the default P0 action representation. Implement delta joint chunks as a fallback/ablation. Defer EE-pose chunks to code-only baselines or later experiments.

## Consequences

The world model can use FK directly. Kinematic feasibility starts with joint limits and velocity/smoothness checks rather than IK. We still need an early ablation on Reach-Narrow comparing absolute vs delta joint chunks for nominal DP3 stability.

## Alternatives considered

- Delta joint chunks first: potentially more stable local control but more drift in multi-chunk imagination.
- EE-pose chunks first: easier geometric interpretation but introduces IK failures too early.
