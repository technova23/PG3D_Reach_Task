# ADR 0003 — DP3 as P0 base policy

Date: 2026-05-16

## Status

Accepted

## Context

The project is about steering 3D point-cloud diffusion policies with executable geometric constraints. DP3 is the preferred base policy for P0, while RISE and CodeDiffuser-style methods are later baselines/extensions.

## Decision

Implement only DP3 for P0. Keep a generic `Policy` interface so RISE or other policies can be plugged in later.

## Consequences

The DP3 fork/mirror is the only required submodule initially. Do not add RISE/MinkowskiEngine dependencies until needed.

## Alternatives considered

- RISE first: relevant but adds MinkowskiEngine/CUDA complexity.
- 3D Diffuser Actor: treat as P2, not needed for MVP.
