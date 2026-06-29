# ADR 0005 — Kinematic point-cloud world model as first-class module

Date: 2026-05-16

## Status

Accepted

## Context

A key project pivot is that, in 3D point-cloud robot policies, the robot portion of the next observation is geometrically predictable from robot state, action chunks, known robot geometry, and robot masks.

## Decision

Implement a kinematic/geometric point-cloud world model as a first-class module, not as a visualization afterthought. P0 world model removes current robot points, renders future robot points from joint chunks, and returns imagined point-cloud rollouts and EEF paths.

## Consequences

Reranking can evaluate constraints over imagined rollouts before simulator execution. This module becomes central to the scientific claim and should be visually validated early.

## Alternatives considered

- No world model: simpler reranking but weaker novelty and less ability to evaluate future multi-chunk violations.
- Learned world model: too much scope for P0.
