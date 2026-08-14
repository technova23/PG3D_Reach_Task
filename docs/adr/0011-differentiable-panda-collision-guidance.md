# ADR 0011 — Differentiable Panda collision guidance

Date: 2026-08-14

## Status

Accepted

## Context

EEF-only ITPS can move the tool center point around an obstacle while another Panda link still
collides. The existing ghost-environment robot clouds are suitable for reranking and evaluation,
but simulator mutation, rendering, segmentation, and NumPy conversion break autograd.

## Decision

Whole-body ITPS uses a pure Torch Panda model backed by collision geometry from the active
ManiSkill `panda_v2.urdf`:

- sample exactly 1024 surface points across link1--link7, the hand, and both fingers;
- reserve at least 32 points per geometry group and allocate the remainder by surface area;
- use deterministic sampling with seed zero by default and log the seed and per-link allocation;
- transform the local points with batched differentiable FK for every diffusion-horizon pose;
- reduce each constraint by its exact worst point across horizon and robot points;
- hold both fingers at the evaluator's fixed `--gripper-open` value;
- exclude fixed link0 from guidance because arm-joint updates cannot move it.

The existing initial-clearance gate, whole-robot executed clearance metric, contact termination,
and PhysX grading continue to include the base. Smooth and hinge point energies remain selectable;
only their spatial and temporal reduction is fixed to the worst point.

## Consequences

Whole-body ITPS is differentiable and does not require camera rendering inside the guidance loop.
The 1024-point surface representation is still an approximation and may miss sufficiently shallow
contacts between samples, so executed ghost-cloud and PhysX safety grading remain authoritative.
This implementation is Panda-specific; other robots require their own validated differentiable
FK and collision-template adapter.

## Alternatives considered

- Differentiate through the ghost simulator: rejected because the render/segmentation path is not
  autograd-compatible.
- Include the fixed base in the maximum energy: rejected because it can dominate the energy while
  producing zero arm-joint gradient.
- Commit a sampled binary point asset: rejected in favor of deterministic startup sampling from
  the active simulator collision model.
