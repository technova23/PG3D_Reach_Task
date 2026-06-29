# ADR 0002 — ManiSkill/SAPIEN as primary simulator

Date: 2026-05-16

## Status

Accepted, supersedes the earlier RLBench simulator decision

## Context

The project needs a controlled simulation environment with point-cloud observations, segmentation
masks, robot proprioception, Gymnasium-style integration, GPU-friendly data collection, and a path
to custom reach and manipulation tasks.

The project proposal now makes ManiSkill/SAPIEN the central simulator. RLBench/PyRep/CoppeliaSim
added setup complexity and compatibility shims that are no longer aligned with the implementation
path.

## Decision

Use ManiSkill/SAPIEN as the primary simulator for P0/P1. Remove RLBench/PyRep/CoppeliaSim as active
dependencies and do not keep RLBench as an alternate backend.

The first task path is:

1. non-rendering smoke with built-in `PickCube-v1` or `PushCube-v1`;
2. narrow ManiSkill reach/custom task if built-in tasks are not sufficient;
3. constrained reach with candidate reranking;
4. pick-and-place after constrained reach works.

## Consequences

Simulator integration work moves to ManiSkill adapters, runbooks, prompts, and smoke scripts. The
repo keeps simulator imports lazy so base tests and package imports do not require ManiSkill,
SAPIEN, Vulkan, or a GPU.

RLBench-specific scripts, tests, compatibility shims, and setup docs are removed from the active
code path. Old RLBench worklog/ADR entries may remain as historical migration context only.

## Alternatives considered

- Keep RLBench as an alternate backend: rejected because the current project needs one lightweight,
  well-documented simulator path and not multi-simulator support.
- DP3's existing environments: easier for the original policy code, but less aligned with the
  custom reach, point-cloud mask, and geometry-overlay path.
