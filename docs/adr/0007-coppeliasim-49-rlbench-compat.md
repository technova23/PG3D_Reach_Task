# ADR 0007 — CoppeliaSim 4.9 compatibility for RLBench observation smoke

Date: 2026-05-16

## Status

Superseded by ADR 0002. Historical context only; RLBench/PyRep/CoppeliaSim are no longer active
pg3d dependencies or backends.

## Context

Upstream PyRep targets older CoppeliaSim APIs, while the workstation simulator is
CoppeliaSim 4.9.0 rev6 on Ubuntu 22.04. Directly using RLBench/PyRep with that
install initially failed because:

- `PyRep.launch(scene_file=...)` left the default scene loaded, so RLBench could not find `Panda`;
- PyRep's single-precision CFFI wrappers corrupted memory against CoppeliaSim 4.9 double-precision
  legacy API symbols;
- CoppeliaSim 4.9's Python add-on launcher needed `zmq` and `cbor2`;
- RLBench reset validation called old path/IK APIs that CoppeliaSim 4.9 has dropped.

The M1 observation adapter needs one ReachTarget observation, not demonstration generation or
training.

## Decision

Support CoppeliaSim 4.9.0 rev6 for the observation-save and smoke paths with local pg3d
compatibility shims:

- launch CoppeliaSim first, then load RLBench `task_design.ttt` with `simLoadScene`;
- route precision-sensitive PyRep wrappers through CoppeliaSim `*_D` symbols;
- point CoppeliaSim's Python launcher at the active pg3d virtualenv and include `pyzmq`/`cbor2`
  in the `rlbench` extra;
- skip RLBench waypoint feasibility validation only for observation-only smoke/save paths.

## Consequences

The ReachTarget observation-save script works with the local 4.9.0 simulator and keeps RLBench
imports lazy. This does not make live RLBench demonstration generation supported on 4.9, because
those paths still depend on removed CoppeliaSim path/IK APIs.

## Alternatives considered

- Require CoppeliaSim 4.1.0 only: simpler but conflicts with the workstation setup and blocks M1.
- Patch the external PyRep install or CoppeliaSim files directly: harder to reproduce and outside
  the repo's controlled source.
- Move to CoppeliaSim's newer remote API now: larger integration change than M1 needs.
