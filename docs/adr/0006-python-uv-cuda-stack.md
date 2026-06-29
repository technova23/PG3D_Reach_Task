# ADR 0006 — uv, Python 3.11, and CUDA 12.9 PyTorch path

Date: 2026-05-16

## Status

Accepted, pending dependency validation

## Context

The target workstation has an RTX 5090. Older DP3 instructions target older CUDA/Python stacks. The project owner wants uv and a new-ish Python version.

## Decision

Use uv with Python 3.11 for the main pg3d environment. Target PyTorch CUDA 12.9 for RTX 5090 because CUDA 12.9 is installed on the workstation. If ManiSkill, SAPIEN, or DP3 force Python 3.10 or a split environment, document it in a new ADR before changing.

## Consequences

The main `pyproject.toml` uses explicit PyTorch CUDA 12.9 sources for the `cu129` extra. The DP3 fork may need dependency cleanup to avoid pulling old torch/gym/numba constraints into the main environment.

## Alternatives considered

- Use DP3's original Python 3.8/CUDA assumptions: likely bad fit for RTX 5090.
- Use conda: acceptable for emergencies, but not preferred for the main workflow.
