# AGENTS.md — pg3d agent instructions

This file is the durable operating guide for Codex and other coding agents working in this repository. Default to read-only analysis unless the user clearly authorizes edits.

## Project objective

`pg3d` studies programmatic geometric guidance for 3D diffusion robot policies. The initial proof of concept is simulation-only:

1. Train/adapt a DP3-style point-cloud diffusion policy on ManiSkill reach demonstrations.
2. Build a kinematic robot-geometry point-cloud world model that imagines future robot point clouds from candidate joint-action chunks.
3. Score handwritten geometric constraints such as `avoid_region` over imagined rollouts.
4. Use candidate rejection/reranking in receding horizon mode to improve combined task-and-constraint success.
5. Only after the reach MVP works, move to pick-and-place, carried-object proxies, no-overflight, energy guidance, and LLM-generated constraints.

Real robot/xArm code is out of scope for now. Keep interfaces robot-agnostic, but do not implement hardware integration.

## Read these docs before coding

Start every substantial task by reading:

- `docs/status.md` — current state and immediate priorities.
- `docs/project_proposal.html` — current research/source-of-truth proposal. Do not skip it.
- `docs/milestones.md` — staged research milestones.
- `docs/architecture/system_architecture.md` — package/module design.
- `docs/adr/` — design decisions that should not be silently changed.
- `docs/runbooks/commands.md` — current commands for setup, test, lint, and smoke checks.

When working on a specific milestone, also read the corresponding file in `docs/prompts/`.

## Engineering style

- Prefer simple, working research code over elaborate frameworks.
- Keep modules small and readable. Avoid clever abstractions unless the milestone needs them. Use type hints, docstrings, and comments where they clarify non-obvious research code.
- Use typed dataclasses or Pydantic models for shared objects like observations, action chunks, constraints, rollouts, and metrics.
- Do not add heavy dependencies casually. If a dependency is needed, document why in the work log.
- Do not import ManiSkill, SAPIEN, rendering/GPU simulator dependencies, or DP3 at package import time. Keep simulator/policy dependencies lazy so CPU-only tests can run.
- Do not modify submodules unless the task explicitly says to do so.
- Do not run long training jobs unless explicitly asked. Create scripts/configs and run smoke-scale checks.
- Keep W&B integration configurable and offline-friendly.

## Python and tooling

- Use `uv` for dependency management.
- Preferred Python: 3.11, unless a dependency forces a documented downgrade.
- Target RTX 5090 / Blackwell with PyTorch CUDA 12.9.
- Use `ruff` for format/lint and `pytest` for lightweight tests.

Common commands:

```bash
uv sync --extra cu129 --extra maniskill --group dev --group notebooks
make test
make lint
make gpu-check
make smoke
make maniskill-check
```

If a command changes, update `docs/runbooks/commands.md` in the same commit.

## Documentation discipline

Every nontrivial Codex task should update at least one of:

- `docs/status.md` with what changed and what remains blocked.
- `docs/worklog/YYYY-MM-DD.md` with a concise work log.
- `docs/adr/NNNN-title.md` for durable design choices.
- `docs/runbooks/commands.md` when setup/run commands change.

Do not let docs become stale. If code behavior and docs disagree, fix both or explicitly call out the mismatch.

## Testing expectations

Use tests for high-leverage correctness, not for professional-software overkill. Prioritize tests for:

- geometry/cost functions,
- constraint serialization,
- action chunk shape conventions,
- world-model FK/point-cloud compositor sanity,
- dataset writer schema validation.

For simulator-dependent code, provide smoke scripts and skip tests gracefully when ManiSkill/SAPIEN or rendering support are unavailable.

## Refactor policy

After each major milestone or after a messy debugging session, run a cleanup pass:

- remove unused code and dead scripts,
- consolidate duplicate utilities,
- update docs and command references,
- keep APIs small,
- avoid accumulating one-off notebooks as source-of-truth.

## Done means

A task is done only when:

1. The requested behavior is implemented or the blocker is clearly documented.
2. Relevant smoke checks/tests have been run, or the reason they cannot run is recorded.
3. `docs/status.md` and any affected runbooks/ADRs are updated.
4. The final response summarizes changed files, commands run, and remaining risks.
