# Refactor and cleanup policy

The repo is research-first, not enterprise-software-first. Still, Codex can create clutter quickly. Run a cleanup pass after every major milestone or after any messy debugging session.

## When to refactor

- After M1, M3, M4, M6, and M7.
- After integrating a submodule or external codebase.
- After a large Codex task that touched many files.
- When docs and commands no longer match reality.

## Cleanup goals

- Remove dead/unused code.
- Merge duplicate utilities.
- Keep public interfaces small.
- Move one-off exploration scripts to `scripts/experiments/` or delete them.
- Update runbooks and status docs.
- Keep tests focused on core geometry, serialization, and shape contracts.

## Refactor prompt

Use `docs/prompts/R_refactor_cleanup.md`.
