# Codex usage playbook

## Operating pattern

Use one Codex thread per coherent task, not one thread for the whole project. Start each milestone with plan mode, then implementation, then review.

Preferred task structure:

1. Ask Codex to read `AGENTS.md`, `docs/status.md`, relevant ADRs, and milestone docs.
2. Ask for a brief implementation plan.
3. Approve/refine the plan.
4. Ask Codex to implement a narrow slice.
5. Ask Codex to run relevant checks.
6. Ask Codex to update docs/work log.
7. Ask Codex to review its own diff before you accept it.

## Prompt shape

Every prompt should include:

- Goal: what to build/change.
- Context: relevant files/docs.
- Constraints: design decisions, dependency limits, no-long-training, no-submodule edits, etc.
- Done when: commands pass and docs update.

## Context management

Keep durable context in files, not chat:

- `AGENTS.md`: stable repo instructions.
- `docs/status.md`: current state.
- `docs/adr/`: design decisions.
- `docs/milestones.md`: roadmap.
- `docs/worklog/`: what changed.
- `docs/runbooks/commands.md`: exact run commands.

When a task finishes, ask Codex to produce a short worklog entry. If it discovered a new repeated rule, update `AGENTS.md` or a runbook.

## Approval and sandboxing

Start conservative. Let Codex read/write the repo, but require approval for:

- installing system packages,
- network access,
- changing submodules,
- deleting large directories,
- running long training jobs,
- pushing to remotes.

## Parallel work

Use separate git worktrees for parallel agents:

```bash
git worktree add ../pg3d-m1-reach -b m1/reach-adapter
git worktree add ../pg3d-m4-world-model -b m4/world-model-v0
```

Do not let two agents edit the same files simultaneously unless the tasks are intentionally coordinated.

## Review checklist

Ask Codex to review against `docs/review_checklist.md` before accepting a diff.
