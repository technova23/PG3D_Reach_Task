# Codex best practices for pg3d

## 1. Put durable instructions in files, not prompts

Use `AGENTS.md` for stable instructions and point Codex to smaller docs for details. Do not paste the entire project history into every prompt.

## 2. Plan before coding on milestone-sized work

For hard tasks, ask for a plan first. Approve or edit the plan, then ask for implementation. This avoids accidental rewrites and keeps research milestones scoped.

## 3. Use the Goal / Context / Constraints / Done-when format

Every task prompt should specify:

- Goal: the exact change.
- Context: files/docs to read.
- Constraints: design decisions and what not to do.
- Done when: checks, docs, and behavior expected.

## 4. Keep one thread per task

Do not run one giant Codex thread for the whole project. Use separate threads/worktrees for repo bootstrap, ManiSkill setup, DP3 integration, world model, constraints, reranking, and eval.

## 5. Keep docs current

Every substantial task should update:

- `docs/status.md`,
- `docs/worklog/YYYY-MM-DD.md`,
- affected runbooks,
- ADRs for durable decisions.

## 6. Run narrow checks

Do not ask Codex to run long training jobs by default. Ask it to run:

- import smoke checks,
- geometry unit tests,
- data-shape tests,
- simulator smoke scripts,
- one-batch training/data-loader smoke tests.

## 7. Review diffs before accepting

Use `docs/prompts/R_review_diff.md` before committing. Pay special attention to dependency changes, submodule changes, path hardcoding, and stale docs.

## 8. Refactor periodically

After each milestone, ask Codex to do a cleanup pass with `docs/prompts/R_refactor_cleanup.md`. This prevents research code from accumulating unused scripts and abandoned APIs.

## 9. Treat dependency changes as design decisions

Changing Python, PyTorch, CUDA, ManiSkill, SAPIEN, or DP3 versions can invalidate previous setup work. Record these changes in runbooks and, if durable, ADRs.

## 10. Keep simulator/policy dependencies lazy

Pure tests should run without ManiSkill, SAPIEN, rendering/GPU simulator dependencies, or DP3 installed. This keeps Codex able to verify geometry and controller code in lightweight environments.
