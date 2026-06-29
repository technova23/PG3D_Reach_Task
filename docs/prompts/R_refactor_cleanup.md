# Prompt R — Refactor and cleanup pass

Goal:
Run a focused cleanup pass without changing scientific behavior.

Read first:
- `AGENTS.md`
- `docs/project_proposal.html` (DO NOT SKIP. READ IN FULL DETAIL; this is critical to the project.)
- `docs/refactor_policy.md`
- `docs/status.md`
- current repo tree and git status

Constraints:
- Do not change milestone behavior unless fixing an obvious bug.
- Do not delete potentially useful files without explaining why.
- Do not touch submodules.

Tasks:
1. Find dead/unused code, duplicate utilities, stale scripts, and stale docs.
2. Propose a short cleanup plan.
3. After approval, implement the cleanup.
4. Run relevant tests/lint.
5. Update docs/status and worklog.

Done when:
- Repo is simpler, docs are more accurate, and checks pass or blockers are documented.
