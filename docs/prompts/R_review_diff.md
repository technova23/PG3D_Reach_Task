# Prompt R — Review current diff

Goal:
CODE REVIEW: Review the current uncommitted diff before I accept-and-commit or reject it.

Read first:
- `AGENTS.md`
- `docs/project_proposal.html` (DO NOT SKIP. READ IN FULL DETAIL; this is critical to the project.)
- `docs/review_checklist.md`
- current git diff

Tasks:
1. Identify correctness risks, scope creep, stale docs, dead code, and fragile assumptions.
2. Check whether commands/tests were run and whether failures were documented.
3. Suggest minimal fixes; do not write code.
4. End with a concise commit message suggestion.
