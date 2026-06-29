# Prompt P01 — Private mirrors and submodules

Goal:
Help set up private dependency mirrors and submodules for pg3d, especially DP3.

Context to read first:
- `AGENTS.md`
- `docs/project_proposal.html` (DO NOT SKIP. READ IN FULL DETAIL; this is critical to the project.)
- `docs/runbooks/dependency_mirroring.md`
- `docs/status.md`
- `docs/adr/0003-dp3-p0-policy.md`

Constraints:
- Do not push to remotes unless I explicitly confirm.
- Do not edit submodule code yet.
- Prefer private mirrors over public forks.
- Use SSH URLs for private submodules.

Tasks:
1. Inspect `.gitmodules` and `external/`.
2. If `external/dp3` is missing, prepare the exact commands I should run to create a private DP3 mirror and add it as a submodule.
3. If `external/dp3` exists, verify submodule status and branch.
4. Update `docs/runbooks/dependency_mirroring.md` with any repo-specific names/branches I provide.
5. Update `docs/status.md` and add a worklog entry.

Done when:
- I have exact commands for mirror creation and submodule setup.
- No accidental public fork/private code leak risk is introduced.
