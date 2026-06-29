# Prompt R — Debug a failure

Goal:
Debug a specific failure without making broad unrelated changes.

Failure:
Paste the exact command, output, traceback, and relevant environment details here.

Read first:
- `AGENTS.md`
- `docs/project_proposal.html` (DO NOT SKIP. READ IN FULL DETAIL; this is critical to the project.)
- `docs/runbooks/commands.md`
- relevant ADRs and modules

Constraints:
- Reproduce the failure first if possible.
- Make the smallest fix that explains the failure.
- Do not change Python/Torch/ManiSkill/SAPIEN versions without documenting why.
- Update the relevant runbook if this is an environment/setup issue.

Tasks:
1. State the likely root causes ranked by probability.
2. Run targeted diagnostics.
3. Implement the minimal fix.
4. Run the failing command again.
5. Update docs/worklog with the failure and fix.
