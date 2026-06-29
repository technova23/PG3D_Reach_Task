# Prompt P10 — First constrained reach evaluation scaffold

Goal:
Create the evaluation scaffold for the first MVP: base DP3 vs rejection vs world-model reranking on constrained reach.

Context to read first:
- `AGENTS.md`
- `docs/project_proposal.html` (DO NOT SKIP. READ IN FULL DETAIL; this is critical to the project.)
- `docs/milestones.md` M7
- `docs/research_brief.md`
- `docs/architecture/system_architecture.md`
- Existing ManiSkill adapter, DP3 adapter, world model, constraints, and controllers.

Constraints:
- Do not presume missing project or implementation details. When in doubt, ask the user a
  clarifying question; they are happy to answer as many questions as needed.
- Fixed seeds and small trial counts first.
- Use W&B, but support `WANDB_MODE=offline`.
- Save constraint instance JSON for every episode.
- Do not over-claim reach results; document code-only baseline strength.

Tasks:
1. Implement constrained reach overlay that samples avoid regions near likely nominal paths.
2. Add evaluation runner for base, rejection, and reranking methods.
3. Add metrics: reach success, constraint satisfaction, combined success, final target distance, min clearance, smoothness, candidate feasibility fraction.
4. Add Wilson confidence intervals or bootstrap intervals.
5. Add W&B logging and local JSONL logs.
6. Add video/plot artifact hooks.
7. Update docs/status and worklog.

Done when:
- A tiny fixed-seed evaluation can run end-to-end or blocker is documented.
- Metrics schema is stable.
