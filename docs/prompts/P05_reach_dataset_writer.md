# Prompt P05 — Reach demo generation and dataset writer

Goal:
Generate/replay ManiSkill reach demonstrations and write a dataset suitable for the pg3d-native DP3 policy.

Context to read first:
- `AGENTS.md`
- `docs/project_proposal.html` (DO NOT SKIP. READ IN FULL DETAIL; this is critical to the project.)
- `docs/milestones.md` M2
- `docs/architecture/system_architecture.md`
- `docs/adr/0002-maniskill-primary-simulator.md`
- `docs/adr/0004-action-representation.md`
- DP3 data loading examples in `external/dp3` if available.

Constraints:
- Start in plan/audit mode only. Do not edit files until the user approves the plan.
- Start with small smoke datasets: 3-5 demos.
- Do not run large generation jobs. It's the user that'll run such jobs.
- Support absolute joint target chunks first and delta joint chunks as fallback.
- Save enough metadata for replay: seed, task variant, action mode, camera config, submodule commit hashes.
- We should clip the pointcloud so we only capture a small area around the robot workspace. The default pointcloud capture in maniskill has several far-off, outlier points.

Tasks:
1. Inspect DP3 dataset expectations in the fork/submodule.
2. Propose the minimal dataset schema for ManiSkill reach.
3. Implement a writer for observation/action sequences.
4. Implement a replay sanity script.
5. Add shape/schema tests independent of ManiSkill where possible.
6. Update docs/status, runbooks, and worklog.

Done when:
- A smoke dataset can be generated or the exact missing simulator blocker is documented.
- The dataset schema is documented clearly enough for DP3 integration.
