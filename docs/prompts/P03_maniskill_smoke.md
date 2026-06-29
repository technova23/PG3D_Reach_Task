# Prompt P03 — ManiSkill built-in task smoke

Goal:
Create the first ManiSkill smoke script for a built-in task such as `PickCube-v1` or `PushCube-v1`.

Context to read first:
- `AGENTS.md`
- `docs/project_proposal.html` (DO NOT SKIP. READ IN FULL DETAIL; this is critical to the project.)
- `docs/status.md`
- `docs/milestones.md` M1
- `docs/runbooks/maniskill_setup.md`
- `docs/adr/0001-reach-first-mvp.md`
- `docs/adr/0002-maniskill-primary-simulator.md`

Constraints:
- Start in plan/audit mode only. Do not edit files until the user approves the plan.
- Keep ManiSkill/SAPIEN/Gymnasium imports lazy.
- Do not implement a full ManiSkill adapter yet.
- Script should fail with a clear message if ManiSkill is missing.
- Use non-rendering `obs_mode="state"` by default.
- Do not require Vulkan or visual assets for the default smoke.

Tasks:
1. Add or update `scripts/check_maniskill.py` to import `gymnasium`, import `mani_skill.envs`, create `PickCube-v1` with `obs_mode="state"` and `num_envs=1`, reset with `seed=0`, print observation/action spaces, step one sampled action if safe, and close.
2. Add a Makefile target if consistent with current structure.
3. Update `docs/runbooks/maniskill_setup.md` with exact commands and observed failure modes.
4. Update `docs/status.md` and add a worklog entry.

Done when:
- The script runs or produces actionable setup errors.
- No package import-time dependency on ManiSkill/SAPIEN is introduced.
