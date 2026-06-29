# Prompt P04 — ManiSkill observation adapter

Goal:
Implement the first `pg3d.envs.maniskill_adapter` observation adapter for ManiSkill reach/built-in task observations.

Context to read first:
- `AGENTS.md`
- `docs/project_proposal.html` (DO NOT SKIP. READ IN FULL DETAIL; this is critical to the project.)
- `docs/architecture/system_architecture.md`
- `docs/milestones.md` M1
- `docs/adr/0001-reach-first-mvp.md`
- `docs/adr/0002-maniskill-primary-simulator.md`
- Existing smoke script from P03.

Constraints:
- Start in plan/audit mode only. Do not edit files until the user approves the plan.
- Policy-visible inputs must be separated from sim GT/eval-only fields.
- Keep ManiSkill/SAPIEN imports lazy.
- Do not train anything.
- Use simple typed dataclasses/Pydantic models.
- Start with state observations; point-cloud/RGB-D/segmentation can be optional if rendering/Vulkan is needed.

Tasks:
1. Inspect existing core data models: `RobotState`, `Observation`, `SimGroundTruth`.
2. Extend them only if needed, then implement adapter functions that convert ManiSkill
   observations/info into these objects.
3. Extract point cloud, optional RGB/features, robot state, robot mask if available, and target position as sim GT/eval context.
4. Add a lightweight observation-save script only if useful; keep rendering/point-cloud modes optional.
5. Add tests for pure data-model/schema logic; simulator tests should skip if ManiSkill is unavailable.
6. Update docs/worklog.

Done when:
- One ManiSkill observation can be adapted or the exact missing simulator/rendering blocker is documented.
- Optional visualization is separate from the default non-rendering smoke. Visualization creates mp4 videos, and can optionally also create rerun-style 3D visualizations (decide if rerun is worth adding as an optional dependency).
- Data shape conventions are documented.
