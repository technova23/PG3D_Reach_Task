# Prompt P08 — Constraint programs v0

Goal:
Implement handwritten constraint objects for reach: `AvoidRegion(target="eef")`, smoothness, and JSON serialization.

Context to read first:
- `AGENTS.md`
- `docs/project_proposal.html` (DO NOT SKIP. READ IN FULL DETAIL; this is critical to the project.)
- `docs/milestones.md` M5
- `docs/architecture/system_architecture.md`
- `docs/research_brief.md`

Constraints:
- Do not presume missing project or implementation details. When in doubt, ask the user a
  clarifying question; they are happy to answer as many questions as needed.
- Python object API first; JSON serialization for replay.
- Simple geometries only: sphere and box initially.
- No full robot-body collision or IK yet.
- Tests should be pure and fast.

Tasks:
1. Add geometry primitives.
2. Add `Constraint` base/protocol and `SceneContext`.
3. Add `AvoidRegion` cost over EEF path.
4. Add smoothness cost.
5. Add serialization/deserialization registry.
6. Add synthetic tests for costs and serialization.
7. Update docs/status and worklog.

Done when:
- Constraint costs behave correctly on trajectories that pass inside/outside regions.
- Constraint configs can be saved and reloaded.
