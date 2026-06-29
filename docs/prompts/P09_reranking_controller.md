# Prompt P09 — Candidate rejection and reranking controller

Goal:
Implement the first composition operators for constrained reach: rejection/filtering and world-model reranking.

Context to read first:
- `AGENTS.md`
- `docs/project_proposal.html` (DO NOT SKIP. READ IN FULL DETAIL; this is critical to the project.)
- `docs/milestones.md` M6
- `docs/architecture/system_architecture.md`
- `docs/research_brief.md`
- Existing world model and constraints modules.

Constraints:
- Do not presume missing project or implementation details. When in doubt, ask the user a
  clarifying question; they are happy to answer as many questions as needed.
- No energy guidance yet.
- K should be configurable, starting with 16 and fallback attempts at 32/64.
- Keep policy interface generic; DP3 adapter can be plugged in later.
- Include rich diagnostics for candidate costs.

Tasks:
1. Define `Policy` protocol with `sample_action_chunks` and optional `score_surrogate`.
2. Implement `BaseController`, `RejectionController`, `RerankingController`.
3. Implement scoring: goal distance, clearance, smoothness, sample-consensus deviation.
4. Add diagnostic objects/logging for per-candidate cost breakdowns.
5. Add tests with a fake policy and fake world model.
6. Update docs/status and worklog.

Done when:
- Tests show the reranker selects the safe candidate in a synthetic reach scenario.
- API is ready to connect to DP3/ManiSkill.
