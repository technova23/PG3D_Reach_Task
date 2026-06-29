# Prompt P00 — Bootstrap pg3d repo scaffold

You are working in the root of the `pg3d` repository.

Goal:
Bootstrap the repo scaffold for a sim-only research project on programmatic geometric guidance for 3D diffusion policies. The initial MVP is constrained ManiSkill reach with DP3, a kinematic point-cloud world model, and candidate reranking.

Context to read first:
- `AGENTS.md`
- `docs/project_proposal.html` (DO NOT SKIP. READ IN FULL DETAIL; this is critical to the project.)
- `docs/status.md`
- `docs/research_brief.md`
- `docs/milestones.md`
- `docs/architecture/system_architecture.md`
- `docs/adr/0001-reach-first-mvp.md`
- `docs/adr/0005-world-model-first-class-module.md`
- `docs/adr/0006-python-uv-cuda-stack.md`

Constraints:
- Use `uv`, Python 3.11, and the existing `pyproject.toml` structure.
- Keep this lightweight; do not add heavy simulator or policy dependencies yet.
- Do not edit submodules in this task.
- Do not implement ManiSkill/DP3 logic yet; just make the base repo clean and runnable.

Tasks:
1. Inspect the scaffold and identify missing basic files.
2. Ensure `make test`, `make lint`, and `make smoke` are meaningful.
3. Add any missing package directories with empty `__init__.py` files for the planned architecture.
4. Add lightweight tests only for import/path sanity.
5. Update `docs/status.md` and add a worklog entry.

Done when:
- `uv sync --extra cu129 --group dev` is expected to work on the RTX 5090 workstation.
- `make test`, `make lint`, and `make smoke` pass locally or failures are documented.
- The final response lists changed files and commands run.
