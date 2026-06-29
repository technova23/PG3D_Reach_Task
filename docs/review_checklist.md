# Review checklist

Use this checklist for Codex self-review and human review.

## Scope

- Does the diff implement the requested milestone slice and avoid unrelated changes?
- Are submodule edits absent unless explicitly requested?
- Did the agent avoid long training jobs unless explicitly requested?

## Correctness

- Are array shapes documented and asserted where useful?
- Are action chunk conventions explicit?
- Are simulator GT fields separated from policy-visible inputs?
- Are lazy imports used for ManiSkill/SAPIEN/DP3?

## Research hygiene

- Are configs and seeds saved for replay?
- Are W&B logs or structured JSON logs included where relevant?
- Are constraint instances serializable?
- Are metrics aligned with task success, constraint satisfaction, and combined success?

## Testing

- Are high-leverage unit tests included for pure geometry/serialization code?
- Are simulator-dependent tests skippable if ManiSkill/SAPIEN or rendering support is unavailable?
- Were `make test` and `make lint` run, or was the reason documented?

## Docs

- Does `docs/status.md` reflect the new state?
- Were run commands updated in `docs/runbooks/commands.md`?
- Is a work log entry present for nontrivial work?
- Is a new ADR needed?

## Cleanliness

- No dead code or unused scripts.
- No notebooks as source of truth.
- No large generated outputs committed.
- No hard-coded personal paths unless clearly marked as examples.
