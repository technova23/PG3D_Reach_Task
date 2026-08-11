# ADR 0010 — Freeze the 75 cm candidate-midpath suite

Date: 2026-08-11

## Status

Accepted

## Context

Successive candidate-midpath pilots used different local episode numbering,
height-dependent clearance filtering, and regenerated obstacle placements. That
made aggregate results useful for debugging but prevented clean paired comparisons
across methods and planning horizons. The 75 cm clearance-safe pilot already
identified ten placements with at least 2 cm initial whole-robot clearance and
recorded their dataset indices, simulator seeds, placement seeds, and exact box
geometry.

## Decision

Freeze `configs/eval/e3_candidate_midpath_75cm_frozen_v1` as the canonical suite
for future candidate-midpath pilot, method-comparison, and planning-horizon
ablations.

The suite fixes:

- output order `0--9` and dataset episode indices
  `305, 317, 974, 986, 1010, 1034, 1069, 1117, 1129, 1138`;
- simulator seeds, policy seeds, source-pool indices, and placement-policy seeds;
- one grounded axis-aligned carton per episode with half-extents
  `[0.055, 0.08, 0.375]` m, yaw zero, and full height 0.75 m;
- the exact XYZ center of every obstacle;
- EEF-target and robot-target constraint variants with identical physical geometry.

Future runs must load the frozen episode-index and constraint files. They must not
regenerate candidate-midpath placements or invoke clearance-safe replacement
selection. Any change to episode identity or physical obstacle geometry requires a
new fixture version and ADR amendment; v1 remains immutable.

This suite is tuning/ablation data. It does not replace or consume the locked
50-episode definitive E3 test split.

## Consequences

- Method and horizon results can be paired by output index, dataset episode,
  simulator seed, policy seed, and obstacle geometry.
- Local `episode_XXX` names are no longer ambiguous within this suite; the fixture
  manifest remains the authoritative mapping to dataset episodes.
- EEF-only methods such as ITPS use `constraints/eef`; exact whole-robot methods use
  `constraints/robot`. Executed whole-robot safety grading remains mandatory.
- The fixed 75 cm obstacle is deliberately difficult. Easier-height studies require
  a separately versioned fixture rather than modifying this one.

## Alternatives considered

- Continue filtering independently for every obstacle height: guarantees valid
  starts but changes the evaluated population and breaks pairing.
- Reuse the original ten pilot episodes: preserves early numbering but includes
  placements that fail the initial-clearance gate.
- Reuse ignored artifact paths directly: operationally convenient but not durable,
  because `artifacts/` is not version-controlled.
