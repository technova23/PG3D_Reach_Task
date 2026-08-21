# ITPS-beam MVP development protocol v1

This directory freezes the six-fixture development protocol described by
`scripts/itps_beam_mvp_report.md`. The fixtures are original U-shape outputs
`000`, `001`, `006`, `007`, `008`, and `009`. Their outcomes were inspected before
this protocol was written, so they are calibration and mechanism-development data,
not a locked evaluation set.

`development_fixture.json` preserves the original dataset, simulator, policy-seed,
and constraint identities while remapping the selected fixtures to contiguous
protocol output indices. Every referenced constraint is protected by SHA-256.

`protocol.json` is the source of truth for scoring scales, calibration grids, mass
estimation, adaptive search, five repeated lineages, and the 300-step evaluation
horizon. A publishable locked evaluation is intentionally deferred.
