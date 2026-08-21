# ITPS-guided search v1

These method records freeze the matched-compute hybrid pilot. `itps_reranking`
evaluates ten independent guided chunks. `itps_beam` uses depth 3, width 2, and
branch factor 2, producing `2 + 4 + 4 = 10` guided expansions per replan.

Episode identity, simulator/policy seeds, obstacle geometry, and robot constraints
remain owned by the referenced immutable U-shape fixture. Select the predeclared
six-episode pilot subset from that fixture without regenerating constraints or
changing geometry.

The JSON files are audit records for evaluator CLI settings; the evaluator remains
the single execution entrypoint rather than gaining a second configuration runner.
