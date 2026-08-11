# Frozen 75 cm candidate-midpath suite

This directory is the canonical source for future **candidate-midpath pilot,
method-comparison, and horizon-ablation runs**. It freezes the ten clearance-safe
75 cm carton placements accepted by
`artifacts/candidate-midpath-clearance-safe-100k-pilot`.

It does not replace the locked 50-episode definitive E3 test split in
`configs/eval/e3_test_episode_indices.txt`.

## Frozen identity

| Output episode | Dataset episode | Simulator seed | Source-pool index | Obstacle center XYZ (m) | Initial robot clearance (m) |
|---:|---:|---:|---:|---|---:|
| 0 | 305 | 48572874 | 2 | `[-0.19806933403015137, 0.09573230147361755, 0.375]` | 0.12291646003723145 |
| 1 | 317 | 48572879 | 3 | `[-0.07605887949466705, -0.0125518087297678, 0.375]` | 0.08478546142578125 |
| 2 | 974 | 48573006 | 10 | `[-0.18901915848255157, -0.057136472314596176, 0.375]` | 0.07897913455963135 |
| 3 | 986 | 48573007 | 11 | `[-0.16298940777778625, -0.01870773732662201, 0.375]` | 0.057202257215976715 |
| 4 | 1010 | 48573016 | 13 | `[-0.1029917299747467, -0.06830453872680664, 0.375]` | 0.18342755734920502 |
| 5 | 1034 | 48573020 | 15 | `[-0.2226373851299286, -0.0004138017538934946, 0.375]` | 0.08854831755161285 |
| 6 | 1069 | 48573030 | 18 | `[-0.1315213441848755, -0.15107932686805725, 0.375]` | 0.0694565698504448 |
| 7 | 1117 | 48573036 | 22 | `[-0.23593541979789734, 0.20035535097122192, 0.375]` | 0.07541734725236893 |
| 8 | 1129 | 48573038 | 23 | `[-0.07847712934017181, -0.04634665697813034, 0.375]` | 0.06874746084213257 |
| 9 | 1138 | 48573041 | 24 | `[-0.0719042643904686, 0.13494063913822174, 0.375]` | 0.11414743959903717 |

All obstacles are grounded axis-aligned cartons with half-extents
`[0.055, 0.08, 0.375]` m, full height 0.75 m, yaw 0, and bottom Z 0.

## Required use

- Use `episode_indices.txt`; do not substitute the original pilot list.
- Load the matching precomputed constraint directory; do not regenerate
  candidate-midpath placements.
- Use `constraints/eef` for EEF-guided base/rejection/reranking/ITPS runs.
- Use `constraints/robot` for exact whole-robot rejection/reranking runs.
- Keep evaluation seed 0 to retain the recorded policy-seed mapping.
- A physical placement or episode change requires a new fixture version.

The EEF and robot constraint files have identical obstacle geometry. They differ
only in the body used by planner guidance. Executed evaluations should continue
to report whole-robot clearance and terminate on geometric or PhysX contact.
