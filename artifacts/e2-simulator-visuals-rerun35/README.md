# Realistic obstacles: simulator and exact policy input

This suite shows each available realistic obstacle in the actual ManiSkill control
environment and pairs it with the exact point-cloud tensor supplied to DP3.

All captures use dataset episode `0`, simulator seed `48572821`, and the locked 100k
EMA checkpoint. The task horizon is 150 steps. A rollout stops after reaching the
goal and completing the required 16-step stable hold; a failure records all 150
steps. Videos contain the reset frame plus one frame per executed step, are 512×512,
and play at the true 20 Hz control rate.

| Family | Outcome | Steps | Video | Exact point-cloud timeline |
| --- | --- | ---: | --- | --- |
| rotated box | stable success | 123 | `box/videos/base/episode_000.mp4` | `box/rerun/base/episode_000.rrd` |
| tall carton | 150-step timeout | 150 | `carton/videos/base/episode_000.mp4` | `carton/rerun/base/episode_000.rrd` |
| vertical cylinder | stable success | 113 | `cylinder/videos/base/episode_000.mp4` | `cylinder/rerun/base/episode_000.rrd` |
| open cabinet | stable success | 115 | `cabinet/videos/base/episode_000.mp4` | `cabinet/rerun/base/episode_000.rrd` |

Each Rerun directory also contains `episode_000.policy_input.npz` and
`episode_000.policy_input.json`. The NPZ point-cloud shape is
`[executed_steps + 1, 1024, 3]`.
Open an RRD with the isolated viewer, for example:

```bash
.venv-rerun35/bin/rerun \
  artifacts/e2-simulator-visuals-rerun35/cabinet/rerun/base/episode_000.rrd
```

The simulator videos and point-cloud timelines are synchronized by control step.
`policy_input/point_cloud` is the complete tensor sent to the policy; its semantic
child entities are visualization-only subsets.

The objects are tabletop-supported: their bottom is at `z=0` and their height is
resolved before actor construction so the top reaches above the selected direct-path
point. For this episode all tops are `z=0.4431` and the path point is `z=0.4231`.
The cabinet is aligned by its back panel so its open interior does not create a
false collision-free placement.
