# Realistic obstacles: simulator and exact policy input

This suite shows each available realistic obstacle in the actual ManiSkill control
environment and pairs it with the exact point-cloud tensor supplied to DP3.

All captures use dataset episode `0`, simulator seed `48572821`, the locked 100k EMA
checkpoint, and 40 executed control steps. Videos contain 41 frames (reset plus one
frame per step), are 512×512, and play at the true 20 Hz control rate.

| Family | Video | Exact point-cloud timeline |
| --- | --- | --- |
| rotated box | `box/videos/base/episode_000.mp4` | `box/rerun/base/episode_000.rrd` |
| tall carton | `carton/videos/base/episode_000.mp4` | `carton/rerun/base/episode_000.rrd` |
| vertical cylinder | `cylinder/videos/base/episode_000.mp4` | `cylinder/rerun/base/episode_000.rrd` |
| open cabinet | `cabinet/videos/base/episode_000.mp4` | `cabinet/rerun/base/episode_000.rrd` |

Each Rerun directory also contains `episode_000.policy_input.npz` and
`episode_000.policy_input.json`. The NPZ point-cloud shape is `[41, 1024, 3]`.
Open an RRD with the isolated viewer, for example:

```bash
.venv-rerun35/bin/rerun \
  artifacts/e2-simulator-visuals-rerun35/cabinet/rerun/base/episode_000.rrd
```

The simulator videos and point-cloud timelines are synchronized by control step.
`policy_input/point_cloud` is the complete tensor sent to the policy; its semantic
child entities are visualization-only subsets.
