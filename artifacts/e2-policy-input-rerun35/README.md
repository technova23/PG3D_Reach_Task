# Exact DP3 policy-input point clouds

These captures use dataset episode `0` / simulator seed `48572821` and the locked
100k EMA checkpoint. Each family contains the initial observation and one executed
step, so every neutral bundle has shape `[2, 1024, 3]`.

| Family | Obstacle points in initial exact tensor | Quota |
| --- | ---: | ---: |
| rotated box | 32 | 32 |
| tall carton | 50 | 32 |
| vertical cylinder | 32 | 32 |
| open cabinet | 92 | 64 |

For each family, open:

- `rerun/base/episode_000.rrd` with Rerun 0.35;
- `rerun/base/episode_000.policy_input.npz` with any NumPy-compatible viewer;
- `rerun/base/episode_000.policy_input.json` for schema, constraint, replan, and
  writer metadata.

`policy_input/point_cloud` is the complete fixed-size tensor sent to DP3, including
the ordered goal-marker tail. The semantic child entities are convenience subsets
of that same tensor and are not additional policy inputs.

Example:

```bash
.venv-rerun35/bin/rerun \
  artifacts/e2-policy-input-rerun35/cabinet/rerun/base/episode_000.rrd
```

The `.npz` arrays are `point_cloud`, `colors`, `valid_mask`, `robot_mask`,
`obstacle_mask`, `scene_mask`, `goal_mask`, `target_position`, `tcp_position`, and
`tcp_clearance`.
