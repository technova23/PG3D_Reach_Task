"""Merge the training-required schema fields into the mixed zarr's metadata.json.

train_dp3_reach.py / rollout_dp3_reach_policy.py read env_id, env_kwargs,
action_mode, crop, and episodes straight off metadata.json (see
load_reach_metadata + _checkpoint_rollout_metadata_and_specs in
scripts/train_dp3_reach.py). The mixed zarr's metadata.json only has the
provenance/saliency fields build_real_sim_mixed_zarr.py wrote -- this fills in
the rest, copied from the sim source (same env/action-space/crop either way),
without touching the existing sources/fields/how_to_remove_real_data/
point_cloud_saliency keys.

"episodes" needs one entry per zarr episode index (699 real + 5000 sim = 5699,
matching meta/episode_ends) so any code indexing metadata["episodes"] by
episode position stays aligned with the zarr.

Real episodes get real per-episode metadata (source, start_idx/end_idx/length,
target_position, start_tcp_pose) computed directly from the mixed zarr's own
data/* arrays -- NOT looked up from Real-data-zarr-setup/episode_target_mapping.json,
whose 5000 entries don't correspond 1:1 to this dataset's 699 golden episodes
(a filtering/success-extraction step sits between the two, so indices aren't
guaranteed aligned beyond the first few episodes checked by hand). Deriving
from the zarr itself is self-consistent by construction: target_position is
verified constant within each episode before being recorded.

Real episodes deliberately get no "seed" key -- they were never generated from
a sim seed, so there is no seed that reproduces them in the simulator. This is
intentional: rollout code builds its candidate seed list as
`[episode["seed"] for episode in metadata["episodes"] if "seed" in episode]`
(scripts/train_dp3_reach.py:930), so entries without a "seed" key are correctly
skipped rather than feeding a fake/meaningless seed into env.reset(). Sim
episodes keep their full original per-episode metadata (goal_pose, seed,
trajectory_family, ...) unchanged.

Only metadata.json is rewritten; the zarr's data/meta arrays are untouched.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import zarr

SIM = Path("/scratch2/skills/pg3d_xarm7_gripper_reach_final.zarr")
MIXED = Path("/scratch2/skills/pg3d_real_sim_mixed.zarr")

COPIED_FIELDS = ["env_id", "env_kwargs", "action_mode", "crop"]


def _real_episode_metadata(real_episodes: int) -> list[dict]:
    root = zarr.open_group(str(MIXED), mode="r")
    data = root["data"]
    episode_ends = np.asarray(root["meta"]["episode_ends"][:real_episodes], dtype=np.int64)
    starts = np.concatenate([np.asarray([0], dtype=np.int64), episode_ends[:-1]])
    target_position = np.asarray(data["target_position"][: int(episode_ends[-1])], dtype=np.float32)
    tcp_pose = np.asarray(data["tcp_pose"][: int(episode_ends[-1])], dtype=np.float32)

    entries = []
    for idx, (start, end) in enumerate(zip(starts, episode_ends)):
        start, end = int(start), int(end)
        segment = target_position[start:end]
        if not np.allclose(segment, segment[0]):
            raise ValueError(
                f"real episode {idx} has non-constant target_position "
                f"(min={segment.min(axis=0)}, max={segment.max(axis=0)}) -- "
                "expected one fixed goal per episode"
            )
        entries.append(
            {
                "source": "real",
                "start_idx": start,
                "end_idx": end,
                "length": end - start,
                "target_position": segment[0].tolist(),
                "start_tcp_pose": tcp_pose[start].tolist(),
            }
        )
    return entries


def main() -> None:
    sim_metadata = json.loads((SIM / "metadata.json").read_text())
    mixed_metadata = json.loads((MIXED / "metadata.json").read_text())

    real_source, sim_source = mixed_metadata["sources"]
    assert real_source["name"] == "real" and sim_source["name"] == "sim"
    real_episodes = real_source["episode_range"][1] - real_source["episode_range"][0]
    sim_episodes = sim_source["episode_range"][1] - sim_source["episode_range"][0]

    sim_episode_list = sim_metadata["episodes"]
    if len(sim_episode_list) != sim_episodes:
        raise ValueError(
            f"sim metadata has {len(sim_episode_list)} episodes but provenance "
            f"says the mixed zarr's sim portion has {sim_episodes} episodes"
        )

    for field in COPIED_FIELDS:
        if field not in sim_metadata:
            raise KeyError(f"sim metadata.json is missing expected field {field!r}")
        mixed_metadata[field] = sim_metadata[field]

    real_episode_list = _real_episode_metadata(real_episodes)
    mixed_metadata["episodes"] = real_episode_list + sim_episode_list

    (MIXED / "metadata.json").write_text(json.dumps(mixed_metadata, indent=2))
    print(f"wrote {len(COPIED_FIELDS)} schema fields + episodes ({len(mixed_metadata['episodes'])} entries)")
    print(f"  real episodes (derived from zarr, no 'seed' key -- excluded from rollout seed pool): {real_episodes}")
    print(f"  sim episodes copied verbatim: {sim_episodes}")


if __name__ == "__main__":
    main()
