from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pg3d.eval import (
    ConditioningBundleStore,
    GuidedProposalTraceWriter,
    action_sha256,
    audit_beam_trace,
    conditioning_window_sha256,
    load_proposal_record,
    verify_replayed_action,
)
from pg3d.world_model import ActionChunk


def _window(offset: float = 0.0) -> list[dict[str, np.ndarray | bool | float]]:
    return [
        {
            "agent_pos": np.asarray([offset, 1.0], dtype=np.float32),
            "point_cloud": np.full((2, 3), offset, dtype=np.float32),
            "success": False,
        },
        {
            "agent_pos": np.asarray([offset + 1.0, 2.0], dtype=np.float32),
            "point_cloud": np.full((2, 3), offset + 1.0, dtype=np.float32),
            "success": True,
        },
    ]


def _chunk() -> ActionChunk:
    return ActionChunk(
        actions=np.arange(16, dtype=np.float32).reshape(2, 8),
        action_mode="abs_joint",
        dt=1.0,
    )


def test_conditioning_bundles_are_deduplicated_and_round_trip(tmp_path: Path) -> None:
    store = ConditioningBundleStore(tmp_path)
    first_hash, first_path = store.store(_window())
    second_hash, second_path = store.store(_window())

    assert first_hash == second_hash == conditioning_window_sha256(_window())
    assert first_path == second_path
    assert len(list(tmp_path.glob("*.npz"))) == 1
    loaded = store.load(first_hash)
    np.testing.assert_array_equal(loaded[1]["agent_pos"], _window()[1]["agent_pos"])


def test_conditioning_bundle_rejects_missing_and_corrupt_data(tmp_path: Path) -> None:
    store = ConditioningBundleStore(tmp_path)
    content_hash, path = store.store(_window())
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        store.load(content_hash)
    with pytest.raises(FileNotFoundError, match="missing"):
        store.load("0" * 64)


def test_proposal_trace_records_hashes_and_replay_diagnostics(tmp_path: Path) -> None:
    writer = GuidedProposalTraceWriter(
        tmp_path,
        run_metadata={"checkpoint_sha256": "a" * 64, "git_revision": "abc"},
    )
    conditioning_hash, _ = writer.store_conditioning(_window())
    chunk = _chunk()
    record = writer.record(
        proposal_id="r0/d1/root/b0",
        purpose="search_expansion",
        replan=0,
        depth=1,
        parent_id="root",
        ancestry=("root",),
        branch_index=0,
        action_chunk=chunk,
        conditioning_hash=conditioning_hash,
        itps_config={"mcmc_steps": 4},
        noise_lineage={"candidate_seed": 7},
        score_data={"feasible": True},
        retained=False,
    )

    assert record["action_sha256"] == action_sha256(chunk)
    assert load_proposal_record(writer.records_path, "r0/d1/root/b0") == record
    assert verify_replayed_action(record, chunk.actions)["exact_action_hash"] is True
    changed = chunk.actions.copy()
    changed[0, 0] += 5e-7
    diagnostic = verify_replayed_action(record, changed)
    assert diagnostic["exact_action_hash"] is False
    assert diagnostic["allclose"] is True


def test_beam_audit_reconstructs_pruning_and_selection() -> None:
    nodes = [
        {
            "node_id": "root/b0",
            "parent_id": "root",
            "feasible": True,
            "score": 0.2,
            "violation_max": 0.0,
            "violation_integral": 0.0,
            "goal_distance": 0.1,
            "retained": True,
            "active_width": 1,
            "observation_hash": "parent-window",
        },
        {
            "node_id": "root/b1",
            "parent_id": "root",
            "feasible": True,
            "score": 0.4,
            "violation_max": 0.0,
            "violation_integral": 0.0,
            "goal_distance": 0.2,
            "retained": False,
            "active_width": 1,
            "observation_hash": "parent-window",
        },
    ]
    result = audit_beam_trace(
        {
            "depths": [{"depth": 1, "nodes": nodes}],
            "selected_node_id": "root/b0",
        }
    )

    assert result["valid"] is True
    nodes[0]["retained"] = False
    nodes[1]["retained"] = True
    assert audit_beam_trace(
        {"depths": [{"depth": 1, "nodes": nodes}], "selected_node_id": "root/b1"}
    )["valid"] is False
