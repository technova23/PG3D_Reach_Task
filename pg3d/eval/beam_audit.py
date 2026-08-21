from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any


def audit_beam_trace(trace: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct every fixed-width prune and final choice from serialized telemetry."""
    errors: list[str] = []
    depths = list(trace.get("depths", []))
    for depth in depths:
        nodes = list(depth.get("nodes", []))
        if not nodes:
            errors.append(f"depth {depth.get('depth')} has no expanded-node telemetry")
            continue
        active_width = int(nodes[0]["active_width"])
        expected = _ordered_nodes(nodes)[:active_width]
        actual = [node for node in nodes if bool(node["retained"])]
        if {node["node_id"] for node in expected} != {node["node_id"] for node in actual}:
            errors.append(f"depth {depth.get('depth')} retention cannot be reconstructed")
        _audit_observation_hashes(nodes, errors=errors, depth=int(depth["depth"]))

    reconstructed_selection = None
    if depths:
        final_nodes = [node for node in depths[-1].get("nodes", []) if node.get("retained")]
        if final_nodes:
            reconstructed_selection = _ordered_nodes(final_nodes)[0]["node_id"]
            selected_node_id = trace.get("selected_node_id")
            if selected_node_id is None:
                selected_node_id = dict(trace.get("selected", {})).get("node_id")
            if reconstructed_selection != selected_node_id:
                errors.append("final selection cannot be reconstructed")
    return {
        "valid": not errors,
        "errors": errors,
        "depths_audited": len(depths),
        "reconstructed_selected_node_id": reconstructed_selection,
    }


def _ordered_nodes(nodes: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    feasible = sorted(
        (node for node in nodes if bool(node["feasible"])),
        key=lambda node: (float(node["score"]), str(node["node_id"])),
    )
    infeasible = sorted(
        (node for node in nodes if not bool(node["feasible"])),
        key=lambda node: (
            float(node["violation_max"]),
            float(node["violation_integral"]),
            float("inf") if node.get("goal_distance") is None else float(node["goal_distance"]),
            str(node["node_id"]),
        ),
    )
    return [*feasible, *infeasible]


def _audit_observation_hashes(
    nodes: list[Mapping[str, Any]],
    *,
    errors: list[str],
    depth: int,
) -> None:
    hashes_by_parent: dict[str, set[str]] = defaultdict(set)
    for node in nodes:
        observation_hash = node.get("observation_hash")
        if observation_hash is not None:
            hashes_by_parent[str(node.get("parent_id"))].add(str(observation_hash))
    for parent_id, hashes in hashes_by_parent.items():
        if len(hashes) != 1:
            errors.append(f"depth {depth} siblings of {parent_id} use different windows")
    parent_hashes = [next(iter(hashes)) for hashes in hashes_by_parent.values() if hashes]
    if len(parent_hashes) != len(set(parent_hashes)):
        errors.append(f"depth {depth} different parents share a conditioning window")
