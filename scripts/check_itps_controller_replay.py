#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare a one-episode guided controller rerun with its original artifacts."
    )
    parser.add_argument("original_dir", type=Path)
    parser.add_argument("replay_dir", type=Path)
    parser.add_argument("--method", required=True)
    parser.add_argument("--episode", type=int, required=True)
    args = parser.parse_args(argv)

    original_metrics = _metric(args.original_dir, args.method, args.episode)
    replay_metrics = _metric(args.replay_dir, args.method, args.episode)
    original_decisions = _decisions(args.original_dir, args.method, args.episode)
    replay_decisions = _decisions(args.replay_dir, args.method, args.episode)
    selection_match = [row["selected_action_sha256"] for row in original_decisions] == [
        row["selected_action_sha256"] for row in replay_decisions
    ]
    executed_match = [row["executed_prefix_sha256"] for row in original_decisions] == [
        row["executed_prefix_sha256"] for row in replay_decisions
    ]
    metric_keys = (
        "steps",
        "replans",
        "termination_reason",
        "stable_goal_reached",
        "stable_combined_success",
        "min_target_distance",
        "final_target_distance",
        "min_clearance",
    )
    metric_mismatches = {
        key: [original_metrics.get(key), replay_metrics.get(key)]
        for key in metric_keys
        if original_metrics.get(key) != replay_metrics.get(key)
    }
    result = {
        "valid": selection_match and executed_match and not metric_mismatches,
        "selection_hashes_match": selection_match,
        "executed_prefix_hashes_match": executed_match,
        "decision_count_original": len(original_decisions),
        "decision_count_replay": len(replay_decisions),
        "metric_mismatches": metric_mismatches,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _metric(root: Path, method: str, episode: int) -> dict[str, Any]:
    matches = [
        row
        for row in _jsonl(root / "metrics.jsonl")
        if row.get("method") == method and int(row.get("episode", -1)) == episode
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one metrics row for {method} episode {episode}")
    return matches[0]


def _decisions(root: Path, method: str, episode: int) -> list[dict[str, Any]]:
    return [
        row
        for row in _jsonl(root / "decisions.jsonl")
        if row.get("method") == method and int(row.get("episode", -1)) == episode
    ]


if __name__ == "__main__":
    raise SystemExit(main())
