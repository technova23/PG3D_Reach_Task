#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize one ITPS-beam MVP stage.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite stage report: {args.output}")
    jobs = _load_jobs(args.input_root)
    summaries = {job_id: _summarize_job(rows) for job_id, rows in jobs.items()}
    ranking = sorted(summaries, key=lambda job_id: _ranking_key(job_id, summaries[job_id]))
    payload = {
        "schema_version": "pg3d.itps_beam_stage_report.v1",
        "classification": "six_fixture_development_evidence",
        "stage": args.stage,
        "jobs": summaries,
        "predeclared_ranking": ranking,
        "selected_config_id": ranking[0] if ranking else None,
        "paired_outcomes": _paired_outcomes(jobs),
        "component_gate": _component_gate(args.stage, summaries, ranking),
        "disclaimer": (
            "Development evidence only; these inspected six fixtures are not a locked test set "
            "and do not establish statistically broad benchmark performance."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _load_jobs(root: Path) -> dict[str, list[dict[str, Any]]]:
    jobs = {}
    for path in sorted(root.glob("*/metrics.jsonl")):
        jobs[path.parent.name] = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        ]
    if not jobs:
        for path in sorted(root.glob("*/stage_report.json")):
            report = json.loads(path.read_text(encoding="utf-8"))
            for job_id, summary in report.get("jobs", {}).items():
                jobs[f"{path.parent.name}/{job_id}"] = [dict(summary)]
    return jobs


def _summarize_job(rows: list[dict[str, Any]]) -> dict[str, Any]:
    episode_rows = [row for row in rows if "episode" in row]
    if not episode_rows:
        return rows[0] if rows else {}
    count = len(episode_rows)
    minimum_distances = [
        float(row.get("min_target_distance", row.get("minimum_target_distance", float("inf"))))
        for row in episode_rows
    ]
    final_distances = [float(row["final_target_distance"]) for row in episode_rows]
    stable = [bool(row["stable_combined_success"]) for row in episode_rows]
    safety = [bool(row["constraint_satisfied"]) for row in episode_rows]
    goal_entry = [row.get("first_goal_entry_step") is not None for row in episode_rows]
    proposal_summary = _proposal_summary(episode_rows)
    return {
        "episodes": count,
        "stable_combined_success_rate": sum(stable) / count,
        "safety_rate": sum(safety) / count,
        "goal_entry_rate": sum(goal_entry) / count,
        "median_minimum_target_distance_m": statistics.median(minimum_distances),
        "mean_final_target_distance_m": statistics.fmean(final_distances),
        "mean_denoiser_evaluations": statistics.fmean(
            float(row.get("denoiser_evaluations", 0)) for row in episode_rows
        ),
        "mean_guided_proposals": statistics.fmean(
            float(row.get("guided_proposals_used", 0)) for row in episode_rows
        ),
        "mean_action_selection_seconds": statistics.fmean(
            float(row.get("action_selection_time_total", 0.0)) for row in episode_rows
        ),
        "optimistic_clearance_error_p95_m": _max_optional(
            row.get("optimistic_clearance_error_p95_m") for row in episode_rows
        ),
        "false_safe_count": _trace_false_safe_count(episode_rows),
        "fallback_count": sum(int(row.get("fallback_count", 0)) for row in episode_rows),
        "worst_fixture_stable_success": min(map(int, stable)),
        "method_config": episode_rows[0].get("method_config"),
        "adaptive_weight_trajectory": [
            update for row in episode_rows for update in row.get("adaptive_weight_updates", [])
        ],
        "posterior_calibration": proposal_summary,
        "pruning": {
            "expanded_nodes": sum(int(row.get("beam_expanded_nodes", 0)) for row in episode_rows),
            "retained_nodes": sum(int(row.get("beam_retained_nodes", 0)) for row in episode_rows),
            "fallbacks": sum(int(row.get("fallback_count", 0)) for row in episode_rows),
        },
        "predicted_vs_executed": {
            "mean_joint_error_rad": _mean_optional(
                row.get("imagined_joint_error_mean_rad") for row in episode_rows
            ),
            "mean_tcp_error_m": _mean_optional(
                row.get("imagined_tcp_error_mean_m") for row in episode_rows
            ),
            "optimistic_clearance_error_p95_m": _max_optional(
                row.get("optimistic_clearance_error_p95_m") for row in episode_rows
            ),
            "false_safe_count": _trace_false_safe_count(episode_rows),
        },
    }


def _ranking_key(job_id: str, summary: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -float(summary.get("stable_combined_success_rate", 0.0)),
        -float(summary.get("safety_rate", 0.0)),
        -float(summary.get("goal_entry_rate", 0.0)),
        float(summary.get("median_minimum_target_distance_m", float("inf"))),
        float(summary.get("mean_final_target_distance_m", float("inf"))),
        float(summary.get("mean_denoiser_evaluations", float("inf"))),
        job_id,
    )


def _paired_outcomes(jobs: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    by_fixture: dict[tuple[int, int], dict[str, Any]] = defaultdict(dict)
    for job_id, rows in jobs.items():
        for row in rows:
            if "episode" not in row:
                continue
            key = (int(row.get("policy_seed", 0)), int(row["episode"]))
            by_fixture[key][job_id] = {
                "stable_combined_success": row["stable_combined_success"],
                "constraint_satisfied": row["constraint_satisfied"],
                "min_target_distance": row.get("min_target_distance"),
                "final_target_distance": row["final_target_distance"],
                "denoiser_evaluations": row.get("denoiser_evaluations"),
            }
    return [
        {"policy_seed": key[0], "episode": key[1], "jobs": values}
        for key, values in sorted(by_fixture.items())
    ]


def _component_gate(
    stage: str,
    summaries: dict[str, dict[str, Any]],
    ranking: list[str],
) -> dict[str, Any]:
    if not ranking:
        return {"retained": False, "reason": "no completed jobs"}
    selected = ranking[0]
    if stage == "mass_ablation" and "no_mass" in summaries:
        baseline = summaries["no_mass"]
        candidate = summaries[selected]
        improved = any(
            candidate[key] > baseline[key]
            for key in (
                "stable_combined_success_rate",
                "safety_rate",
                "worst_fixture_stable_success",
            )
        )
        return {"retained": selected != "no_mass" and improved, "selected": selected}
    if stage in {"adaptive_weights", "adaptive_search"}:
        baseline_id = "fixed_mass" if stage == "adaptive_weights" else "fixed_compute"
        if baseline_id in summaries:
            return {
                "retained": _ranking_key(selected, summaries[selected])
                < _ranking_key(baseline_id, summaries[baseline_id]),
                "selected": selected,
                "baseline": baseline_id,
            }
    return {"retained": True, "selected": selected}


def _trace_false_safe_count(rows: list[dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        path = row.get("imagined_execution_trace")
        if path is None or not Path(path).is_file():
            continue
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            count += int(json.loads(line)["feasibility_confusion"]["false_safe"])
    return count


def _max_optional(values: Any) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return max(finite) if finite else None


def _mean_optional(values: Any) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return statistics.fmean(finite) if finite else None


def _proposal_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    records = []
    for trace in sorted({str(row["proposal_trace"]) for row in rows if row.get("proposal_trace")}):
        path = Path(trace)
        if path.is_file():
            records.extend(
                json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            )
    mass = [
        record["score_data"]
        for record in records
        if record.get("purpose") == "mass_probe" and record.get("score_data") is not None
    ]
    return {
        "proposal_records": len(records),
        "mass_probe_records": len(mass),
        "viable_mass_probes": sum(bool(item["feasible"]) for item in mass),
        "mean_mass_posterior_lcb": _mean_optional(item.get("mass_posterior_lcb") for item in mass),
    }


if __name__ == "__main__":
    raise SystemExit(main())
