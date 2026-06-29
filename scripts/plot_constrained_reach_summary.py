from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from pg3d.eval import success_rate_ci_rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = success_rate_ci_rows(summary, methods=args.methods)
    if not rows:
        print("No method summaries found to plot.", file=sys.stderr)
        return 1
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(
            f"Failed to import matplotlib for plotting: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2

    output = args.output or args.summary.parent / "plots" / "comparative_success_ci.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    plots_dir = output.parent

    methods = _unique([str(row["method"]) for row in rows])
    metrics = _unique([str(row["metric"]) for row in rows])
    labels = {
        str(row["metric"]): str(row["label"])
        for row in rows
    }
    by_key = {
        (str(row["method"]), str(row["metric"])): row
        for row in rows
    }

    x = np.arange(len(methods))
    width = min(0.8 / max(len(metrics), 1), 0.25)
    fig, ax = plt.subplots(figsize=args.figsize)
    for idx, metric in enumerate(metrics):
        values: list[float] = []
        yerr: list[list[float]] = [[], []]
        for method in methods:
            row = by_key[(method, metric)]
            values.append(float(row["rate"]))
            yerr[0].append(float(row["err_low"]))
            yerr[1].append(float(row["err_high"]))
        ax.bar(
            x + (idx - (len(metrics) - 1) / 2.0) * width,
            values,
            width,
            label=labels[metric],
            yerr=yerr,
            capsize=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Rate with Wilson 95% CI")
    ax.set_title(args.title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=args.dpi)
    plt.close(fig)
    print(output)

    episode_rows = _episode_rows(summary, methods=args.methods)
    if episode_rows:
        outcome_rows = _outcome_summary_rows(episode_rows, methods=methods)
        counts_output = args.counts_output or plots_dir / "outcome_counts.png"
        distance_output = args.distance_output or plots_dir / "failure_distances.png"
        report_output = args.report_output or plots_dir / "outcome_report.json"
        csv_output = args.csv_output or plots_dir / "outcome_report.csv"

        _write_outcome_count_plot(
            counts_output,
            outcome_rows=outcome_rows,
            methods=methods,
            dpi=args.dpi,
            figsize=tuple(args.figsize),
        )
        print(counts_output)
        _write_failure_distance_plot(
            distance_output,
            outcome_rows=outcome_rows,
            methods=methods,
            dpi=args.dpi,
            figsize=tuple(args.figsize),
        )
        print(distance_output)
        _write_json_report(report_output, outcome_rows)
        print(report_output)
        _write_csv_report(csv_output, outcome_rows)
        print(csv_output)
        _print_outcome_report(outcome_rows)
    else:
        print("Summary has no per-episode rows; skipped outcome-count report.", file=sys.stderr)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot constrained-reach success rates with Wilson confidence intervals."
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--title", default="Constrained reach validation comparison")
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--figsize", type=float, nargs=2, default=(10.0, 5.0))
    parser.add_argument("--counts-output", type=Path, default=None)
    parser.add_argument("--distance-output", type=Path, default=None)
    parser.add_argument("--report-output", type=Path, default=None)
    parser.add_argument("--csv-output", type=Path, default=None)
    return parser.parse_args(argv)


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _episode_rows(
    summary: dict[str, Any],
    *,
    methods: list[str] | None,
) -> list[dict[str, Any]]:
    rows = summary.get("episodes", [])
    if not isinstance(rows, list):
        return []
    if methods is None:
        return [row for row in rows if isinstance(row, dict)]
    allowed = set(methods)
    return [
        row
        for row in rows
        if isinstance(row, dict) and str(row.get("method")) in allowed
    ]


def _outcome_summary_rows(
    rows: list[dict[str, Any]],
    *,
    methods: list[str],
) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in methods}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)

    result: list[dict[str, Any]] = []
    for method in methods:
        method_rows = by_method.get(method, [])
        total = len(method_rows)
        reached = [row for row in method_rows if bool(row.get("reach_success"))]
        avoided = [row for row in method_rows if bool(row.get("constraint_satisfied"))]
        failed_goal = [row for row in method_rows if not bool(row.get("reach_success"))]
        failed_constraint = [
            row for row in method_rows if not bool(row.get("constraint_satisfied"))
        ]
        both_success = [
            row
            for row in method_rows
            if bool(row.get("reach_success")) and bool(row.get("constraint_satisfied"))
        ]
        reached_but_violated = [
            row
            for row in method_rows
            if bool(row.get("reach_success")) and not bool(row.get("constraint_satisfied"))
        ]
        missed_but_avoided = [
            row
            for row in method_rows
            if not bool(row.get("reach_success")) and bool(row.get("constraint_satisfied"))
        ]
        failed_both = [
            row
            for row in method_rows
            if not bool(row.get("reach_success")) and not bool(row.get("constraint_satisfied"))
        ]
        goal_distances = [
            value
            for row in failed_goal
            if (value := _optional_float(row.get("final_target_distance"))) is not None
        ]
        violation_depths = [
            -clearance
            for row in failed_constraint
            if (clearance := _optional_float(row.get("min_clearance"))) is not None
        ]
        violation_depths = [max(0.0, value) for value in violation_depths]
        result.append(
            {
                "method": method,
                "episodes": total,
                "goal_reached": len(reached),
                "goal_not_reached": len(failed_goal),
                "constraint_avoided": len(avoided),
                "constraint_not_avoided": len(failed_constraint),
                "both_success": len(both_success),
                "reached_but_constraint_failed": len(reached_but_violated),
                "goal_failed_but_constraint_avoided": len(missed_but_avoided),
                "both_failed": len(failed_both),
                "goal_failure_final_distance": _stats(goal_distances),
                "constraint_failure_violation_depth": _stats(violation_depths),
            }
        )
    return result


def _write_outcome_count_plot(
    output: Path,
    *,
    outcome_rows: list[dict[str, Any]],
    methods: list[str],
    dpi: int,
    figsize: tuple[float, float],
) -> None:
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    by_method = {str(row["method"]): row for row in outcome_rows}
    categories = [
        ("both_success", "Reached + avoided"),
        ("reached_but_constraint_failed", "Reached, violated"),
        ("goal_failed_but_constraint_avoided", "Missed, avoided"),
        ("both_failed", "Missed + violated"),
    ]
    x = np.arange(len(methods))
    bottoms = np.zeros(len(methods), dtype=np.float32)
    fig, ax = plt.subplots(figsize=figsize)
    for key, label in categories:
        values = np.asarray(
            [float(by_method.get(method, {}).get(key, 0)) for method in methods],
            dtype=np.float32,
        )
        ax.bar(x, values, bottom=bottoms, label=label)
        bottoms += values
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Episode count")
    ax.set_title("Constrained reach outcome counts")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def _write_failure_distance_plot(
    output: Path,
    *,
    outcome_rows: list[dict[str, Any]],
    methods: list[str],
    dpi: int,
    figsize: tuple[float, float],
) -> None:
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    by_method = {str(row["method"]): row for row in outcome_rows}
    x = np.arange(len(methods))
    width = 0.35
    goal_values = [
        float(
            by_method.get(method, {})
            .get("goal_failure_final_distance", {})
            .get("mean", 0.0)
            or 0.0
        )
        for method in methods
    ]
    violation_values = [
        float(
            by_method.get(method, {})
            .get("constraint_failure_violation_depth", {})
            .get("mean", 0.0)
            or 0.0
        )
        for method in methods
    ]
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width / 2.0, goal_values, width, label="Mean final goal distance if missed")
    ax.bar(x + width / 2.0, violation_values, width, label="Mean violation depth if violated")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Distance (m)")
    ax.set_title("Failure distance magnitudes")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def _write_json_report(output: Path, outcome_rows: list[dict[str, Any]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(outcome_rows, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv_report(output: Path, outcome_rows: list[dict[str, Any]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "method",
        "episodes",
        "goal_reached",
        "goal_not_reached",
        "constraint_avoided",
        "constraint_not_avoided",
        "both_success",
        "reached_but_constraint_failed",
        "goal_failed_but_constraint_avoided",
        "both_failed",
        "goal_failure_final_distance_mean",
        "goal_failure_final_distance_median",
        "goal_failure_final_distance_max",
        "constraint_failure_violation_depth_mean",
        "constraint_failure_violation_depth_median",
        "constraint_failure_violation_depth_max",
    ]
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in outcome_rows:
            goal_stats = row["goal_failure_final_distance"]
            violation_stats = row["constraint_failure_violation_depth"]
            flat = {
                key: row.get(key)
                for key in columns
                if not key.startswith("goal_failure_")
                and not key.startswith("constraint_failure_")
            }
            flat.update(
                {
                    "goal_failure_final_distance_mean": goal_stats["mean"],
                    "goal_failure_final_distance_median": goal_stats["median"],
                    "goal_failure_final_distance_max": goal_stats["max"],
                    "constraint_failure_violation_depth_mean": violation_stats["mean"],
                    "constraint_failure_violation_depth_median": violation_stats["median"],
                    "constraint_failure_violation_depth_max": violation_stats["max"],
                }
            )
            writer.writerow(flat)


def _print_outcome_report(outcome_rows: list[dict[str, Any]]) -> None:
    for row in outcome_rows:
        goal_stats = row["goal_failure_final_distance"]
        violation_stats = row["constraint_failure_violation_depth"]
        print(
            "outcome "
            f"method={row['method']} episodes={row['episodes']} "
            f"goal_not_reached={row['goal_not_reached']} "
            f"constraint_not_avoided={row['constraint_not_avoided']} "
            f"both_failed={row['both_failed']} "
            f"missed_but_avoided={row['goal_failed_but_constraint_avoided']} "
            f"reached_but_violated={row['reached_but_constraint_failed']} "
            f"both_success={row['both_success']} "
            f"goal_fail_mean_dist={_format_optional(goal_stats['mean'])} "
            f"constraint_fail_mean_depth={_format_optional(violation_stats['mean'])}"
        )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
    }


def _format_optional(value: Any) -> str:
    numeric = _optional_float(value)
    if numeric is None:
        return "nan"
    return f"{numeric:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())

