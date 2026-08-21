#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pg3d.composition import ConvexScoreWeights, add_mass_weight, simplex_weights
from pg3d.eval import ConditioningBundleStore, load_guided_fixture_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs/eval/e10_itps_beam_mvp_v1/protocol.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts/itps-beam-mvp-v1"
STAGES = (
    "smoke",
    "lookahead",
    "fixed_weight_calibration",
    "mass_ablation",
    "adaptive_weights",
    "adaptive_search",
    "repeated_lineages",
    "decision",
)


@dataclass(frozen=True)
class ProtocolJob:
    job_id: str
    method: str
    horizon: int
    score_mode: str
    weights: ConvexScoreWeights
    lineage: int = 0
    fixture_subset: tuple[int, ...] | None = None
    uncertainty_allocation: bool = False
    dynamic_width: bool = False
    route_diversity: bool = False
    adaptive_rate: float = 0.25
    adaptive_temperature: float = 1.0
    qualitative_artifacts: bool = False


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.resolve()
    config = load_protocol(config_path)
    fixture_path = _resolve(config["fixture_manifest"])
    fixture = load_guided_fixture_manifest(fixture_path, repo_root=REPO_ROOT)
    output_root = args.output_root.resolve()
    stage_root = output_root / args.stage
    config_hash = _file_sha256(config_path)
    fixture_hash = _file_sha256(fixture_path)
    checkpoint = Path(config["checkpoint"]).resolve()
    checkpoint_hash = _file_sha256(checkpoint)

    if args.stage == "decision":
        return _run_report(stage_root=output_root, stage=args.stage, dry_run=args.dry_run)

    jobs = stage_jobs(args.stage, config)
    for job in jobs:
        job_root = stage_root / job.job_id
        command = evaluation_command(config, config_path, fixture_path, job_root, job)
        if args.dry_run:
            print(shlex.join(command))
            continue
        if job_root.exists():
            if not args.resume:
                raise FileExistsError(f"refusing to reuse protocol output: {job_root}")
            completion = job_root / "stage_complete.json"
            if not completion.is_file():
                raise RuntimeError(f"cannot resume incomplete job: {job_root}")
            _validate_completion(
                completion,
                config_hash=config_hash,
                fixture_hash=fixture_hash,
                checkpoint_hash=checkpoint_hash,
            )
            print(f"resume: validated and skipped {job.job_id}", flush=True)
            continue
        job_root.mkdir(parents=True)
        snapshot = {
            "schema_version": "pg3d.itps_beam_job.v1",
            "classification": "development_only",
            "stage": args.stage,
            "job": _job_json(job),
            "command": command,
            "protocol_path": str(config_path),
            "protocol_sha256": config_hash,
            "fixture_path": str(fixture_path),
            "fixture_sha256": fixture_hash,
            "checkpoint_sha256": checkpoint_hash,
        }
        (job_root / "protocol_snapshot.json").write_text(
            json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        result = subprocess.run(command, cwd=REPO_ROOT, check=False)
        if result.returncode != 0:
            return result.returncode
        validation = validate_job_output(
            job_root,
            expected_episodes=(
                len(job.fixture_subset) if job.fixture_subset else len(fixture.episodes)
            ),
            expected_method=job.method,
            smoke=args.stage == "smoke",
        )
        completion = {
            **snapshot,
            "validation": validation,
            "metrics_sha256": _file_sha256(job_root / "metrics.jsonl"),
        }
        (job_root / "stage_complete.json").write_text(
            json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if args.dry_run:
        return _run_report(stage_root=stage_root, stage=args.stage, dry_run=True)
    return _run_report(stage_root=stage_root, stage=args.stage, dry_run=False)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute the frozen ITPS-beam MVP protocol.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def load_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "pg3d.itps_beam_mvp_protocol.v1":
        raise ValueError("unsupported ITPS-beam protocol schema")
    if payload.get("classification") != "development_only":
        raise ValueError("ITPS-beam MVP must remain development-only")
    shared = dict(payload["shared"])
    expected = {
        "max_steps": 300,
        "post_success_steps": 16,
        "goal_threshold_m": 0.025,
        "executed_actions_per_replan": 8,
        "constraint_target": "robot",
        "geometry_mode": "exact",
        "hard_clearance_m": 0.03,
        "lineages": [0, 1, 2, 3, 4],
    }
    for key, value in expected.items():
        if shared.get(key) != value:
            raise ValueError(f"frozen protocol setting changed: {key}")
    h3 = dict(payload["methods"]["h3"])
    if h3.get("planning_horizon_chunks") != 3 or h3.get("guided_search_expansions") != 10:
        raise ValueError("H3 must use physical depth three and exactly ten expansions")
    if dict(payload["methods"]["h1"]).get("planning_horizon_chunks") != 1:
        raise ValueError("H1 physical depth must be one")
    return payload


def stage_jobs(stage: str, config: dict[str, Any]) -> list[ProtocolJob]:
    task = _task_weights(config)
    mass = add_mass_weight(task, 0.25)
    h1 = lambda job_id, mode, weights, **kwargs: ProtocolJob(  # noqa: E731
        job_id, "itps_reranking", 1, mode, weights, **kwargs
    )
    h3 = lambda job_id, mode, weights, **kwargs: ProtocolJob(  # noqa: E731
        job_id, "itps_beam", 3, mode, weights, **kwargs
    )
    if stage == "smoke":
        return [
            h1("h1", "fixed_task", task, fixture_subset=(0,)),
            h3("h3", "fixed_task", task, fixture_subset=(0,)),
        ]
    if stage == "lookahead":
        return [
            ProtocolJob("itps", "itps", 1, "avoidance_only", task),
            h1("h1_fixed", "fixed_task", task),
            h3("h3_fixed", "fixed_task", task),
            h1("h1_avoidance", "avoidance_only", task),
            h3("h3_avoidance", "avoidance_only", task),
        ]
    if stage == "fixed_weight_calibration":
        return [
            h3(f"simplex_{index:02d}", "fixed_task", weights)
            for index, weights in enumerate(simplex_weights(0.25))
        ]
    if stage == "mass_ablation":
        jobs = [h3("no_mass", "fixed_task", task)]
        for mass_weight in (0.10, 0.25, 0.40):
            weights = add_mass_weight(task, mass_weight)
            jobs.extend(
                [
                    h3(f"mean_wm{mass_weight:.2f}", "mass_mean", weights),
                    h3(f"lcb_wm{mass_weight:.2f}", "mass_lcb", weights),
                ]
            )
        return jobs
    if stage == "adaptive_weights":
        return [
            h3("fixed_mass", "mass_lcb", mass),
            *[
                h3(
                    f"rho{rho:.2f}_t{temperature:.1f}",
                    "adaptive_mass",
                    mass,
                    adaptive_rate=rho,
                    adaptive_temperature=temperature,
                )
                for rho in (0.10, 0.25, 0.50)
                for temperature in (0.5, 1.0)
            ],
        ]
    if stage == "adaptive_search":
        variants = {
            "fixed_compute": {},
            "uncertainty": {"uncertainty_allocation": True},
            "dynamic_width": {"dynamic_width": True},
            "diversity": {"route_diversity": True},
            "combined": {
                "uncertainty_allocation": True,
                "dynamic_width": True,
                "route_diversity": True,
            },
        }
        return [h3(name, "adaptive_mass", mass, **options) for name, options in variants.items()]
    if stage == "repeated_lineages":
        jobs = []
        for lineage in config["shared"]["lineages"]:
            artifacts = lineage == 0
            jobs.extend(
                [
                    ProtocolJob(
                        f"itps_l{lineage}",
                        "itps",
                        1,
                        "avoidance_only",
                        task,
                        lineage=lineage,
                        qualitative_artifacts=artifacts,
                    ),
                    h3(
                        f"fixed_mass_l{lineage}",
                        "mass_lcb",
                        mass,
                        lineage=lineage,
                        qualitative_artifacts=artifacts,
                    ),
                    h3(
                        f"combined_l{lineage}",
                        "adaptive_mass",
                        mass,
                        lineage=lineage,
                        uncertainty_allocation=True,
                        dynamic_width=True,
                        route_diversity=True,
                        qualitative_artifacts=artifacts,
                    ),
                ]
            )
        return jobs
    raise ValueError(f"stage {stage!r} does not launch evaluator jobs")


def evaluation_command(
    config: dict[str, Any],
    config_path: Path,
    fixture_path: Path,
    output_dir: Path,
    job: ProtocolJob,
) -> list[str]:
    shared = config["shared"]
    itps = config["itps"]
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/eval_constrained_reach.py"),
        "--checkpoint",
        str(Path(config["checkpoint"]).resolve()),
        "--checkpoint-model",
        str(config["checkpoint_model"]),
        "--dataset",
        str(Path(config["dataset"]).resolve()),
        "--fixture-manifest",
        str(fixture_path),
        "--source",
        "dataset",
        "--output-dir",
        str(output_dir),
        "--methods",
        job.method,
        "--device",
        str(shared["device"]),
        "--seed",
        str(job.lineage),
        "--max-steps",
        str(shared["max_steps"]),
        "--post-success-steps",
        str(shared["post_success_steps"]),
        "--goal-thresh",
        str(shared["goal_threshold_m"]),
        "--planning-horizon-chunks",
        str(job.horizon),
        "--execution-horizon-chunks",
        "1",
        "--geometry-mode",
        "exact",
        "--constraint-target",
        "robot",
        "--guided-candidates",
        "10",
        "--beam-width",
        "2",
        "--beam-branch-factor",
        "2",
        "--score-config",
        str(config_path),
        "--score-mode",
        job.score_mode,
        "--score-weights",
        *(str(value) for value in job.weights.as_tuple()),
        "--max-guided-proposals",
        str(config["mass_estimator"]["max_guided_proposals_per_replan"]),
        "--adaptive-rate",
        str(job.adaptive_rate),
        "--adaptive-temperature",
        str(job.adaptive_temperature),
        "--itps-guide-ratio",
        str(itps["guide_ratio"]),
        "--itps-mcmc-steps",
        str(itps["mcmc_steps"]),
        "--itps-energy",
        str(itps["energy"]),
        "--itps-barrier-temperature",
        str(itps["barrier_temperature"]),
        "--itps-robot-points",
        str(itps["robot_points"]),
        "--itps-robot-sample-seed",
        str(itps["robot_sample_seed"]),
        "--ddim-eta",
        str(itps["eta"]),
        "--embody-obstacle",
        "--obstacle-family",
        "u_shape",
        "--robot-clearance-metric",
        "--profile",
        "--allow-failure",
    ]
    if job.fixture_subset:
        command.extend(["--fixture-output-indices", *(str(value) for value in job.fixture_subset)])
    if job.uncertainty_allocation:
        command.append("--uncertainty-allocation")
    if job.dynamic_width:
        command.append("--dynamic-beam-width")
    if job.route_diversity:
        command.append("--route-diversity")
    if job.qualitative_artifacts:
        command.extend(["--video", "--rerun", "--artifact-selection", "all"])
    return command


def validate_job_output(
    root: Path,
    *,
    expected_episodes: int,
    expected_method: str,
    smoke: bool,
) -> dict[str, Any]:
    metrics_path = root / "metrics.jsonl"
    summary_path = root / "summary.json"
    if not metrics_path.is_file() or not summary_path.is_file():
        raise ValueError(f"job output is incomplete: {root}")
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != expected_episodes or {row["method"] for row in rows} != {expected_method}:
        raise ValueError("job metrics do not match the expected method/fixture count")
    if any(int(row["steps"]) > 300 for row in rows):
        raise ValueError("job exceeded the frozen 300-step horizon")
    if expected_method == "itps_beam" and any(
        int(row["beam_expanded_nodes_per_replan"]) != 10
        for row in rows
        if not row["method_config"]["adaptive_beam"]["dynamic_width"]
    ):
        raise ValueError("fixed H3 did not expand exactly ten search nodes per replan")
    if smoke and any(int(row["denoiser_evaluations_per_replan"]) != 4000 for row in rows):
        raise ValueError("smoke job did not use 4,000 denoiser evaluations per replan")
    proposal_records = 0
    for row in rows:
        trace = row.get("proposal_trace")
        if trace is None:
            raise ValueError("guided job is missing proposal trace")
        for line in Path(trace).read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            store = ConditioningBundleStore(Path(record["conditioning_bundle"]).parent)
            store.load(str(record["conditioning_hash"]))
            proposal_records += 1
    return {
        "episodes": len(rows),
        "proposal_records": proposal_records,
        "metrics_sha256": _file_sha256(metrics_path),
        "summary_sha256": _file_sha256(summary_path),
    }


def _validate_completion(
    path: Path,
    *,
    config_hash: str,
    fixture_hash: str,
    checkpoint_hash: str,
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "protocol_sha256": config_hash,
        "fixture_sha256": fixture_hash,
        "checkpoint_sha256": checkpoint_hash,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"resume artifact hash mismatch: {key}")
    metrics = path.parent / "metrics.jsonl"
    if _file_sha256(metrics) != payload.get("metrics_sha256"):
        raise ValueError("resume metrics hash mismatch")


def _run_report(*, stage_root: Path, stage: str, dry_run: bool) -> int:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/report_itps_beam_mvp.py"),
        "--input-root",
        str(stage_root),
        "--stage",
        stage,
        "--output",
        str(stage_root / "stage_report.json"),
    ]
    if dry_run:
        print(shlex.join(command))
        return 0
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


def _task_weights(config: dict[str, Any]) -> ConvexScoreWeights:
    raw = config["scoring"]["default_fixed_weights"]
    return ConvexScoreWeights(
        goal=float(raw["goal"]),
        clearance=float(raw["clearance"]),
        smoothness=float(raw["smoothness"]),
        mass=0.0,
    )


def _job_json(job: ProtocolJob) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "method": job.method,
        "horizon": job.horizon,
        "score_mode": job.score_mode,
        "weights": job.weights.to_json(),
        "lineage": job.lineage,
        "fixture_subset": job.fixture_subset,
        "uncertainty_allocation": job.uncertainty_allocation,
        "dynamic_width": job.dynamic_width,
        "route_diversity": job.route_diversity,
        "adaptive_rate": job.adaptive_rate,
        "adaptive_temperature": job.adaptive_temperature,
        "qualitative_artifacts": job.qualitative_artifacts,
    }


def _resolve(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
