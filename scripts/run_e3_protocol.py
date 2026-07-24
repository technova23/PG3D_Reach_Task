#!/usr/bin/env python3
"""Run the locked E3 constraint builder and comparison without CLI drift."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs/eval/e3_protocol.json"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.resolve()
    config = load_protocol(config_path)
    population = dict(config["populations"][args.population])
    constraints_output = resolve_repo_path(population["constraints_output_dir"])
    evaluation_output = resolve_repo_path(population["evaluation_output_dir"])
    locked_episode_indices = read_episode_indices(resolve_repo_path(config["episode_indices_file"]))
    if len(locked_episode_indices) != int(config["expected_test_episodes"]):
        raise ValueError("locked E3 episode file count does not match the protocol")

    if args.phase in {"build", "all"}:
        ensure_output_available(
            constraints_output,
            allow_existing=args.allow_existing_output,
        )
        command = builder_command(config, population, constraints_output)
        if run_command(command, dry_run=args.dry_run) != 0:
            return 1

    if args.phase in {"eval", "all"}:
        manifest_path = constraints_output / "manifest.json"
        if args.dry_run and not manifest_path.is_file():
            print(
                "constraint manifest not built; evaluation preview uses symbolic "
                "resolved half-extents",
                flush=True,
            )
            manifest = {
                "constraint_config": {
                    "resolved_box_half_extents": [
                        "<manifest-hx>",
                        "<manifest-hy>",
                        "<manifest-hz>",
                    ]
                }
            }
        else:
            manifest = load_constraint_manifest(
                constraints_output,
                expected_path_source=str(population["path_source"]),
                minimum_selected=int(population["minimum_selected_episodes"]),
                expected_episode_indices=locked_episode_indices,
                minimum_initial_clearance=float(
                    config["constraint_builder"]["initial_robot_clearance_margin"]
                ),
            )
        ensure_output_available(
            evaluation_output,
            allow_existing=args.allow_existing_output,
        )
        command = evaluation_command(
            config,
            population,
            constraints_output,
            evaluation_output,
            manifest,
        )
        if args.dry_run:
            return run_command(command, dry_run=True)
        evaluation_output.mkdir(parents=True, exist_ok=True)
        write_protocol_snapshot(
            evaluation_output,
            config_path=config_path,
            config=config,
            population_name=args.population,
            constraint_manifest=manifest,
            command=command,
        )
        return run_command(command, dry_run=False)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute one population from the immutable E3 comparison protocol."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--population",
        choices=["full_distribution", "nominal_base_success"],
        required=True,
    )
    parser.add_argument("--phase", choices=["build", "eval", "all"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Permit reuse/overwrite of protocol output directories (off by default).",
    )
    return parser.parse_args(argv)


def load_protocol(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "pg3d.e3_protocol.v1":
        raise ValueError("unsupported E3 protocol schema")
    required_populations = {"full_distribution", "nominal_base_success"}
    if set(config.get("populations", {})) != required_populations:
        raise ValueError("E3 protocol must define exactly the two declared populations")
    artifacts = dict(config.get("artifacts", {}))
    if not (
        artifacts.get("video")
        and artifacts.get("rerun")
        and artifacts.get("selection") == "all"
        and artifacts.get("require_mp4_rerun_pair_per_method_episode")
        and artifacts.get("require_embedded_identity")
    ):
        raise ValueError("E3 protocol requires labeled MP4/Rerun pairs for every row")
    evaluation = dict(config.get("evaluation", {}))
    if evaluation.get("methods") != ["base", "rejection", "reranking", "itps"]:
        raise ValueError("E3 protocol methods or paired order changed")
    if not evaluation.get("terminate_on_obstacle_contact"):
        raise ValueError("E3 protocol must terminate on physical obstacle contact")
    return config


def builder_command(
    config: dict[str, Any],
    population: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    settings = dict(config["constraint_builder"])
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/build_nominal_path_constraints.py"),
        "--dataset",
        str(resolve_repo_path(config["dataset"])),
        "--checkpoint",
        str(resolve_repo_path(config["checkpoint"])),
        "--checkpoint-model",
        str(config["checkpoint_model"]),
        "--path-source",
        str(population["path_source"]),
        "--episode-indices-file",
        str(resolve_repo_path(config["episode_indices_file"])),
        "--output-dir",
        str(output_dir),
        "--device",
        str(config["evaluation"]["device"]),
        "--max-steps",
        str(settings["max_steps"]),
        "--post-success-steps",
        str(settings["post_success_steps"]),
        "--path-fraction",
        str(settings["path_fraction"]),
        "--path-height-margin",
        str(settings["path_height_margin"]),
        "--initial-robot-clearance-margin",
        str(settings["initial_robot_clearance_margin"]),
        "--path-fraction-search-min",
        str(settings["path_fraction_search_min"]),
        "--path-fraction-search-max",
        str(settings["path_fraction_search_max"]),
        "--path-fraction-search-step",
        str(settings["path_fraction_search_step"]),
        "--anchor-offset-max-fraction",
        str(settings["anchor_offset_max_fraction"]),
        "--anchor-offset-step-fraction",
        str(settings["anchor_offset_step_fraction"]),
        "--avoid-shape",
        str(settings["avoid_shape"]),
        "--avoid-box-half-extents",
        *(str(value) for value in settings["box_half_extents"]),
        "--obstacle-yaw-deg",
        str(settings["obstacle_yaw_deg"]),
        "--support-plane-z",
        str(settings["support_plane_z"]),
        "--min-successes",
        str(population["minimum_selected_episodes"]),
    ]
    return command


def evaluation_command(
    config: dict[str, Any],
    population: dict[str, Any],
    constraints_output: Path,
    evaluation_output: Path,
    constraint_manifest: dict[str, Any],
) -> list[str]:
    settings = dict(config["evaluation"])
    artifacts = dict(config["artifacts"])
    builder = dict(config["constraint_builder"])
    resolved_extents = constraint_manifest["constraint_config"]["resolved_box_half_extents"]
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/eval_constrained_reach.py"),
        "--dataset",
        str(resolve_repo_path(config["dataset"])),
        "--checkpoint",
        str(resolve_repo_path(config["checkpoint"])),
        "--checkpoint-model",
        str(config["checkpoint_model"]),
        "--methods",
        *(str(method) for method in settings["methods"]),
        "--source",
        "dataset",
        "--episode-indices-file",
        str(constraints_output / "episode_indices.txt"),
        "--constraints-dir",
        str(constraints_output / "constraints"),
        "--output-dir",
        str(evaluation_output),
        "--device",
        str(settings["device"]),
        "--seed",
        str(settings["seed"]),
        "--max-steps",
        str(settings["max_steps"]),
        "--post-success-steps",
        str(settings["post_success_steps"]),
        "--planning-horizon-chunks",
        str(settings["planning_horizon_chunks"]),
        "--execution-horizon-chunks",
        str(settings["execution_horizon_chunks"]),
        "--geometry-mode",
        str(settings["geometry_mode"]),
        "--constraint-target",
        str(settings["constraint_target"]),
        "--k-schedule",
        *(str(value) for value in settings["k_schedule"]),
        "--policy-batch-size",
        str(settings["policy_batch_size"]),
        "--itps-guide-ratio",
        str(settings["itps_guide_ratio"]),
        "--itps-mcmc-steps",
        str(settings["itps_mcmc_steps"]),
        "--itps-energy",
        str(settings["itps_energy"]),
        "--itps-barrier-temperature",
        str(settings["itps_barrier_temperature"]),
        "--avoid-shape",
        str(builder["avoid_shape"]),
        "--avoid-box-half-extents",
        *(str(value) for value in resolved_extents),
        "--embody-obstacle",
        "--obstacle-family",
        str(settings["obstacle_family"]),
        "--obstacle-yaw-deg",
        str(builder["obstacle_yaw_deg"]),
        "--obstacle-support-plane-z",
        str(builder["support_plane_z"]),
        "--obstacle-point-quota",
        str(settings["obstacle_point_quota"]),
        "--robot-clearance-stride",
        str(settings["robot_clearance_stride"]),
        "--precomputed-initial-clearance-margin",
        str(builder["initial_robot_clearance_margin"]),
        "--paired-bootstrap-samples",
        str(settings["paired_bootstrap_samples"]),
        "--paired-bootstrap-seed",
        str(settings["paired_bootstrap_seed"]),
        "--artifact-selection",
        str(artifacts["selection"]),
        "--allow-failure",
    ]
    if settings["robot_clearance_metric"]:
        command.append("--robot-clearance-metric")
    if settings["terminate_on_obstacle_contact"]:
        command.append("--terminate-on-obstacle-contact")
    for enabled, flag in (
        (artifacts["video"], "--video"),
        (artifacts["rerun"], "--rerun"),
        (artifacts["plots"], "--plots"),
        (artifacts["profile"], "--profile"),
        (artifacts["sync_cuda_timers"], "--sync-cuda-timers"),
    ):
        if enabled:
            command.append(flag)
    return command


def load_constraint_manifest(
    output_dir: Path,
    *,
    expected_path_source: str,
    minimum_selected: int,
    expected_episode_indices: list[int],
    minimum_initial_clearance: float,
) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"constraint manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("path_source") != expected_path_source:
        raise ValueError("constraint manifest path source does not match protocol")
    if int(manifest.get("attempted_episodes", -1)) != len(expected_episode_indices):
        raise ValueError("constraint manifest does not cover the locked test population")
    attempted_indices = [int(row["dataset_episode_index"]) for row in manifest.get("attempts", [])]
    if attempted_indices != expected_episode_indices:
        raise ValueError("constraint manifest episodes or order do not match the locked test set")
    selected_count = int(manifest.get("selected_episodes", -1))
    if selected_count < minimum_selected:
        raise ValueError("constraint manifest selected too few episodes")
    selected = list(manifest.get("selected", []))
    if len(selected) != selected_count:
        raise ValueError("constraint manifest selected rows are incomplete")
    invalid_clearances = [
        float(row.get("initial_robot_clearance", float("-inf")))
        for row in selected
        if float(row.get("initial_robot_clearance", float("-inf"))) + 1e-8
        < minimum_initial_clearance
    ]
    if invalid_clearances:
        raise ValueError("constraint manifest violates the initial robot clearance gate")
    if any(float(row["discrete_min_clearance"]) >= 0.0 for row in selected):
        raise ValueError("constraint manifest contains an obstacle that misses its source path")
    remapped_indices = read_episode_indices(output_dir / "episode_indices.txt")
    selected_indices = [int(row["dataset_episode_index"]) for row in selected]
    if remapped_indices != selected_indices:
        raise ValueError("remapped constraint episode indices do not match the manifest")
    constraint_files = sorted((output_dir / "constraints").glob("episode_*.json"))
    if len(constraint_files) != selected_count:
        raise ValueError("serialized constraint file count does not match the manifest")
    resolved = manifest.get("constraint_config", {}).get("resolved_box_half_extents")
    if not isinstance(resolved, list) or len(resolved) != 3:
        raise ValueError("constraint manifest lacks resolved shared box geometry")
    return manifest


def ensure_output_available(path: Path, *, allow_existing: bool) -> None:
    if allow_existing or not path.exists():
        return
    if path.is_dir() and not any(path.iterdir()):
        return
    raise FileExistsError(
        f"refusing to overwrite nonempty protocol output: {path}; "
        "choose a new protocol version or pass --allow-existing-output explicitly"
    )


def write_protocol_snapshot(
    output_dir: Path,
    *,
    config_path: Path,
    config: dict[str, Any],
    population_name: str,
    constraint_manifest: dict[str, Any],
    command: list[str],
) -> None:
    raw_config = config_path.read_bytes()
    snapshot = {
        "schema_version": "pg3d.e3_protocol_snapshot.v1",
        "protocol_config": config,
        "protocol_config_path": str(config_path),
        "protocol_config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "population": population_name,
        "constraint_manifest_path": str(
            resolve_repo_path(config["populations"][population_name]["constraints_output_dir"])
            / "manifest.json"
        ),
        "constraint_manifest_sha256": hashlib.sha256(
            (
                resolve_repo_path(config["populations"][population_name]["constraints_output_dir"])
                / "manifest.json"
            ).read_bytes()
        ).hexdigest(),
        "resolved_box_half_extents": constraint_manifest["constraint_config"][
            "resolved_box_half_extents"
        ],
        "command": command,
    }
    (output_dir / "protocol_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_command(command: list[str], *, dry_run: bool) -> int:
    print(shlex.join(command), flush=True)
    if dry_run:
        return 0
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    return int(completed.returncode)


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def read_episode_indices(path: Path) -> list[int]:
    indices: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.split("#", 1)[0].strip()
        if value:
            indices.append(int(value))
    return indices


if __name__ == "__main__":
    raise SystemExit(main())
