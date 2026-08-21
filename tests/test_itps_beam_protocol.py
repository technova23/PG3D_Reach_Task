from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.report_itps_beam_mvp import _ranking_key
from scripts.run_itps_beam_mvp import (
    evaluation_command,
    load_protocol,
    stage_jobs,
)


def _config_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "configs/eval/e10_itps_beam_mvp_v1/protocol.json"
    )


def test_protocol_declares_fixed_h1_h3_and_calibration_grids() -> None:
    config = load_protocol(_config_path())

    assert len(stage_jobs("fixed_weight_calibration", config)) == 15
    assert len(stage_jobs("mass_ablation", config)) == 7
    assert len(stage_jobs("adaptive_weights", config)) == 7
    assert len(stage_jobs("repeated_lineages", config)) == 15
    assert {job.horizon for job in stage_jobs("smoke", config)} == {1, 3}


def test_protocol_command_freezes_depth_budget_and_lineage(tmp_path: Path) -> None:
    config_path = _config_path()
    config = load_protocol(config_path)
    h3 = stage_jobs("smoke", config)[1]
    command = evaluation_command(
        config,
        config_path,
        config_path.parent / "development_fixture.json",
        tmp_path / "output",
        h3,
    )

    assert command[command.index("--planning-horizon-chunks") + 1] == "3"
    assert command[command.index("--beam-width") + 1] == "2"
    assert command[command.index("--beam-branch-factor") + 1] == "2"
    assert command[command.index("--max-guided-proposals") + 1] == "20"
    assert command[command.index("--fixture-output-indices") + 1] == "0"


def test_protocol_rejects_changed_horizon(tmp_path: Path) -> None:
    payload = json.loads(_config_path().read_text(encoding="utf-8"))
    payload["shared"]["max_steps"] = 299
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="max_steps"):
        load_protocol(path)


def test_predeclared_ranking_uses_outcomes_then_compute_then_id() -> None:
    common = {
        "stable_combined_success_rate": 0.5,
        "safety_rate": 1.0,
        "goal_entry_rate": 0.5,
        "median_minimum_target_distance_m": 0.02,
        "mean_final_target_distance_m": 0.03,
    }
    cheaper = {**common, "mean_denoiser_evaluations": 100.0}
    expensive = {**common, "mean_denoiser_evaluations": 200.0}

    assert _ranking_key("b", cheaper) < _ranking_key("a", expensive)
