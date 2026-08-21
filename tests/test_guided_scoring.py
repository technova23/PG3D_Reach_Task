from __future__ import annotations

import json
from pathlib import Path

import pytest

from pg3d.composition import (
    ConvexScoreWeights,
    GuidedScoreConfig,
    simplex_weights,
)
from pg3d.eval import load_guided_fixture_manifest


def test_normalized_guided_score_clips_all_terms() -> None:
    config = GuidedScoreConfig(
        mode="fixed_task",
        weights=ConvexScoreWeights(goal=0.5, clearance=0.25, smoothness=0.25),
    )

    terms = config.terms(
        goal_distance_m=1.5,
        min_clearance_m=0.055,
        smoothness_rad2=-1.0,
    )

    assert terms.goal == 1.0
    assert terms.clearance == pytest.approx(0.5)
    assert terms.smoothness == 0.0
    assert config.feasible_score(terms, avoidance_penalty=99.0) == pytest.approx(0.625)


def test_avoidance_only_preserves_historical_penalty() -> None:
    config = GuidedScoreConfig(mode="avoidance_only")
    terms = config.terms(
        goal_distance_m=0.7,
        min_clearance_m=0.031,
        smoothness_rad2=0.1,
    )

    assert config.feasible_score(terms, avoidance_penalty=3.25) == 3.25


def test_convex_weights_and_quarter_simplex() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        ConvexScoreWeights(goal=0.5, clearance=0.5, smoothness=0.5)
    with pytest.raises(ValueError, match="non-negative"):
        ConvexScoreWeights(goal=1.1, clearance=-0.1, smoothness=0.0)

    weights = simplex_weights(0.25)
    assert len(weights) == 15
    assert len({item.as_tuple() for item in weights}) == 15
    assert all(sum(item.as_tuple()) == pytest.approx(1.0) for item in weights)


def test_fixture_manifest_rejects_changed_constraint(tmp_path: Path) -> None:
    constraint = tmp_path / "constraint.json"
    constraint.write_text("{}", encoding="utf-8")
    manifest_path = tmp_path / "fixture.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "pg3d.itps_beam_fixture.v1",
                "classification": "development_only",
                "name": "test",
                "dataset": "dataset.zarr",
                "checkpoint": "checkpoint.pt",
                "episodes": [
                    {
                        "output_index": 0,
                        "source_output_index": 0,
                        "dataset_episode_index": 305,
                        "simulator_seed": 1,
                        "policy_seed": 2,
                        "constraint_file": str(constraint),
                        "constraint_sha256": "incorrect",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="constraint hash mismatch"):
        load_guided_fixture_manifest(manifest_path, repo_root=tmp_path)


def test_canonical_fixture_manifest_is_development_only() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = load_guided_fixture_manifest(
        root / "configs/eval/e10_itps_beam_mvp_v1/development_fixture.json",
        repo_root=root,
    )

    assert manifest.classification == "development_only"
    assert [episode.source_output_index for episode in manifest.episodes] == [0, 1, 6, 7, 8, 9]
    assert [episode.dataset_episode_index for episode in manifest.episodes] == [
        305,
        317,
        1069,
        1117,
        1129,
        1138,
    ]
