from pathlib import Path

import numpy as np
import pytest

from scripts.replay_u_shape_planner_paths import _replay_env_kwargs, parse_args


def test_replay_defaults_to_saved_position_ik_paths_and_three_cm_margin() -> None:
    args = parse_args([])

    assert args.planner_dir == Path(
        "artifacts/e10-u-shape-conventional-position-ik-margin3cm-extended-v1"
    )
    assert args.output_dir == Path("artifacts/e10-u-shape-planner-path-executions-v1")
    assert args.episodes is None
    assert args.safety_margin == pytest.approx(0.03)


def test_replay_rejects_negative_safety_margin() -> None:
    with pytest.raises(ValueError, match="safety-margin"):
        parse_args(["--safety-margin", "-0.01"])


def test_replay_environment_enables_rgb_rendering() -> None:
    kwargs = _replay_env_kwargs(
        {"env_kwargs": {"render_mode": None, "obs_mode": "state"}},
        obstacle_half_extents=np.asarray([0.1, 0.08, 0.375]),
    )

    assert kwargs["render_mode"] == "rgb_array"
    assert kwargs["obs_mode"] == "pointcloud"
    assert kwargs["pg3d_obstacle_family"] == "u_shape"
