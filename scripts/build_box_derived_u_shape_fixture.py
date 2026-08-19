"""Build planner-free U-shape previews from the frozen E3 box placements.

Each source box is treated as an immutable anchor. The replacement U keeps its
configured center, outer width, depth, and 0.75 m height, while its opening faces
the recorded episode start. Versioned per-episode geometry is read from a JSON
guidance file; finalized guidance must not be edited in place.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from pg3d.envs.obstacles import transform_box_component, u_shape_components
from pg3d.eval import load_episode_constraints, save_episode_constraints
from pg3d.eval.u_shape_placement import (
    UShapeCandidate,
    box_derived_u_shape_candidate,
    component_geometry,
    u_shape_constraints,
)
from pg3d.utils.serialization import jsonable
from scripts.rollout_dp3_reach_policy import save_rerun_timeline

DEFAULT_GUIDANCE = Path("configs/eval/e10_u_shape_box_derived_guidance_v1.json")
DEFAULT_CONFIG_DIR = Path("configs/eval/e10_u_shape_box_derived_review_v1")
DEFAULT_ARTIFACT_DIR = Path("artifacts/e10-u-shape-box-derived-review-v1")
EXPECTED_EPISODES = (305, 317, 974, 986, 1010, 1034, 1069, 1117, 1129, 1138)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    guidance = json.loads(args.guidance.read_text(encoding="utf-8"))
    source_path = Path(guidance["source_fixture"])
    source = json.loads(source_path.read_text(encoding="utf-8"))
    _validate_inputs(source, guidance)
    _prepare_output(args.config_dir, overwrite=args.overwrite)
    _prepare_output(args.artifact_dir, overwrite=args.overwrite)

    dataset = Path(source["dataset"])
    arrays = _open_dataset_arrays(dataset)
    episode_ends = np.asarray(arrays["episode_ends"][:], dtype=np.int64)
    guidance_by_index = {int(item["output_index"]): item for item in guidance["episodes"]}
    records: list[dict[str, Any]] = []
    for source_episode in source["episodes"]:
        output_index = int(source_episode["output_index"])
        dataset_index = int(source_episode["dataset_episode_index"])
        start_index = 0 if dataset_index == 0 else int(episode_ends[dataset_index - 1])
        end_index = int(episode_ends[dataset_index])
        nominal = np.asarray(arrays["eef_pos"][start_index:end_index], dtype=np.float32)
        if len(nominal) < 9:
            raise RuntimeError(f"dataset episode {dataset_index} has no complete first chunk")
        start = np.asarray(arrays["tcp_pose"][start_index], dtype=np.float32)[:3]
        goal = np.asarray(arrays["target_position"][start_index], dtype=np.float32)
        first_chunk_distance = float(np.sum(np.linalg.norm(np.diff(nominal[:9], axis=0), axis=1)))
        candidate, source_box = _candidate_from_source_box(
            source_path=source_path,
            source_episode=source_episode,
            guidance=guidance_by_index[output_index],
            start=start,
            first_chunk_distance=first_chunk_distance,
        )
        record = _episode_record(
            source_episode=source_episode,
            candidate=candidate,
            source_box=source_box,
            guidance=guidance_by_index[output_index],
            nominal=nominal,
            start=start,
            goal=goal,
        )
        eef_constraints = u_shape_constraints(
            candidate,
            target="eef",
            name_prefix="u_shape_box_derived",
        )
        robot_constraints = u_shape_constraints(
            candidate,
            target="robot",
            name_prefix="u_shape_box_derived",
        )
        _assert_matched_geometry(eef_constraints, robot_constraints)
        constraint_name = f"episode_{output_index:03d}.json"
        save_episode_constraints(
            args.config_dir / "constraints" / "eef" / constraint_name, eef_constraints
        )
        save_episode_constraints(
            args.config_dir / "constraints" / "robot" / constraint_name,
            robot_constraints,
        )
        surface = _u_surface_points(candidate, spacing=0.012)
        timeline = [
            {
                "point_cloud": np.asarray(arrays["point_cloud"][start_index], dtype=np.float32),
                "point_valid_mask": np.asarray(arrays["point_valid_mask"][start_index], dtype=bool),
                "robot_mask": np.asarray(arrays["robot_mask"][start_index], dtype=bool),
                "target_position": goal,
                "tcp_pose": np.asarray(arrays["tcp_pose"][start_index], dtype=np.float32),
            }
        ]
        rerun_path = args.artifact_dir / "rerun" / f"episode_{output_index:03d}.rrd"
        save_rerun_timeline(
            rerun_path,
            timeline,
            constraints=robot_constraints,
            recording_identity={
                "fixture_id": guidance["fixture_id"],
                "episode": output_index,
                "dataset_episode_index": dataset_index,
                "status": (
                    "object_geometry_finalized"
                    if guidance["object_geometry_status"] == "finalized"
                    else "geometry_only_inspection_candidate"
                ),
                "planner_run": False,
            },
            inspection={
                "nominal_tcp_path": nominal,
                "obstacle_surface_points": surface,
                "start_position": start,
                "goal_position": goal,
                "summary": record,
            },
        )
        _save_topdown(
            args.artifact_dir / "topdown" / f"episode_{output_index:03d}.png",
            candidate=candidate,
            source_box=source_box,
            nominal=nominal,
            start=start,
            goal=goal,
            title=f"Box-derived U review — episode {output_index:03d}",
        )
        records.append(record)
        print(
            f"[box-derived-u] episode={output_index:03d} dataset={dataset_index} "
            f"size={record['full_size_m']} yaw={record['yaw_deg']:.2f}deg",
            flush=True,
        )

    _write_suite(
        config_dir=args.config_dir,
        artifact_dir=args.artifact_dir,
        source=source,
        source_path=source_path,
        guidance=guidance,
        guidance_path=args.guidance,
        records=records,
    )
    print(
        f"built {len(records)} planner-free U previews in {args.config_dir} and "
        f"{args.artifact_dir}",
        flush=True,
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guidance", type=Path, default=DEFAULT_GUIDANCE)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _prepare_output(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {path}; pass --overwrite")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _validate_inputs(source: dict[str, Any], guidance: dict[str, Any]) -> None:
    if guidance.get("object_geometry_status") not in {"review_candidate", "finalized"}:
        raise ValueError("object_geometry_status must be 'review_candidate' or 'finalized'")
    episodes = source.get("episodes", [])
    indices = tuple(int(item["dataset_episode_index"]) for item in episodes)
    if indices != EXPECTED_EPISODES:
        raise ValueError(f"source episode order changed: {indices}")
    if [int(item["output_index"]) for item in episodes] != list(range(10)):
        raise ValueError("source output indices must be 0 through 9")
    guided = guidance.get("episodes", [])
    if [int(item["output_index"]) for item in guided] != list(range(10)):
        raise ValueError("guidance output indices must be exactly 0 through 9")
    for item in guided:
        size = np.asarray(item["full_size_xy_m"], dtype=np.float64)
        delta = np.asarray(item["center_delta_xy_m"], dtype=np.float64)
        yaw_delta = float(item["yaw_delta_deg"])
        if size.shape != (2,) or np.any(size <= 0.0) or not np.isfinite(size).all():
            raise ValueError("full_size_xy_m must contain two positive finite values")
        if delta.shape != (2,) or not np.isfinite(delta).all() or not np.isfinite(yaw_delta):
            raise ValueError("guidance center/yaw adjustments must be finite")


def _open_dataset_arrays(dataset: Path) -> dict[str, Any]:
    def open_array(relative: str) -> Any:
        return zarr.open_array(str(dataset / relative), mode="r")

    return {
        "episode_ends": open_array("meta/episode_ends"),
        "eef_pos": open_array("data/eef_pos"),
        "tcp_pose": open_array("data/tcp_pose"),
        "target_position": open_array("data/target_position"),
        "point_cloud": open_array("data/point_cloud"),
        "point_valid_mask": open_array("data/point_valid_mask"),
        "robot_mask": open_array("data/robot_mask"),
    }


def _candidate_from_source_box(
    *,
    source_path: Path,
    source_episode: dict[str, Any],
    guidance: dict[str, Any],
    start: np.ndarray,
    first_chunk_distance: float,
) -> tuple[UShapeCandidate, dict[str, Any]]:
    constraint_path = (
        source_path.parent / "constraints" / "robot" / str(source_episode["constraint_file"])
    )
    constraints = load_episode_constraints(constraint_path)
    if len(constraints) != 1 or getattr(constraints[0].region, "region_type", None) != "box":
        raise ValueError(f"expected exactly one source box in {constraint_path}")
    source_region = constraints[0].region
    source_center = np.asarray(source_region.center, dtype=np.float32)
    source_half_extents = np.asarray(source_region.half_extents, dtype=np.float32)
    fixture_center = np.asarray(source_episode["center"], dtype=np.float32)
    if not np.allclose(source_center, fixture_center, atol=1e-7):
        raise ValueError(f"source fixture/constraint center mismatch in {constraint_path}")
    center = source_center.copy()
    center[:2] += np.asarray(guidance["center_delta_xy_m"], dtype=np.float32)
    full_size_xy = np.asarray(guidance["full_size_xy_m"], dtype=np.float32)
    half_extents = np.asarray([0.5 * full_size_xy[0], 0.5 * full_size_xy[1], 0.375])
    candidate = box_derived_u_shape_candidate(
        center,
        half_extents,
        start=start,
        first_chunk_distance=first_chunk_distance,
        yaw_delta=float(np.deg2rad(float(guidance["yaw_delta_deg"]))),
    )
    return candidate, {
        "center": source_center.astype(float).tolist(),
        "half_extents": source_half_extents.astype(float).tolist(),
        "full_size": (2.0 * source_half_extents).astype(float).tolist(),
        "yaw_rad": float(source_region.yaw),
    }


def _assert_matched_geometry(eef: list[Any], robot: list[Any]) -> None:
    eef_geometry = [item.region.to_json() for item in eef]
    robot_geometry = [item.region.to_json() for item in robot]
    if eef_geometry != robot_geometry:
        raise RuntimeError("EEF and robot U constraints do not share physical geometry")


def _episode_record(
    *,
    source_episode: dict[str, Any],
    candidate: UShapeCandidate,
    source_box: dict[str, Any],
    guidance: dict[str, Any],
    nominal: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
) -> dict[str, Any]:
    constraints = u_shape_constraints(candidate, target="robot", name_prefix="u_shape_box_derived")
    back = next(item for item in constraints if item.name.endswith("_back"))
    direct = np.linspace(start, goal, 2049, dtype=np.float32)
    output_index = int(source_episode["output_index"])
    usable_cavity_depth = float((26.0 / 15.0) * candidate.half_extents[1])
    return {
        "output_index": output_index,
        "dataset_episode_index": int(source_episode["dataset_episode_index"]),
        "simulator_seed": int(source_episode["simulator_seed"]),
        "policy_seed": int(source_episode["policy_seed"]),
        "root_center": candidate.root_center.astype(float).tolist(),
        "yaw_rad": float(candidate.yaw),
        "yaw_deg": float(np.rad2deg(candidate.yaw)),
        "envelope_half_extents_m": candidate.half_extents.astype(float).tolist(),
        "full_size_m": candidate.full_size.astype(float).tolist(),
        "mouth_width_m": float(candidate.mouth_width),
        "usable_cavity_depth_m": usable_cavity_depth,
        "mouth_distance_m": float(candidate.mouth_distance),
        "back_distance_m": float(candidate.back_distance),
        "mouth_first_chunk_ratio": float(candidate.mouth_chunk_fraction),
        "back_first_chunk_ratio": float(candidate.back_chunk_ratio),
        "lateral_offset_from_start_m": float(candidate.lateral_offset),
        "first_chunk_distance_m": float(candidate.chunk_distance),
        "start_position": np.asarray(start, dtype=float).tolist(),
        "goal_position": np.asarray(goal, dtype=float).tolist(),
        "direct_path_back_clearance_m": float(np.min(back.region.signed_distance(direct))),
        "nominal_path_back_clearance_m": float(
            np.min(back.region.signed_distance(np.asarray(nominal, dtype=np.float32)))
        ),
        "source_box": source_box,
        "guidance": guidance,
        "components": component_geometry(candidate),
        "eef_constraint_file": f"constraints/eef/episode_{output_index:03d}.json",
        "robot_constraint_file": f"constraints/robot/episode_{output_index:03d}.json",
        "rerun_file": f"rerun/episode_{output_index:03d}.rrd",
        "topdown_file": f"topdown/episode_{output_index:03d}.png",
        "planner_status": "not_run",
        "whole_robot_clearance_status": "not_validated",
        "policy_obstacle_visibility_status": "not_measured",
    }


def _u_surface_points(candidate: UShapeCandidate, *, spacing: float) -> np.ndarray:
    clouds = []
    for component in u_shape_components(candidate.half_extents):
        center, yaw = transform_box_component(
            component,
            center=candidate.root_center,
            yaw=candidate.yaw,
        )
        clouds.append(
            _box_surface_points(
                center,
                np.asarray(component.half_extents, dtype=np.float32),
                yaw,
                spacing=spacing,
            )
        )
    return np.concatenate(clouds, axis=0).astype(np.float32)


def _box_surface_points(
    center: np.ndarray,
    half_extents: np.ndarray,
    yaw: float,
    *,
    spacing: float,
) -> np.ndarray:
    axes = [
        np.linspace(-half, half, max(2, int(math.ceil(2.0 * half / spacing)) + 1))
        for half in half_extents
    ]
    faces = []
    for fixed_axis in range(3):
        other = [axis for axis in range(3) if axis != fixed_axis]
        grid_a, grid_b = np.meshgrid(axes[other[0]], axes[other[1]], indexing="ij")
        for sign in (-1.0, 1.0):
            points = np.zeros((grid_a.size, 3), dtype=np.float32)
            points[:, fixed_axis] = sign * half_extents[fixed_axis]
            points[:, other[0]] = grid_a.reshape(-1)
            points[:, other[1]] = grid_b.reshape(-1)
            faces.append(points)
    local = np.concatenate(faces, axis=0)
    rotation = np.asarray(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
        dtype=np.float32,
    )
    local[:, :2] = local[:, :2] @ rotation.T
    return local + np.asarray(center, dtype=np.float32)


def _save_topdown(
    path: Path,
    *,
    candidate: UShapeCandidate,
    source_box: dict[str, Any],
    nominal: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    _add_box_patch(
        ax,
        np.asarray(source_box["center"], dtype=np.float32),
        np.asarray(source_box["half_extents"], dtype=np.float32),
        float(source_box["yaw_rad"]),
        facecolor="none",
        edgecolor="#e58f26",
        linestyle="--",
        linewidth=2,
        label="source box envelope",
    )
    for index, component in enumerate(u_shape_components(candidate.half_extents)):
        center, yaw = transform_box_component(
            component, center=candidate.root_center, yaw=candidate.yaw
        )
        _add_box_patch(
            ax,
            center,
            np.asarray(component.half_extents, dtype=np.float32),
            yaw,
            facecolor="#5b8ecb",
            edgecolor="#173b68",
            label="proposed U" if index == 0 else None,
        )
    ax.plot(nominal[:, 0], nominal[:, 1], "--", color="gray", label="dataset nominal TCP")
    ax.scatter(start[0], start[1], color="red", label="start", zorder=4)
    ax.scatter(goal[0], goal[1], color="green", label="goal", zorder=4)
    ax.set_title(title)
    ax.set_xlabel("world X (m)")
    ax.set_ylabel("world Y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _add_box_patch(
    ax: Any,
    center: np.ndarray,
    half_extents: np.ndarray,
    yaw: float,
    **kwargs: Any,
) -> None:
    from matplotlib.patches import Polygon

    hx, hy = half_extents[:2]
    local = np.asarray([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]], dtype=np.float32)
    rotation = np.asarray(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
        dtype=np.float32,
    )
    world = local @ rotation.T + center[:2]
    ax.add_patch(Polygon(world, closed=True, **kwargs))


def _write_suite(
    *,
    config_dir: Path,
    artifact_dir: Path,
    source: dict[str, Any],
    source_path: Path,
    guidance: dict[str, Any],
    guidance_path: Path,
    records: list[dict[str, Any]],
) -> None:
    geometry_status = str(guidance["object_geometry_status"])
    fixture_status = (
        "object_geometry_finalized"
        if geometry_status == "finalized"
        else "geometry_only_inspection_candidate"
    )
    indices = [int(item["dataset_episode_index"]) for item in records]
    (config_dir / "episode_indices.txt").write_text(
        "# Box-derived E10 U review. Local output order 0--9.\n"
        + "\n".join(str(index) for index in indices)
        + "\n",
        encoding="utf-8",
    )
    fixture = {
        "schema_version": "pg3d.e10_u_shape_box_derived_review.v1",
        "fixture_id": guidance["fixture_id"],
        "status": fixture_status,
        "object_geometry": {
            "status": geometry_status,
            "finalized_on": guidance.get("finalized_on"),
            "change_policy": guidance.get("change_policy"),
        },
        "source_fixture": str(source_path),
        "guidance_file": str(guidance_path),
        "dataset": source["dataset"],
        "checkpoint": source["checkpoint"],
        "episode_count": len(records),
        "rules": guidance["shared_rules"],
        "validation": {
            "motion_planner_run": False,
            "live_physx_run": False,
            "whole_robot_clearance_validated": False,
            "policy_obstacle_visibility_measured": False,
            "comparison_trials_run": [],
            "rerun_policy_cloud_source": "saved obstacle-free dataset reset",
            "rerun_u_surface_source": "deterministic proposed-geometry surface sample",
        },
        "episodes": records,
    }
    (config_dir / "fixture.json").write_text(
        json.dumps(jsonable(fixture), indent=2, sort_keys=True), encoding="utf-8"
    )
    (artifact_dir / "placement_report.json").write_text(
        json.dumps(jsonable(fixture), indent=2, sort_keys=True), encoding="utf-8"
    )
    readme_status = (
        """The ten U-object dimensions, centers, yaws, component geometry, and matched constraint
files are finalized as version 1. Do not edit the v1 guidance or generated fixture in place; create
a new versioned fixture for any later geometry change.
"""
        if geometry_status == "finalized"
        else "These geometries remain inspection candidates and may still be revised.\n"
    )
    (config_dir / "README.md").write_text(
        """# Box-derived E10 U-shape fixture

This fixture replaces each frozen E3 box with a U having the same initial outer envelope and
center before the recorded versioned guidance adjustments. Its opening faces the episode start.

"""
        + readme_status
        + """
The Rerun policy cloud is the saved obstacle-free
dataset reset; `inspection/proposed_u_surface` is a deterministic sample of the proposed geometry.
The geometry-only review artifact itself does not embed the later MPLib paths, live PhysX results,
whole-arm clearance results, or any ITPS/reranking comparison outcome. Object finalization is
separate from per-episode benchmark validity; consult the non-convex experiment plan for the
predeclared validity strata.
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
