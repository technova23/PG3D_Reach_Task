from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from pg3d.envs.maniskill_adapter.dataset import load_reach_metadata
from pg3d.policies.dp3.goal_markers import (
    DEFAULT_GOAL_MARKER_POINTS,
    DEFAULT_GOAL_MARKER_RADIUS,
    goal_marker_offsets,
    goal_marker_points,
)
from pg3d.utils.serialization import jsonable


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = zarr.open_group(str(args.dataset), mode="r")
    data = root["data"]
    metadata = load_reach_metadata(args.dataset)
    target_key = _target_key(data)
    required = {"point_cloud", target_key, "tcp_pose"}
    missing = required.difference(data.keys())
    if missing:
        raise ValueError(f"dataset missing required arrays: {sorted(missing)}")

    total_rows = int(data["point_cloud"].shape[0])
    row_indices = _sample_rows(
        total_rows,
        sample_count=args.samples,
        seed=args.seed,
        mode=args.sample_mode,
        start=args.start,
    )
    arrays = _read_rows(data, target_key=target_key, row_indices=row_indices)
    report = audit_rows(
        arrays,
        row_indices=row_indices,
        dataset=args.dataset,
        metadata=metadata,
        marker_points=args.goal_marker_points,
        marker_radius=args.goal_marker_radius,
        centroid_threshold=args.centroid_threshold,
        suspicious_far_distance=args.suspicious_far_distance,
        suspicious_tiny_distance=args.suspicious_tiny_distance,
    )
    text = json.dumps(jsonable(report), indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")

    if args.strict:
        failures = report["assertions"]["failures"]
        if failures:
            raise AssertionError("; ".join(str(failure) for failure in failures))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit consistency between stored goal markers, target_position, and TCP pose."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-mode", choices=["random", "linspace", "contiguous"], default="random")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--goal-marker-points", type=int, default=DEFAULT_GOAL_MARKER_POINTS)
    parser.add_argument("--goal-marker-radius", type=float, default=DEFAULT_GOAL_MARKER_RADIUS)
    parser.add_argument("--centroid-threshold", type=float, default=0.01)
    parser.add_argument("--suspicious-far-distance", type=float, default=2.0)
    parser.add_argument("--suspicious-tiny-distance", type=float, default=0.0001)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if args.start < 0:
        raise ValueError("--start must be non-negative")
    if args.goal_marker_points < 0:
        raise ValueError("--goal-marker-points must be non-negative")
    if args.goal_marker_radius < 0.0:
        raise ValueError("--goal-marker-radius must be non-negative")
    return args


def audit_rows(
    arrays: dict[str, np.ndarray],
    *,
    row_indices: np.ndarray,
    dataset: Path,
    metadata: dict[str, Any],
    marker_points: int,
    marker_radius: float,
    centroid_threshold: float,
    suspicious_far_distance: float,
    suspicious_tiny_distance: float,
) -> dict[str, Any]:
    point_cloud = arrays["point_cloud"].astype(np.float32, copy=False)
    goal_xyz = arrays["goal_xyz"].astype(np.float32, copy=False)
    tcp_pose = arrays["tcp_pose"].astype(np.float32, copy=False)
    valid_mask = arrays.get("point_valid_mask")
    if valid_mask is None:
        valid_mask = np.ones(point_cloud.shape[:2], dtype=bool)
    else:
        valid_mask = valid_mask.astype(bool, copy=False)

    ee_position = tcp_pose[:, :3]
    goal_rel = goal_xyz - ee_position
    goal_rel_norm = np.linalg.norm(goal_rel, axis=1)
    markers = point_cloud[:, -marker_points:, :] if marker_points else np.zeros((len(row_indices), 0, 3))
    marker_centroid = (
        markers.mean(axis=1) if marker_points else np.zeros_like(goal_xyz)
    )
    centroid_error = np.linalg.norm(marker_centroid - goal_xyz, axis=1)
    expected_markers = goal_marker_points(
        goal_xyz,
        num_points=marker_points,
        radius=marker_radius,
    )
    expected_marker_max_abs_error = (
        np.max(np.abs(markers - expected_markers), axis=(1, 2))
        if marker_points
        else np.zeros((len(row_indices),), dtype=np.float32)
    )
    distances_to_goal = np.linalg.norm(point_cloud - goal_xyz[:, None, :], axis=-1)
    nearest_distance = np.min(np.where(valid_mask, distances_to_goal, np.inf), axis=1)

    centroid_bad = np.flatnonzero(centroid_error > centroid_threshold)
    far_bad = np.flatnonzero(goal_rel_norm > suspicious_far_distance)
    tiny_bad = np.flatnonzero(goal_rel_norm < suspicious_tiny_distance)
    finite_failures = [
        name
        for name, value in {
            "goal_xyz": goal_xyz,
            "ee_position": ee_position,
            "goal_rel": goal_rel,
            "point_cloud": point_cloud,
        }.items()
        if not np.isfinite(value).all()
    ]
    crop = metadata.get("crop", {}) if isinstance(metadata.get("crop", {}), dict) else {}
    bounds = np.asarray(crop.get("bounds", []), dtype=np.float32)
    point_cloud_valid = point_cloud[valid_mask]

    failures: list[str] = []
    if centroid_bad.size:
        failures.append(f"{centroid_bad.size} marker centroids exceed {centroid_threshold}m")
    if far_bad.size:
        failures.append(f"{far_bad.size} goal_rel magnitudes exceed {suspicious_far_distance}m")
    if tiny_bad.size:
        failures.append(f"{tiny_bad.size} goal_rel magnitudes below {suspicious_tiny_distance}m")
    for name in finite_failures:
        failures.append(f"{name} contains non-finite values")

    return {
        "dataset": str(dataset),
        "sample_count": int(len(row_indices)),
        "sample_rows": row_indices.astype(int).tolist(),
        "coordinate_frame_trace": {
            "point_cloud_xyz": "ManiSkill pointcloud.xyzw[:, :3], cropped/padded in environment/world coordinates; not normalized in zarr.",
            "goal_marker_points": "Stored as final K point-cloud rows, generated as target_position + deterministic offsets in same coordinates; generation-time saliency injection clips markers to crop bounds.",
            "target_position": "ManiSkill obs['extra']['goal_pos']; same environment/world coordinates as point cloud and TCP.",
            "goal_xyz": "Alias of target_position used by explicit goal encoder.",
            "tcp_pose[:3]": "ManiSkill obs['extra']['tcp_pose'] position; same environment/world coordinates.",
            "ee_position": "Alias of tcp_pose[:3].",
            "goal_rel": "goal_xyz - ee_position in environment/world coordinate differences.",
        },
        "metadata": {
            "env_id": metadata.get("env_id"),
            "action_mode": metadata.get("action_mode"),
            "crop": crop,
            "point_cloud_saliency": metadata.get("point_cloud_saliency"),
        },
        "marker_generation": {
            "marker_points": int(marker_points),
            "marker_radius": float(marker_radius),
            "offset_centroid": goal_marker_offsets(
                num_points=marker_points,
                radius=marker_radius,
            ).mean(axis=0).astype(float).tolist()
            if marker_points
            else [0.0, 0.0, 0.0],
            "note": (
                "Centroid error includes the deterministic offset pattern centroid. "
                "For exact marker equality, inspect expected_marker_max_abs_error."
            ),
        },
        "marker_centroid_error": {
            **_summary(centroid_error),
            "threshold": float(centroid_threshold),
            "count_gt_threshold": int(centroid_bad.size),
            "example_rows": row_indices[centroid_bad[:10]].astype(int).tolist(),
        },
        "expected_marker_max_abs_error": {
            **_summary(expected_marker_max_abs_error),
            "count_gt_1cm": int(np.count_nonzero(expected_marker_max_abs_error > 0.01)),
        },
        "goal_rel_norm": {
            **_summary(goal_rel_norm),
            "count_gt_far_threshold": int(far_bad.size),
            "count_lt_tiny_threshold": int(tiny_bad.size),
            "example_far_rows": row_indices[far_bad[:10]].astype(int).tolist(),
            "example_tiny_rows": row_indices[tiny_bad[:10]].astype(int).tolist(),
        },
        "nearest_point_to_goal": _summary(nearest_distance),
        "ranges": {
            "point_cloud_valid_min": (
                point_cloud_valid.min(axis=0).astype(float).tolist()
                if point_cloud_valid.size
                else None
            ),
            "point_cloud_valid_max": (
                point_cloud_valid.max(axis=0).astype(float).tolist()
                if point_cloud_valid.size
                else None
            ),
            "goal_xyz_min": goal_xyz.min(axis=0).astype(float).tolist(),
            "goal_xyz_max": goal_xyz.max(axis=0).astype(float).tolist(),
            "ee_position_min": ee_position.min(axis=0).astype(float).tolist(),
            "ee_position_max": ee_position.max(axis=0).astype(float).tolist(),
            "crop_bounds": bounds.astype(float).tolist() if bounds.shape == (3, 2) else None,
        },
        "assertions": {
            "finite_goal_xyz": bool(np.isfinite(goal_xyz).all()),
            "finite_ee_position": bool(np.isfinite(ee_position).all()),
            "finite_goal_rel": bool(np.isfinite(goal_rel).all()),
            "finite_point_cloud": bool(np.isfinite(point_cloud).all()),
            "all_centroid_errors_below_threshold": bool(np.all(centroid_error < centroid_threshold)),
            "failures": failures,
        },
        "recommended_assertions": [
            "assert np.isfinite(goal_xyz).all()",
            "assert np.isfinite(ee_position).all()",
            "assert np.isfinite(goal_rel).all()",
            f"assert centroid_error < {centroid_threshold}",
        ],
        "recommended_insertion_points": {
            "dataset_loading": "ReachSequenceDataset.__getitem__ after reading target_position and tcp_pose.",
            "rollout_inference": "obs_window_to_torch after stacking target_position/tcp_pose into goal_xyz/ee_position.",
            "encoder_forward": "DP3Encoder.forward immediately before goal_encoder(goal_rel).",
        },
    }


def _read_rows(data: Any, *, target_key: str, row_indices: np.ndarray) -> dict[str, np.ndarray]:
    arrays = {
        "point_cloud": np.asarray(data["point_cloud"][:], dtype=np.float32)[row_indices],
        "goal_xyz": np.asarray(data[target_key][:], dtype=np.float32)[row_indices],
        "tcp_pose": np.asarray(data["tcp_pose"][:], dtype=np.float32)[row_indices],
    }
    if "point_valid_mask" in data:
        arrays["point_valid_mask"] = np.asarray(data["point_valid_mask"][:], dtype=bool)[
            row_indices
        ]
    return arrays


def _sample_rows(
    total_rows: int,
    *,
    sample_count: int,
    seed: int,
    mode: str,
    start: int,
) -> np.ndarray:
    if total_rows <= 0:
        raise ValueError("dataset has no rows")
    count = min(sample_count, total_rows)
    if mode == "random":
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(total_rows, size=count, replace=False)).astype(np.int64)
    if mode == "linspace":
        return np.linspace(0, total_rows - 1, count).round().astype(np.int64)
    if mode == "contiguous":
        if start >= total_rows:
            raise ValueError(f"--start {start} is outside dataset with {total_rows} rows")
        stop = min(start + count, total_rows)
        return np.arange(start, stop, dtype=np.int64)
    raise ValueError(f"unsupported sample mode {mode!r}")


def _target_key(data: Any) -> str:
    if "target_position" in data:
        return "target_position"
    if "goal_pos" in data:
        return "goal_pos"
    raise ValueError("dataset has neither /data/target_position nor /data/goal_pos")


def _summary(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"min": float("nan"), "mean": float("nan"), "median": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "min": float(np.min(finite)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


if __name__ == "__main__":
    raise SystemExit(main())
