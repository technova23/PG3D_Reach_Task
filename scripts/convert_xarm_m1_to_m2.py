#!/usr/bin/env python
"""Convert xArm7 simulator artifacts from M1 world frame to M2 base frame.

M1 xArm sim placed the robot base at ``[-0.615, 0, 0]`` in simulator world
coordinates. M2 makes the robot base frame the simulator world frame, matching
the real xArm data convention. This script is intentionally one-way:

    M2 XYZ = M1 XYZ + [0.615, 0, 0]

It shifts only Cartesian positions. Joint states, actions, masks, radii,
extents, axes, and orientations are preserved.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from pg3d.envs.xarm_adapter.reach_config import M1_TO_M2_XYZ_SHIFT

XYZ_ARRAY_KEYS = {
    "point_cloud",
    "target_position",
    "goal_pos",
    "eef_pos",
    "start_position",
    "tcp_position",
}
POSE_ARRAY_KEYS = {
    "tcp_pose",
}
COORDINATE_FRAME = "xarm7_base_m2"
CONVERTED_FROM = "xarm7_sim_m1"


def default_output_path(path: Path) -> Path:
    """Return a sibling output path with an ``_m2`` suffix."""
    if path.suffix:
        return path.with_name(f"{path.stem}_m2{path.suffix}")
    return path.with_name(f"{path.name}_m2")


def convert_zarr(
    input_zarr: Path | str,
    output_zarr: Path | str | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    """Copy a Zarr dataset and shift supported Cartesian arrays into M2."""
    input_path = Path(input_zarr)
    output_path = default_output_path(input_path) if output_zarr is None else Path(output_zarr)
    if not input_path.exists():
        raise FileNotFoundError(f"input zarr does not exist: {input_path}")
    _copytree(input_path, output_path, overwrite=overwrite)

    root = zarr.open_group(str(output_path), mode="a")
    data = root["data"]
    shifted: list[str] = []
    for key in sorted(XYZ_ARRAY_KEYS):
        if key in data:
            _shift_xyz_array(data[key])
            shifted.append(key)
    for key in sorted(POSE_ARRAY_KEYS):
        if key in data:
            _shift_pose_array(data[key])
            shifted.append(f"{key}[:3]")

    root.attrs["coordinate_frame"] = COORDINATE_FRAME
    root.attrs["converted_from"] = CONVERTED_FROM
    root.attrs["m1_to_m2_xyz_shift"] = M1_TO_M2_XYZ_SHIFT.astype(float).tolist()
    root.attrs["shifted_arrays"] = shifted
    _update_metadata_json(output_path, shifted_arrays=shifted)
    return output_path


def convert_constraints_dir(
    input_dir: Path | str,
    output_dir: Path | str | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    """Copy a directory of constraint JSON files and shift coordinates into M2."""
    input_path = Path(input_dir)
    output_path = default_output_path(input_path) if output_dir is None else Path(output_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"constraints dir does not exist: {input_path}")
    _copytree(input_path, output_path, overwrite=overwrite)

    for path in sorted(output_path.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        shifted = shift_constraints_payload(payload)
        path.write_text(json.dumps(shifted, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def shift_constraints_payload(payload: Any) -> Any:
    """Shift one loaded constraint JSON payload from M1 to M2 coordinates."""
    if isinstance(payload, list):
        return [shift_constraint_config(item) for item in payload]
    if isinstance(payload, dict):
        return shift_constraint_config(payload)
    raise ValueError(f"constraint payload must be a dict or list, got {type(payload).__name__}")


def shift_constraint_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a shifted copy of one JSON-safe constraint config."""
    shifted = dict(config)
    if "region" in shifted and isinstance(shifted["region"], dict):
        shifted["region"] = shift_region_config(shifted["region"])
    if shifted.get("type") == "cartesian_pose" and "target_position" in shifted:
        shifted["target_position"] = _shift_vector(shifted["target_position"], dims=3)
    if shifted.get("type") == "cartesian_pose":
        metadata = dict(shifted.get("metadata", {}))
        metadata.setdefault("coordinate_frame", COORDINATE_FRAME)
        metadata.setdefault("converted_from", CONVERTED_FROM)
        shifted["metadata"] = metadata
    return shifted


def shift_region_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a shifted copy of one JSON-safe region config."""
    shifted = dict(config)
    region_type = shifted.get("type")
    if region_type in {"sphere", "box", "cylinder"} and "center" in shifted:
        shifted["center"] = _shift_vector(shifted["center"], dims=3)
    elif region_type == "rect2d" and "center" in shifted:
        shifted["center"] = _shift_vector(shifted["center"], dims=2)
    return shifted


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert xArm7 M1 simulator Zarr/constraint artifacts to M2 base frame."
    )
    parser.add_argument("--input-zarr", type=Path, default=None)
    parser.add_argument("--output-zarr", type=Path, default=None)
    parser.add_argument("--constraints-dir", type=Path, default=None)
    parser.add_argument("--output-constraints-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.input_zarr is None and args.constraints_dir is None:
        raise ValueError("provide --input-zarr, --constraints-dir, or both")
    if args.output_zarr is not None and args.input_zarr is None:
        raise ValueError("--output-zarr requires --input-zarr")
    if args.output_constraints_dir is not None and args.constraints_dir is None:
        raise ValueError("--output-constraints-dir requires --constraints-dir")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.input_zarr is not None:
        out_zarr = convert_zarr(
            args.input_zarr,
            args.output_zarr,
            overwrite=args.overwrite,
        )
        print(f"converted zarr: {args.input_zarr} -> {out_zarr}")
    if args.constraints_dir is not None:
        out_constraints = convert_constraints_dir(
            args.constraints_dir,
            args.output_constraints_dir,
            overwrite=args.overwrite,
        )
        print(f"converted constraints: {args.constraints_dir} -> {out_constraints}")
    return 0


def _copytree(input_path: Path, output_path: Path, *, overwrite: bool) -> None:
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")
        shutil.rmtree(output_path)
    shutil.copytree(input_path, output_path)


def _shift_xyz_array(array: Any) -> None:
    values = np.asarray(array[:], dtype=np.float32)
    if values.shape[-1] != 3:
        raise ValueError(f"{array.name} must end with XYZ dimension 3, got {values.shape}")
    array[:] = (values + M1_TO_M2_XYZ_SHIFT.reshape((1,) * (values.ndim - 1) + (3,))).astype(
        np.float32
    )


def _shift_pose_array(array: Any) -> None:
    values = np.asarray(array[:], dtype=np.float32)
    if values.shape[-1] < 3:
        raise ValueError(f"{array.name} must have at least 3 pose values, got {values.shape}")
    values[..., :3] += M1_TO_M2_XYZ_SHIFT.reshape((1,) * (values.ndim - 1) + (3,))
    array[:] = values.astype(np.float32)


def _shift_vector(value: Any, *, dims: int) -> list[float]:
    vector = np.asarray(value, dtype=np.float32).reshape(dims)
    if dims == 3:
        vector = vector + M1_TO_M2_XYZ_SHIFT
    elif dims == 2:
        vector = vector + M1_TO_M2_XYZ_SHIFT[:2]
    else:
        raise ValueError(f"unsupported vector dims {dims}")
    return vector.astype(float).tolist()


def _update_metadata_json(output_path: Path, *, shifted_arrays: list[str]) -> None:
    metadata_path = output_path / "metadata.json"
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    shifted_metadata_fields = metadata.get("shifted_metadata_fields")
    if shifted_metadata_fields is None:
        shifted_metadata_fields = _shift_metadata_coordinates(metadata)
    metadata["coordinate_frame"] = COORDINATE_FRAME
    metadata["converted_from"] = CONVERTED_FROM
    metadata["m1_to_m2_xyz_shift"] = M1_TO_M2_XYZ_SHIFT.astype(float).tolist()
    metadata["shifted_arrays"] = list(shifted_arrays)
    metadata["shifted_metadata_fields"] = list(shifted_metadata_fields)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def _shift_metadata_coordinates(metadata: dict[str, Any]) -> list[str]:
    """Shift known absolute Cartesian fields in reach-dataset metadata."""
    shifted: list[str] = []

    crop = metadata.get("crop")
    if isinstance(crop, dict) and "bounds" in crop:
        bounds = np.asarray(crop["bounds"], dtype=np.float32)
        if bounds.shape != (3, 2):
            raise ValueError(f"metadata crop.bounds must have shape (3, 2), got {bounds.shape}")
        crop["bounds"] = (bounds + M1_TO_M2_XYZ_SHIFT.reshape(3, 1)).astype(float).tolist()
        shifted.append("crop.bounds")

    env_kwargs = metadata.get("env_kwargs")
    if isinstance(env_kwargs, dict):
        if "goal_center" in env_kwargs:
            env_kwargs["goal_center"] = _shift_vector(env_kwargs["goal_center"], dims=3)
            shifted.append("env_kwargs.goal_center")
        goal_regions = env_kwargs.get("goal_regions")
        if isinstance(goal_regions, list):
            for region in goal_regions:
                if isinstance(region, dict) and "center" in region:
                    region["center"] = _shift_vector(region["center"], dims=3)
            if any(isinstance(region, dict) and "center" in region for region in goal_regions):
                shifted.append("env_kwargs.goal_regions[*].center")

    episodes = metadata.get("episodes")
    if isinstance(episodes, list):
        field_counts: dict[str, int] = {}
        for episode in episodes:
            if not isinstance(episode, dict):
                continue
            for key in ("target_position",):
                if key in episode:
                    episode[key] = _shift_vector(episode[key], dims=3)
                    field_counts[key] = field_counts.get(key, 0) + 1
            for key in ("start_tcp_pose", "goal_pose"):
                if key in episode:
                    episode[key] = _shift_metadata_pose(episode[key])
                    field_counts[key] = field_counts.get(key, 0) + 1
            start_sampling = episode.get("start_sampling")
            if isinstance(start_sampling, dict):
                for key in ("actual_position", "sampled_position"):
                    if key in start_sampling:
                        start_sampling[key] = _shift_vector(start_sampling[key], dims=3)
                        path = f"start_sampling.{key}"
                        field_counts[path] = field_counts.get(path, 0) + 1
            waypoints = episode.get("trajectory_waypoints")
            if isinstance(waypoints, list):
                episode["trajectory_waypoints"] = [
                    _shift_vector(waypoint, dims=3) for waypoint in waypoints
                ]
                field_counts["trajectory_waypoints"] = (
                    field_counts.get("trajectory_waypoints", 0) + 1
                )
        shifted.extend(
            f"episodes[*].{field} ({count} episodes)"
            for field, count in sorted(field_counts.items())
        )
    return shifted


def _shift_metadata_pose(value: Any) -> list[float]:
    pose = np.asarray(value, dtype=np.float32).reshape(-1)
    if pose.size < 3:
        raise ValueError(f"metadata pose must have at least 3 values, got {pose.shape}")
    shifted = pose.copy()
    shifted[:3] += M1_TO_M2_XYZ_SHIFT
    return shifted.astype(float).tolist()


if __name__ == "__main__":
    raise SystemExit(main())
