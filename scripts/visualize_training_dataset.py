from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pg3d.envs.maniskill_adapter.dataset import load_reach_metadata
from pg3d.eval import AvoidOverlayConfig, direct_path_avoid_region
from pg3d.utils.serialization import jsonable as _jsonable
from pg3d.viz.constraints import avoid_region_line_visuals


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import rerun as rr
    except Exception as exc:
        print(f"Failed to import rerun: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Install viz deps or pass --rerun none", file=sys.stderr)
        rr = None

    import zarr

    metadata = load_reach_metadata(args.dataset)
    root = zarr.open_group(str(args.dataset), mode="r")
    data = root["data"]
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    episode_starts = np.concatenate([np.asarray([0], dtype=np.int64), episode_ends[:-1]])
    if args.episode_index < 0 or args.episode_index >= len(episode_ends):
        raise IndexError(
            f"--episode-index {args.episode_index} is outside dataset episode range "
            f"[0, {len(episode_ends) - 1}]"
        )

    episode_metadata = list(metadata.get("episodes", []))
    selected_indices = _selected_episode_indices(
        episode_index=args.episode_index,
        same_seed_group=args.same_seed_group,
        episode_metadata=episode_metadata,
        episode_count=len(episode_ends),
    )
    trajectories = [
        _load_dataset_trajectory(
            data=data,
            episode_metadata=episode_metadata,
            episode_index=idx,
            start=int(episode_starts[idx]),
            end=int(episode_ends[idx]),
        )
        for idx in selected_indices
    ]
    trajectories = [traj for traj in trajectories if traj["tcp_path"].shape[0] >= 2]
    if args.success_only:
        before = len(trajectories)
        trajectories = [traj for traj in trajectories if bool(traj["success"])]
        print(f"kept successful dataset trajectories: {len(trajectories)}/{before}", flush=True)
    if not trajectories:
        raise RuntimeError("no dataset trajectories selected for plotting")

    first = trajectories[0]
    start_tcp = np.asarray(first["tcp_path"][0], dtype=np.float32).reshape(3)
    target = np.asarray(first["target_position"], dtype=np.float32).reshape(3)
    constraint = direct_path_avoid_region(
        start_tcp=start_tcp,
        target_position=target,
        config=AvoidOverlayConfig(
            radius=args.avoid_radius,
            min_radius=args.avoid_min_radius,
            margin=args.avoid_margin,
            weight=1.0,
        ),
    )

    if args.rerun is not None:
        if rr is None:
            raise RuntimeError("rerun output requested but rerun import failed")
        _write_rerun(
            rr=rr,
            output=args.rerun,
            trajectories=trajectories,
            constraint=constraint,
            scene_points=_initial_scene_points(data, int(episode_starts[selected_indices[0]])),
        )
    if args.video is not None:
        _write_video(
            output=args.video,
            trajectories=trajectories,
            constraint=constraint,
            scene_points=_initial_scene_points(data, int(episode_starts[selected_indices[0]])),
            fps=args.video_fps,
        )

    summary = {
        "dataset": str(args.dataset),
        "episode_index": args.episode_index,
        "same_seed_group": args.same_seed_group,
        "selected_episode_indices": selected_indices,
        "trajectory_count": len(trajectories),
        "trajectories": trajectories,
        "avoid_region": _constraint_summary(constraint),
        "rerun": str(args.rerun) if args.rerun is not None else None,
        "video": str(args.video) if args.video is not None else None,
    }
    summary_path = args.summary or _default_summary_path(args)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True), encoding="utf-8")
    print(
        "saved training dataset trajectory visualization: "
        f"episodes={selected_indices} trajectories={len(trajectories)}",
        flush=True,
    )
    if args.rerun is not None:
        print(f"saved rerun: {args.rerun}")
    if args.video is not None:
        print(f"saved video: {args.video}")
    print(f"saved summary: {summary_path}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize recorded training zarr TCP trajectories with a virtual avoid sphere."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument(
        "--same-seed-group",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="plot all dataset episodes sharing the selected episode seed",
    )
    parser.add_argument("--success-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--avoid-radius", type=float, default=0.08)
    parser.add_argument("--avoid-min-radius", type=float, default=0.08)
    parser.add_argument("--avoid-margin", type=float, default=0.0)
    parser.add_argument(
        "--rerun",
        type=str,
        default="artifacts/training_dataset_trajectories/episode0_dataset_trajectories.rrd",
        help="Rerun output path; pass none to disable",
    )
    parser.add_argument(
        "--video",
        type=str,
        default="artifacts/training_dataset_trajectories/episode0_dataset_trajectories.mp4",
        help="MP4 output path; pass none to disable",
    )
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--video-fps", type=int, default=10)
    args = parser.parse_args(argv)
    args.rerun = None if args.rerun.lower() in {"", "none", "null", "off"} else Path(args.rerun)
    args.video = None if args.video.lower() in {"", "none", "null", "off"} else Path(args.video)
    if args.avoid_radius <= 0 or args.avoid_min_radius <= 0:
        raise ValueError("avoid radii must be positive")
    if args.avoid_margin < 0:
        raise ValueError("--avoid-margin must be non-negative")
    if args.video_fps <= 0:
        raise ValueError("--video-fps must be positive")
    return args


def _selected_episode_indices(
    *,
    episode_index: int,
    same_seed_group: bool,
    episode_metadata: list[dict[str, Any]],
    episode_count: int,
) -> list[int]:
    if not same_seed_group:
        return [episode_index]
    selected_meta = episode_metadata[episode_index] if episode_index < len(episode_metadata) else {}
    seed = selected_meta.get("seed")
    if seed is None:
        return [episode_index]
    return [
        idx
        for idx in range(episode_count)
        if idx < len(episode_metadata) and episode_metadata[idx].get("seed") == seed
    ]


def _load_dataset_trajectory(
    *,
    data: Any,
    episode_metadata: list[dict[str, Any]],
    episode_index: int,
    start: int,
    end: int,
) -> dict[str, Any]:
    tcp_path = np.asarray(data["tcp_pose"][start:end, :3], dtype=np.float32)
    target_positions = np.asarray(data["target_position"][start:end], dtype=np.float32)
    success = np.asarray(data["success"][start:end], dtype=bool) if "success" in data else np.zeros((end - start,), dtype=bool)
    meta = episode_metadata[episode_index] if episode_index < len(episode_metadata) else {}
    first_success_step = int(np.argmax(success) + 1) if bool(np.any(success)) else None
    return {
        "episode_index": episode_index,
        "seed": meta.get("seed"),
        "steps": int(max(tcp_path.shape[0] - 1, 0)),
        "success": bool(np.any(success) or meta.get("success", False)),
        "first_success_step": first_success_step,
        "tcp_path": tcp_path,
        "target_position": target_positions[-1] if target_positions.size else np.zeros(3, dtype=np.float32),
        "final_distance": meta.get("final_distance"),
        "source": "training_dataset",
    }


def _initial_scene_points(data: Any, row_index: int) -> dict[str, np.ndarray]:
    valid = np.asarray(data["point_valid_mask"][row_index], dtype=bool)
    points = np.asarray(data["point_cloud"][row_index], dtype=np.float32)[valid]
    robot_mask = np.asarray(data["robot_mask"][row_index], dtype=bool)[valid]
    return {"points": points[~robot_mask]}


def _write_rerun(
    *,
    rr: Any,
    output: Path,
    trajectories: list[dict[str, Any]],
    constraint: Any,
    scene_points: dict[str, np.ndarray],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    rr.init("pg3d_training_dataset_trajectories", spawn=False)
    rr.save(str(output))
    rr.set_time_sequence("step", 0)
    points = scene_points["points"]
    if points.size:
        rr.log("world/point_cloud", rr.Points3D(points, colors=[150, 150, 150], radii=0.003))
    first_path = trajectories[0]["tcp_path"]
    target = np.asarray(trajectories[0]["target_position"], dtype=np.float32).reshape(1, 3)
    start = first_path[0:1].astype(np.float32)
    rr.log("world/goal", rr.Points3D(target, colors=[0, 255, 0], radii=0.018))
    rr.log("world/start_tcp", rr.Points3D(start, colors=[255, 220, 0], radii=0.014))
    for visual in avoid_region_line_visuals([constraint]):
        rr.log(
            f"world/constraints/{visual.name}",
            rr.LineStrips3D(visual.line_strips, colors=visual.color),
            static=True,
        )
    palette = _palette(len(trajectories))
    for idx, trajectory in enumerate(trajectories):
        path = np.asarray(trajectory["tcp_path"], dtype=np.float32)
        if path.shape[0] < 2:
            continue
        color = tuple(int(v) for v in palette[idx][:3])
        rr.log(
            f"world/training_dataset/episode_{int(trajectory['episode_index']):04d}",
            rr.LineStrips3D([path], colors=color),
            static=True,
        )
        rr.log(
            f"world/training_dataset/end_{int(trajectory['episode_index']):04d}",
            rr.Points3D(path[-1:].astype(np.float32), colors=color, radii=0.01),
            static=True,
        )
    rr.disconnect()


def _write_video(
    *,
    output: Path,
    trajectories: list[dict[str, Any]],
    constraint: Any,
    scene_points: dict[str, np.ndarray],
    fps: int,
) -> None:
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    paths = [np.asarray(traj["tcp_path"], dtype=np.float32) for traj in trajectories]
    target = np.asarray(trajectories[0]["target_position"], dtype=np.float32).reshape(3)
    start = paths[0][0]
    center = np.asarray(constraint.region.center, dtype=np.float32).reshape(3)
    radius = float(constraint.region.radius)
    bounds = _plot_bounds(paths=paths, start=start, target=target, center=center, radius=radius)
    palette = plt.cm.tab20(np.linspace(0.0, 1.0, max(len(paths), 1)))
    frames = []
    frame_count = max(24, fps * 4)
    for frame_idx in range(frame_count):
        fig = plt.figure(figsize=(8.0, 6.0), dpi=140)
        ax = fig.add_subplot(111, projection="3d")
        scene = scene_points["points"]
        if scene.size:
            ax.scatter(scene[:, 0], scene[:, 1], scene[:, 2], s=2, c="#9ca3af", alpha=0.16)
        _plot_sphere_wire(ax, center=center, radius=radius, color="#ff4010")
        ax.scatter([start[0]], [start[1]], [start[2]], c="gold", s=55, edgecolors="black")
        ax.scatter([target[0]], [target[1]], [target[2]], c="limegreen", s=70, edgecolors="black")
        for idx, path in enumerate(paths):
            color = palette[idx % len(palette)]
            ax.plot(path[:, 0], path[:, 1], path[:, 2], color=color, linewidth=2.0, alpha=0.95)
            ax.scatter(path[-1:, 0], path[-1:, 1], path[-1:, 2], color=color, s=22)
        ax.set_xlim(bounds[0])
        ax.set_ylim(bounds[1])
        ax.set_zlim(bounds[2])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title("Training dataset trajectories with virtual avoid sphere")
        ax.view_init(elev=24, azim=-70 + 360.0 * frame_idx / frame_count)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(_canvas_rgb_array(fig.canvas))
        plt.close(fig)
    imageio.mimsave(output, frames, fps=fps, macro_block_size=16)


def _plot_bounds(
    *,
    paths: list[np.ndarray],
    start: np.ndarray,
    target: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    sphere_extents = np.asarray(
        [
            center + np.asarray([radius, 0.0, 0.0], dtype=np.float32),
            center - np.asarray([radius, 0.0, 0.0], dtype=np.float32),
            center + np.asarray([0.0, radius, 0.0], dtype=np.float32),
            center - np.asarray([0.0, radius, 0.0], dtype=np.float32),
            center + np.asarray([0.0, 0.0, radius], dtype=np.float32),
            center - np.asarray([0.0, 0.0, radius], dtype=np.float32),
        ],
        dtype=np.float32,
    )
    points = np.concatenate([*paths, start.reshape(1, 3), target.reshape(1, 3), sphere_extents], axis=0)
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    mid = (mins + maxs) * 0.5
    span = max(float(np.max(maxs - mins)) * 1.15, 0.16)
    half = span * 0.5
    return (
        (float(mid[0] - half), float(mid[0] + half)),
        (float(mid[1] - half), float(mid[1] + half)),
        (float(mid[2] - half), float(mid[2] + half)),
    )


def _palette(count: int) -> list[tuple[int, int, int]]:
    base = [
        (70, 130, 255),
        (255, 150, 40),
        (80, 210, 120),
        (210, 90, 255),
        (255, 90, 120),
        (80, 220, 220),
        (235, 205, 70),
        (130, 110, 255),
        (255, 105, 210),
        (105, 190, 90),
        (245, 120, 80),
        (120, 220, 170),
    ]
    return [base[idx % len(base)] for idx in range(max(count, 1))]


def _canvas_rgb_array(canvas: Any) -> np.ndarray:
    width, height = canvas.get_width_height()
    if hasattr(canvas, "buffer_rgba"):
        rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
        return rgba[:, :, :3].copy()
    if hasattr(canvas, "tostring_rgb"):
        return np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
    if hasattr(canvas, "tostring_argb"):
        argb = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8).reshape(height, width, 4)
        return argb[:, :, [1, 2, 3]].copy()
    raise AttributeError("Matplotlib canvas cannot export RGB pixels")


def _plot_sphere_wire(ax: Any, *, center: np.ndarray, radius: float, color: str) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 80)
    circles = [
        np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=1),
        np.stack([np.cos(theta), np.zeros_like(theta), np.sin(theta)], axis=1),
        np.stack([np.zeros_like(theta), np.cos(theta), np.sin(theta)], axis=1),
    ]
    for circle in circles:
        pts = center.reshape(1, 3) + radius * circle
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, linewidth=1.2, alpha=0.9)


def _constraint_summary(constraint: Any) -> dict[str, Any]:
    region = constraint.region
    return {
        "name": constraint.name,
        "center": np.asarray(region.center, dtype=np.float32).tolist(),
        "radius": float(region.radius),
        "margin": float(constraint.margin),
    }


def _default_summary_path(args: argparse.Namespace) -> Path:
    if args.rerun is not None:
        return args.rerun.with_suffix(".summary.json")
    if args.video is not None:
        return args.video.with_suffix(".summary.json")
    return Path("artifacts/training_dataset_trajectories/episode0_dataset_trajectories.summary.json")


if __name__ == "__main__":
    raise SystemExit(main())
