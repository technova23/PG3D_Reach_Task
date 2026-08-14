"""Export neutral pg3d policy-input bundles with the isolated Rerun 0.35 SDK."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import rerun as rr

REQUIRED_RERUN_VERSION = "0.35.0"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args()
    if rr.__version__ != REQUIRED_RERUN_VERSION:
        raise RuntimeError(
            f"exporter requires rerun-sdk=={REQUIRED_RERUN_VERSION}, got {rr.__version__}"
        )
    if args.validate is not None:
        return _validate(args.validate)
    if args.bundle is None or args.metadata is None or args.output is None:
        parser.error("--bundle, --metadata, and --output are required for export")
    export(args.bundle, args.metadata, args.output)
    return _validate(args.output)


def export(bundle_path: Path, metadata_path: Path, output_path: Path) -> None:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(bundle_path, allow_pickle=False) as bundle:
        points = bundle["point_cloud"]
        colors = bundle["colors"]
        valid = bundle["valid_mask"]
        robot = bundle["robot_mask"]
        obstacle = bundle["obstacle_mask"]
        scene = bundle["scene_mask"]
        goal = bundle["goal_mask"]
        target = bundle["target_position"]
        tcp = bundle["tcp_position"]
        clearance = bundle["tcp_clearance"]
        itps_robot_points = (
            bundle["itps_robot_points"].copy() if "itps_robot_points" in bundle.files else None
        )
        itps_robot_link_indices = (
            bundle["itps_robot_link_indices"].copy()
            if "itps_robot_link_indices" in bundle.files
            else None
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rr.RecordingStream("pg3d_dp3_reach_policy_rollout") as recording:
        recording.save(output_path)
        recording.log(
            "recording/provenance",
            rr.TextDocument(
                json.dumps(
                    {
                        "rerun_writer_version": rr.__version__,
                        "neutral_bundle": str(bundle_path),
                        "policy_tensor_shape": list(points.shape[1:]),
                    },
                    sort_keys=True,
                )
            ),
            static=True,
        )
        recording.log(
            "recording/identity",
            rr.TextDocument(json.dumps(metadata.get("recording_identity", {}), sort_keys=True)),
            static=True,
        )
        for visual in metadata.get("constraint_visuals", []):
            recording.log(
                f"world/constraints/{visual['name']}",
                rr.LineStrips3D(visual["line_strips"], colors=visual["color"]),
                static=True,
            )
        executed: list[np.ndarray] = []
        for step in range(points.shape[0]):
            recording.set_time("step", sequence=step)
            recording.log(
                "policy_input/point_cloud",
                rr.Points3D(points[step], colors=colors[step]),
            )
            for name, mask, color in (
                ("robot_points", robot[step] & valid[step], [0, 128, 255]),
                ("obstacle_points", obstacle[step] & valid[step], [180, 90, 20]),
                ("scene_points", scene[step] & valid[step], [160, 160, 160]),
                ("goal_marker_points", goal[step] & valid[step], [0, 255, 0]),
            ):
                selected = points[step][mask]
                if selected.size:
                    recording.log(f"policy_input/{name}", rr.Points3D(selected, colors=color))
            recording.log("world/goal", rr.Points3D(target[step].reshape(1, 3), colors=[0, 255, 0]))
            recording.log("world/tcp", rr.Points3D(tcp[step].reshape(1, 3), colors=[255, 220, 0]))
            executed.append(tcp[step])
            if len(executed) >= 2:
                recording.log(
                    "world/executed_tcp_path",
                    rr.LineStrips3D([np.asarray(executed)], colors=[255, 220, 0]),
                )
            if np.isfinite(clearance[step]):
                recording.log("metrics/min_clearance_m", rr.Scalars(clearance[step]))
                recording.log(
                    "metrics/constraint_violation",
                    rr.Scalars(float(clearance[step] < 0.0)),
                )
        for replan in metadata.get("replans", []):
            recording.set_time("step", sequence=int(replan["step"]))
            replan_index = int(replan["replan_index"])
            for candidate in replan.get("candidates", []):
                color = [40, 200, 80] if candidate.get("feasible") else [220, 60, 40]
                recording.log(
                    f"planning/replan_{replan_index:03d}/candidates/{int(candidate['index']):03d}",
                    rr.LineStrips3D([candidate["eef_path"]], colors=color),
                )
            selected = replan.get("selected_eef_path")
            if selected is not None:
                recording.log(
                    f"planning/replan_{replan_index:03d}/selected",
                    rr.LineStrips3D([selected], colors=[255, 0, 255], radii=0.004),
                )
            bundle_index = replan.get("itps_robot_points_bundle_index")
            if bundle_index is not None:
                if itps_robot_points is None or itps_robot_link_indices is None:
                    raise RuntimeError("ITPS replan metadata has no bundled robot geometry")
                link_palette = np.asarray(
                    [
                        [31, 119, 180],
                        [255, 127, 14],
                        [44, 160, 44],
                        [214, 39, 40],
                        [148, 103, 189],
                        [140, 86, 75],
                        [227, 119, 194],
                        [127, 127, 127],
                        [188, 189, 34],
                        [23, 190, 207],
                    ],
                    dtype=np.uint8,
                )
                colors_by_link = link_palette[itps_robot_link_indices]
                rollout = itps_robot_points[int(bundle_index)]
                for horizon_index, cloud in enumerate(rollout):
                    recording.log(
                        f"planning/replan_{replan_index:03d}/robot/horizon_{horizon_index:03d}",
                        rr.Points3D(cloud, colors=colors_by_link, radii=0.003),
                    )
                for worst in replan.get("itps_worst_points", []):
                    recording.log(
                        f"planning/replan_{replan_index:03d}/worst_points/constraint_"
                        f"{int(worst['constraint_index']):03d}",
                        rr.Points3D([worst["position"]], colors=[255, 0, 0], radii=0.008),
                    )


def _validate(path: Path) -> int:
    cli = Path(sys.executable).with_name("rerun")
    result = subprocess.run(
        [str(cli), "rrd", "print", str(path)],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
    )
    if result.returncode:
        sys.stderr.write(result.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
