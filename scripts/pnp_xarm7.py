#!/usr/bin/env python
"""Pick-and-place for xArm7 + gripper, built up in stages.

Fresh script, deliberately independent of the reach-eval-derived
``eval_pose_variety_pick_and_place.py``. The scene is
``pg3d.envs.xarm_adapter.pnp_env``: one red 7cm cube to pick, one green
marker where it goes.

Stages, run one at a time via ``--mode`` so each is verified before the next
is trusted:

* ``scene``  -- STAGE 1 (this one). Reset the env a few times and save a
  single tiled PNG of the spawns. No policy, no checkpoint, no motion: purely
  "does the scene look right and do objects spawn sensibly across the
  workspace". Everything downstream depends on this being right, so it is
  looked at first.
* ``tune-gripper`` -- STAGE 2. Sweep gripper force/close settings against a
  scripted top-down grasp and report which combinations actually hold the
  cube through a lift. No learned policy involved: this isolates the physics.
* ``pick``   -- STAGE 3. Checkpoint-steered approach + scripted grasp + lift.
* ``pick-place`` -- STAGE 4. Adds the transport-and-release half.

Only ``scene`` is implemented so far -- the later modes exit with a clear
message rather than pretending to run.

Stage 1 usage:
    python scripts/pnp_xarm7.py --mode scene --out artifacts/pnp/scene.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _save_png(path: Path, image: np.ndarray) -> None:
    """Write an RGB uint8 array to `path`, via whichever imaging lib exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio

        imageio.imwrite(str(path), image)
        return
    except Exception:  # noqa: BLE001
        pass
    try:
        from PIL import Image

        Image.fromarray(image).save(str(path))
        return
    except Exception:  # noqa: BLE001
        pass
    import cv2

    cv2.imwrite(str(path), image[:, :, ::-1])  # RGB -> BGR


def _tile(images: list[np.ndarray], cols: int) -> np.ndarray:
    """Tile equal-sized frames into one image, padding the last row."""
    if len(images) == 1:
        return images[0]
    height, width = images[0].shape[:2]
    blank = np.zeros_like(images[0])
    rows = []
    for start in range(0, len(images), cols):
        row = images[start : start + cols]
        row = row + [blank] * (cols - len(row))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def _label(image: np.ndarray, text: str) -> np.ndarray:
    """Stamp a small caption in the corner, if cv2 is available."""
    try:
        import cv2

        out = np.ascontiguousarray(image.copy())
        cv2.rectangle(out, (0, 0), (150, 26), (0, 0, 0), -1)
        cv2.putText(
            out, text, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2
        )
        return out
    except Exception:  # noqa: BLE001
        return image


def run_scene(args: argparse.Namespace) -> int:
    """STAGE 1: render the scene across a few random spawns."""
    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to import ManiSkill: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    from pg3d.envs.xarm_adapter.pnp_env import (
        cube_position,
        goal_position,
        register_pnp_envs,
    )
    from pg3d.utils.arrays import frame_to_numpy

    register_pnp_envs()

    env = gym.make(
        "PG3DPnP-XArm7-Gripper-v0",
        render_mode="rgb_array",
        obs_mode="state",
        cube_edge=args.cube_edge,
        goal_marker_radius=args.goal_marker_radius,
        spawn_margin=args.spawn_margin,
        min_object_separation=args.min_object_separation,
    )
    frames: list[np.ndarray] = []
    try:
        bounds = env.unwrapped._spawn_bounds_xy()
        print(
            f"spawn box (world XY): x=[{bounds[0, 0]:.3f}, {bounds[0, 1]:.3f}] "
            f"y=[{bounds[1, 0]:.3f}, {bounds[1, 1]:.3f}]  "
            f"(reach box inset by --spawn-margin={args.spawn_margin})"
        )
        print(
            f"cube: {args.cube_edge * 100:.0f}cm edge, red | "
            f"goal marker: {args.goal_marker_radius * 100:.0f}cm radius, green | "
            f"min separation {args.min_object_separation}m"
        )
        for sample_idx in range(args.num_samples):
            env.reset(seed=args.seed + sample_idx)
            cube = cube_position(env)
            goal = goal_position(env)
            separation = float(np.linalg.norm(cube[:2] - goal[:2]))
            print(
                f"  [{sample_idx}] cube={np.round(cube, 3).tolist()} "
                f"goal={np.round(goal, 3).tolist()} separation={separation:.3f}m"
            )
            frame = frame_to_numpy(env.render())
            frames.append(_label(frame, f"#{sample_idx} sep={separation:.2f}m"))
    finally:
        env.close()

    image = _tile(frames, cols=args.cols)
    _save_png(args.out, image)
    print(f"\nwrote scene image: {args.out}  ({image.shape[1]}x{image.shape[0]})")
    return 0


def _not_implemented(mode: str) -> int:
    print(
        f"--mode {mode} is not implemented yet. Stages are built and verified "
        "one at a time: run --mode scene first and confirm the setup looks "
        "right, then gripper tuning, then pick, then pick-place.",
        file=sys.stderr,
    )
    return 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="xArm7 pick-and-place, staged: scene -> gripper tuning -> pick -> place."
    )
    p.add_argument(
        "--mode",
        default="scene",
        choices=["scene", "tune-gripper", "pick", "pick-place"],
        help="Which stage to run. Only 'scene' is implemented so far.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--cube-edge",
        type=float,
        default=0.07,
        help="Cube side length in metres (0.07 = the 7cm cube).",
    )
    p.add_argument(
        "--goal-marker-radius",
        type=float,
        default=0.03,
        help="Radius of the green goal marker sphere. This is also the base "
        "env's goal_thresh, but pick/place success here is judged from the "
        "cube's own pose, so it only affects visibility.",
    )
    p.add_argument(
        "--spawn-margin",
        type=float,
        default=0.06,
        help="Inset (m) applied to the arm's reach box before sampling spawn "
        "positions -- the extreme corners are near-singular and leave no room "
        "for a tilted approach.",
    )
    p.add_argument(
        "--min-object-separation",
        type=float,
        default=0.20,
        help="Minimum XY distance (m) between cube and goal, so every episode "
        "is a real relocation.",
    )
    # scene-mode only
    p.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="scene mode: how many random spawns to render into the tiled image.",
    )
    p.add_argument(
        "--cols", type=int, default=2, help="scene mode: tiling columns."
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/pnp/scene.png"),
        help="scene mode: where to write the tiled setup image.",
    )
    args = p.parse_args(argv)
    if args.cube_edge <= 0:
        raise ValueError("--cube-edge must be positive")
    if args.goal_marker_radius <= 0:
        raise ValueError("--goal-marker-radius must be positive")
    if args.spawn_margin < 0:
        raise ValueError("--spawn-margin must be non-negative")
    if args.min_object_separation < 0:
        raise ValueError("--min-object-separation must be non-negative")
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.cols <= 0:
        raise ValueError("--cols must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "scene":
        return run_scene(args)
    return _not_implemented(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
