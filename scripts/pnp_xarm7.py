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


_DOWN_QUAT_WXYZ = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)


def _finger_separation(env: Any) -> float:
    """Distance between the two finger links' origins, metres.

    A proxy for jaw opening, not the exact inner-face gap (the finger links'
    own thickness sits inside it), but the OPEN-minus-CLOSED delta is the real
    stroke, and the open value bounds how wide an object can possibly fit.
    Worth printing before any force sweep: if a 7cm cube simply does not fit
    between the jaws, no amount of force tuning will ever grasp it, and the
    sweep would just be measuring failure for the wrong reason.
    """
    from pg3d.utils.arrays import to_numpy

    links = {link.name: link for link in env.unwrapped.agent.robot.get_links()}
    left, right = links.get("left_finger"), links.get("right_finger")
    if left is None or right is None:
        return float("nan")
    lp = to_numpy(left.pose.p).reshape(-1, 3)[0]
    rp = to_numpy(right.pose.p).reshape(-1, 3)[0]
    return float(np.linalg.norm(lp - rp))


def _gripper_qvel(env: Any) -> float:
    """Max |qvel| over the gripper joints (indices 7+), rad/s.

    The whole reason gripper_force_limit sat at 0.1 for so long was a
    runaway-velocity blowup (documented in agents.py: 0.2 -> 57 rad/s in 8
    steps). Any force value this sweep recommends has to be checked against
    that, not just against grasp success -- a config that holds the cube but
    destabilises the joint is not usable.
    """
    from pg3d.utils.arrays import to_numpy

    qvel = to_numpy(env.unwrapped.agent.robot.get_qvel()).reshape(-1)
    return float(np.max(np.abs(qvel[7:]))) if qvel.shape[0] > 7 else 0.0


def _arm_qvel(env: Any) -> float:
    from pg3d.utils.arrays import to_numpy

    qvel = to_numpy(env.unwrapped.agent.robot.get_qvel()).reshape(-1)
    return float(np.max(np.abs(qvel[:7])))


def _plan_and_replay(
    env: Any,
    *,
    position: np.ndarray,
    quat: np.ndarray,
    gripper_value: float,
    max_steps: int,
    frames: list[np.ndarray] | None,
    stats: dict[str, float],
) -> bool:
    """Plan a screw-motion move to (position, quat) and replay it. False if unplannable."""
    import sapien

    from dataset_generation.write_maniskill_reach_dataset import (
        _format_sim_action,
        _move_to_pose_with_screw,
        _pose_with_orientation,
    )
    from pg3d.envs.xarm_adapter.motionplanner import XArm7GripperMotionPlanningSolver
    from pg3d.utils.arrays import bool_any, frame_to_numpy

    solver = XArm7GripperMotionPlanningSolver(
        env,
        debug=False,
        vis=False,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
    )
    try:
        plan = _move_to_pose_with_screw(
            solver,
            _pose_with_orientation(sapien, position=position, quat=quat),
            suppress_output=True,
        )
    finally:
        solver.close()
    if plan == -1 or "position" not in plan:
        return False

    for planned_qpos in np.asarray(plan["position"], dtype=np.float32)[:max_steps]:
        action = _format_sim_action(env, planned_qpos, gripper_action=gripper_value)
        _obs, _reward, _term, truncated, _info = env.step(action)
        stats["max_gripper_qvel"] = max(stats["max_gripper_qvel"], _gripper_qvel(env))
        stats["max_arm_qvel"] = max(stats["max_arm_qvel"], _arm_qvel(env))
        if frames is not None:
            frames.append(frame_to_numpy(env.render()))
        if bool_any(truncated):
            break
    return True


def _scripted_grasp_trial(
    env: Any,
    *,
    args: argparse.Namespace,
    frames: list[np.ndarray] | None,
) -> dict[str, Any]:
    """One scripted top-down pick: approach, descend, close, lift. No policy.

    Deliberately policy-free. The point is to measure the GRIPPER, so every
    other source of error is removed: the approach is a planned straight-line
    descent onto a cube whose exact pose is read from the sim, straight down
    (zero tilt, so no fingertip lever-arm offset), centred by construction. If
    a grasp fails here it is the physics, not the policy and not alignment.
    """
    from dataset_generation.write_maniskill_reach_dataset import _hold_sim_action
    from pg3d.envs.xarm_adapter.pnp_env import cube_position
    from pg3d.utils.arrays import bool_any, frame_to_numpy

    stats = {"max_gripper_qvel": 0.0, "max_arm_qvel": 0.0}
    cube_start = cube_position(env)
    grasp_p = cube_start.copy()  # TCP target = cube centre, straight down
    pregrasp_p = grasp_p + np.array([0.0, 0.0, args.pregrasp_height], dtype=np.float32)
    lift_p = grasp_p + np.array([0.0, 0.0, args.lift_height], dtype=np.float32)

    row: dict[str, Any] = {
        "cube_start": cube_start.tolist(),
        "reached_pregrasp": False,
        "reached_grasp": False,
        "lift_planned": False,
    }

    if not _plan_and_replay(
        env, position=pregrasp_p, quat=_DOWN_QUAT_WXYZ,
        gripper_value=args.gripper_open_value, max_steps=args.move_max_steps,
        frames=frames, stats=stats,
    ):
        row.update(stats, outcome="pregrasp_unplannable", cube_lift=0.0, success=False)
        return row
    row["reached_pregrasp"] = True

    if not _plan_and_replay(
        env, position=grasp_p, quat=_DOWN_QUAT_WXYZ,
        gripper_value=args.gripper_open_value, max_steps=args.move_max_steps,
        frames=frames, stats=stats,
    ):
        row.update(stats, outcome="descent_unplannable", cube_lift=0.0, success=False)
        return row
    row["reached_grasp"] = True

    cube_before_close = cube_position(env)
    row["approach_drift"] = float(
        np.linalg.norm(cube_before_close[:2] - cube_start[:2])
    )

    # Ramp the close rather than snapping to the target: a 1e5-stiffness PD
    # spring given a full-travel step change is exactly what produced the
    # historical qvel blowup, and it also pops the cube out on contact.
    hold_qpos = np.asarray(env.unwrapped.agent.robot.get_qpos()).reshape(-1)[:7]
    for step in range(args.close_steps):
        frac = (step + 1) / float(args.close_steps)
        value = args.gripper_open_value + frac * (
            args.gripper_close_value - args.gripper_open_value
        )
        action = _hold_sim_action(env, gripper_open=value, qpos=hold_qpos)
        _obs, _reward, _term, truncated, _info = env.step(action)
        stats["max_gripper_qvel"] = max(stats["max_gripper_qvel"], _gripper_qvel(env))
        stats["max_arm_qvel"] = max(stats["max_arm_qvel"], _arm_qvel(env))
        if frames is not None:
            frames.append(frame_to_numpy(env.render()))
        if bool_any(truncated):
            break
    row["finger_separation_closed"] = _finger_separation(env)

    cube_before_lift = cube_position(env)
    row["lift_planned"] = _plan_and_replay(
        env, position=lift_p, quat=_DOWN_QUAT_WXYZ,
        gripper_value=args.gripper_close_value, max_steps=args.move_max_steps,
        frames=frames, stats=stats,
    )

    cube_end = cube_position(env)
    cube_lift = float(cube_end[2] - cube_before_lift[2])
    success = bool(row["lift_planned"] and cube_lift >= args.success_lift_fraction * args.lift_height)
    row.update(
        stats,
        cube_lift=cube_lift,
        cube_end=cube_end.tolist(),
        success=success,
        outcome="held" if success else ("dropped_or_missed" if row["lift_planned"] else "lift_unplannable"),
    )
    return row


def run_tune_gripper(args: argparse.Namespace) -> int:
    """STAGE 2: sweep gripper force settings against a scripted top-down grasp."""
    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to import ManiSkill: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    import json

    from pg3d.envs.xarm_adapter.agents import XArm7Gripper
    from pg3d.envs.xarm_adapter.pnp_env import register_pnp_envs
    from pg3d.utils.arrays import frame_to_numpy
    from scripts.rollout_dp3_reach_policy import save_video

    register_pnp_envs()

    baseline_force = XArm7Gripper.gripper_force_limit
    baseline_stiffness = XArm7Gripper.gripper_stiffness
    print(
        f"baseline XArm7Gripper: force_limit={baseline_force} "
        f"stiffness={baseline_stiffness} damping={XArm7Gripper.gripper_damping}"
    )
    print(
        f"sweeping force_limit over {args.force_limits} "
        f"x stiffness {args.stiffnesses or [baseline_stiffness]}  "
        f"({args.trials} trials each, cube edge {args.cube_edge * 100:.0f}cm)\n"
    )

    stiffness_values = args.stiffnesses or [baseline_stiffness]
    results: list[dict[str, Any]] = []
    geometry_reported = False

    try:
        for stiffness in stiffness_values:
            for force_limit in args.force_limits:
                # These are class attributes read when the agent builds its
                # controller configs, so they MUST be set before gym.make --
                # mutating them on a live env would have no effect.
                XArm7Gripper.gripper_force_limit = force_limit
                XArm7Gripper.gripper_stiffness = stiffness

                env = gym.make(
                    "PG3DPnP-XArm7-Gripper-v0",
                    render_mode="rgb_array",
                    obs_mode="state",
                    cube_edge=args.cube_edge,
                    spawn_margin=args.spawn_margin,
                    min_object_separation=args.min_object_separation,
                )
                trials: list[dict[str, Any]] = []
                try:
                    for trial_idx in range(args.trials):
                        env.reset(seed=args.seed + trial_idx)
                        if not geometry_reported:
                            # Measured once, at reset, with the gripper at its
                            # rest (closed) keyframe -- then again after the
                            # first trial's open command inside the trial.
                            print(
                                f"finger separation at rest/closed keyframe: "
                                f"{_finger_separation(env):.4f}m"
                            )
                        record = (
                            args.video_dir is not None and trial_idx == 0
                        )
                        frames = [frame_to_numpy(env.render())] if record else None
                        row = _scripted_grasp_trial(env, args=args, frames=frames)
                        row["trial"] = trial_idx
                        trials.append(row)
                        if not geometry_reported:
                            print(
                                f"finger separation once closed on the cube: "
                                f"{row.get('finger_separation_closed', float('nan')):.4f}m "
                                f"(cube edge {args.cube_edge:.3f}m)"
                            )
                            geometry_reported = True
                        if record and frames:
                            args.video_dir.mkdir(parents=True, exist_ok=True)
                            path = (
                                args.video_dir
                                / f"force{force_limit:g}_stiff{stiffness:g}.mp4"
                            )
                            save_video(path, frames, fps=args.video_fps)
                            print(f"  video: {path}")
                finally:
                    env.close()

                held = [t for t in trials if t["success"]]
                lifts = [t["cube_lift"] for t in trials]
                drifts = [t.get("approach_drift", 0.0) for t in trials]
                summary = {
                    "force_limit": force_limit,
                    "stiffness": stiffness,
                    "trials": len(trials),
                    "held": len(held),
                    "success_rate": len(held) / max(len(trials), 1),
                    "mean_lift": float(np.mean(lifts)) if lifts else 0.0,
                    "max_gripper_qvel": max(t["max_gripper_qvel"] for t in trials),
                    "max_arm_qvel": max(t["max_arm_qvel"] for t in trials),
                    "mean_approach_drift": float(np.mean(drifts)) if drifts else 0.0,
                    "outcomes": {
                        outcome: sum(1 for t in trials if t["outcome"] == outcome)
                        for outcome in sorted({t["outcome"] for t in trials})
                    },
                    "episodes": trials,
                }
                results.append(summary)
                flag = " UNSTABLE" if summary["max_gripper_qvel"] > args.qvel_warn else ""
                print(
                    f"  force={force_limit:<6g} stiff={stiffness:<8g} "
                    f"held {summary['held']}/{summary['trials']} "
                    f"({100 * summary['success_rate']:.0f}%)  "
                    f"mean_lift={summary['mean_lift']:.4f}m  "
                    f"max|qvel| gripper={summary['max_gripper_qvel']:.2f} "
                    f"arm={summary['max_arm_qvel']:.2f}rad/s{flag}  "
                    f"{summary['outcomes']}"
                )
    finally:
        XArm7Gripper.gripper_force_limit = baseline_force
        XArm7Gripper.gripper_stiffness = baseline_stiffness

    print("\n── Sweep summary (sorted: most reliable, then most stable)")
    stable = [r for r in results if r["max_gripper_qvel"] <= args.qvel_warn]
    ranked = sorted(
        stable or results,
        key=lambda r: (-r["success_rate"], r["max_gripper_qvel"]),
    )
    for r in ranked:
        print(
            f"   force_limit={r['force_limit']:<6g} stiffness={r['stiffness']:<8g} "
            f"success={100 * r['success_rate']:5.1f}%  "
            f"max gripper |qvel|={r['max_gripper_qvel']:.2f} rad/s"
        )
    if not stable:
        print(
            f"   NOTE: every config exceeded --qvel-warn={args.qvel_warn} rad/s. "
            "Ranking above is unfiltered and none of these is safe to adopt "
            "as-is -- the joint is being driven unstable."
        )
    best = ranked[0] if ranked else None
    if best is not None:
        print(
            f"\n   -> best: gripper_force_limit = {best['force_limit']:g}"
            + (f", gripper_stiffness = {best['stiffness']:g}" if len(stiffness_values) > 1 else "")
            + f"  ({100 * best['success_rate']:.0f}% held, "
            f"peak gripper |qvel| {best['max_gripper_qvel']:.2f} rad/s)"
        )
        if best["success_rate"] < 1.0:
            print(
                "      Not 100% -- check the recorded video before adopting it; "
                "a partial hold rate here (policy-free, perfectly centred, "
                "zero tilt) is a floor, and the real pick will do worse."
            )
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(results, indent=2))
        print(f"\nwrote sweep summary: {args.summary_json}")
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
    # tune-gripper mode
    p.add_argument(
        "--force-limits",
        type=float,
        nargs="+",
        default=[0.1, 5.0, 20.0, 50.0, 100.0],
        help="tune-gripper: gripper_force_limit values to sweep. Includes 0.1 "
        "(the long-standing untested default) as a control so the sweep shows "
        "what it actually does rather than assuming.",
    )
    p.add_argument(
        "--stiffnesses",
        type=float,
        nargs="+",
        default=None,
        help="tune-gripper: optional gripper_stiffness values to cross with "
        "--force-limits. Defaults to leaving stiffness at its current value "
        "(1-D sweep over force only).",
    )
    p.add_argument(
        "--trials",
        type=int,
        default=5,
        help="tune-gripper: scripted grasp attempts per config, each on a "
        "freshly sampled cube position.",
    )
    p.add_argument("--pregrasp-height", type=float, default=0.12, help="tune-gripper: standoff above the cube (m).")
    p.add_argument("--lift-height", type=float, default=0.15, help="tune-gripper: vertical lift (m).")
    p.add_argument("--close-steps", type=int, default=15, help="tune-gripper: steps to ramp the gripper shut.")
    p.add_argument("--gripper-open-value", type=float, default=-1.0)
    p.add_argument("--gripper-close-value", type=float, default=1.0)
    p.add_argument("--move-max-steps", type=int, default=120, help="tune-gripper: replay budget per planned move.")
    p.add_argument(
        "--success-lift-fraction",
        type=float,
        default=0.5,
        help="tune-gripper: cube must rise this fraction of --lift-height to count as held.",
    )
    p.add_argument(
        "--qvel-warn",
        type=float,
        default=5.0,
        help="tune-gripper: gripper |qvel| (rad/s) above which a config is "
        "treated as unstable and excluded from the recommendation. The 0.1 "
        "default exists because of a historical blowup to ~57 rad/s.",
    )
    p.add_argument("--video-dir", type=Path, default=None, help="tune-gripper: record trial 0 of each config here.")
    p.add_argument("--video-fps", type=int, default=15)
    p.add_argument("--summary-json", type=Path, default=None)
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
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    if any(f < 0 for f in args.force_limits):
        raise ValueError("--force-limits must be non-negative")
    if args.stiffnesses is not None and any(s <= 0 for s in args.stiffnesses):
        raise ValueError("--stiffnesses must be positive")
    if args.pregrasp_height <= 0:
        raise ValueError("--pregrasp-height must be positive")
    if args.lift_height <= 0:
        raise ValueError("--lift-height must be positive")
    if args.close_steps <= 0:
        raise ValueError("--close-steps must be positive")
    if args.move_max_steps <= 0:
        raise ValueError("--move-max-steps must be positive")
    if not (0.0 <= args.success_lift_fraction <= 1.0):
        raise ValueError("--success-lift-fraction must be in [0.0, 1.0]")
    if args.qvel_warn <= 0:
        raise ValueError("--qvel-warn must be positive")
    if args.video_fps <= 0:
        raise ValueError("--video-fps must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "scene":
        return run_scene(args)
    if args.mode == "tune-gripper":
        return run_tune_gripper(args)
    return _not_implemented(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
