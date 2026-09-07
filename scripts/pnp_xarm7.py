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
* ``pick``   -- STAGE 3 (this one, now implemented). The trained DP3 reach
  checkpoint is actually rolled out (reranking-steered, same mechanism as the
  reach evals) to approach a cube spawned anywhere in the workspace, always
  in straight-down orientation -- no orientation variety, no tilt-realign.
  The grasp itself (close + lift) stays scripted because the checkpoint has
  no gripper output at all, using the settings ``tune-gripper`` recommended.
* ``pick-place`` -- STAGE 4 (implemented). Everything ``pick`` does, then
  transports the held cube to the green goal marker on the table -- the same
  policy rollout, gripper closed -- releases it, and scores where it actually
  came to rest against a tolerance the marker is drawn at.

All four stages are implemented.

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


def _set_gripper_force_limit(env: Any, force_limit: float) -> None:
    """Bump ``drive_joint``'s live force limit, bypassing the controller config.

    ``XArm7Gripper.gripper_force_limit`` (agents.py) is a class attribute only
    read when the controller config is BUILT, at env construction -- setting
    it after the env exists does nothing to the running sim. Getting a
    different force for the CLOSE phase (must stay soft: 0.05, tuned so
    asymmetric first contact doesn't punch the cube out before both fingers
    are seated) versus the HOLD/TRANSPORT phase (needs real margin against
    inertial loads once the cube is already seated -- no more asymmetric-
    contact risk at that point) means reaching past the config and poking the
    live PhysX joint drive directly.

    SAPIEN has used both ``set_drive_property`` (singular) and
    ``set_drive_properties`` (plural) as the method name across versions this
    repo has touched; try both and raise rather than silently no-op if
    neither exists -- this runs unattended on enigma, where a silently
    ignored force bump would look identical to a real one right up until the
    next slip.
    """
    from pg3d.envs.xarm_adapter.agents import XArm7Gripper

    joint = env.unwrapped.agent.robot.active_joints_map["drive_joint"]
    kwargs = dict(
        stiffness=XArm7Gripper.gripper_stiffness,
        damping=XArm7Gripper.gripper_damping,
        force_limit=float(force_limit),
    )
    for method_name in ("set_drive_property", "set_drive_properties"):
        method = getattr(joint, method_name, None)
        if method is not None:
            method(**kwargs)
            return
    raise AttributeError(
        f"drive_joint ({type(joint)!r}) exposes neither set_drive_property nor "
        "set_drive_properties -- can't apply the transport-phase force bump. "
        "Check the installed SAPIEN version's joint drive API."
    )


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
    baseline_damping = XArm7Gripper.gripper_damping
    print(
        f"baseline XArm7Gripper: force_limit={baseline_force} "
        f"stiffness={baseline_stiffness} damping={baseline_damping}"
    )

    stiffness_values = args.stiffnesses or [baseline_stiffness]
    damping_values = args.dampings or [baseline_damping]
    grid_size = len(args.force_limits) * len(stiffness_values) * len(damping_values)
    print(
        f"sweeping force_limit over {args.force_limits} "
        f"x stiffness {stiffness_values} "
        f"x damping {damping_values}  "
        f"({grid_size} configs x {args.trials} trials, cube edge {args.cube_edge * 100:.0f}cm)\n"
    )

    results: list[dict[str, Any]] = []
    geometry_reported = False
    geometry_warned = False

    try:
        for damping in damping_values:
            for stiffness in stiffness_values:
                for force_limit in args.force_limits:
                    # These are class attributes read when the agent builds
                    # its controller configs, so they MUST be set before
                    # gym.make -- mutating them on a live env has no effect.
                    XArm7Gripper.gripper_force_limit = force_limit
                    XArm7Gripper.gripper_stiffness = stiffness
                    XArm7Gripper.gripper_damping = damping

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
                                # Measured once, at reset, with the gripper at
                                # its rest keyframe -- then again after the
                                # first trial's close command inside the trial.
                                rest_gap = _finger_separation(env)
                                print(
                                    f"finger separation at rest keyframe: {rest_gap:.4f}m "
                                    f"vs cube edge {args.cube_edge:.3f}m"
                                )
                                if rest_gap < args.cube_edge:
                                    geometry_warned = True
                                    print(
                                        f"  *** WARNING: the jaws' rest opening "
                                        f"({rest_gap * 100:.1f}cm) is NARROWER than "
                                        f"the cube ({args.cube_edge * 100:.0f}cm). The "
                                        "gripper physically cannot take this cube "
                                        "between its fingers -- force/stiffness "
                                        "cannot fix that, only a smaller --cube-edge "
                                        "or a wider gripper open-command/keyframe can. "
                                        "Every result below is confounded by this "
                                        "until it's resolved."
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
                                    / f"force{force_limit:g}_stiff{stiffness:g}_damp{damping:g}.mp4"
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
                        "damping": damping,
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
                        f"damp={damping:<6g} "
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
        XArm7Gripper.gripper_damping = baseline_damping

    if geometry_warned:
        print(
            f"\n*** Every config above was run with the jaws unable to fit "
            f"around a {args.cube_edge * 100:.0f}cm cube. Re-run with a "
            "smaller --cube-edge (or fix the gripper's open keyframe/limits) "
            "before trusting any force/stiffness/damping recommendation."
        )

    print("\n── Sweep summary (sorted: most reliable, then most stable)")
    stable = [r for r in results if r["max_gripper_qvel"] <= args.qvel_warn]
    ranked = sorted(
        stable or results,
        key=lambda r: (-r["success_rate"], r["max_gripper_qvel"]),
    )
    for r in ranked:
        print(
            f"   force_limit={r['force_limit']:<6g} stiffness={r['stiffness']:<8g} "
            f"damping={r['damping']:<6g} "
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
            + (f", gripper_damping = {best['damping']:g}" if len(damping_values) > 1 else "")
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


def run_pick(args: argparse.Namespace, *, place_enabled: bool = False) -> int:
    """STAGE 3 (pick) / STAGE 4 (pick-place), sharing one implementation.

    Deliberately NOT policy-steered yet and deliberately NOT orientation-
    aware: every approach is straight down (``_DOWN_QUAT_WXYZ``), exactly the
    grasp shape ``tune-gripper`` validated. The only thing this stage adds
    over ``tune-gripper`` is that the APPROACH is now the trained DP3 reach
    checkpoint, steered by the same reranking mechanism the reach evals use
    (K candidate action chunks per replan, scored by imagined rollout against
    a CartesianPoseConstraint, lowest-cost one executed) -- not a planned
    trajectory. The cube also spawns anywhere in the workspace each episode
    (``pnp_env``'s own reset sampler), so this answers "can the policy get to
    the cube wherever it is, and does the tuned grasp then hold it".

    Deliberately fixed straight-down orientation for every approach
    (``_DOWN_QUAT_WXYZ``), no orientation variety and no tilt-realign retry:
    tilt is what made the earlier pick-and-place eval hard to read (rotation
    error amplifies at the fingertips, ~3.5cm off the TCP), and at zero tilt
    a realign step would have nothing to correct toward anyway. Approach
    variety comes later, once straight-down-anywhere is solid.

    Structure per episode, mirroring the pick half of
    ``eval_pose_variety_pick_and_place.py`` with the place half and the
    orientation machinery removed:

    1. POLICY REACH to a pre-grasp standoff ``--pregrasp-height`` above the
       cube, gripper held open. The standoff exists because a learned reach
       flown straight to cube height arrives still moving laterally with the
       claws wide open and broadsides the cube.
    2. SCRIPTED VERTICAL DESCENT onto the cube (straight down == the approach
       axis here), so the fingers only ever move parallel to themselves near
       the cube.
    3. SCRIPTED CLOSE (ramped over ``--close-steps``) then LIFT, using the
       gripper settings ``tune-gripper`` recommended. The checkpoint has no
       gripper output at all (see agents.py), so close/lift can only ever be
       scripted -- that is not a shortcut, it is the checkpoint's actual
       action space.

    With ``place_enabled`` (STAGE 4, ``--mode pick-place``), a successful pick
    continues into:

    4. POLICY TRANSPORT to the green goal marker on the table -- the SAME
       reranking rollout as the approach, gripper held closed, its EMA seeded
       from the lift's last action so the arm doesn't snap at the handoff.
    5. SCRIPTED RELEASE (ramped open) then a settle window, after which the
       cube's own pose is scored: it must land within ``--place-radius`` of
       the goal AND come to rest at its natural resting height. The goal
       marker is drawn at exactly ``--place-radius``, so the visible green
       target on the table IS the scored tolerance area.

    Both stages share this one function because the pick half is identical --
    duplicating it would just let the two drift apart.
    """
    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
        import sapien
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to import ManiSkill/SAPIEN: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    import json

    from dataset_generation.write_maniskill_reach_dataset import (
        _format_sim_action,
        _hold_sim_action,
        _move_to_pose_with_screw,
        _pose_with_orientation,
        _refresh_obs_after_manual_qpos,
        _tcp_pose,
    )
    from pg3d.constraints import CartesianPoseConstraint
    from pg3d.envs.maniskill_adapter import ManiSkillGhostPandaGeometryProvider
    from pg3d.envs.maniskill_adapter.dataset import load_reach_metadata
    from pg3d.envs.xarm_adapter.agents import XArm7Gripper
    from pg3d.envs.xarm_adapter.motionplanner import XArm7GripperMotionPlanningSolver
    from pg3d.envs.xarm_adapter.pnp_env import cube_position, goal_position, register_pnp_envs
    from pg3d.envs.xarm_adapter.reach_env import ROBOT_BASE_POSE
    from pg3d.eval import TimingRecorder, scene_context_for_constraints
    from pg3d.policies.dp3.checkpoint import load_reach_policy_from_checkpoint
    from pg3d.utils.arrays import bool_any, frame_to_numpy
    from pg3d.utils.devices import select_device
    from pg3d.world_model import GeometricWorldModel
    from scripts.eval_constrained_reach import DP3ChunkPolicyAdapter, _env_kwargs, _select_decision
    from scripts.eval_pose_variety_checkpoint_reachability import (
        _quat_angular_distance_rad,
        _set_site_pose,
    )
    from scripts.rollout_dp3_reach_policy import (
        _action_mode,
        append_obs_window,
        crop_config_from_metadata,
        make_initial_obs_window,
        policy_action_to_sim_action,
        rollout_observation_entry,
        save_video,
    )

    if args.checkpoint is None or args.dataset is None:
        print(
            "--mode pick needs --checkpoint (the trained DP3 reach policy) and "
            "--dataset (read ONLY for metadata.json: the observation/crop/action "
            "config the checkpoint was trained on).",
            file=sys.stderr,
        )
        return 2

    metadata = load_reach_metadata(args.dataset)
    device = select_device(args.device)
    policy = load_reach_policy_from_checkpoint(args.checkpoint, device=device)
    action_mode = _action_mode(str(metadata.get("action_mode", "abs_joint")))
    crop_config = crop_config_from_metadata(metadata)

    register_pnp_envs()

    baseline_force = XArm7Gripper.gripper_force_limit
    baseline_stiffness = XArm7Gripper.gripper_stiffness
    baseline_damping = XArm7Gripper.gripper_damping
    if args.gripper_force_limit is not None:
        XArm7Gripper.gripper_force_limit = args.gripper_force_limit
    if args.gripper_stiffness is not None:
        XArm7Gripper.gripper_stiffness = args.gripper_stiffness
    if args.gripper_damping is not None:
        XArm7Gripper.gripper_damping = args.gripper_damping
    # This is the CLOSE-phase force -- the value the sweep tuned for
    # asymmetric first-contact safety. Captured here (after any CLI
    # override above) so the post-lift bump below has the right value to
    # restore to before release.
    close_force_limit = XArm7Gripper.gripper_force_limit

    # One episode is: policy reach + scripted descent + close + lift, plus (in
    # pick-place) the policy transport + release + settle. The env step limit
    # has to cover all of it or a late phase gets truncated by the env rather
    # than judged on its merits.
    meta_max_steps = metadata.get("env_kwargs", {}).get("max_episode_steps")
    place_budget = (
        int(args.post_lift_settle_steps)
        + int(args.place_max_steps)
        + int(args.release_steps)
        + int(args.settle_steps)
        if place_enabled
        else 0
    )
    env_max_steps = max(
        int(args.max_steps)
        + int(args.descent_max_steps)
        + int(args.close_steps)
        + int(args.lift_max_steps)
        + place_budget,
        int(meta_max_steps or 0),
    )
    # Cube's natural resting height, used to tell "placed on the table" from
    # "still in the jaws" / "fell off". Table surface is at the robot base's
    # own Z -- see XARM7_REACH_BOX_BASE's dz convention in reach_config.py.
    cube_rest_z = float(ROBOT_BASE_POSE.p[2]) + args.cube_edge / 2.0
    stage = "pick-place" if place_enabled else "pick"
    print(
        f"{stage}: force_limit={XArm7Gripper.gripper_force_limit} "
        f"stiffness={XArm7Gripper.gripper_stiffness} "
        f"damping={XArm7Gripper.gripper_damping}\n"
        f"      {args.episodes} episodes, cube edge {args.cube_edge * 100:.0f}cm, "
        f"straight-down approach only, env step limit {env_max_steps}\n"
        f"      policy={args.checkpoint}"
    )
    if place_enabled:
        print(
            f"      place tolerance {args.place_radius:.3f}m (the green goal "
            f"marker is drawn at exactly this radius, so what you see on the "
            f"table IS the scored zone); cube rests at Z={cube_rest_z:.3f}\n"
        )
    else:
        print()

    sim_env_kwargs = _env_kwargs(metadata, render_mode="rgb_array", max_episode_steps=env_max_steps)
    sim_env_kwargs["cube_edge"] = args.cube_edge
    sim_env_kwargs["spawn_margin"] = args.spawn_margin
    sim_env_kwargs["min_object_separation"] = args.min_object_separation
    # Draw the goal marker at the scoring radius so the visible green target
    # on the table is literally the tolerance area, not a decorative dot of
    # some unrelated size.
    if place_enabled:
        sim_env_kwargs["goal_marker_radius"] = args.place_radius
    sim_env = gym.make("PG3DPnP-XArm7-Gripper-v0", **sim_env_kwargs)
    # Ghost env supplies the world model's geometry; it is the env the
    # checkpoint was TRAINED on (no cube), never stepped, only queried.
    ghost_env = gym.make(str(metadata["env_id"]), **_env_kwargs(metadata, render_mode=None))

    adapter = DP3ChunkPolicyAdapter(
        policy, action_mode=action_mode, device=device, policy_batch_size=args.policy_batch_size
    )
    provider = ManiSkillGhostPandaGeometryProvider(
        ghost_env, task_name=str(metadata.get("env_id", "unknown")), crop_bounds=crop_config.bounds
    )
    world_model = GeometricWorldModel(provider)
    rng = np.random.default_rng(args.seed)

    def _policy_reach_to(
        *,
        sim_entry: dict[str, Any],
        obs_window: Any,
        frames: list[np.ndarray] | None,
        target_position: np.ndarray,
        max_steps: int,
        gripper_value: float,
        phase: str,
        initial_ema_action: np.ndarray | None = None,
        ema_warmup_steps: int = 0,
    ) -> tuple[bool, dict[str, Any], Any, int, float, float, bool, np.ndarray | None]:
        """Steer the checkpoint to `target_position` (straight-down orientation).

        Same reranking mechanism as the reach evals -- this is the actual DP3
        rollout, not a planner. Shared by both the pick approach (gripper
        open, empty) and the place transport (gripper closed, carrying the
        cube): identical machinery, only the target, step budget, and held
        gripper command differ. Returns (reached, sim_entry, obs_window,
        steps, pos_err, rot_err, truncated_early, last_ema_action); `frames`
        is mutated in place so the caller keeps one continuous video across
        phases.

        `initial_ema_action` seeds the EMA instead of starting from the raw
        first policy action. Needed at the pick->place handoff: the close+lift
        phase is a scripted, non-EMA open-loop replay, so without a seed the
        place phase's first blended action snaps from wherever that replay
        left off straight to a raw policy output against this gripper's stiff
        PD controller (stiffness=1e5) -- which in the earlier pick-and-place
        eval launched the cube ~0.30m in a single rendered frame.

        `ema_warmup_steps` (only meaningful together with `initial_ema_action`)
        ramps the EMA blend weight up from a low floor
        (``args.transport_ema_warmup_alpha``) to the configured
        ``args.action_ema_alpha`` over that many steps, instead of jumping
        straight to the full weight on step 1. A single 50/50 blend still
        lets through half of whatever the raw policy predicts, and the very
        first prediction here is exactly where this checkpoint is furthest
        out of distribution (see the caller: the observation window is only
        now being repointed at a brand-new, far-away goal after several
        pick-phase steps at the OLD goal -- this checkpoint was trained as a
        single-goal-per-episode reach policy, never on a goal teleporting
        mid-rollout). A low-force gripper (see agents.py, tuned soft
        specifically so the jaws yield into the cube rather than crushing
        through it) has no margin to also absorb one bad, large first
        command -- it yields to that impact too, and the cube is punched out
        under a nominally-still-closed target.
        """
        constraint = CartesianPoseConstraint(
            target_position=target_position,
            target_orientation=_DOWN_QUAT_WXYZ,
            position_tolerance=args.position_tolerance,
            rotation_tolerance=args.rotation_tolerance,
            weight=1.0,
            name=f"pnp_{phase}",
        )
        scene = scene_context_for_constraints(
            target_position=target_position,
            constraints=[constraint],
            metadata={"phase": phase},
        )
        timer = TimingRecorder(enabled=False)
        ema_sim_action: np.ndarray | None = (
            None if initial_ema_action is None else np.asarray(initial_ema_action, dtype=np.float32)
        )
        reached_goal = False
        truncated_early = False
        pos_err = float("inf")
        rot_err = float("inf")
        total_steps = 0
        was_training = policy.training
        policy.eval()
        try:
            while total_steps < max_steps:
                decision = _select_decision(
                    method="reranking",
                    adapter=adapter,
                    world_model=world_model,
                    provider=provider,
                    current_entry=sim_entry,
                    obs_window=obs_window,
                    scene=scene,
                    constraints=[constraint],
                    crop_config=crop_config,
                    goal_thresh=args.position_tolerance,
                    planning_horizon_chunks=1,
                    geometry_mode="exact",
                    k_schedule=tuple(args.k_schedule),
                    match_current_robot_points=True,
                    rng=rng,
                    timer=timer,
                )
                steps_to_execute = min(
                    decision.selected_chunk.horizon,
                    int(policy.n_action_steps),
                    max_steps - total_steps,
                )
                for policy_action in decision.selected_chunk.actions[:steps_to_execute]:
                    sim_action = policy_action_to_sim_action(
                        policy_action,
                        np.asarray(sim_entry["agent_pos"], dtype=np.float32),
                        action_mode=action_mode,
                        sim_action_dim=int(np.prod(sim_env.action_space.shape)),
                        low=getattr(sim_env.action_space, "low", None),
                        high=getattr(sim_env.action_space, "high", None),
                        gripper_open=gripper_value,
                    )
                    if initial_ema_action is not None and total_steps < ema_warmup_steps:
                        # Ramp the blend weight up from a low floor instead of
                        # jumping straight to the full weight -- bounds the
                        # max per-step delta right after a seeded handoff
                        # regardless of how far off the raw prediction is.
                        warmup_frac = (total_steps + 1) / float(max(ema_warmup_steps, 1))
                        effective_alpha = args.transport_ema_warmup_alpha + warmup_frac * (
                            args.action_ema_alpha - args.transport_ema_warmup_alpha
                        )
                    else:
                        effective_alpha = args.action_ema_alpha
                    if ema_sim_action is None or effective_alpha >= 1.0:
                        ema_sim_action = sim_action
                    else:
                        ema_sim_action = (
                            effective_alpha * sim_action
                            + (1.0 - effective_alpha) * ema_sim_action
                        )
                    sim_obs, _reward, _term, truncated, sim_info = sim_env.step(ema_sim_action)
                    total_steps += 1
                    sim_entry = rollout_observation_entry(
                        sim_obs, sim_info, env=sim_env, crop_config=crop_config
                    )
                    obs_window = append_obs_window(
                        obs_window, sim_entry, n_obs_steps=int(policy.n_obs_steps)
                    )
                    if frames is not None:
                        frames.append(frame_to_numpy(sim_env.render()))

                    tcp = _tcp_pose(sim_env.unwrapped)
                    pos_err = float(np.linalg.norm(tcp[:3].astype(np.float64) - target_position))
                    rot_err = _quat_angular_distance_rad(
                        tcp[3:7].astype(np.float64), _DOWN_QUAT_WXYZ.astype(np.float64)
                    )
                    if pos_err <= args.position_tolerance and rot_err <= args.rotation_tolerance:
                        reached_goal = True
                        break
                    if bool_any(truncated):
                        truncated_early = True
                        break
                if truncated_early or reached_goal:
                    break
        finally:
            if was_training:
                policy.train()
        return (
            reached_goal,
            sim_entry,
            obs_window,
            total_steps,
            pos_err,
            rot_err,
            truncated_early,
            ema_sim_action,
        )

    def _replay_planned_move(
        *,
        position: np.ndarray,
        gripper_value: float,
        max_steps: int,
        frames: list[np.ndarray] | None,
    ) -> tuple[bool, np.ndarray | None]:
        """Plan+replay a straight-line screw move (descent, lift). Scripted, not policy."""
        solver = XArm7GripperMotionPlanningSolver(
            sim_env,
            debug=False,
            vis=False,
            base_pose=sim_env.unwrapped.agent.robot.pose,
            visualize_target_grasp_pose=False,
            print_env_info=False,
        )
        try:
            plan = _move_to_pose_with_screw(
                solver,
                _pose_with_orientation(sapien, position=position, quat=_DOWN_QUAT_WXYZ),
                suppress_output=True,
            )
        finally:
            solver.close()
        if plan == -1 or "position" not in plan:
            return False, None
        last_action: np.ndarray | None = None
        for planned_qpos in np.asarray(plan["position"], dtype=np.float32)[:max_steps]:
            action = _format_sim_action(sim_env, planned_qpos, gripper_action=gripper_value)
            _obs, _reward, _term, truncated, _info = sim_env.step(action)
            last_action = np.asarray(action, dtype=np.float32)
            if frames is not None:
                frames.append(frame_to_numpy(sim_env.render()))
            if bool_any(truncated):
                break
        return True, last_action

    rows: list[dict[str, Any]] = []
    if args.video_dir is not None:
        args.video_dir.mkdir(parents=True, exist_ok=True)

    try:
        rest_gap: float | None = None
        for episode_idx in range(args.episodes):
            _reset_obs, reset_info = sim_env.reset(
                seed=args.seed + episode_idx, options={"reconfigure": True}
            )
            if rest_gap is None:
                rest_gap = _finger_separation(sim_env)
                if rest_gap < args.cube_edge:
                    print(
                        f"*** WARNING: jaw rest opening ({rest_gap * 100:.1f}cm) is "
                        f"narrower than the cube ({args.cube_edge * 100:.0f}cm) -- "
                        "results below are confounded by geometry, not the policy. "
                        "See tune-gripper.",
                        file=sys.stderr,
                    )

            cube_start = cube_position(sim_env)
            grasp_position = cube_start.copy()
            pregrasp_position = grasp_position + np.array(
                [0.0, 0.0, args.pregrasp_height], dtype=np.float32
            )
            # Read the env's own sampled goal BEFORE goal_site gets repurposed
            # as the pick's conditioning signal below -- this is where the cube
            # has to end up, and pnp_env already guarantees it is at least
            # --min-object-separation from the cube, so every episode is a real
            # relocation.
            place_target = goal_position(sim_env)
            # Release slightly above the cube's own resting height so the cube
            # is let go just clear of the table rather than pressed into it.
            release_position = place_target + np.array(
                [0.0, 0.0, args.place_height_offset], dtype=np.float32
            )
            row: dict[str, Any] = {
                "episode": episode_idx,
                "cube_start": cube_start.tolist(),
                "pregrasp_position": pregrasp_position.tolist(),
                "place_target": place_target.tolist(),
            }

            # goal_site IS the policy's goal-conditioning signal (see
            # pnp_env's docstring / PG3DReachEnv._get_obs_extra), so pointing
            # it at the pre-grasp standoff is how the checkpoint is told where
            # to go. One actor serves both jobs, so the green marker rides the
            # ACTIVE target: it hovers at the cube's approach point during the
            # pick, then returns to the table goal for the place phase.
            _set_site_pose(sim_env, "goal_site", pregrasp_position)
            update_render = getattr(sim_env.unwrapped.scene, "update_render", None)
            if callable(update_render):
                update_render()
            sim_obs, sim_info = _refresh_obs_after_manual_qpos(
                sim_env, info=reset_info, gripper_open=args.gripper_open_value
            )
            sim_entry = rollout_observation_entry(
                sim_obs, sim_info, env=sim_env, crop_config=crop_config
            )
            obs_window = make_initial_obs_window(sim_entry, n_obs_steps=int(policy.n_obs_steps))

            conditioned_on = np.asarray(sim_entry["target_position"], dtype=np.float64)
            if not np.allclose(conditioned_on, pregrasp_position.astype(np.float64), atol=1e-3):
                print(
                    f"  [ep {episode_idx:02d}] WARNING — policy is conditioned on "
                    f"{conditioned_on.round(3).tolist()} but the pre-grasp target is "
                    f"{pregrasp_position.astype(np.float64).round(3).tolist()}; "
                    "goal_site did not take the new pose"
                )

            record = args.video_dir is not None and (
                args.video_all or episode_idx < args.video_first_n
            )
            frames: list[np.ndarray] | None = (
                [frame_to_numpy(sim_env.render())] if record else None
            )

            # --- 1. POLICY REACH to the pre-grasp standoff.
            (
                reached,
                sim_entry,
                obs_window,
                reach_steps,
                pos_err,
                rot_err,
                truncated_early,
                _pick_ema_action,
            ) = _policy_reach_to(
                sim_entry=sim_entry,
                obs_window=obs_window,
                frames=frames,
                target_position=pregrasp_position,
                max_steps=args.max_steps,
                gripper_value=args.gripper_open_value,
                phase="pick_pregrasp",
            )
            row.update(
                reach_steps=reach_steps,
                reach_pos_err=float(pos_err),
                reach_rot_err_deg=float(np.degrees(rot_err)),
                reach_reached=bool(reached),
            )
            if not reached:
                reason = (
                    f"truncated at step {reach_steps}"
                    if truncated_early
                    else f"step budget ({args.max_steps}) exhausted"
                )
                row.update(outcome="reach_failed", cube_lift=0.0, success=False)
                print(
                    f"  [ep {episode_idx:02d}] cube={np.round(cube_start, 3).tolist()}  "
                    f"REACH FAILED — {reason}, pos_err={pos_err:.4f} "
                    f"rot_err={np.degrees(rot_err):.1f}deg"
                )
                rows.append(row)
                _save_pick_video(args, save_video, frames, episode_idx, row)
                continue

            # --- 2. SCRIPTED VERTICAL DESCENT onto the cube.
            descent_ok, _ = _replay_planned_move(
                position=grasp_position,
                gripper_value=args.gripper_open_value,
                max_steps=args.descent_max_steps,
                frames=frames,
            )
            if not descent_ok:
                row.update(outcome="descent_unplannable", cube_lift=0.0, success=False)
                print(
                    f"  [ep {episode_idx:02d}] cube={np.round(cube_start, 3).tolist()}  "
                    f"reach OK ({reach_steps} steps) but DESCENT UNPLANNABLE"
                )
                rows.append(row)
                _save_pick_video(args, save_video, frames, episode_idx, row)
                continue

            tcp_at_grasp = _tcp_pose(sim_env.unwrapped)[:2]
            cube_now = cube_position(sim_env)
            centering_err = float(np.linalg.norm(tcp_at_grasp - cube_now[:2]))
            approach_drift = float(np.linalg.norm(cube_now[:2] - cube_start[:2]))
            row.update(grasp_centering_err=centering_err, approach_drift=approach_drift)

            # --- 3. SCRIPTED CLOSE (ramped) then LIFT.
            hold_qpos = np.asarray(sim_env.unwrapped.agent.robot.get_qpos()).reshape(-1)[:7]
            for close_step in range(args.close_steps):
                frac = (close_step + 1) / float(args.close_steps)
                value = args.gripper_open_value + frac * (
                    args.gripper_close_value - args.gripper_open_value
                )
                action = _hold_sim_action(sim_env, gripper_open=value, qpos=hold_qpos)
                _obs, _reward, _term, truncated, _info = sim_env.step(action)
                if frames is not None:
                    frames.append(frame_to_numpy(sim_env.render()))
                if bool_any(truncated):
                    break

            cube_before_lift = cube_position(sim_env)
            lift_position = grasp_position + np.array(
                [0.0, 0.0, args.lift_height], dtype=np.float32
            )
            lift_ok, last_lift_action = _replay_planned_move(
                position=lift_position,
                gripper_value=args.gripper_close_value,
                max_steps=args.lift_max_steps,
                frames=frames,
            )
            cube_after_lift = cube_position(sim_env)
            cube_lift = float(cube_after_lift[2] - cube_before_lift[2])
            picked_ok = bool(
                lift_ok and cube_lift >= args.success_lift_fraction * args.lift_height
            )
            row.update(
                cube_lift=cube_lift,
                cube_after_lift=cube_after_lift.tolist(),
                lift_planned=lift_ok,
                picked=picked_ok,
            )

            if not place_enabled:
                row.update(
                    success=picked_ok,
                    cube_end=cube_after_lift.tolist(),
                    outcome="picked"
                    if picked_ok
                    else ("dropped_or_missed" if lift_ok else "lift_unplannable"),
                )
                print(
                    f"  [ep {episode_idx:02d}] cube={np.round(cube_start, 3).tolist()}  "
                    f"reach OK ({reach_steps} steps, pos_err={pos_err:.4f})  "
                    f"centering={centering_err:.4f}m drift={approach_drift:.4f}m  "
                    f"lift={cube_lift:.4f}m  outcome={row['outcome']}"
                )
                rows.append(row)
                _save_pick_video(args, save_video, frames, episode_idx, row)
                continue

            # Nothing to transport if the pick didn't hold -- the gripper is
            # closed on empty air, or the cube fell during the lift.
            if not picked_ok:
                row.update(
                    success=False,
                    cube_end=cube_after_lift.tolist(),
                    outcome="pick_failed",
                )
                print(
                    f"  [ep {episode_idx:02d}] cube={np.round(cube_start, 3).tolist()}  "
                    f"reach OK ({reach_steps} steps)  centering={centering_err:.4f}m  "
                    f"PICK FAILED — cube rose only {cube_lift:.4f}m "
                    f"(needed >={args.success_lift_fraction * args.lift_height:.4f}m)"
                )
                rows.append(row)
                _save_pick_video(args, save_video, frames, episode_idx, row)
                continue

            # Stiffen the grip now that the cube is confirmed seated (lift
            # just succeeded, both fingers symmetrically loaded) -- the
            # asymmetric-contact punch-out risk that keeps CLOSE force at
            # 0.05 no longer applies once we're just holding, and 0.05 only
            # has margin for an instantaneous grasp check, not for holding
            # weight over many steps: frame-by-frame video review (E:/
            # place_chk/pnp) showed the cube sinking to table height DURING
            # the post-lift settle hold below -- arm stationary the whole
            # time, no transport motion yet -- i.e. a slow static gravity
            # slip through the fingers, not a dynamic/inertial one. Moving
            # the bump to BEFORE the settle loop (rather than after, as it
            # first was) is what actually covers that window. No local sim
            # to tune this against (eval runs on enigma) -- 0.25 is an
            # engineering estimate (~5x the static-hold floor, a fraction of
            # the 20 that punched cubes out during CLOSE) rather than a swept
            # value; revisit with a slip-vs-force sweep (same shape as
            # tune-gripper) if slips persist or a wider margin turns out safe.
            _set_gripper_force_limit(sim_env, args.transport_gripper_force_limit)

            # --- Post-lift SETTLE: hold the current qpos, gripper still
            # closed, for a few steps before handing control back to the
            # policy. The lift is an open-loop scripted replay of a planned
            # trajectory that can end abruptly with nonzero joint velocity --
            # letting that die out here means the place phase's first policy
            # step isn't also fighting leftover motion from the lift.
            lift_hold_qpos = np.asarray(sim_env.unwrapped.agent.robot.get_qpos()).reshape(-1)[:7]
            for _settle_step in range(args.post_lift_settle_steps):
                action = _hold_sim_action(
                    sim_env, gripper_open=args.gripper_close_value, qpos=lift_hold_qpos
                )
                sim_obs, _reward, _term, truncated, sim_info = sim_env.step(action)
                if frames is not None:
                    frames.append(frame_to_numpy(sim_env.render()))
                if bool_any(truncated):
                    break

            # Bisection check: is the cube still up here, or did it already
            # slip during LIFT or during this very SETTLE hold, before the
            # transport ever starts? A video's own goal_site sphere sits
            # right on top of the gripper here and fully occludes a mid-air
            # cube for this whole window, so pixels alone can't answer this
            # -- ground truth can.
            cube_after_settle = cube_position(sim_env)
            held_after_settle = bool(
                (cube_after_settle[2] - cube_rest_z)
                >= 0.5 * args.success_lift_fraction * args.lift_height
            )
            row.update(
                cube_z_after_settle=float(cube_after_settle[2]),
                held_after_settle=held_after_settle,
            )
            if not held_after_settle:
                print(
                    f"  [ep {episode_idx:02d}] DROPPED during LIFT/SETTLE -- "
                    f"cube already at z={cube_after_settle[2]:.4f} "
                    f"(rest={cube_rest_z:.4f}) before transport even starts"
                )

            # --- 4. POLICY TRANSPORT to the goal marker, cube still held.
            # Same reranking rollout as the pick approach; the green marker
            # (and with it the policy's goal conditioning) goes back to the
            # table goal it was sampled at.
            _set_site_pose(sim_env, "goal_site", place_target)
            if callable(update_render):
                update_render()
            sim_obs, sim_info = _refresh_obs_after_manual_qpos(
                sim_env, info=sim_info, gripper_open=args.gripper_close_value
            )
            sim_entry = rollout_observation_entry(
                sim_obs, sim_info, env=sim_env, crop_config=crop_config
            )
            # Reinitialize the window (like the START of the episode) rather
            # than appending one new entry into a window still mostly full of
            # PICK-phase history (conditioned on the old, pregrasp goal). This
            # checkpoint was trained as a single-goal-per-episode reach
            # policy; a window mixing two different goals is exactly the kind
            # of input it never saw, and is what was producing an erratic,
            # sometimes violent, first prediction at this handoff.
            obs_window = make_initial_obs_window(sim_entry, n_obs_steps=int(policy.n_obs_steps))
            (
                place_reached,
                sim_entry,
                obs_window,
                place_steps,
                place_pos_err,
                place_rot_err,
                place_truncated,
                _place_ema_action,
            ) = _policy_reach_to(
                sim_entry=sim_entry,
                obs_window=obs_window,
                frames=frames,
                target_position=release_position,
                max_steps=args.place_max_steps,
                gripper_value=args.gripper_close_value,
                phase="place_transport",
                # Seeded from the lift's last raw action: the close+lift is a
                # scripted open-loop replay, so an unseeded EMA here blends a
                # raw policy action against whatever that replay left off at.
                initial_ema_action=last_lift_action,
                ema_warmup_steps=args.transport_ema_warmup_steps,
            )
            row.update(
                place_steps=place_steps,
                place_pos_err=float(place_pos_err),
                place_rot_err_deg=float(np.degrees(place_rot_err)),
                place_reached=bool(place_reached),
            )

            # `place_pos_err` above is the ARM's own tracking error against
            # the release target -- it says nothing about whether the cube
            # was still between the fingers when the arm got there. A low
            # pos_err with a large final cube XY error (checked below) is
            # exactly what an empty-handed arm converging on the goal looks
            # like, so check the cube's height HERE, before release, while
            # we still know whether the transport carried it or dropped it
            # somewhere along the way.
            cube_after_transport = cube_position(sim_env)
            # Lenient vs. the pick's own lift-success bar: grip compliance
            # can sag a held cube a little without it having actually
            # dropped, so this only needs to rule out "already on the table".
            held_through_transport = bool(
                (cube_after_transport[2] - cube_rest_z)
                >= 0.5 * args.success_lift_fraction * args.lift_height
            )
            row.update(
                cube_z_after_transport=float(cube_after_transport[2]),
                held_through_transport=held_through_transport,
            )

            # Drop the grip back to the tuned CLOSE/RELEASE force before
            # opening -- the transport-phase bump above is deliberately not
            # in effect for the release ramp, which was tuned against 0.05.
            _set_gripper_force_limit(sim_env, close_force_limit)

            # --- 5. RELEASE (ramped open), then settle. Deliberately runs
            # even if the transport did not converge: releasing wherever the
            # arm actually ended up gets scored honestly, and stops the cube
            # being carried into the next episode's reset.
            hold_qpos = np.asarray(sim_env.unwrapped.agent.robot.get_qpos()).reshape(-1)[:7]
            for release_step in range(args.release_steps):
                frac = (release_step + 1) / float(args.release_steps)
                value = args.gripper_close_value + frac * (
                    args.gripper_open_value - args.gripper_close_value
                )
                action = _hold_sim_action(sim_env, gripper_open=value, qpos=hold_qpos)
                _obs, _reward, _term, truncated, _info = sim_env.step(action)
                if frames is not None:
                    frames.append(frame_to_numpy(sim_env.render()))
                if bool_any(truncated):
                    break
            # Let the cube actually land and stop before measuring it --
            # scoring mid-fall would read as a miss even for a good release.
            for _settle_step in range(args.settle_steps):
                action = _hold_sim_action(
                    sim_env, gripper_open=args.gripper_open_value, qpos=hold_qpos
                )
                _obs, _reward, _term, truncated, _info = sim_env.step(action)
                if frames is not None:
                    frames.append(frame_to_numpy(sim_env.render()))
                if bool_any(truncated):
                    break

            cube_end = cube_position(sim_env)
            place_xy_err = float(np.linalg.norm(cube_end[:2] - place_target[:2]))
            resting_z_err = float(abs(cube_end[2] - cube_rest_z))
            # Placed = landed inside the tolerance disc AND actually settled at
            # its natural resting height. The height check is what separates a
            # real placement from a cube still wedged in the jaws, or one that
            # bounced off the table edge.
            settled = resting_z_err <= args.place_settle_tolerance
            placed_ok = bool(place_xy_err <= args.place_radius and settled)
            outcome = "placed" if placed_ok else ("not_settled" if not settled else "missed_goal")
            # A dropped-in-flight miss is a different failure than a
            # genuinely-carried-but-overshot one, and looks identical in the
            # xy_err/settled numbers alone -- relabel it using the
            # held_through_transport check above so the two aren't conflated.
            if not placed_ok and not row["held_through_transport"]:
                outcome = "dropped_in_transport"
            row.update(
                cube_end=cube_end.tolist(),
                place_xy_err=place_xy_err,
                cube_resting_z_err=resting_z_err,
                cube_settled=settled,
                success=placed_ok,
                outcome=outcome,
            )
            print(
                f"  [ep {episode_idx:02d}] cube={np.round(cube_start, 3).tolist()} "
                f"-> goal={np.round(place_target, 3).tolist()}  "
                f"pick OK (lift={cube_lift:.4f}m)  "
                f"transport {'OK' if place_reached else 'NOT CONVERGED'} "
                f"({place_steps} steps, pos_err={place_pos_err:.4f}, "
                f"held={row['held_through_transport']})  "
                f"final XY err={place_xy_err:.4f}m (tol {args.place_radius:.3f})  "
                f"outcome={row['outcome']}"
            )
            rows.append(row)
            _save_pick_video(args, save_video, frames, episode_idx, row)
    finally:
        sim_env.close()
        ghost_env.close()
        XArm7Gripper.gripper_force_limit = baseline_force
        XArm7Gripper.gripper_stiffness = baseline_stiffness
        XArm7Gripper.gripper_damping = baseline_damping

    reached_rows = [r for r in rows if r.get("reach_reached")]
    picked_rows = [r for r in rows if r.get("picked")]
    succeeded = [r for r in rows if r["success"]]
    lifts = [r["cube_lift"] for r in rows]
    outcomes = {
        outcome: sum(1 for r in rows if r["outcome"] == outcome)
        for outcome in sorted({r["outcome"] for r in rows})
    }
    total = max(len(rows), 1)
    label = "placed" if place_enabled else "picked"
    print(
        f"\n── {stage} summary: {label} {len(succeeded)}/{len(rows)} "
        f"({100 * len(succeeded) / total:.0f}%)\n"
        f"   funnel: policy reach converged {len(reached_rows)}/{len(rows)}"
        f"  ->  picked {len(picked_rows)}/{len(rows)}"
        + (
            f"  ->  placed {len(succeeded)}/{len(rows)}"
            if place_enabled
            else ""
        )
        + f"\n   mean_lift={float(np.mean(lifts)) if lifts else 0.0:.4f}m"
        f"\n   outcomes: {outcomes}"
    )
    if place_enabled:
        placed_rows = [r for r in rows if r.get("place_xy_err") is not None]
        if placed_rows:
            errs = [r["place_xy_err"] for r in placed_rows]
            transported = [r for r in placed_rows if r.get("place_reached")]
            print(
                f"   of the {len(placed_rows)} episode(s) that got as far as a "
                f"release: transport converged {len(transported)}/{len(placed_rows)}, "
                f"final XY error mean={float(np.mean(errs)):.4f}m "
                f"min={float(np.min(errs)):.4f}m max={float(np.max(errs)):.4f}m "
                f"(tolerance {args.place_radius:.3f}m)"
            )
    if reached_rows and len(picked_rows) < len(reached_rows):
        print(
            f"   NOTE: {len(reached_rows) - len(picked_rows)} episode(s) reached the "
            "pre-grasp pose but still failed to pick -- that is a grasp/descent "
            "problem, not a policy-steering one. Check grasp_centering_err and "
            "approach_drift in those rows before touching reach settings."
        )
    if place_enabled and picked_rows and len(succeeded) < len(picked_rows):
        print(
            f"   NOTE: {len(picked_rows) - len(succeeded)} episode(s) picked the cube "
            "but did not place it. 'missed_goal' means the transport ended too far "
            "from the marker (a steering problem -- check place_pos_err); "
            "'not_settled' means the cube never came to rest at table height "
            "(it stuck in the jaws, or bounced away -- check the video)."
        )
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(rows, indent=2))
        print(f"wrote pick summary: {args.summary_json}")
    return 0


def _save_pick_video(
    args: argparse.Namespace,
    save_video: Any,
    frames: list[np.ndarray] | None,
    episode_idx: int,
    row: dict[str, Any],
) -> None:
    """Write one episode's video (if recorded) and record its path on the row."""
    if not frames or args.video_dir is None:
        return
    path = args.video_dir / f"pick_ep{episode_idx:02d}_{row['outcome']}.mp4"
    save_video(path, frames, fps=args.video_fps)
    row["video"] = str(path)
    print(f"    video: {path}")


def _not_implemented(mode: str) -> int:
    print(
        f"--mode {mode} is not implemented yet. scene, tune-gripper, and pick "
        "are done, in that order -- pick-place is next.",
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
        help="Which stage to run: scene -> tune-gripper -> pick -> pick-place.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--cube-edge",
        type=float,
        default=0.04,
        help="Cube side length in metres (0.04 = the 4cm cube; the original "
        "7cm default was wider than the gripper's measured 5.47cm rest "
        "opening -- see tune-gripper's finger-separation warning).",
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
        default=[0.05, 0.1, 0.2, 0.5, 1.0],
        help="tune-gripper: gripper_force_limit values to sweep. Narrowed "
        "around 0.1 -- an earlier wider sweep (0.1/5/20/50/100) found every "
        "value above 0.1 drove the joint unstable (>100 rad/s) before even "
        "reaching the cube, and 0.1 itself only held 40%%.",
    )
    p.add_argument(
        "--stiffnesses",
        type=float,
        nargs="+",
        default=[100_000.0, 20_000.0, 5_000.0, 1_000.0, 200.0],
        help="tune-gripper: gripper_stiffness values to cross with "
        "--force-limits (full grid, not independent sweeps). Defaults to a "
        "descending sweep from the current baseline (1e5): a PD spring that "
        "stiff snapping onto rigid contact is a plausible cause of the "
        "qvel blowups seen at every force_limit so far, independent of force. "
        "Pass a single value to pin stiffness and sweep force alone.",
    )
    p.add_argument(
        "--dampings",
        type=float,
        nargs="+",
        default=None,
        help="tune-gripper: optional gripper_damping values to cross in as a "
        "third sweep axis. Defaults to leaving damping at its current value "
        "-- add this only after force/stiffness narrow things down, the grid "
        "size multiplies.",
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
    p.add_argument("--video-dir", type=Path, default=None, help="tune-gripper/pick: directory to record videos into.")
    p.add_argument("--video-fps", type=int, default=15)
    p.add_argument("--summary-json", type=Path, default=None)
    # pick mode
    p.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="pick: number of pick attempts, each at a freshly sampled cube "
        "position spanning the whole workspace.",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="pick: trained DP3 reach checkpoint to roll out for the approach. Required for --mode pick.",
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="pick: dataset whose metadata.json describes the observation/crop/"
        "action config the checkpoint was trained on. Read for metadata ONLY -- "
        "never for episode content, and never for the env id (the env is always "
        "this script's own PG3DPnP scene, so the cube is actually present). "
        "Required for --mode pick.",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="pick: torch device for the policy -- 'auto', 'cpu', 'cuda', etc. (default: auto-select).",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=300,
        help="pick: step budget for the policy reach to the pre-grasp standoff.",
    )
    p.add_argument(
        "--position-tolerance",
        type=float,
        default=0.02,
        help="pick: pre-grasp reach counts as converged within this position error (m).",
    )
    p.add_argument(
        "--rotation-tolerance",
        type=float,
        default=0.15,
        help="pick: and within this orientation error (rad) of straight-down.",
    )
    p.add_argument(
        "--k-schedule",
        type=int,
        nargs="+",
        default=[8],
        help="pick: reranking candidates per replan (the DP3 steering mechanism).",
    )
    p.add_argument(
        "--action-ema-alpha",
        type=float,
        default=0.5,
        help="pick: EMA blend on policy actions (1.0 = raw policy action, no smoothing). "
        "Smoothing matters against this gripper's stiff PD controller.",
    )
    p.add_argument("--policy-batch-size", type=int, default=8, help="pick: policy inference batch size.")
    p.add_argument(
        "--lift-max-steps",
        type=int,
        default=80,
        help="pick: replay budget for the scripted lift.",
    )
    p.add_argument(
        "--descent-max-steps",
        type=int,
        default=80,
        help="pick: replay budget for the scripted vertical descent from the "
        "pre-grasp standoff onto the cube.",
    )
    # pick-place mode (everything above applies too -- the pick half is shared)
    p.add_argument(
        "--place-radius",
        type=float,
        default=0.05,
        help="pick-place: the cube counts as placed if it settles within this "
        "distance (m, XY) of the goal marker. The marker is DRAWN at this "
        "radius, so the visible green target on the table is the scored zone.",
    )
    p.add_argument(
        "--place-height-offset",
        type=float,
        default=0.02,
        help="pick-place: release the cube this far above its natural resting "
        "height, so it is let go just clear of the table rather than pressed "
        "into it.",
    )
    p.add_argument(
        "--place-settle-tolerance",
        type=float,
        default=0.03,
        help="pick-place: after release the cube's height must be within this "
        "of its natural resting height to count as settled. This is what "
        "separates a real placement from one still wedged in the jaws.",
    )
    p.add_argument(
        "--place-max-steps",
        type=int,
        default=300,
        help="pick-place: step budget for the policy transport to the goal.",
    )
    p.add_argument(
        "--release-steps",
        type=int,
        default=15,
        help="pick-place: steps to ramp the gripper back open, mirroring --close-steps.",
    )
    p.add_argument(
        "--settle-steps",
        type=int,
        default=25,
        help="pick-place: steps to let the cube land and stop before scoring it. "
        "Measuring mid-fall would read a good release as a miss.",
    )
    p.add_argument(
        "--post-lift-settle-steps",
        type=int,
        default=10,
        help="pick-place: steps to hold still (gripper closed) right after the "
        "lift, before the policy takes back control for the transport. Lets "
        "any residual velocity from the lift's open-loop scripted replay die "
        "out first, so the transport's first policy step isn't also fighting "
        "leftover motion.",
    )
    p.add_argument(
        "--transport-ema-warmup-steps",
        type=int,
        default=15,
        help="pick-place: ramp the transport phase's EMA blend weight up from "
        "--transport-ema-warmup-alpha to --action-ema-alpha over this many "
        "steps, instead of using the full weight from step 1. Bounds the max "
        "per-step move right after the pick->place handoff -- the point in "
        "the rollout where this checkpoint (trained single-goal-per-episode) "
        "is most out of distribution, since the goal just teleported to a "
        "brand-new position while still holding the cube.",
    )
    p.add_argument(
        "--transport-ema-warmup-alpha",
        type=float,
        default=0.1,
        help="pick-place: EMA blend weight at the very start of the transport "
        "warmup ramp (see --transport-ema-warmup-steps). Low = trust the "
        "seeded lift action more initially; must be < --action-ema-alpha or "
        "the ramp does nothing.",
    )
    p.add_argument(
        "--transport-gripper-force-limit",
        type=float,
        default=0.25,
        help="pick-place: live drive_joint force_limit used for HOLD/TRANSPORT "
        "and RELEASE, applied right after the post-lift settle (see "
        "_set_gripper_force_limit) and dropped back to the CLOSE force before "
        "the release ramp. The CLOSE force (0.05) is tuned soft so asymmetric "
        "first contact doesn't punch the cube out before both fingers are "
        "seated -- that risk is gone once the lift has confirmed a symmetric "
        "grasp, but 0.05 alone only has margin for gravity, not the inertial "
        "loads a moving arm adds, which is what was causing slips at random "
        "points along the transport path. No local sim to sweep this against "
        "(eval runs on enigma) -- 0.25 is an engineering estimate (~5x the "
        "static-hold floor, well under the 20 that punched cubes out during "
        "CLOSE), not a tuned value. Revisit with a slip-vs-force sweep if "
        "slips persist or a wider margin turns out safe.",
    )
    p.add_argument(
        "--gripper-force-limit",
        type=float,
        default=None,
        help="pick: override XArm7Gripper.gripper_force_limit for this run "
        "(default: leave the class default from agents.py, i.e. the "
        "tune-gripper-recommended 0.05).",
    )
    p.add_argument("--gripper-stiffness", type=float, default=None, help="pick: override gripper_stiffness.")
    p.add_argument("--gripper-damping", type=float, default=None, help="pick: override gripper_damping.")
    p.add_argument(
        "--video-all",
        action="store_true",
        help="pick: record every episode instead of just the first --video-first-n.",
    )
    p.add_argument(
        "--video-first-n",
        type=int,
        default=3,
        help="pick: with --video-dir set (and --video-all not set), record only this many leading episodes.",
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
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    if any(f < 0 for f in args.force_limits):
        raise ValueError("--force-limits must be non-negative")
    if args.stiffnesses is not None and any(s <= 0 for s in args.stiffnesses):
        raise ValueError("--stiffnesses must be positive")
    if args.dampings is not None and any(d < 0 for d in args.dampings):
        raise ValueError("--dampings must be non-negative")
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
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.gripper_force_limit is not None and args.gripper_force_limit < 0:
        raise ValueError("--gripper-force-limit must be non-negative")
    if args.transport_gripper_force_limit <= 0:
        raise ValueError("--transport-gripper-force-limit must be positive")
    if args.gripper_stiffness is not None and args.gripper_stiffness <= 0:
        raise ValueError("--gripper-stiffness must be positive")
    if args.gripper_damping is not None and args.gripper_damping < 0:
        raise ValueError("--gripper-damping must be non-negative")
    if args.video_first_n < 0:
        raise ValueError("--video-first-n must be non-negative")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.position_tolerance <= 0:
        raise ValueError("--position-tolerance must be positive")
    if args.rotation_tolerance <= 0:
        raise ValueError("--rotation-tolerance must be positive")
    if not args.k_schedule or any(k <= 0 for k in args.k_schedule):
        raise ValueError("--k-schedule entries must be positive")
    if not (0.0 < args.action_ema_alpha <= 1.0):
        raise ValueError("--action-ema-alpha must be in (0.0, 1.0]")
    if args.policy_batch_size <= 0:
        raise ValueError("--policy-batch-size must be positive")
    if args.lift_max_steps <= 0:
        raise ValueError("--lift-max-steps must be positive")
    if args.descent_max_steps <= 0:
        raise ValueError("--descent-max-steps must be positive")
    if args.place_radius <= 0:
        raise ValueError("--place-radius must be positive")
    if args.place_settle_tolerance <= 0:
        raise ValueError("--place-settle-tolerance must be positive")
    if args.place_max_steps <= 0:
        raise ValueError("--place-max-steps must be positive")
    if args.release_steps <= 0:
        raise ValueError("--release-steps must be positive")
    if args.settle_steps <= 0:
        raise ValueError("--settle-steps must be positive")
    if args.post_lift_settle_steps <= 0:
        raise ValueError("--post-lift-settle-steps must be positive")
    if args.transport_ema_warmup_steps < 0:
        raise ValueError("--transport-ema-warmup-steps must be non-negative")
    if not (0.0 < args.transport_ema_warmup_alpha <= 1.0):
        raise ValueError("--transport-ema-warmup-alpha must be in (0.0, 1.0]")
    if args.mode == "pick-place" and args.transport_ema_warmup_alpha >= args.action_ema_alpha:
        raise ValueError(
            "--transport-ema-warmup-alpha must be < --action-ema-alpha "
            "(otherwise the warmup ramp does nothing)"
        )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "scene":
        return run_scene(args)
    if args.mode == "tune-gripper":
        return run_tune_gripper(args)
    if args.mode == "pick":
        return run_pick(args, place_enabled=False)
    if args.mode == "pick-place":
        return run_pick(args, place_enabled=True)
    return _not_implemented(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
