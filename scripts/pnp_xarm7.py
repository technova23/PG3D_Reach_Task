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
* ``pick-place`` -- STAGE 4. Adds the transport-and-release half.

``pick-place`` is not implemented yet -- it exits with a clear message
rather than pretending to run.

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


def run_pick(args: argparse.Namespace) -> int:
    """STAGE 3: scripted top-down pick, cube spawned anywhere in the workspace.

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
    from pg3d.envs.xarm_adapter.pnp_env import cube_position, register_pnp_envs
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

    # One episode is: policy reach + scripted descent + close + lift. The env
    # step limit has to cover all of it or a late phase gets truncated by the
    # env rather than judged on its merits.
    meta_max_steps = metadata.get("env_kwargs", {}).get("max_episode_steps")
    env_max_steps = max(
        int(args.max_steps)
        + int(args.descent_max_steps)
        + int(args.close_steps)
        + int(args.lift_max_steps),
        int(meta_max_steps or 0),
    )
    print(
        f"pick: force_limit={XArm7Gripper.gripper_force_limit} "
        f"stiffness={XArm7Gripper.gripper_stiffness} "
        f"damping={XArm7Gripper.gripper_damping}\n"
        f"      {args.episodes} episodes, cube edge {args.cube_edge * 100:.0f}cm, "
        f"straight-down approach only, env step limit {env_max_steps}\n"
        f"      policy={args.checkpoint}\n"
    )

    sim_env_kwargs = _env_kwargs(metadata, render_mode="rgb_array", max_episode_steps=env_max_steps)
    sim_env_kwargs["cube_edge"] = args.cube_edge
    sim_env_kwargs["spawn_margin"] = args.spawn_margin
    sim_env_kwargs["min_object_separation"] = args.min_object_separation
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
    ) -> tuple[bool, dict[str, Any], Any, int, float, float, bool]:
        """Steer the checkpoint to `target_position` (straight-down orientation).

        Same reranking mechanism as the reach evals -- this is the actual DP3
        rollout, not a planner. Returns (reached, sim_entry, obs_window,
        steps, pos_err, rot_err, truncated_early); `frames` is mutated in
        place so the caller keeps one continuous video across phases.
        """
        constraint = CartesianPoseConstraint(
            target_position=target_position,
            target_orientation=_DOWN_QUAT_WXYZ,
            position_tolerance=args.position_tolerance,
            rotation_tolerance=args.rotation_tolerance,
            weight=1.0,
            name="pnp_pick_pregrasp",
        )
        scene = scene_context_for_constraints(
            target_position=target_position,
            constraints=[constraint],
            metadata={"phase": "pick_pregrasp"},
        )
        timer = TimingRecorder(enabled=False)
        ema_sim_action: np.ndarray | None = None
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
                        gripper_open=args.gripper_open_value,
                    )
                    if ema_sim_action is None or args.action_ema_alpha >= 1.0:
                        ema_sim_action = sim_action
                    else:
                        ema_sim_action = (
                            args.action_ema_alpha * sim_action
                            + (1.0 - args.action_ema_alpha) * ema_sim_action
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
        return (reached_goal, sim_entry, obs_window, total_steps, pos_err, rot_err, truncated_early)

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
            row: dict[str, Any] = {
                "episode": episode_idx,
                "cube_start": cube_start.tolist(),
                "pregrasp_position": pregrasp_position.tolist(),
            }

            # goal_site IS the policy's goal-conditioning signal (see
            # pnp_env's docstring / PG3DReachEnv._get_obs_extra), so pointing
            # it at the pre-grasp standoff is how the checkpoint is told where
            # to go. Cosmetically it also moves the green marker onto the
            # cube's approach point for the duration of the pick.
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
            ) = _policy_reach_to(
                sim_entry=sim_entry,
                obs_window=obs_window,
                frames=frames,
                target_position=pregrasp_position,
                max_steps=args.max_steps,
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
            lift_ok, _ = _replay_planned_move(
                position=lift_position,
                gripper_value=args.gripper_close_value,
                max_steps=args.lift_max_steps,
                frames=frames,
            )
            cube_end = cube_position(sim_env)
            cube_lift = float(cube_end[2] - cube_before_lift[2])
            success = bool(
                lift_ok and cube_lift >= args.success_lift_fraction * args.lift_height
            )
            row.update(
                cube_lift=cube_lift,
                cube_end=cube_end.tolist(),
                lift_planned=lift_ok,
                success=success,
                outcome="picked" if success else ("dropped_or_missed" if lift_ok else "lift_unplannable"),
            )
            print(
                f"  [ep {episode_idx:02d}] cube={np.round(cube_start, 3).tolist()}  "
                f"reach OK ({reach_steps} steps, pos_err={pos_err:.4f})  "
                f"centering={centering_err:.4f}m drift={approach_drift:.4f}m  "
                f"lift={cube_lift:.4f}m  outcome={row['outcome']}"
            )
            rows.append(row)
            _save_pick_video(args, save_video, frames, episode_idx, row)
    finally:
        sim_env.close()
        ghost_env.close()
        XArm7Gripper.gripper_force_limit = baseline_force
        XArm7Gripper.gripper_stiffness = baseline_stiffness
        XArm7Gripper.gripper_damping = baseline_damping

    picked = [r for r in rows if r["success"]]
    reached_rows = [r for r in rows if r.get("reach_reached")]
    lifts = [r["cube_lift"] for r in rows]
    outcomes = {
        outcome: sum(1 for r in rows if r["outcome"] == outcome)
        for outcome in sorted({r["outcome"] for r in rows})
    }
    print(
        f"\n── pick summary: picked {len(picked)}/{len(rows)} "
        f"({100 * len(picked) / max(len(rows), 1):.0f}%)  |  "
        f"policy reach converged {len(reached_rows)}/{len(rows)}  |  "
        f"mean_lift={float(np.mean(lifts)) if lifts else 0.0:.4f}m\n"
        f"   outcomes: {outcomes}"
    )
    if reached_rows and len(picked) < len(reached_rows):
        print(
            f"   NOTE: {len(reached_rows) - len(picked)} episode(s) reached the "
            "pre-grasp pose but still failed to pick -- that is a grasp/descent "
            "problem, not a policy-steering one. Check grasp_centering_err and "
            "approach_drift in those rows before touching reach settings."
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
        help="Which stage to run. 'scene', 'tune-gripper', and 'pick' are implemented; 'pick-place' is not yet.",
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
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "scene":
        return run_scene(args)
    if args.mode == "tune-gripper":
        return run_tune_gripper(args)
    if args.mode == "pick":
        return run_pick(args)
    return _not_implemented(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
