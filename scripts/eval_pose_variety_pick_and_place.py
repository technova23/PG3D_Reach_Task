#!/usr/bin/env python
"""Checkpoint-based pick-and-place eval for xArm7 + gripper.

Totally inspired by (and reuses most of the machinery of)
``eval_pose_variety_checkpoint_reachability.py``. The difference is what the
episode is FOR: instead of an abstract start/goal position pair, a single
real, physically simulated cube (``pg3d.envs.xarm_adapter.pick_env``) is
spawned. The checkpoint is steered -- via the exact same reranking mechanism
as the reach eval, reused twice per episode -- first to an IK-verified
pre-grasp pose at the cube, then (once picked up) to a second,
INDEPENDENTLY sampled place pose with its own random position and
orientation, and released onto a 7cm-radius (default) landing-zone marker on
the table. Both the pick and the place are verified against ground truth:
grasp success from the cube's own SAPIEN pose rising with the gripper on
lift, place success from that same pose settling within the landing zone
after release.

Why hardcoded close/open, not learned: the checkpoint being evaluated has no
gripper output at all -- ``XArm7Gripper.agent_pos_joint_indices`` strips
gripper qpos out of the observation, and its action space validates to width
7 unless ``action_includes_gripper=True`` is set on the agent, which it is
not (see agents.py's own docstring, which says a future pick-and-place
policy should override this). So "grasp" and "release" are not things this
checkpoint can predict; it can only be steered to a pose, then a scripted
close/lift or open/release takes over -- exactly the design confirmed for
this repo's PnP plan.

Design, mirroring eval_pose_variety_checkpoint_reachability.py's structure:

* One "cube position" = one fixed cube XY placement on the table. For that
  position, run --episodes-per-position episodes that reuse the SAME cube
  position but each get their own INDEPENDENT, FPS-diversified grasp-APPROACH
  orientation (tilt away from straight-down) -- the cube itself always rests
  flat (only the gripper's approach angle varies; see the design discussion
  this script came out of). The grasp target position is the cube's center
  plus --grasp-height-offset -- fixed offset above cube center, straight
  down onto the top face by default.
* PLACE position is sampled fresh per episode too -- a separate random point
  on the table, at least --min-grasp-place-distance from the cube -- with its
  own independent random approach orientation (--place-orientation-cone-deg).
  This is the "sample another random point as goal, and some random
  orientation" for the place step.
* Robot START is always the env's own fixed rest configuration
  (``agent.keyframes["rest"].qpos``, gripper facing straight down) -- the
  SAME configuration every episode resets to, not a separately sampled
  Cartesian pose. "From its natural position facing downwards, the robot
  must be able to go towards the cube and pick it up."
* Every (grasp, place) pose is IK-verified with mplib (ground truth) before
  any policy rollout is attempted, exactly like the reach eval: each pose
  individually reachable AND both rest->grasp and grasp->place inter-feasible
  (a collision-free plan actually connects them), or the episode is skipped
  and counted, never silently dropped.
* Markers: start_site (red) sits at the cube -- there's no separate start
  pose left to mark, so it's repurposed as "what's being picked up".
  goal_site (green) tracks whichever target is currently active: the grasp
  pose during the pick phase, the place pose (== the place_site landing-zone
  cylinder's own position) during the place phase.
* PICK phase, in three parts, because a single learned reach straight to the
  grasp pose does not work: the policy arrives at cube height still moving
  laterally with the claws wide open, and broadsides the cube several cm away
  before anything closes (observed: TCP within 0.008m of the grasp target yet
  0.044m off the cube's live centre -- only possible if the cube moved).
  (a) APPROACH: rejection-free reranking (K candidate DP3 action chunks per
  replan, scored by imagined-rollout distance to a CartesianPoseConstraint,
  lowest cost executed), gripper held OPEN via --gripper-open-value, aimed at
  a pre-grasp STANDOFF --pregrasp-height back along the gripper's own approach
  axis. (b) DESCENT: a scripted straight-line screw motion down that same axis
  onto the grasp pose, so near the cube the fingers only ever move parallel to
  themselves. (c) CENTRING CHECK against the cube's LIVE position: if the
  fingers are further than --grasp-centering-tolerance off centre, the
  approach tilt is shrunk toward straight-down (see _slerp_towards_down) by
  an amount CALCULATED from how far the previous attempt actually missed by
  -- not a random guess: rotation error amplifies at the fingertips (they sit
  ~3.5cm+ from the TCP -- see the URDF), so less tilt directly means less
  lateral fingertip offset for the same TCP convergence -- and the whole
  approach is re-flown, retargeted on wherever the cube is now, up to
  --grasp-realign-attempts times. Only a centred grasp proceeds to the close;
  an uncentred one is recorded as grasp_misaligned rather than attempted.
  Then: CLOSE (hold qpos, ramp gripper to
  --gripper-close-value, --close-steps settle steps) then LIFT (a fresh
  mplib-planned straight vertical move of --lift-height, gripper still
  commanded closed). Pick succeeds iff the cube's own SAPIEN pose rose by at
  least --grasp-success-lift-fraction of --lift-height -- i.e. it actually
  came along with the gripper, not just proximity.
* Only if the pick succeeded: PLACE phase, the SAME reranking mechanism
  again, this time toward the place pose, gripper still held closed
  (--place-max-steps budget), its EMA seeded from the lift phase's last raw
  action so the first place-reach step doesn't snap from the scripted lift's
  open-loop replay straight to a raw policy action against this gripper's
  stiff PD controller -- observed launching the cube ~0.30m in a single
  rendered frame otherwise. Then RELEASE (hold qpos, gripper opens to
  --gripper-open-value, --release-steps settle steps) wherever the arm ended
  up -- even a failed place-reach still releases and gets scored, rather than
  carrying the cube back into the next episode's reset. Place succeeds iff
  the place-reach converged AND the cube's final position lands within
  --place-radius of the place target (XY) AND settles back near its natural
  resting height (not still airborne, not fallen through/off the table).

Usage:
    python scripts/eval_pose_variety_pick_and_place.py \\
        --checkpoint artifacts/checkpoints/xarm7_reach_run1/final_step_00100000.pt \\
        --dataset artifacts/pg3d_xarm7_gripper_reach_5k.zarr \\
        --num-cube-positions 4 --episodes-per-position 5 \\
        --video-dir artifacts/pick_and_place_eval \\
        --summary-json artifacts/pick_and_place_eval/summary.json

``--dataset`` is used ONLY to read metadata.json (env_kwargs/crop
config/action_mode) -- the exact observation shape the checkpoint was
trained on -- never to source episode content or the env_id itself (the
env_id is always this script's own PG3DPick-XArm7-Gripper-Workspace-v0, so
the cube is actually present).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from pg3d.utils.arrays import to_numpy as _to_numpy
from scripts.eval_pose_variety_checkpoint_reachability import (
    _build_ik_env_and_planner,
    _orientation_variety_stats_deg,
    _quat_angular_distance_rad,
    _set_site_pose,
    _tilt_from_down_deg,
)

_PICK_ENV_ID = "PG3DPick-XArm7-Gripper-Workspace-v0"


def _cube_world_position(env: Any) -> np.ndarray:
    """Read the cube actor's current ground-truth world position.

    Mirrors dataset_generation.write_maniskill_reach_dataset._robot_base_position's
    own pattern (to_numpy(...).reshape(-1, 3)[0]) rather than indexing
    ``.pose.p[0, 2]`` directly -- ``.p`` may be a CUDA torch tensor on the GPU
    sim backend or a plain numpy array on CPU, and to_numpy() is the one
    conversion already proven safe for both across this codebase.
    """
    cube = getattr(env.unwrapped, "cube", None)
    if cube is None:
        raise AttributeError("env.unwrapped has no 'cube' actor -- wrong env id?")
    return _to_numpy(cube.pose.p).reshape(-1, 3)[0].astype(np.float32)


def _approach_axis(quat_wxyz: np.ndarray) -> np.ndarray:
    """Unit vector the gripper "points along" for a SAPIEN [w,x,y,z] orientation.

    The gripper's own local +Z is its approach axis: the canonical
    straight-down tabletop orientation [0,1,0,0] (wxyz, a 180deg flip about
    x) maps local +Z to world -Z, i.e. pointing down at the table -- which is
    exactly what "facing downwards" means for this robot's rest pose. So
    rotating [0,0,1] by the target orientation gives the direction the
    fingers travel when the arm closes distance onto an object.
    """
    from scipy.spatial.transform import Rotation

    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    rot = Rotation.from_quat([q[1], q[2], q[3], q[0]])  # wxyz -> xyzw
    axis = rot.apply(np.array([0.0, 0.0, 1.0]))
    return (axis / max(float(np.linalg.norm(axis)), 1e-9)).astype(np.float32)


def _pregrasp_position(
    grasp_position: np.ndarray, quat_wxyz: np.ndarray, standoff: float
) -> np.ndarray:
    """Back the grasp pose off by `standoff` metres ALONG THE APPROACH AXIS.

    Not straight up in world Z: for a tilted approach those differ, and only
    the along-axis offset makes the final descent a pure translation down the
    fingers' own axis. That is the whole point -- the observed failure was the
    policy flying to the grasp pose laterally at cube height, so the wide-open
    claws swept sideways INTO the cube and knocked it several cm away before
    anything closed (pos 00 / ep 01: TCP within 0.008m of the grasp target,
    yet 0.044m off the cube's live centre -- only possible if the cube moved).
    Approaching a standoff point first, then descending along the approach
    axis, means the fingers only ever move parallel to themselves near the
    cube and cannot broadside it.
    """
    grasp_position = np.asarray(grasp_position, dtype=np.float32)
    if standoff <= 0.0:
        return grasp_position.copy()
    return (grasp_position - _approach_axis(quat_wxyz) * float(standoff)).astype(np.float32)


_DOWN_QUAT_WXYZ = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)


def _slerp_towards_down(quat_wxyz: np.ndarray, weight: float) -> np.ndarray:
    """Rotate a SAPIEN [w,x,y,z] orientation a fraction `weight` of the way to
    straight-down, via quaternion slerp (weight=0 -> unchanged, weight=1 ->
    exactly straight-down).

    This is the CALCULATED correction for an off-centre grasp, replacing a
    random orientation jitter: rotation error amplifies at the fingertips
    (they sit ~3.5cm+ from the TCP origin -- see the URDF's
    left_finger/right_finger joint offsets, and agents.py's XArm7Gripper
    docstring), so less tilt directly means less lateral fingertip
    displacement for the same TCP convergence. There is no need to guess a
    direction: shrinking toward the one orientation with zero lever-arm
    effect (straight down) is the correction indicated by the physics,
    scaled by how badly the previous attempt missed (see
    --grasp-realign-tilt-gain / --grasp-realign-max-tilt-shrink).
    """
    from scipy.spatial.transform import Rotation, Slerp

    weight = float(np.clip(weight, 0.0, 1.0))
    if weight <= 0.0:
        return np.asarray(quat_wxyz, dtype=np.float32)
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    current = Rotation.from_quat([q[1], q[2], q[3], q[0]])
    down = Rotation.from_quat(
        [_DOWN_QUAT_WXYZ[1], _DOWN_QUAT_WXYZ[2], _DOWN_QUAT_WXYZ[3], _DOWN_QUAT_WXYZ[0]]
    )
    slerp = Slerp([0.0, 1.0], Rotation.concatenate([current, down]))
    result = slerp([weight])[0]
    xyzw = result.as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float32)


def _sample_cube_position(
    rng: np.random.Generator,
    *,
    bounds_world: np.ndarray,
) -> np.ndarray:
    """Sample one cube XY position; Z is fixed to bounds_world's own Z row."""
    bounds = np.asarray(bounds_world, dtype=np.float32).reshape(3, 2)
    return rng.uniform(bounds[:, 0], bounds[:, 1]).astype(np.float32)


def _sample_place_position(
    rng: np.random.Generator,
    *,
    bounds_world: np.ndarray,
    reference: np.ndarray,
    min_distance: float,
    resample_attempts: int,
) -> np.ndarray:
    """Sample a place position, >= min_distance from `reference` (the cube).

    With the CLI default (--position-resample-attempts 1) this loop only ever
    runs once, so `min_distance` was previously decorative -- whatever the
    single draw landed on was returned regardless of distance (observed:
    pos 01's grasp/place were 4.3cm apart against a requested
    --min-grasp-place-distance of 15cm). Falling back to the FARTHEST
    candidate seen, rather than just the last one, means a too-low
    `resample_attempts` degrades gracefully toward "best effort" instead of
    silently ignoring the constraint outright.
    """
    bounds = np.asarray(bounds_world, dtype=np.float32).reshape(3, 2)
    best = None
    best_distance = -1.0
    for _ in range(max(resample_attempts, 1)):
        candidate = rng.uniform(bounds[:, 0], bounds[:, 1]).astype(np.float32)
        distance = float(np.linalg.norm(candidate - reference))
        if distance >= min_distance:
            return candidate
        if distance > best_distance:
            best, best_distance = candidate, distance
    return best


def _sync_pick_markers(sim_env: Any, *, marker_pos: np.ndarray, target_pos: np.ndarray) -> None:
    """Point start_site (red) near the cube, goal_site (green) at the current target.

    start_site is now purely informational -- the robot's actual start is
    always its fixed rest configuration (see main()'s use of rest_qpos), not
    a separate sampled Cartesian pose, so there is no longer a meaningful
    "start position" to mark. Repurposed to sit at/near the cube instead, so
    the video makes clear what's being picked up. goal_site is NOT cosmetic:
    same as eval_pose_variety_checkpoint_reachability._sync_pose_variety_markers,
    its pose is what PG3DReachEnv._get_obs_extra exposes as the policy's
    actual goal-conditioning signal -- grasp_position during the pick phase,
    place_position (== the place_site cylinder's own position) during place.
    """
    _set_site_pose(sim_env, "start_site", marker_pos)
    _set_site_pose(sim_env, "goal_site", target_pos)
    update_render = getattr(sim_env.unwrapped.scene, "update_render", None)
    if callable(update_render):
        update_render()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from dataset_generation.write_maniskill_reach_dataset import (
        _farthest_point_select_orientations,
        _format_sim_action,
        _hold_sim_action,
        _move_to_pose_with_screw,
        _plan_multisegment_trajectory,
        _pose_with_orientation,
        _refresh_obs_after_manual_qpos,
        _resolve_reachable_orientation,
        _sample_broad_orientation_sapien,
        _tcp_pose,
    )
    from pg3d.constraints import CartesianPoseConstraint
    from pg3d.envs.maniskill_adapter import ManiSkillGhostPandaGeometryProvider
    from pg3d.envs.maniskill_adapter.dataset import load_reach_metadata
    from pg3d.envs.xarm_adapter.pick_env import register_pg3d_xarm7_pick_envs, set_cube_pose
    from pg3d.envs.xarm_adapter.reach_config import XARM7_REACH_WORKSPACE_BOUNDS
    from pg3d.eval import TimingRecorder, scene_context_for_constraints
    from pg3d.policies.dp3.checkpoint import load_reach_policy_from_checkpoint
    from pg3d.utils.arrays import bool_any, frame_to_numpy
    from pg3d.utils.devices import select_device
    from pg3d.world_model import GeometricWorldModel
    from scripts.eval_constrained_reach import DP3ChunkPolicyAdapter, _env_kwargs, _select_decision
    from scripts.rollout_dp3_reach_policy import (
        _action_mode,
        append_obs_window,
        crop_config_from_metadata,
        make_initial_obs_window,
        policy_action_to_sim_action,
        rollout_observation_entry,
        save_video,
    )

    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
        import sapien
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to import ManiSkill/SAPIEN: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    metadata = load_reach_metadata(args.dataset)
    device = select_device(args.device)
    policy = load_reach_policy_from_checkpoint(args.checkpoint, device=device)
    action_mode = _action_mode(str(metadata.get("action_mode", "abs_joint")))
    crop_config = crop_config_from_metadata(metadata)

    try:
        # Plain reach env for IK/inter-feasibility pre-verification -- the
        # cube's own collision geometry isn't needed for this: the grasp
        # target pose being tested IS the cube's location, so verifying the
        # arm can reach that Cartesian point/orientation is the same check
        # whether or not a cube collision mesh sits there too.
        ik_env, ik_solver, planner, rest_qpos = _build_ik_env_and_planner("gripper")
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to build IK env/planner: {type(exc).__name__}: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1

    register_pg3d_xarm7_pick_envs()
    from pg3d.envs.xarm_adapter.motionplanner import XArm7GripperMotionPlanningSolver

    def _run_reach_phase(
        *,
        sim_entry: dict[str, Any],
        obs_window: Any,
        frames: list[np.ndarray],
        target_position: np.ndarray,
        target_quat: np.ndarray,
        constraint_name: str,
        max_steps: int,
        gripper_value: float,
        rng: np.random.Generator,
        initial_ema_action: np.ndarray | None = None,
    ) -> tuple[bool, dict[str, Any], Any, int, float, float, bool, np.ndarray | None]:
        """Run one reach-via-reranking phase toward (target_position, target_quat).

        Shared by both the pick-approach reach and the place-approach reach --
        identical reranking mechanism, only the target pose, step budget, and
        held gripper command differ. Returns (reached, sim_entry, obs_window,
        total_steps, pos_err, rot_err, truncated_early, last_ema_action);
        `frames` is mutated in place so the caller keeps one continuous video
        across phases.

        `initial_ema_action`, if given, seeds the EMA instead of starting from
        the raw first policy action. Needed at the pick->place handoff: the
        close+lift phase is a scripted, non-EMA replay of a planned
        trajectory, so without a seed the place phase's very first blended
        action snaps from wherever that replay left off straight to the
        policy's raw first output against this gripper's stiff PD controller
        (stiffness=1e5) -- observed throwing the cube ~0.30m across one
        rendered frame (pos 01 / ep 01: cube held firmly through frame 92,
        gone by 94, landed 0.30m away by 102 -- a launch, not a drop).
        """
        constraint = CartesianPoseConstraint(
            target_position=target_position,
            target_orientation=target_quat,
            position_tolerance=args.position_tolerance,
            rotation_tolerance=args.rotation_tolerance,
            weight=1.0,
            name=constraint_name,
        )
        scene = scene_context_for_constraints(
            target_position=target_position,
            constraints=[constraint],
            metadata={"phase": constraint_name},
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
                    if ema_sim_action is None or args.action_ema_alpha >= 1.0:
                        ema_sim_action = sim_action
                    else:
                        ema_sim_action = (
                            args.action_ema_alpha * sim_action
                            + (1.0 - args.action_ema_alpha) * ema_sim_action
                        )
                    sim_obs, _reward, _terminated, truncated, sim_info = sim_env.step(ema_sim_action)
                    total_steps += 1
                    sim_entry = rollout_observation_entry(
                        sim_obs, sim_info, env=sim_env, crop_config=crop_config
                    )
                    obs_window = append_obs_window(
                        obs_window, sim_entry, n_obs_steps=int(policy.n_obs_steps)
                    )
                    frames.append(frame_to_numpy(sim_env.render()))

                    tcp = _tcp_pose(sim_env.unwrapped)
                    pos_err = float(np.linalg.norm(tcp[:3].astype(np.float64) - target_position))
                    rot_err = _quat_angular_distance_rad(
                        tcp[3:7].astype(np.float64), target_quat.astype(np.float64)
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

    def _descend_along_approach(
        *,
        target_position: np.ndarray,
        quat: np.ndarray,
        frames: list[np.ndarray],
        gripper_value: float,
    ) -> bool:
        """Scripted straight-line descent from the pre-grasp standoff to the grasp pose.

        Planned with the same screw-motion planner the lift already uses, so
        the TCP travels a straight SE(3) line -- which, since the standoff was
        offset purely along the approach axis, is a pure translation down that
        axis. The fingers therefore only ever move parallel to themselves in
        the cube's vicinity and cannot sweep into it sideways, which is what
        the learned reach alone was doing. Returns False if no plan was found
        (the caller records that as its own outcome rather than pretending the
        gripper is where it should be).
        """
        if args.pregrasp_height <= 0.0:
            return True
        descent_planner = XArm7GripperMotionPlanningSolver(
            sim_env,
            debug=False,
            vis=False,
            base_pose=sim_env.unwrapped.agent.robot.pose,
            visualize_target_grasp_pose=False,
            print_env_info=False,
        )
        try:
            descent_plan = _move_to_pose_with_screw(
                descent_planner,
                _pose_with_orientation(sapien, position=target_position, quat=quat),
                suppress_output=True,
            )
        finally:
            descent_planner.close()
        if descent_plan == -1 or "position" not in descent_plan:
            return False
        descent_positions = np.asarray(descent_plan["position"], dtype=np.float32)[
            : args.descent_max_steps
        ]
        for planned_qpos in descent_positions:
            descent_action = _format_sim_action(
                sim_env, planned_qpos, gripper_action=gripper_value
            )
            _obs, _reward, _terminated, truncated, _info = sim_env.step(descent_action)
            frames.append(frame_to_numpy(sim_env.render()))
            if bool_any(truncated):
                break
        return True

    _meta_max_steps = metadata.get("env_kwargs", {}).get("max_episode_steps")
    # The pick phase can run up to (1 + --grasp-realign-attempts) times -- one
    # initial approach plus a re-approach per orientation nudge -- each of
    # which is a full reach budget plus a scripted descent, so the env's step
    # limit has to cover the worst case or a late realign attempt would be
    # truncated by the env rather than judged on its merits.
    _pick_attempts = 1 + int(args.grasp_realign_attempts)
    _env_max_steps = max(
        _pick_attempts * (int(args.max_steps) + int(args.descent_max_steps))
        + int(args.close_steps)
        + int(args.lift_max_steps)
        + int(args.place_max_steps)
        + int(args.release_steps),
        int(_meta_max_steps or 0),
    )
    print(
        f"metadata env_kwargs max_episode_steps={_meta_max_steps!r}, "
        f"pick+place budget={_env_max_steps} -> env limit set to {_env_max_steps}"
    )
    sim_env_kwargs = _env_kwargs(metadata, render_mode="rgb_array", max_episode_steps=_env_max_steps)
    sim_env_kwargs["place_radius"] = args.place_radius
    sim_env = gym.make(_PICK_ENV_ID, **sim_env_kwargs)
    ghost_env = gym.make(str(metadata["env_id"]), **_env_kwargs(metadata, render_mode=None))

    adapter = DP3ChunkPolicyAdapter(
        policy, action_mode=action_mode, device=device, policy_batch_size=args.policy_batch_size
    )
    provider = ManiSkillGhostPandaGeometryProvider(
        ghost_env, task_name=str(metadata.get("env_id", "unknown")), crop_bounds=crop_config.bounds
    )
    world_model = GeometricWorldModel(provider)

    rng = np.random.default_rng(args.seed)
    base_bounds = np.asarray(XARM7_REACH_WORKSPACE_BOUNDS, dtype=np.float32).copy()
    base_z_offset = float(base_bounds[2, 0]) - 0.05  # recover base_position[2]
    table_z = base_z_offset
    cube_rest_z = table_z + args.cube_half_size
    grasp_target_z = cube_rest_z + args.grasp_height_offset

    # Cube XY sampled from the normal reach workspace footprint, but Z is
    # pinned to the table surface -- the cube always rests flat, never floats.
    cube_bounds = base_bounds.copy()
    cube_bounds[2, 0] = cube_rest_z
    cube_bounds[2, 1] = cube_rest_z

    # PLACE target is also on the table -- same Z convention as the cube's
    # own resting height (the place goal IS "put the cube back down here").
    place_target_z = cube_rest_z + args.place_height_offset
    place_bounds = base_bounds.copy()
    place_bounds[2, 0] = place_target_z
    place_bounds[2, 1] = place_target_z

    print(
        f"cube rests at world Z={cube_rest_z:.3f} (table Z={table_z:.3f} + "
        f"cube_half_size={args.cube_half_size}); grasp target Z={grasp_target_z:.3f} "
        f"(+ --grasp-height-offset={args.grasp_height_offset}); "
        f"place target Z={place_target_z:.3f} (+ --place-height-offset="
        f"{args.place_height_offset}), place radius={args.place_radius}m; "
        f"start = robot's fixed rest configuration (rest_qpos, facing down) -- "
        f"NOT a separately sampled Cartesian pose (see main()'s design note)"
    )

    if args.video_dir is not None:
        args.video_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    counts = {
        "total_episodes": 0,
        "ik_failed": 0,
        "motion_plan_infeasible": 0,
        "reach_reached": 0,
        "descent_failed": 0,
        "grasp_misaligned": 0,
        "pick_succeeded": 0,
        "place_succeeded": 0,
        "pick_and_place_succeeded": 0,
    }

    try:
        for pos_idx in range(args.num_cube_positions):
            cube_center = _sample_cube_position(rng, bounds_world=cube_bounds)
            grasp_position = cube_center.copy()
            grasp_position[2] = grasp_target_z
            place_position = _sample_place_position(
                rng,
                bounds_world=place_bounds,
                reference=grasp_position,
                min_distance=args.min_grasp_place_distance,
                resample_attempts=args.position_resample_attempts,
            )
            print(
                f"\n=== cube position {pos_idx:02d} — center={cube_center.tolist()} "
                f"place={place_position.tolist()} "
                f"(robot starts every episode from its fixed rest pose) ==="
            )

            grasp_pool = _farthest_point_select_orientations(
                [
                    _sample_broad_orientation_sapien(rng, cone_deg=args.orientation_cone_deg)
                    for _ in range(max(args.episodes_per_position * 4, 12))
                ],
                k=args.episodes_per_position,
                rng=rng,
            )
            # Place-approach orientation: sampled the same way as grasp -- an
            # independent random orientation per episode (this is the "some
            # random orientation" for the place pose you asked for).
            place_pool = _farthest_point_select_orientations(
                [
                    _sample_broad_orientation_sapien(rng, cone_deg=args.place_orientation_cone_deg)
                    for _ in range(max(args.episodes_per_position * 4, 12))
                ],
                k=args.episodes_per_position,
                rng=rng,
            )
            print("  grasp-approach orientation variety (pairwise spread, degrees):")
            stats = _orientation_variety_stats_deg(grasp_pool)
            tilts = ", ".join(f"{_tilt_from_down_deg(q):.0f}" for q in grasp_pool)
            print(
                f"    min={stats['min_deg']:.1f} mean={stats['mean_deg']:.1f} "
                f"max={stats['max_deg']:.1f}  (tilt-from-down per episode: [{tilts}] deg)"
            )
            print("  place-approach orientation variety (pairwise spread, degrees):")
            stats = _orientation_variety_stats_deg(place_pool)
            tilts = ", ".join(f"{_tilt_from_down_deg(q):.0f}" for q in place_pool)
            print(
                f"    min={stats['min_deg']:.1f} mean={stats['mean_deg']:.1f} "
                f"max={stats['max_deg']:.1f}  (tilt-from-down per episode: [{tilts}] deg)"
            )

            pos_counts = {
                "total": 0,
                "ik_failed": 0,
                "motion_plan_infeasible": 0,
                "reach_reached": 0,
                "descent_failed": 0,
                "grasp_misaligned": 0,
                "pick_succeeded": 0,
                "place_succeeded": 0,
                "pick_and_place_succeeded": 0,
            }
            for episode_idx in range(args.episodes_per_position):
                resolved: dict[str, tuple[np.ndarray, np.ndarray]] = {}
                failure_label: str | None = None
                for label, position, primary_quat, cone_deg in (
                    ("grasp", grasp_position, grasp_pool[episode_idx], args.orientation_cone_deg),
                    ("place", place_position, place_pool[episode_idx], args.place_orientation_cone_deg),
                ):
                    result = _resolve_reachable_orientation(
                        planner=planner,
                        env=ik_env,
                        sapien=sapien,
                        position=position,
                        primary_quat=primary_quat,
                        seed_qpos=rest_qpos,
                        rng=rng,
                        cone_deg=cone_deg,
                        extra_attempts=args.extra_orientation_attempts,
                        suppress_planner_output=True,
                    )
                    if result is None:
                        failure_label = label
                        break
                    resolved[label] = result

                counts["total_episodes"] += 1
                pos_counts["total"] += 1
                row: dict[str, Any] = {
                    "cube_position_idx": pos_idx,
                    "episode": episode_idx,
                    "cube_center": cube_center.tolist(),
                    "grasp_position": grasp_position.tolist(),
                    "place_position": place_position.tolist(),
                }
                if failure_label is not None:
                    counts["ik_failed"] += 1
                    pos_counts["ik_failed"] += 1
                    row["outcome"] = f"{failure_label}_ik_failed"
                    print(f"  [ep {episode_idx}] SKIP — {failure_label} pose not IK-reachable")
                    all_rows.append(row)
                    continue

                grasp_quat = resolved["grasp"][1]
                place_quat = resolved["place"][1]
                row["grasp_quat"] = grasp_quat.tolist()
                row["place_quat"] = place_quat.tolist()
                pregrasp_position = _pregrasp_position(
                    grasp_position, grasp_quat, args.pregrasp_height
                )
                row["pregrasp_position"] = pregrasp_position.tolist()

                # "From its natural rest position facing downwards, the robot
                # must be able to go towards the cube and pick it up" -- verify
                # a collision-free plan exists from rest_qpos (the SAME fixed
                # configuration every episode resets to) to the grasp pose,
                # rather than from a separately sampled/resolved start pose.
                # Both waypoints of the actual approach are verified in one
                # multisegment plan: rest -> pregrasp standoff -> grasp. Just
                # checking rest -> grasp would validate a path this script no
                # longer executes.
                approach_poses = [
                    _pose_with_orientation(sapien, position=pregrasp_position, quat=grasp_quat)
                ]
                if args.pregrasp_height > 0.0:
                    approach_poses.append(
                        _pose_with_orientation(sapien, position=grasp_position, quat=grasp_quat)
                    )
                inter_feasible_plan = _plan_multisegment_trajectory(
                    planner=planner,
                    env=ik_env,
                    poses=approach_poses,
                    start_qpos=rest_qpos,
                    suppress_planner_output=True,
                    smooth_trajectory=True,
                )
                if inter_feasible_plan is None:
                    counts["motion_plan_infeasible"] += 1
                    pos_counts["motion_plan_infeasible"] += 1
                    row["outcome"] = "motion_plan_infeasible"
                    print(
                        f"  [ep {episode_idx}] SKIP — grasp pose is IK-reachable, but no "
                        "collision-free plan connects it from the robot's rest pose "
                        "via the pre-grasp standoff"
                    )
                    all_rows.append(row)
                    continue

                # Second inter-feasibility check: from (approximately) the
                # resolved grasp configuration -- the last qpos of the plan
                # above -- to the place pose. Approximates "after lifting the
                # cube straight up, can the arm still get to the place pose"
                # without re-planning through the exact lift waypoint; a
                # short vertical lift from an already-feasible grasp qpos is
                # very unlikely to be the thing that breaks feasibility to a
                # nearby place pose, so this is a fair proxy, not exact.
                # _plan_multisegment_trajectory returns (positions, status) --
                # a tuple, NOT the raw dict _move_to_pose_with_screw gives.
                inter_feasible_positions, _inter_feasible_status = inter_feasible_plan
                grasp_end_qpos = np.asarray(inter_feasible_positions[-1], dtype=np.float32)
                place_feasible_plan = _plan_multisegment_trajectory(
                    planner=planner,
                    env=ik_env,
                    poses=[_pose_with_orientation(sapien, position=place_position, quat=place_quat)],
                    start_qpos=grasp_end_qpos,
                    suppress_planner_output=True,
                    smooth_trajectory=True,
                )
                if place_feasible_plan is None:
                    counts["motion_plan_infeasible"] += 1
                    pos_counts["motion_plan_infeasible"] += 1
                    row["outcome"] = "place_motion_plan_infeasible"
                    print(
                        f"  [ep {episode_idx}] SKIP — grasp and place pose each "
                        "individually IK-reachable, but no collision-free plan connects "
                        "them (not inter-feasible)"
                    )
                    all_rows.append(row)
                    continue
                print(
                    f"  [ep {episode_idx}] reachable and inter-feasible from rest — proceeding. "
                    f"tilt-from-down: grasp={_tilt_from_down_deg(grasp_quat):.0f}deg "
                    f"place={_tilt_from_down_deg(place_quat):.0f}deg"
                )

                # No explicit qpos override here -- PG3DReachXArm7Env._initialize_episode
                # already resets the robot to its own agent.keyframes["rest"].qpos on
                # every reset (see reach_env.py), which is the exact same rest_qpos
                # this episode was just IK-verified from. "The robot's natural
                # position facing downwards" IS the env's own default reset state.
                _reset_obs, reset_info = sim_env.reset(
                    seed=args.seed + pos_idx * 1000 + episode_idx, options={"reconfigure": True}
                )
                set_cube_pose(sim_env, cube_center)
                _set_site_pose(sim_env, "place_site", place_position)
                _sync_pick_markers(sim_env, marker_pos=cube_center, target_pos=pregrasp_position)
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
                        f"  [ep {episode_idx}] WARNING — policy goal-conditioning is "
                        f"{conditioned_on.round(3).tolist()} but the pre-grasp target is "
                        f"{pregrasp_position.astype(np.float64).round(3).tolist()}; "
                        "goal_site did not take the new pose"
                    )

                frames = [frame_to_numpy(sim_env.render())]

                # --- PICK APPROACH, retried with a CALCULATED orientation
                # correction if the fingers don't end up centred on the cube.
                #
                # Each attempt is: learned reach to the pre-grasp STANDOFF (so
                # the policy never flies in at cube height with the claws wide
                # open -- that was broadsiding the cube and knocking it several
                # cm away), then a scripted straight-line descent down the
                # gripper's own approach axis onto the grasp pose, then a
                # measurement of how well centred the fingers actually are on
                # the cube's LIVE position. If that measurement fails, the
                # approach orientation is shrunk toward straight-down (see
                # _slerp_towards_down) by an amount proportional to the
                # measured miss -- not a random guess -- and the whole
                # approach is re-flown, retargeted on wherever the cube is
                # NOW, since a previous attempt may have moved it.
                attempt_quat = grasp_quat
                attempt_grasp_position = grasp_position.copy()
                attempt_pregrasp = pregrasp_position.copy()
                grasp_centering_err = float("inf")
                cube_disturbance = 0.0
                reached_goal = False
                descent_ok = True
                realign_used = 0
                total_steps = 0
                pos_err = float("inf")
                rot_err = float("inf")
                truncated_early = False
                for attempt in range(args.grasp_realign_attempts + 1):
                    if attempt > 0:
                        realign_used = attempt
                        live_cube = _cube_world_position(sim_env)
                        attempt_grasp_position = np.array(
                            [live_cube[0], live_cube[1], grasp_target_z], dtype=np.float32
                        )
                        # Calculated, not random: shrink the PREVIOUS attempt's
                        # tilt toward straight-down (zero fingertip lever arm)
                        # by an amount proportional to how far off-centre that
                        # attempt actually measured, capped so an
                        # IK-verified-adjacent orientation is never abandoned
                        # in one step.
                        shrink = min(
                            args.grasp_realign_max_tilt_shrink,
                            args.grasp_realign_tilt_gain
                            * grasp_centering_err
                            / max(args.grasp_centering_tolerance, 1e-6),
                        )
                        prev_tilt = _tilt_from_down_deg(attempt_quat)
                        attempt_quat = _slerp_towards_down(attempt_quat, shrink)
                        attempt_pregrasp = _pregrasp_position(
                            attempt_grasp_position, attempt_quat, args.pregrasp_height
                        )
                        print(
                            f"  [ep {episode_idx}] realign attempt {attempt}/"
                            f"{args.grasp_realign_attempts} — previous miss was "
                            f"{grasp_centering_err:.4f}m, shrinking tilt "
                            f"{shrink * 100:.0f}% toward straight-down "
                            f"({prev_tilt:.0f}deg -> {_tilt_from_down_deg(attempt_quat):.0f}deg), "
                            f"retargeted on the cube's current position"
                        )
                        _sync_pick_markers(
                            sim_env, marker_pos=live_cube, target_pos=attempt_pregrasp
                        )
                        sim_obs, sim_info = _refresh_obs_after_manual_qpos(
                            sim_env, info=sim_info, gripper_open=args.gripper_open_value
                        )
                        sim_entry = rollout_observation_entry(
                            sim_obs, sim_info, env=sim_env, crop_config=crop_config
                        )
                        obs_window = append_obs_window(
                            obs_window, sim_entry, n_obs_steps=int(policy.n_obs_steps)
                        )

                    (
                        reached_goal,
                        sim_entry,
                        obs_window,
                        total_steps,
                        pos_err,
                        rot_err,
                        truncated_early,
                        _pick_ema_action,
                    ) = _run_reach_phase(
                        sim_entry=sim_entry,
                        obs_window=obs_window,
                        frames=frames,
                        target_position=attempt_pregrasp,
                        target_quat=attempt_quat,
                        constraint_name="pick_and_place_grasp",
                        max_steps=args.max_steps,
                        gripper_value=args.gripper_open_value,
                        rng=rng,
                    )
                    if not reached_goal:
                        reason = (
                            f"episode truncated at step {total_steps}"
                            if truncated_early
                            else f"step budget ({args.max_steps}) exhausted"
                        )
                        print(
                            f"  [ep {episode_idx}] pre-grasp reach did NOT converge — {reason}. "
                            f"final pos_err={pos_err:.4f} rot_err={np.degrees(rot_err):.1f}deg -- "
                            "skipping grasp attempt"
                        )
                        break

                    print(
                        f"  [ep {episode_idx}] pre-grasp reach converged at step {total_steps} "
                        f"(pos_err={pos_err:.4f} rot_err={np.degrees(rot_err):.1f}deg) — "
                        f"descending {args.pregrasp_height:.3f}m along the approach axis"
                    )
                    descent_ok = _descend_along_approach(
                        target_position=attempt_grasp_position,
                        quat=attempt_quat,
                        frames=frames,
                        gripper_value=args.gripper_open_value,
                    )
                    if not descent_ok:
                        print(
                            f"  [ep {episode_idx}] descent motion plan FAILED from the "
                            "pre-grasp standoff onto the grasp pose"
                        )
                        break

                    tcp_at_grasp = _tcp_pose(sim_env.unwrapped)[:2]
                    cube_now = _cube_world_position(sim_env)
                    grasp_centering_err = float(np.linalg.norm(tcp_at_grasp - cube_now[:2]))
                    # How far the approach itself shoved the cube. Non-zero here
                    # means the open claws made contact on the way in -- the
                    # thing the pre-grasp standoff + axial descent exists to
                    # prevent -- so it's reported explicitly rather than left to
                    # be inferred from the centring error.
                    cube_disturbance = float(np.linalg.norm(cube_now[:2] - cube_center[:2]))
                    print(
                        f"  [ep {episode_idx}] at grasp depth: TCP is "
                        f"{grasp_centering_err:.4f}m from the cube's XY centre "
                        f"(allowed <={args.grasp_centering_tolerance}m); the approach "
                        f"moved the cube {cube_disturbance:.4f}m from where it was placed"
                    )
                    if grasp_centering_err <= args.grasp_centering_tolerance:
                        break

                row["reach_total_steps"] = total_steps
                row["reach_final_pos_err"] = float(pos_err)
                row["reach_final_rot_err_deg"] = float(np.degrees(rot_err))
                row["realign_attempts_used"] = realign_used
                row["grasp_centering_err"] = grasp_centering_err
                row["cube_disturbance"] = cube_disturbance
                row["grasp_quat_used"] = attempt_quat.tolist()

                if not reached_goal:
                    row["outcome"] = "reach_failed"
                    all_rows.append(row)
                    if args.video_dir is not None:
                        video_path = (
                            args.video_dir
                            / f"pos_{pos_idx:02d}_episode_{episode_idx:02d}_reach_failed.mp4"
                        )
                        save_video(video_path, frames, fps=args.video_fps)
                        row["video"] = str(video_path)
                        print(f"  [ep {episode_idx}] outcome={row['outcome']}  video: {video_path}")
                    continue

                counts["reach_reached"] += 1
                pos_counts["reach_reached"] += 1

                if not descent_ok:
                    row["outcome"] = "descent_failed"
                    counts["descent_failed"] += 1
                    pos_counts["descent_failed"] += 1
                    all_rows.append(row)
                    if args.video_dir is not None:
                        video_path = (
                            args.video_dir
                            / f"pos_{pos_idx:02d}_episode_{episode_idx:02d}_descent_failed.mp4"
                        )
                        save_video(video_path, frames, fps=args.video_fps)
                        row["video"] = str(video_path)
                        print(f"  [ep {episode_idx}] outcome={row['outcome']}  video: {video_path}")
                    continue

                # --- Pre-close alignment gate: the fingers are at grasp depth
                # but still not centred on the cube, and every realign attempt
                # has been spent. Closing here would clip the cube off-centre
                # and knock it away rather than straddle it, so the close is
                # skipped and the failure recorded as what it actually is.
                if grasp_centering_err > args.grasp_centering_tolerance:
                    row["outcome"] = "grasp_misaligned"
                    counts["grasp_misaligned"] += 1
                    pos_counts["grasp_misaligned"] += 1
                    print(
                        f"  [ep {episode_idx}] SKIP close -- TCP still "
                        f"{grasp_centering_err:.4f}m off the cube's actual XY "
                        f"center (allowed <={args.grasp_centering_tolerance}m) after "
                        f"{realign_used} realign attempt(s); closing here would "
                        "clip the cube off-center rather than grasp it"
                    )
                    if args.video_dir is not None:
                        video_path = (
                            args.video_dir
                            / f"pos_{pos_idx:02d}_episode_{episode_idx:02d}_misaligned.mp4"
                        )
                        save_video(video_path, frames, fps=args.video_fps)
                        row["video"] = str(video_path)
                        print(f"  [ep {episode_idx}] outcome={row['outcome']}  video: {video_path}")
                    all_rows.append(row)
                    continue

                # --- Hardcoded grasp: this checkpoint has no gripper output at all
                # (see module docstring) -- close then lift are scripted, not policy.
                cube_z_before = float(_cube_world_position(sim_env)[2])
                current_qpos = np.asarray(
                    sim_env.unwrapped.agent.robot.get_qpos()
                ).reshape(-1)[:7]
                # Ramp the gripper command gradually from open to fully closed
                # over --close-steps, rather than snapping straight to
                # --gripper-close-value on step 1. Commanding the full closed
                # target immediately -- against this gripper's very stiff PD
                # controller (stiffness=1e5, see XArm7Gripper in agents.py) --
                # was slamming shut and popping the cube out from the sudden
                # contact impulse ("squeezing it out") instead of settling
                # into a stable grip around it.
                for close_step in range(args.close_steps):
                    close_frac = (close_step + 1) / float(args.close_steps)
                    gripper_value = args.gripper_open_value + close_frac * (
                        args.gripper_close_value - args.gripper_open_value
                    )
                    close_action = _hold_sim_action(
                        sim_env, gripper_open=gripper_value, qpos=current_qpos
                    )
                    sim_obs, _reward, _terminated, truncated, sim_info = sim_env.step(close_action)
                    frames.append(frame_to_numpy(sim_env.render()))
                    if bool_any(truncated):
                        break

                sim_planner = XArm7GripperMotionPlanningSolver(
                    sim_env,
                    debug=False,
                    vis=False,
                    base_pose=sim_env.unwrapped.agent.robot.pose,
                    visualize_target_grasp_pose=False,
                    print_env_info=False,
                )
                # Lift from where the grasp ACTUALLY happened (attempt_*), not
                # the originally sampled pose -- a realign attempt may have
                # retargeted both the position and the orientation.
                lift_position = attempt_grasp_position.copy()
                lift_position[2] = grasp_target_z + args.lift_height
                lift_plan_failed = False
                try:
                    lift_plan = _move_to_pose_with_screw(
                        sim_planner,
                        _pose_with_orientation(sapien, position=lift_position, quat=attempt_quat),
                        suppress_output=True,
                    )
                finally:
                    sim_planner.close()
                if lift_plan == -1 or "position" not in lift_plan:
                    lift_plan_failed = True
                    print(f"  [ep {episode_idx}] lift motion plan FAILED — cube likely still grasped in place")
                    final_lift_action: np.ndarray | None = None
                else:
                    lift_positions = np.asarray(lift_plan["position"], dtype=np.float32)[
                        : args.lift_max_steps
                    ]
                    final_lift_action = None
                    for planned_qpos in lift_positions:
                        lift_action = _format_sim_action(
                            sim_env, planned_qpos, gripper_action=args.gripper_close_value
                        )
                        sim_obs, _reward, _terminated, truncated, sim_info = sim_env.step(
                            lift_action
                        )
                        final_lift_action = np.asarray(lift_action, dtype=np.float32)
                        frames.append(frame_to_numpy(sim_env.render()))
                        if bool_any(truncated):
                            break

                cube_z_after_lift = float(_cube_world_position(sim_env)[2])
                cube_lift = cube_z_after_lift - cube_z_before
                pick_succeeded = (
                    not lift_plan_failed
                    and cube_lift >= args.grasp_success_lift_fraction * args.lift_height
                )
                row["cube_z_before_close"] = cube_z_before
                row["cube_z_after_lift"] = cube_z_after_lift
                row["cube_lift_m"] = cube_lift
                row["lift_plan_failed"] = lift_plan_failed
                row["pick_succeeded"] = pick_succeeded
                print(
                    f"  [ep {episode_idx}] pick_{'succeeded' if pick_succeeded else 'failed'} "
                    f"— cube rose {cube_lift:.4f}m (needed "
                    f">={args.grasp_success_lift_fraction * args.lift_height:.4f}m)"
                )
                if pick_succeeded:
                    counts["pick_succeeded"] += 1
                    pos_counts["pick_succeeded"] += 1

                if not pick_succeeded:
                    # Nothing to place if the pick itself didn't hold -- the
                    # gripper is closed on empty air, or the cube fell during
                    # the lift.
                    row["outcome"] = "pick_failed"
                    if args.video_dir is not None:
                        video_path = (
                            args.video_dir / f"pos_{pos_idx:02d}_episode_{episode_idx:02d}.mp4"
                        )
                        save_video(video_path, frames, fps=args.video_fps)
                        row["video"] = str(video_path)
                        print(f"  [ep {episode_idx}] outcome={row['outcome']}  video: {video_path}")
                    all_rows.append(row)
                    continue

                # --- Place phase: reach (via the SAME reranking mechanism) from
                # the lifted pose to the place pose, cube still held (gripper
                # commanded closed the whole way). Re-point the goal-conditioning
                # marker at the place pose first -- exactly the same "goal_site
                # is the actual conditioning signal" mechanism used for the pick
                # phase -- then re-read the observation before starting the
                # reach loop, since sim_entry/obs_window are still stale from
                # the last close/lift step (which didn't touch goal_site).
                _sync_pick_markers(sim_env, marker_pos=cube_center, target_pos=place_position)
                sim_obs, sim_info = _refresh_obs_after_manual_qpos(
                    sim_env, info=sim_info, gripper_open=args.gripper_close_value
                )
                sim_entry = rollout_observation_entry(
                    sim_obs, sim_info, env=sim_env, crop_config=crop_config
                )
                obs_window = append_obs_window(
                    obs_window, sim_entry, n_obs_steps=int(policy.n_obs_steps)
                )
                (
                    place_reached_goal,
                    sim_entry,
                    obs_window,
                    place_total_steps,
                    place_pos_err,
                    place_rot_err,
                    place_truncated_early,
                    _place_ema_action,
                ) = _run_reach_phase(
                    sim_entry=sim_entry,
                    obs_window=obs_window,
                    frames=frames,
                    target_position=place_position,
                    target_quat=place_quat,
                    constraint_name="pick_and_place_place",
                    max_steps=args.place_max_steps,
                    gripper_value=args.gripper_close_value,
                    rng=rng,
                    # Seed the EMA from the lift phase's last raw action rather
                    # than starting fresh: the close+lift replay is an
                    # open-loop, non-EMA scripted trajectory, so an unseeded
                    # EMA here would blend a raw first policy action against
                    # whatever qpos/gripper command the lift left off at --
                    # against this gripper's stiff PD controller that showed
                    # up as the arm snapping straight down in a single frame
                    # and launching the cube ~0.30m (pos 01 / ep 01: held
                    # through frame 92, gone by 94).
                    initial_ema_action=final_lift_action,
                )
                row["place_reach_total_steps"] = place_total_steps
                row["place_reach_final_pos_err"] = float(place_pos_err)
                row["place_reach_final_rot_err_deg"] = float(np.degrees(place_rot_err))

                if not place_reached_goal:
                    reason = (
                        f"episode truncated at step {place_total_steps}"
                        if place_truncated_early
                        else f"step budget ({args.place_max_steps}) exhausted"
                    )
                    print(
                        f"  [ep {episode_idx}] place-reach did NOT converge — {reason}. "
                        f"final pos_err={place_pos_err:.4f} "
                        f"rot_err={np.degrees(place_rot_err):.1f}deg -- releasing in place"
                    )
                    row["outcome"] = "place_reach_failed"
                else:
                    print(
                        f"  [ep {episode_idx}] place-reach converged at step {place_total_steps} "
                        f"(pos_err={place_pos_err:.4f} rot_err={np.degrees(place_rot_err):.1f}deg)"
                    )

                # --- Hardcoded release: open the gripper wherever the arm ended
                # up (converged at the place pose, or not -- releasing is still
                # the right thing to do rather than carrying the cube back to
                # reset holding it, since a failed place-reach still ends
                # SOMEWHERE and dropping there is at least observable/scored).
                release_qpos = np.asarray(
                    sim_env.unwrapped.agent.robot.get_qpos()
                ).reshape(-1)[:7]
                for _ in range(args.release_steps):
                    release_action = _hold_sim_action(
                        sim_env, gripper_open=args.gripper_open_value, qpos=release_qpos
                    )
                    sim_obs, _reward, _terminated, truncated, sim_info = sim_env.step(release_action)
                    frames.append(frame_to_numpy(sim_env.render()))
                    if bool_any(truncated):
                        break

                cube_final = _cube_world_position(sim_env)
                place_xy_err = float(np.linalg.norm(cube_final[:2] - place_position[:2]))
                # "Resting" tolerance on Z: within 2 cube-half-sizes of the
                # cube's own natural resting height -- generous enough to allow
                # a small bounce/settle, tight enough to exclude "still held
                # aloft" or "fell through/off the table".
                cube_resting = abs(float(cube_final[2]) - cube_rest_z) <= 2.0 * args.cube_half_size
                place_succeeded = (
                    place_reached_goal
                    and place_xy_err <= args.place_radius
                    and cube_resting
                )
                row["cube_final_position"] = cube_final.tolist()
                row["place_xy_err"] = place_xy_err
                row["cube_resting"] = cube_resting
                row["place_succeeded"] = place_succeeded
                pick_and_place_succeeded = pick_succeeded and place_succeeded
                row["pick_and_place_succeeded"] = pick_and_place_succeeded
                row["outcome"] = (
                    "pick_and_place_succeeded" if pick_and_place_succeeded else "place_failed"
                )
                if place_succeeded:
                    counts["place_succeeded"] += 1
                    pos_counts["place_succeeded"] += 1
                if pick_and_place_succeeded:
                    counts["pick_and_place_succeeded"] += 1
                    pos_counts["pick_and_place_succeeded"] += 1
                print(
                    f"  [ep {episode_idx}] {row['outcome']} — cube landed "
                    f"{place_xy_err:.4f}m from place target (radius={args.place_radius}m), "
                    f"resting={cube_resting}"
                )

                if args.video_dir is not None:
                    video_path = args.video_dir / f"pos_{pos_idx:02d}_episode_{episode_idx:02d}.mp4"
                    save_video(video_path, frames, fps=args.video_fps)
                    row["video"] = str(video_path)
                    print(f"  [ep {episode_idx}] outcome={row['outcome']}  video: {video_path}")

                all_rows.append(row)

            attempted = pos_counts["total"] - pos_counts["ik_failed"] - pos_counts["motion_plan_infeasible"]
            print(
                f"  — position {pos_idx:02d}: pick_and_place_succeeded="
                f"{pos_counts['pick_and_place_succeeded']}/{attempted} attempted "
                f"({100 * pos_counts['pick_and_place_succeeded'] / max(attempted, 1):.1f}%), "
                f"pick_succeeded={pos_counts['pick_succeeded']}/{attempted}, "
                f"place_succeeded={pos_counts['place_succeeded']}/{attempted}, "
                f"reach_reached={pos_counts['reach_reached']}/{attempted}, "
                f"ik_unreachable={pos_counts['ik_failed']}/{pos_counts['total']}, "
                f"not_inter_feasible={pos_counts['motion_plan_infeasible']}/{pos_counts['total']}"
            )
    finally:
        ik_solver.close()
        ik_env.close()
        sim_env.close()
        ghost_env.close()

    attempted_total = (
        counts["total_episodes"] - counts["ik_failed"] - counts["motion_plan_infeasible"]
    )
    print("\n── Summary (checkpoint pick-and-place robustness across all cube positions)")
    print(f"   total episodes            : {counts['total_episodes']}")
    print(f"   IK-unreachable (skipped)  : {counts['ik_failed']}")
    print(f"   not inter-feasible (skip) : {counts['motion_plan_infeasible']}")
    print(f"   attempted by checkpoint   : {attempted_total}")
    print(
        f"   reach converged           : {counts['reach_reached']}/{attempted_total} "
        f"({100 * counts['reach_reached'] / max(attempted_total, 1):.1f}%)"
    )
    print(
        f"   descent plan failed       : {counts['descent_failed']}/{counts['reach_reached']} "
        "(pre-grasp reached, but no straight-line plan down onto the grasp pose)"
    )
    print(
        f"   grasp misaligned (skipped): {counts['grasp_misaligned']}/{counts['reach_reached']} "
        "(at grasp depth the TCP still wasn't centered on the cube after every "
        "realign attempt -- close was never attempted)"
    )
    print(
        f"   pick succeeded            : {counts['pick_succeeded']}/{attempted_total} "
        f"({100 * counts['pick_succeeded'] / max(attempted_total, 1):.1f}% of attempted)"
    )
    print(
        f"   place succeeded           : {counts['place_succeeded']}/{attempted_total} "
        f"({100 * counts['place_succeeded'] / max(attempted_total, 1):.1f}% of attempted)"
    )
    print(
        f"   pick AND place succeeded  : {counts['pick_and_place_succeeded']}/{attempted_total} "
        f"({100 * counts['pick_and_place_succeeded'] / max(attempted_total, 1):.1f}% of attempted, "
        f"{100 * counts['pick_and_place_succeeded'] / max(counts['total_episodes'], 1):.1f}% of all)"
    )

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps({"counts": counts, "episodes": all_rows}, indent=2))
        print(f"\nwrote summary: {args.summary_json}")

    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Checkpoint-based pick-and-place eval: DP3 + reranking steered from an "
            "IK-verified start pose to an IK-verified grasp pose at a real, "
            "physically simulated cube, then a hardcoded close+lift."
        )
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Used only for metadata.json (env_kwargs/crop/action_mode) -- the "
        "observation shape the checkpoint was trained on. No episode content is read, "
        "and env_id is ignored (this script always uses its own pick env id).",
    )
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument(
        "--num-cube-positions", type=int, default=4, help="Distinct cube XY placements."
    )
    p.add_argument(
        "--episodes-per-position",
        type=int,
        default=5,
        help="Grasp-approach orientation variants per cube position.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--orientation-cone-deg",
        type=float,
        default=45.0,
        help=(
            "Max tilt (degrees) of sampled start/grasp-approach orientations away "
            "from straight-down. Kept tighter than the pure-reach eval's 60deg "
            "default -- a heavily tilted parallel-jaw approach loses grasp margin "
            "around a small cube even when the TCP pose itself is reachable."
        ),
    )
    p.add_argument("--extra-orientation-attempts", type=int, default=3)
    p.add_argument(
        "--cube-half-size",
        type=float,
        default=0.02,
        help=(
            "Half-side-length of the spawned cube, meters (0.02 -> a 4cm cube). NOT "
            "verified against the real xArm gripper's actual jaw opening -- that "
            "spec isn't recorded in this repo; re-tune before trusting sim-to-real."
        ),
    )
    p.add_argument(
        "--grasp-height-offset",
        type=float,
        default=0.0,
        help=(
            "Extra Z offset (meters) from the cube's CENTER to the reach target. "
            "0.0 means the TCP target is the cube's vertical center -- correct for "
            "link_tcp (centered between the finger tips) to straddle the cube when "
            "the jaws close, per the 'fixed offset above cube center, straight down' "
            "design. Positive values approach from higher up before descending."
        ),
    )
    p.add_argument(
        "--pregrasp-height",
        type=float,
        default=0.08,
        help="Standoff distance (m) BACK ALONG THE GRIPPER'S APPROACH AXIS at which "
        "the learned reach phase actually aims. The policy flies to that standoff "
        "point, then a scripted straight-line descent runs down the approach axis "
        "onto the grasp pose. Without it the policy arrives at cube height moving "
        "laterally with the claws wide open, which broadsides the cube and knocks it "
        "several cm away before anything closes. Set 0.0 to disable (reach flies "
        "directly to the grasp pose, the old behaviour).",
    )
    p.add_argument(
        "--descent-max-steps",
        type=int,
        default=40,
        help="Replay budget for the scripted pre-grasp descent trajectory.",
    )
    p.add_argument(
        "--grasp-realign-attempts",
        type=int,
        default=3,
        help="How many times to re-fly the approach with a CALCULATED orientation "
        "correction when the fingers end up off-centre on the cube at grasp depth. "
        "Each retry also re-targets on the cube's LIVE position, since a previous "
        "attempt may have moved it. 0 disables retrying (fail straight to "
        "grasp_misaligned).",
    )
    p.add_argument(
        "--grasp-realign-tilt-gain",
        type=float,
        default=0.5,
        help="How much to shrink the approach tilt toward straight-down per realign "
        "attempt, as a multiple of (measured centring error / "
        "--grasp-centering-tolerance). Rotation error amplifies at the fingertips "
        "(they sit ~3.5cm+ from the TCP -- see the URDF), so a smaller tilt directly "
        "means a smaller lever-arm offset for the same TCP convergence; this is a "
        "computed correction from the actual measured miss, not a random guess.",
    )
    p.add_argument(
        "--grasp-realign-max-tilt-shrink",
        type=float,
        default=0.7,
        help="Cap (fraction, 0-1) on how much of the remaining tilt a single realign "
        "attempt can remove, so an orientation is never collapsed straight to "
        "vertical in one step -- it stays close to the IK-verified, "
        "inter-feasible orientation the episode was validated with, converging "
        "toward straight-down only as fast as repeated misses justify.",
    )
    p.add_argument(
        "--place-radius",
        type=float,
        default=0.07,
        help="Radius (m) of the place landing-zone marker. The cube must come to "
        "rest within this XY distance of the sampled place target to count as a "
        "successful place.",
    )
    p.add_argument(
        "--place-height-offset",
        type=float,
        default=0.0,
        help="Extra Z offset (m) from the table surface to the place target -- "
        "same convention as --grasp-height-offset. 0.0 means the TCP target puts "
        "the cube's center back at its normal resting height.",
    )
    p.add_argument(
        "--place-orientation-cone-deg",
        type=float,
        default=45.0,
        help="Max tilt (degrees) of the sampled place-approach orientation away "
        "from straight-down -- independent of --orientation-cone-deg so the pick "
        "and place approaches can be tuned separately. This is the 'some random "
        "orientation' for the place pose.",
    )
    p.add_argument(
        "--min-grasp-place-distance",
        type=float,
        default=0.15,
        help="Minimum required separation (m) between the grasp position and the "
        "sampled place position -- ensures a genuine relocation is being tested, "
        "not placing the cube back where it was picked up.",
    )
    p.add_argument(
        "--place-max-steps",
        type=int,
        default=120,
        help="Sim-step budget for the reach-to-place-pose phase (cube held), "
        "before it's called failed and the cube is released wherever it ended up.",
    )
    p.add_argument(
        "--release-steps",
        type=int,
        default=10,
        help="Settle steps holding the place-converged (or place-timed-out) qpos "
        "while the gripper opens to release the cube.",
    )
    p.add_argument(
        "--position-resample-attempts",
        type=int,
        default=20,
        help="How many times to redraw the place position before accepting the "
        "farthest candidate seen. The old default of 1 made --min-grasp-place-"
        "distance decorative -- a single draw was returned regardless of how "
        "close it landed to the cube (observed: 4.3cm apart against a requested "
        "15cm minimum).",
    )
    p.add_argument("--position-tolerance", type=float, default=0.02)
    p.add_argument("--rotation-tolerance", type=float, default=0.35, help="Radians.")
    p.add_argument(
        "--grasp-centering-tolerance",
        type=float,
        default=0.012,
        help="Max allowed XY distance (m) between the TCP and the cube's LIVE "
        "position at reach convergence, checked right before committing to the "
        "close. A reach that 'converged' within --position-tolerance/"
        "--rotation-tolerance can still be off-center enough at the fingertips "
        "(rotation error amplifies with finger length) to clip the cube instead "
        "of straddling it -- this gate skips that doomed close instead of "
        "attempting it blindly.",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=120,
        help="Sim-step budget for the reach-to-grasp-pose phase before it's called failed.",
    )
    p.add_argument(
        "--action-ema-alpha",
        type=float,
        default=0.6,
        help="EMA smoothing factor applied to each executed sim action during the "
        "reach phase. 1.0 = no smoothing; lower = heavier smoothing.",
    )
    p.add_argument("--k-schedule", type=int, nargs="+", default=[16, 32, 64])
    p.add_argument("--policy-batch-size", type=int, default=64)
    p.add_argument(
        "--gripper-open-value",
        type=float,
        default=-1.0,
        help="Normalized gripper command held during the reach/approach phase. "
        "-1.0 = fully open (controller's configured lower bound) -- wider than the "
        "reach dataset's semi-open bake, appropriate for actually clearing the cube "
        "on approach.",
    )
    p.add_argument(
        "--gripper-close-value",
        type=float,
        default=1.0,
        help="Normalized gripper command used for the hardcoded close+lift. 1.0 = "
        "fully closed (the controller's configured upper bound, which already backs "
        "off the true hard joint limit by a safety margin -- see "
        "XArm7Gripper._GRIPPER_CLOSED in agents.py).",
    )
    p.add_argument(
        "--close-steps",
        type=int,
        default=15,
        help="Settle steps holding the reach-converged qpos while the gripper closes.",
    )
    p.add_argument("--lift-height", type=float, default=0.15, help="Meters, vertical.")
    p.add_argument(
        "--lift-max-steps",
        type=int,
        default=60,
        help="Replay budget for the planned lift trajectory.",
    )
    p.add_argument(
        "--grasp-success-lift-fraction",
        type=float,
        default=0.5,
        help="The cube's own SAPIEN pose must rise by at least this fraction of "
        "--lift-height for the episode to count as a successful grasp.",
    )
    p.add_argument("--video-dir", type=Path, default=None)
    p.add_argument("--video-fps", type=int, default=10)
    p.add_argument("--summary-json", type=Path, default=None)
    args = p.parse_args(argv)
    if args.num_cube_positions <= 0:
        raise ValueError("--num-cube-positions must be positive")
    if args.episodes_per_position <= 0:
        raise ValueError("--episodes-per-position must be positive")
    if args.extra_orientation_attempts < 0:
        raise ValueError("--extra-orientation-attempts must be non-negative")
    if args.cube_half_size <= 0:
        raise ValueError("--cube-half-size must be positive")
    if args.place_radius <= 0:
        raise ValueError("--place-radius must be positive")
    if args.pregrasp_height < 0.0:
        raise ValueError("--pregrasp-height must be non-negative (0.0 disables it)")
    if args.descent_max_steps <= 0:
        raise ValueError("--descent-max-steps must be positive")
    if args.grasp_realign_attempts < 0:
        raise ValueError("--grasp-realign-attempts must be non-negative")
    if args.grasp_realign_tilt_gain < 0.0:
        raise ValueError("--grasp-realign-tilt-gain must be non-negative")
    if not (0.0 <= args.grasp_realign_max_tilt_shrink <= 1.0):
        raise ValueError("--grasp-realign-max-tilt-shrink must be in [0.0, 1.0]")
    if args.min_grasp_place_distance < 0.0:
        raise ValueError("--min-grasp-place-distance must be non-negative")
    if args.place_max_steps <= 0:
        raise ValueError("--place-max-steps must be positive")
    if args.release_steps < 0:
        raise ValueError("--release-steps must be non-negative")
    if args.position_resample_attempts <= 0:
        raise ValueError("--position-resample-attempts must be positive")
    if args.position_tolerance < 0.0 or args.rotation_tolerance < 0.0:
        raise ValueError("tolerances must be non-negative")
    if args.grasp_centering_tolerance < 0.0:
        raise ValueError("--grasp-centering-tolerance must be non-negative")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if not (0.0 < args.action_ema_alpha <= 1.0):
        raise ValueError("--action-ema-alpha must be in (0.0, 1.0]")
    if args.close_steps < 0:
        raise ValueError("--close-steps must be non-negative")
    if args.lift_height <= 0:
        raise ValueError("--lift-height must be positive")
    if args.lift_max_steps <= 0:
        raise ValueError("--lift-max-steps must be positive")
    if not (0.0 <= args.grasp_success_lift_fraction <= 1.0):
        raise ValueError("--grasp-success-lift-fraction must be in [0.0, 1.0]")
    if args.video_fps <= 0:
        raise ValueError("--video-fps must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(main())
