"""Replay one recorded episode from a pg3d xArm7 reach zarr on the REAL xArm7.

Open-loop joint replay only — no camera/point-cloud is needed or used, since
`sim_action`/`state` are pure joint-space commands independent of observation.
The gripper is closed once before the arm trajectory starts (it never moves
during data-gen: the sim gripper is held rigidly closed for the whole episode).

Timing: the planner samples every trajectory at exactly `control_timestep`
(1/control_freq = 50ms, verified against the env), and the constant-speed
retiming done at data-gen time (`_retime_trajectory_constant_speed` in
write_maniskill_reach_dataset.py) preserves that same row count/spacing — it
only resamples *which* qpos sits at each already-fixed-time slot. So row `i`
of `sim_action` is always meant to land at `t = i * 50ms`; there is no
separate per-row timestamp to load, just this one constant.

Requires the UFactory SDK: `pip install xarm-python-sdk`.

Example:
    # Sanity check shape/timing/joint-limits, no hardware needed:
    python scripts/replay_xarm7_trajectory_real.py \\
        --zarr artifacts/pg3d_xarm7_gripper_reach.zarr --episode-idx 0 --dry-run

    # Real run:
    python scripts/replay_xarm7_trajectory_real.py \\
        --zarr artifacts/pg3d_xarm7_gripper_reach.zarr --episode-idx 0 \\
        --ip 192.168.1.xxx --gripper-closed-pos 300

    # Cautious first hardware test at half speed (2x slower, never exceeds
    # velocity/accel limits the trajectory implies — only makes it slower):
    python scripts/replay_xarm7_trajectory_real.py \\
        --zarr artifacts/pg3d_xarm7_gripper_reach.zarr --episode-idx 0 \\
        --ip 192.168.1.xxx --gripper-closed-pos 300 --time-scale 2.0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import zarr

CONTROL_FREQ_HZ = 20.0  # must match the sim's control_freq (verified: 20Hz, 50ms/step)
HOMING_SPEED_RAD_S = 0.35  # slow, conservative speed for the initial move-to-start
JOINT_LIMITS_RAD = np.array(  # xArm7 hard joint limits, radians (from the URDF)
    [
        [-6.2832, 6.2832],
        [-2.0590, 2.0944],
        [-6.2832, 6.2832],
        [-0.1920, 3.9270],
        [-6.2832, 6.2832],
        [-1.6930, 3.1416],
        [-6.2832, 6.2832],
    ]
)


def _load_episode(zarr_path: Path, episode_idx: int) -> dict[str, np.ndarray]:
    z = zarr.open(str(zarr_path), mode="r")
    ends = np.asarray(z["meta/episode_ends"])
    if not (0 <= episode_idx < len(ends)):
        raise ValueError(f"episode_idx must be in [0, {len(ends)}), got {episode_idx}")
    start = 0 if episode_idx == 0 else int(ends[episode_idx - 1])
    end = int(ends[episode_idx])
    sim_action = np.asarray(z["data/sim_action"][start:end], dtype=np.float32)
    state = np.asarray(z["data/state"][start:end], dtype=np.float32)
    # Episode acceptance means *some* step hit the goal (marked True there), not
    # necessarily the last recorded step — replay continues through settle/hold
    # after that, and per-step success can flicker near the threshold boundary.
    success = bool(np.asarray(z["data/success"][start:end]).any())
    family_id = int(np.asarray(z["data/trajectory_family_id"][start:end])[0, 0])
    return {
        "sim_action": sim_action,  # (T, 7) absolute joint targets, radians, joint1..joint7
        "state": state,  # (T, 13) realized qpos: [:7] arm, [7:] gripper (constant)
        "success": success,
        "family_id": family_id,
        "length": end - start,
    }


def _check_joint_limits(arm_angles: np.ndarray) -> None:
    lo, hi = JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1]
    violations = np.flatnonzero(np.any((arm_angles < lo) | (arm_angles > hi), axis=1))
    if violations.size:
        raise ValueError(
            f"{violations.size} step(s) exceed xArm7 joint limits, e.g. row "
            f"{int(violations[0])} = {arm_angles[violations[0]].round(4).tolist()}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", type=Path, required=True)
    parser.add_argument("--episode-idx", type=int, required=True)
    parser.add_argument("--ip", type=str, default=None, help="xArm controller IP (required unless --dry-run)")
    parser.add_argument(
        "--gripper-closed-pos",
        type=int,
        default=None,
        help=(
            "xArm gripper SDK position (0-850 range) matching the sim's closed pose. "
            "Bench-calibrate this once: jog the real gripper closed and read back its "
            "position via arm.get_gripper_position(). Omit to skip gripper control entirely."
        ),
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help=(
            "Multiply the 50ms per-step interval by this factor (e.g. 2.0 = replay at "
            "half speed). >1 is always safe (slower, never exceeds the recorded "
            "velocity profile); <1 risks exceeding real joint velocity/accel limits."
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt before moving real hardware.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the trajectory summary and validate joint limits without connecting to hardware.",
    )
    args = parser.parse_args(argv)

    ep = _load_episode(args.zarr, args.episode_idx)
    arm_angles = ep["sim_action"][:, :7]  # (T, 7), radians, joint1..joint7
    _check_joint_limits(arm_angles)
    print(
        f"episode {args.episode_idx}: family={ep['family_id']} length={ep['length']} "
        f"success={ep['success']}",
        flush=True,
    )
    print(f"start qpos (rad): {arm_angles[0].round(4).tolist()}", flush=True)
    print(f"goal qpos  (rad): {arm_angles[-1].round(4).tolist()}", flush=True)
    dt = (1.0 / CONTROL_FREQ_HZ) * args.time_scale
    print(f"replay cadence: {1.0 / dt:.2f} Hz ({dt * 1000.0:.0f} ms/step)", flush=True)

    if args.dry_run:
        print("--dry-run: joint limits OK, not connecting to hardware.", flush=True)
        return 0

    if args.ip is None:
        print("error: --ip is required unless --dry-run", file=sys.stderr)
        return 1

    if not args.yes:
        reply = input(
            f"About to move REAL xArm7 at {args.ip} through {arm_angles.shape[0]} steps "
            f"({dt * arm_angles.shape[0]:.1f}s). Workspace clear, E-stop in reach? [y/N] "
        )
        if reply.strip().lower() != "y":
            print("aborted.", flush=True)
            return 1

    from xarm.wrapper import XArmAPI  # local import: only required for a real run

    arm = XArmAPI(args.ip)
    realized = np.full_like(arm_angles, np.nan)
    try:
        arm.clean_error()
        arm.clean_warn()
        arm.motion_enable(enable=True)
        arm.set_mode(0)  # position mode: blocking, internally-smoothed single moves
        arm.set_state(0)  # ready/sport state

        if args.gripper_closed_pos is not None:
            arm.set_gripper_enable(True)
            arm.set_gripper_mode(0)
            arm.set_gripper_position(args.gripper_closed_pos, wait=True)
            print(f"gripper closed to {args.gripper_closed_pos}", flush=True)

        # Move slowly to the episode's first joint config before streaming — never
        # snap directly from the arm's current pose into the recorded trajectory.
        code = arm.set_servo_angle(
            angle=arm_angles[0].tolist(),
            speed=HOMING_SPEED_RAD_S,
            is_radian=True,
            wait=True,
        )
        if code != 0:
            print(f"error: move-to-start failed, code={code}", file=sys.stderr)
            return 1
        realized[0] = np.asarray(arm.get_servo_angle(is_radian=True)[1][:7])
        print("at start pose, streaming trajectory...", flush=True)

        # Servo motion mode (mode=1): each set_servo_angle_j call is one control-cycle
        # target, not internally re-smoothed — the caller supplies the smooth stream.
        # This matches how sim_action was generated (one abs-joint target per 50ms step).
        arm.set_mode(1)
        arm.set_state(0)
        t_start = time.perf_counter()
        for step_idx in range(1, arm_angles.shape[0]):
            code = arm.set_servo_angle_j(angle=arm_angles[step_idx].tolist(), is_radian=True)
            if code != 0:
                print(f"warning: step {step_idx} rejected, code={code}", file=sys.stderr)
            realized[step_idx] = np.asarray(arm.get_servo_angle(is_radian=True)[1][:7])
            # Absolute deadline schedule (not accumulated per-step sleep) so timing
            # doesn't drift from per-call overhead over a long trajectory.
            deadline = t_start + step_idx * dt
            sleep_for = deadline - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)

        arm.set_mode(0)
        arm.set_state(0)
        print("replay complete.", flush=True)
    except KeyboardInterrupt:
        print("interrupted — stopping arm.", file=sys.stderr)
        arm.set_state(4)  # stop
        raise
    finally:
        arm.disconnect()

    # Fidelity check: how closely did the real arm track the recorded sim trajectory?
    valid = ~np.isnan(realized).any(axis=1)
    if valid.any():
        err = np.abs(realized[valid] - arm_angles[valid])
        print(
            f"tracking error (rad): mean={err.mean():.4f} max={err.max():.4f} "
            f"final-pose error={np.abs(realized[valid][-1] - arm_angles[-1]).max():.4f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
