from __future__ import annotations

import numpy as np

from pg3d.world_model.types import ActionChunk, Array, as_float_array


def interpret_joint_chunk(
    action_chunk: ActionChunk,
    start_q: Array,
    *,
    controlled_dof: int | None = None,
) -> Array:
    """Convert a joint-action chunk into an imagined joint-state trajectory.

    P07 uses prefix semantics: an action with dimension `A` controls the first `A` joints of
    `start_q`, while any trailing joints hold their previous values. This matches the first
    Panda reach convention where 7D arm labels are interpreted against 9D qpos.
    """
    q0 = as_float_array(start_q, name="start_q", ndim=1)
    if q0.shape[0] <= 0:
        raise ValueError("start_q must contain at least one joint")

    action_dim = action_chunk.action_dim
    controlled = action_dim if controlled_dof is None else int(controlled_dof)
    if controlled <= 0:
        raise ValueError("controlled_dof must be positive")
    if controlled != action_dim:
        raise ValueError(
            f"controlled_dof must match action_dim for P07, got {controlled} and {action_dim}"
        )
    if controlled > q0.shape[0]:
        raise ValueError(
            f"action_dim {controlled} cannot exceed start_q dof {q0.shape[0]}"
        )

    trajectory = np.repeat(q0.reshape(1, -1), action_chunk.horizon, axis=0).astype(
        np.float32,
        copy=True,
    )
    current = q0.astype(np.float32, copy=True)
    for step_idx, action in enumerate(action_chunk.actions):
        if action_chunk.action_mode == "abs_joint":
            current[:controlled] = action[:controlled]
        elif action_chunk.action_mode == "delta_joint":
            current[:controlled] = current[:controlled] + action[:controlled]
        else:  # ActionChunk validation should make this unreachable.
            raise ValueError(f"unsupported action_mode {action_chunk.action_mode!r}")
        trajectory[step_idx] = current
    return trajectory
