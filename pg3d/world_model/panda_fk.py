from __future__ import annotations

import math

import torch

_JOINT_ORIGINS = (
    ((0.0, 0.0, 0.333), (0.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (-math.pi / 2.0, 0.0, 0.0)),
    ((0.0, -0.316, 0.0), (math.pi / 2.0, 0.0, 0.0)),
    ((0.0825, 0.0, 0.0), (math.pi / 2.0, 0.0, 0.0)),
    ((-0.0825, 0.384, 0.0), (-math.pi / 2.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (math.pi / 2.0, 0.0, 0.0)),
    ((0.088, 0.0, 0.0), (math.pi / 2.0, 0.0, 0.0)),
)

PANDA_MOVABLE_COLLISION_LINKS = (
    "panda_link1",
    "panda_link2",
    "panda_link3",
    "panda_link4",
    "panda_link5",
    "panda_link6",
    "panda_link7",
    "panda_hand",
    "panda_leftfinger",
    "panda_rightfinger",
)


def panda_link_transforms(
    q: torch.Tensor,
    world_from_base: torch.Tensor | None = None,
    *,
    gripper_open: float | torch.Tensor = 0.04,
) -> torch.Tensor:
    """Return world transforms for Panda links with controllable collision geometry.

    The returned tensor has shape ``[..., 10, 4, 4]`` and follows
    :data:`PANDA_MOVABLE_COLLISION_LINKS`. The fixed base link is intentionally omitted because
    arm-joint guidance cannot move it. ``gripper_open`` is the position of each mirrored Panda
    finger joint in meters and may be scalar or broadcastable with ``q.shape[:-1]``.
    """
    if q.ndim < 1 or q.shape[-1] != 7:
        raise ValueError(f"q must have shape [..., 7], got {tuple(q.shape)}")

    transform = _world_from_base(q, world_from_base)
    link_transforms = []
    for joint_index, (xyz, rpy) in enumerate(_JOINT_ORIGINS):
        transform = transform @ _fixed_transform(q, xyz=xyz, rpy=rpy)
        transform = transform @ _rotation_z(q[..., joint_index])
        link_transforms.append(transform)

    # panda_joint8 and panda_hand_joint are fixed.
    hand_transform = transform @ _fixed_transform(q, xyz=(0.0, 0.0, 0.107))
    hand_transform = hand_transform @ _fixed_transform(q, rpy=(0.0, 0.0, -math.pi / 4.0))
    link_transforms.append(hand_transform)

    finger_position = _broadcast_scalar(
        gripper_open,
        like=q,
        name="gripper_open",
    )
    finger_origin = hand_transform @ _fixed_transform(q, xyz=(0.0, 0.0, 0.0584))
    left_finger = finger_origin @ _translation_y(finger_position)
    right_finger = finger_origin @ _translation_y(-finger_position)
    link_transforms.extend((left_finger, right_finger))
    return torch.stack(link_transforms, dim=-3)


def panda_end_effector_position(
    q: torch.Tensor,
    world_from_base: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ManiSkill Panda TCP positions for joint vectors ``[..., 7]``.

    The chain mirrors ManiSkill's ``panda_v2.urdf`` through ``panda_hand_tcp``.
    ``world_from_base`` is an optional homogeneous transform with shape ``[4, 4]``
    or a batch shape broadcastable with ``q.shape[:-1]``.
    """
    hand_transform = panda_link_transforms(q, world_from_base)[..., 7, :, :]
    tcp_transform = hand_transform @ _fixed_transform(q, xyz=(0.0, 0.0, 0.1034))
    return tcp_transform[..., :3, 3]


def _world_from_base(
    q: torch.Tensor,
    world_from_base: torch.Tensor | None,
) -> torch.Tensor:
    batch_shape = q.shape[:-1]
    if world_from_base is None:
        transform = torch.eye(4, device=q.device, dtype=q.dtype)
    else:
        if world_from_base.shape[-2:] != (4, 4):
            raise ValueError(
                f"world_from_base must end in shape [4, 4], got {tuple(world_from_base.shape)}"
            )
        transform = world_from_base.to(device=q.device, dtype=q.dtype)
    try:
        return torch.broadcast_to(transform, (*batch_shape, 4, 4)).clone()
    except RuntimeError as exc:
        raise ValueError(
            f"world_from_base shape {tuple(transform.shape)} is not broadcastable "
            f"with q batch shape {tuple(batch_shape)}"
        ) from exc


def _broadcast_scalar(
    value: float | torch.Tensor,
    *,
    like: torch.Tensor,
    name: str,
) -> torch.Tensor:
    scalar = torch.as_tensor(value, device=like.device, dtype=like.dtype)
    try:
        return torch.broadcast_to(scalar, like.shape[:-1])
    except RuntimeError as exc:
        raise ValueError(
            f"{name} shape {tuple(scalar.shape)} is not broadcastable with q batch shape "
            f"{tuple(like.shape[:-1])}"
        ) from exc


def _fixed_transform(
    like: torch.Tensor,
    *,
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    matrix = torch.tensor(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, xyz[0]],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, xyz[1]],
            [-sp, cp * sr, cp * cr, xyz[2]],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=like.device,
        dtype=like.dtype,
    )
    return torch.broadcast_to(matrix, (*like.shape[:-1], 4, 4))


def _rotation_z(theta: torch.Tensor) -> torch.Tensor:
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    zeros = torch.zeros_like(theta)
    ones = torch.ones_like(theta)
    row0 = torch.stack((cos_theta, -sin_theta, zeros, zeros), dim=-1)
    row1 = torch.stack((sin_theta, cos_theta, zeros, zeros), dim=-1)
    row2 = torch.stack((zeros, zeros, ones, zeros), dim=-1)
    row3 = torch.stack((zeros, zeros, zeros, ones), dim=-1)
    return torch.stack((row0, row1, row2, row3), dim=-2)


def _translation_y(distance: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(distance)
    ones = torch.ones_like(distance)
    row0 = torch.stack((ones, zeros, zeros, zeros), dim=-1)
    row1 = torch.stack((zeros, ones, zeros, distance), dim=-1)
    row2 = torch.stack((zeros, zeros, ones, zeros), dim=-1)
    row3 = torch.stack((zeros, zeros, zeros, ones), dim=-1)
    return torch.stack((row0, row1, row2, row3), dim=-2)
