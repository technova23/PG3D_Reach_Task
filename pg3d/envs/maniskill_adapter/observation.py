from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pg3d.envs.maniskill_adapter.types import Observation, RobotState, SimGroundTruth
from pg3d.utils.arrays import to_numpy as _to_numpy

Array = np.ndarray


@dataclass(frozen=True)
class SegmentationContext:
    """Segmentation ids needed to derive masks from a ManiSkill point cloud."""

    robot_ids: frozenset[int] = frozenset()
    object_ids: Mapping[str, frozenset[int]] = field(default_factory=dict)


def adapt_observation(
    obs: Mapping[str, Any],
    *,
    info: Mapping[str, Any] | None = None,
    env: Any | None = None,
    segmentation_context: SegmentationContext | None = None,
    task_name: str = "unknown",
    batch_index: int = 0,
) -> Observation:
    """Convert one ManiSkill observation into pg3d's observation boundary object.

    ManiSkill imports are intentionally avoided here. The optional ``env`` object is only inspected
    duck-typed for segmentation ids when point-cloud segmentation is present.
    """
    if segmentation_context is None and env is not None:
        segmentation_context = segmentation_context_from_env(env)

    agent = _mapping(obs.get("agent"), name="obs['agent']")
    extra = _mapping(obs.get("extra", {}), name="obs['extra']")
    pointcloud = obs.get("pointcloud")

    agent_pos_indices = _agent_pos_joint_indices(env)
    joint_positions = _batch_item(agent["qpos"], batch_index=batch_index, name="agent.qpos")
    joint_velocities = (
        _batch_item(agent["qvel"], batch_index=batch_index, name="agent.qvel")
        if "qvel" in agent
        else None
    )
    if agent_pos_indices is not None:
        joint_positions = joint_positions[agent_pos_indices]
        if joint_velocities is not None:
            joint_velocities = joint_velocities[agent_pos_indices]

    robot_state = RobotState(
        joint_positions=joint_positions,
        joint_velocities=joint_velocities,
        tcp_pose=(
            _batch_item(extra["tcp_pose"], batch_index=batch_index, name="extra.tcp_pose")
            if "tcp_pose" in extra
            else None
        ),
        gripper_open=_gripper_open(agent.get("qpos"), batch_index=batch_index),
        metadata={"source": "maniskill", "batch_index": batch_index},
    )

    point_cloud, point_features = _adapt_pointcloud(
        pointcloud, batch_index=batch_index
    )
    robot_mask, object_masks = _derive_masks(
        point_features.get("segmentation"), segmentation_context=segmentation_context
    )

    sim_gt = SimGroundTruth(
        task_name=task_name,
        target_position=(
            _batch_item(extra["goal_pos"], batch_index=batch_index, name="extra.goal_pos")
            if "goal_pos" in extra
            else None
        ),
        success=_bool_from_info(info, "success", batch_index=batch_index),
        metadata={
            "info": _summarize_info(info or {}, batch_index=batch_index),
            "extra_keys": sorted(str(key) for key in extra.keys()),
        },
    )

    return Observation(
        point_cloud=point_cloud,
        point_features=point_features,
        robot_state=robot_state,
        robot_mask=robot_mask,
        object_masks=object_masks,
        sim_gt=sim_gt,
        metadata={
            "source": "maniskill",
            "obs_keys": sorted(str(key) for key in obs.keys()),
            "batch_index": batch_index,
            "segmentation_context": _segmentation_context_summary(segmentation_context),
        },
    )


def segmentation_context_from_env(env: Any) -> SegmentationContext:
    """Collect robot and common task object segmentation ids from a live ManiSkill env."""
    unwrapped = getattr(env, "unwrapped", env)
    robot_ids = _ids_from_links(getattr(getattr(unwrapped, "agent", None), "robot", None))
    object_ids: dict[str, frozenset[int]] = {}
    for name in ("cube", "goal_site"):
        ids = _ids_from_actor(getattr(unwrapped, name, None))
        if ids:
            object_ids[name] = ids
    return SegmentationContext(robot_ids=frozenset(robot_ids), object_ids=object_ids)


def _adapt_pointcloud(
    pointcloud: Any,
    *,
    batch_index: int,
) -> tuple[Array, dict[str, Array]]:
    if pointcloud is None:
        return np.zeros((0, 3), dtype=np.float32), {}
    pointcloud_map = _mapping(pointcloud, name="obs['pointcloud']")
    if "xyzw" not in pointcloud_map:
        raise ValueError("pointcloud observation must include 'xyzw'")

    xyzw = _batch_item(pointcloud_map["xyzw"], batch_index=batch_index, name="pointcloud.xyzw")
    if xyzw.ndim != 2 or xyzw.shape[1] < 3:
        raise ValueError(f"pointcloud.xyzw must have shape [N, >=3], got {xyzw.shape}")
    point_cloud = xyzw[:, :3].astype(np.float32, copy=True)

    features: dict[str, Array] = {}
    if "rgb" in pointcloud_map:
        rgb = _batch_item(pointcloud_map["rgb"], batch_index=batch_index, name="pointcloud.rgb")
        features["rgb"] = rgb.astype(np.uint8, copy=False)
    if "segmentation" in pointcloud_map:
        segmentation = _batch_item(
            pointcloud_map["segmentation"],
            batch_index=batch_index,
            name="pointcloud.segmentation",
        )
        if segmentation.ndim == 2 and segmentation.shape[1] == 1:
            segmentation = segmentation[:, 0]
        features["segmentation"] = segmentation.astype(np.int64, copy=False)
    return point_cloud, features


def _derive_masks(
    segmentation: Array | None,
    *,
    segmentation_context: SegmentationContext | None,
) -> tuple[Array | None, dict[str, Array]]:
    if segmentation is None or segmentation_context is None:
        return None, {}

    robot_mask = (
        np.isin(segmentation, list(segmentation_context.robot_ids))
        if segmentation_context.robot_ids
        else None
    )
    object_masks = {
        name: np.isin(segmentation, list(ids))
        for name, ids in segmentation_context.object_ids.items()
        if ids
    }
    return robot_mask, object_masks


def _batch_item(value: Any, *, batch_index: int, name: str) -> Array:
    array = _to_numpy(value)
    if array.ndim == 0:
        return array.reshape(1)
    if array.ndim >= 2:
        if batch_index >= array.shape[0]:
            raise IndexError(
                f"{name} batch_index={batch_index} is out of range for shape {array.shape}"
            )
        return np.asarray(array[batch_index])
    return array


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}")
    return value


def _agent_pos_joint_indices(env: Any | None) -> list[int] | None:
    """Return the agent-declared qpos subset for agent_pos, or None for all of it.

    Some agents (e.g. XArm7Gripper) have qpos entries that are pure URDF <mimic>
    followers of another joint already in qpos -- redundant information that
    would otherwise bloat agent_pos with copies of the same value. Such agents
    declare ``agent_pos_joint_indices`` (a list of qpos indices to keep); agents
    that don't declare it (the default) keep the full qpos vector unchanged.
    """
    agent = getattr(getattr(env, "unwrapped", env), "agent", None)
    return getattr(agent, "agent_pos_joint_indices", None)


def _gripper_open(qpos: Any, *, batch_index: int) -> float | None:
    if qpos is None:
        return None
    qpos_array = _batch_item(qpos, batch_index=batch_index, name="agent.qpos")
    if qpos_array.shape[0] < 2:
        return None
    return float(np.mean(qpos_array[-2:]))


def _bool_from_info(
    info: Mapping[str, Any] | None,
    key: str,
    *,
    batch_index: int,
) -> bool | None:
    if info is None or key not in info:
        return None
    value = _batch_item(info[key], batch_index=batch_index, name=f"info.{key}")
    return bool(np.asarray(value).reshape(-1)[0])


def _summarize_info(info: Mapping[str, Any], *, batch_index: int) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in info.items():
        array = _to_numpy(value)
        if array.ndim == 0:
            item = array.item()
        elif array.shape[0] > batch_index:
            item = array[batch_index]
            item = item.item() if np.asarray(item).ndim == 0 else np.asarray(item).tolist()
        else:
            item = array.tolist()
        summary[str(key)] = item
    return summary


def _ids_from_links(robot: Any) -> set[int]:
    if robot is None:
        return set()
    links = getattr(robot, "links", None)
    if links is None and hasattr(robot, "get_links"):
        links = robot.get_links()
    if links is None:
        return set()
    ids: set[int] = set()
    for link in links:
        ids.update(_ids_from_actor(link))
    return ids


def _ids_from_actor(actor: Any) -> frozenset[int]:
    if actor is None:
        return frozenset()
    raw_id = getattr(actor, "per_scene_id", None)
    if raw_id is None:
        raw_id = getattr(actor, "id", None)
    if raw_id is None:
        return frozenset()
    values = np.asarray(_to_numpy(raw_id)).reshape(-1)
    return frozenset(int(value) for value in values)


def _segmentation_context_summary(
    segmentation_context: SegmentationContext | None,
) -> dict[str, Any]:
    if segmentation_context is None:
        return {"available": False}
    return {
        "available": True,
        "robot_ids": sorted(segmentation_context.robot_ids),
        "object_ids": {
            name: sorted(ids) for name, ids in segmentation_context.object_ids.items()
        },
    }
