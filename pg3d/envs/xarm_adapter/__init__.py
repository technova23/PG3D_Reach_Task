"""xArm7 adapter: ManiSkill agents + reach envs for pg3d.

Sibling of :mod:`pg3d.envs.maniskill_adapter` (Panda). Importing either
register function imports the env module which registers all agents and env ids.

- :func:`register_pg3d_xarm7_reach_envs`        — no-gripper (TCP = link_eef)
- :func:`register_pg3d_xarm7_gripper_reach_envs` — with gripper (TCP = link_tcp)

Both share the same dataset schema, DP3 training, constraints, and eval.
"""

from __future__ import annotations


def register_pg3d_xarm7_reach_envs() -> None:
    """Register ``xarm7_nogripper`` agent and ``PG3DReach-XArm7-*`` env ids."""
    from pg3d.envs.xarm_adapter import reach_env  # noqa: F401


def register_pg3d_xarm7_gripper_reach_envs() -> None:
    """Register ``xarm7_gripper`` agent and ``PG3DReach-XArm7-Gripper-*`` env ids."""
    from pg3d.envs.xarm_adapter import reach_env  # noqa: F401


def register_pg3d_xarm7_robotiq_reach_envs() -> None:
    """Register ``xarm7_robotiq`` agent and ``PG3DReach-XArm7-Robotiq-*`` env ids."""
    from pg3d.envs.xarm_adapter import reach_env  # noqa: F401


__all__ = [
    "register_pg3d_xarm7_reach_envs",
    "register_pg3d_xarm7_gripper_reach_envs",
    "register_pg3d_xarm7_robotiq_reach_envs",
]
