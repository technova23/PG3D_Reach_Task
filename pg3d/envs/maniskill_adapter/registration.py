from __future__ import annotations

_REGISTERED_REACH_ENVS = False


def register_pg3d_reach_envs() -> None:
    """Register pg3d custom ManiSkill reach tasks lazily."""
    global _REGISTERED_REACH_ENVS
    if _REGISTERED_REACH_ENVS:
        return
    import pg3d.envs.maniskill_adapter.reach_env  # noqa: F401

    _REGISTERED_REACH_ENVS = True
