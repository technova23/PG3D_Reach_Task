from __future__ import annotations

import sys
from typing import Any


def main() -> int:
    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
    except Exception as exc:
        print(
            f"Failed to import ManiSkill/Gymnasium: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print(
            "Install with: "
            "uv sync --extra cu129 --extra maniskill --group dev --group notebooks"
        )
        return 2

    env: Any | None = None
    try:
        env = gym.make("PickCube-v1", obs_mode="state", num_envs=1)
        obs, info = env.reset(seed=0)
        print("env: PickCube-v1")
        print(f"observation_space: {env.observation_space}")
        print(f"action_space: {env.action_space}")
        print(f"reset_obs: {_summarize(obs)}")
        info_keys = sorted(info.keys()) if isinstance(info, dict) else type(info).__name__
        print(f"reset_info_keys: {info_keys}")

        action = env.action_space.sample()
        step_result = env.step(action)
        step_obs = step_result[0]
        print(f"step_obs: {_summarize(step_obs)}")
    except Exception as exc:
        print(f"ManiSkill smoke failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if env is not None:
            env.close()
            print("closed env")
    return 0


def _summarize(value: Any) -> Any:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None:
        return {"shape": tuple(shape), "dtype": str(dtype)}
    if isinstance(value, dict):
        return {str(key): _summarize(item) for key, item in value.items()}
    return type(value).__name__


if __name__ == "__main__":
    raise SystemExit(main())
