from __future__ import annotations

import importlib
import subprocess
import sys

FORBIDDEN_MODULES = [
    "dexart",
    "gym",
    "metaworld",
    "mj_envs",
    "mjrl",
    "mujoco_py",
    "numba",
    "pytorch3d",
    "mani_skill",
    "sapien",
]


def test_pg3d_import_does_not_load_torch_or_dp3() -> None:
    code = """
import importlib
import sys

importlib.import_module("pg3d")
assert "torch" not in sys.modules
assert "pg3d.policies.dp3" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_dp3_import_does_not_load_sim_dependencies() -> None:
    importlib.import_module("pg3d.policies.dp3")

    for module in FORBIDDEN_MODULES:
        assert module not in sys.modules
