from __future__ import annotations

import importlib
import subprocess
import sys

import pg3d

PLANNED_PACKAGES = [
    "pg3d.envs",
    "pg3d.envs.maniskill_adapter",
    "pg3d.policies",
    "pg3d.world_model",
    "pg3d.constraints",
    "pg3d.composition",
    "pg3d.baselines",
    "pg3d.eval",
    "pg3d.viz",
    "pg3d.logging",
    "pg3d.utils",
]


def test_package_imports() -> None:
    assert pg3d.__version__


def test_planned_packages_import_without_sim_dependencies() -> None:
    for package in PLANNED_PACKAGES:
        importlib.import_module(package)


def test_maniskill_adapter_import_keeps_simulator_lazy() -> None:
    code = """
import importlib
import sys

importlib.import_module("pg3d.envs.maniskill_adapter")
assert "mani_skill" not in sys.modules
assert "sapien" not in sys.modules
assert "gymnasium" not in sys.modules
assert "rerun" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)
