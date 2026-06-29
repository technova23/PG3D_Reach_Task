from __future__ import annotations

import importlib

mods = [
    "numpy",
    "scipy",
    "pydantic",
    "omegaconf",
    "zarr",
    "trimesh",
    "wandb",
    "pg3d",
    "pg3d.policies.dp3",
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
for mod in mods:
    importlib.import_module(mod)
    print(f"ok: {mod}")
