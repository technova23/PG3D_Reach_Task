"""pg3d-native DP3 policy core.

This package ports a narrow, simulation-free slice of the MIT-licensed
3D Diffusion Policy implementation into pg3d. The external DP3 submodule
remains a reference during migration; runtime code should import this package.
"""

from pg3d.policies.dp3.noise_lineage import ITPSNoiseDraw, ITPSNoiseLineage, stable_lineage_seed
from pg3d.policies.dp3.policy import DP3, SimpleDP3
from pg3d.policies.dp3.reach_dataset import ReachDatasetConfig, ReachSequenceDataset
from pg3d.policies.dp3.synthetic import make_synthetic_batch, make_tiny_policy

__all__ = [
    "DP3",
    "ITPSNoiseDraw",
    "ITPSNoiseLineage",
    "ReachDatasetConfig",
    "ReachSequenceDataset",
    "SimpleDP3",
    "make_synthetic_batch",
    "make_tiny_policy",
    "stable_lineage_seed",
]
