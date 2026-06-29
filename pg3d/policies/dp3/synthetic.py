from __future__ import annotations

import torch

from pg3d.policies.dp3.policy import DP3Batch, SimpleDP3


def tiny_shape_meta(
    num_points: int = 32,
    point_dim: int = 3,
    state_dim: int = 7,
    action_dim: int = 7,
) -> dict[str, dict[str, dict[str, list[int]]]]:
    """Return the minimal shape metadata needed to construct a tiny DP3 policy."""
    return {
        "obs": {
            "point_cloud": {"shape": [num_points, point_dim]},
            "agent_pos": {"shape": [state_dim]},
        },
        "action": {"shape": [action_dim]},
    }


def make_tiny_policy(
    *,
    horizon: int = 4,
    n_obs_steps: int = 2,
    n_action_steps: int = 2,
    num_points: int = 32,
    state_dim: int = 7,
    action_dim: int = 7,
    num_inference_steps: int = 2,
    goal_marker_points: int = 0,
) -> SimpleDP3:
    """Build a small DP3 instance suitable for CPU/GPU smoke tests."""
    return SimpleDP3(
        shape_meta=tiny_shape_meta(
            num_points=num_points,
            state_dim=state_dim,
            action_dim=action_dim,
        ),
        horizon=horizon,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        num_inference_steps=num_inference_steps,
        encoder_output_dim=16,
        diffusion_step_embed_dim=32,
        down_dims=(32, 64),
        kernel_size=3,
        n_groups=8,
        goal_marker_points=goal_marker_points,
        pointcloud_encoder_cfg={
            "out_channels": 16,
            "use_layernorm": True,
            "final_norm": "layernorm",
        },
    )


def make_synthetic_batch(
    *,
    batch_size: int = 2,
    horizon: int = 4,
    n_obs_steps: int = 2,
    num_points: int = 32,
    point_dim: int = 3,
    state_dim: int = 7,
    action_dim: int = 7,
    device: torch.device | str = "cpu",
) -> DP3Batch:
    """Create synthetic point-cloud observations and action chunks."""
    return {
        "obs": {
            "point_cloud": torch.randn(
                batch_size,
                n_obs_steps,
                num_points,
                point_dim,
                device=device,
            ),
            "agent_pos": torch.randn(batch_size, n_obs_steps, state_dim, device=device),
        },
        "action": torch.randn(batch_size, horizon, action_dim, device=device),
    }
