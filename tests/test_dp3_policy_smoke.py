from __future__ import annotations

import torch

from pg3d.policies.dp3 import make_synthetic_batch, make_tiny_policy


def test_tiny_dp3_policy_instantiates() -> None:
    policy = make_tiny_policy()

    assert sum(param.numel() for param in policy.parameters()) > 0
    assert policy.action_dim == 7
    assert policy.obs_feature_dim > 0


def test_tiny_dp3_policy_supports_ordered_goal_marker_branch() -> None:
    policy = make_tiny_policy(goal_marker_points=4)

    assert policy.goal_marker_points == 4
    assert policy.obs_encoder.goal_marker_mlp is not None
    assert policy.obs_feature_dim > 80


def test_tiny_dp3_predict_action_cpu() -> None:
    torch.manual_seed(0)
    policy = make_tiny_policy()
    batch = make_synthetic_batch()

    result = policy.predict_action(batch["obs"])

    assert result["action"].shape == (2, 2, 7)
    assert result["action_pred"].shape == (2, 4, 7)
    assert torch.isfinite(result["action"]).all()
    assert torch.isfinite(result["action_pred"]).all()


def test_tiny_dp3_compute_loss_and_optimizer_step_cpu() -> None:
    torch.manual_seed(0)
    policy = make_tiny_policy()
    batch = make_synthetic_batch()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)

    loss, loss_dict = policy.compute_loss(batch)
    loss.backward()
    grad_norm = sum(
        param.grad.detach().abs().sum().item()
        for param in policy.parameters()
        if param.grad is not None
    )
    optimizer.step()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss_dict["bc_loss"] >= 0.0
    assert grad_norm > 0.0
