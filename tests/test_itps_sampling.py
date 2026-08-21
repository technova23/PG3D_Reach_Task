from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from pg3d.policies.dp3 import ITPSNoiseLineage, SimpleDP3
from pg3d.policies.dp3.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from pg3d.policies.dp3.reach_dataset import reach_shape_meta


class _RecordingZeroModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.samples: list[torch.Tensor] = []

    def forward(
        self,
        *,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: torch.Tensor | None,
    ) -> torch.Tensor:
        del timestep, global_cond
        self.samples.append(sample.detach().clone())
        return torch.zeros_like(sample)


class _RecordingScheduler:
    def __init__(self) -> None:
        self.timesteps = torch.tensor([1, 0], dtype=torch.long)
        self.step_calls: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        self.add_noise_calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def set_timesteps(self, count: int) -> None:
        assert count == 2

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        *,
        generator: torch.Generator | None,
    ) -> SimpleNamespace:
        assert generator is not None
        self.step_calls.append(
            (int(timestep), model_output.detach().clone(), sample.detach().clone())
        )
        return SimpleNamespace(
            pred_original_sample=sample + 1.0,
            prev_sample=sample - 10.0,
        )

    def add_noise(
        self,
        original: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        self.add_noise_calls.append((original.detach().clone(), timesteps.detach().clone()))
        return original + torch.zeros_like(noise) + 100.0


def test_itps_repeats_at_same_noise_level_then_advances(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = _tiny_policy()
    model = _RecordingZeroModel()
    scheduler = _RecordingScheduler()
    policy.model = model
    monkeypatch.setattr(policy, "_make_itps_scheduler", lambda: scheduler)

    condition = torch.zeros((1, 2, 2), dtype=torch.float32)
    condition[0, 0, 0] = 7.0
    mask = torch.zeros_like(condition, dtype=torch.bool)
    mask[0, 0, 0] = True

    def energy(sample: torch.Tensor) -> torch.Tensor:
        return sample[..., 1].sum(dim=1)

    original_scheduler = policy.noise_scheduler
    result = policy.stochastic_sample(
        condition,
        mask,
        generator=torch.Generator().manual_seed(4),
        guidance_fn=energy,
        guide_ratio=60.0,
        mcmc_steps=2,
    )

    assert policy.noise_scheduler is original_scheduler
    assert len(model.samples) == 4
    assert len(scheduler.step_calls) == 4
    assert len(scheduler.add_noise_calls) == 2
    assert [call[0] for call in scheduler.step_calls] == [1, 1, 0, 0]
    assert all(sample[0, 0, 0].item() == 7.0 for sample in model.samples)
    torch.testing.assert_close(model.samples[1][..., 1], model.samples[0][..., 1] + 101.0)
    torch.testing.assert_close(model.samples[2][..., 1], model.samples[1][..., 1] - 10.0)
    torch.testing.assert_close(model.samples[3][..., 1], model.samples[2][..., 1] + 101.0)
    assert result[0, 0, 0].item() == 7.0

    for timestep, guided_output, _sample in scheduler.step_calls[:2]:
        assert timestep == 1
        torch.testing.assert_close(guided_output[..., 1], torch.full((1, 2), 60.0))
    for timestep, unguided_output, _sample in scheduler.step_calls[2:]:
        assert timestep == 0
        torch.testing.assert_close(unguided_output, torch.zeros_like(unguided_output))


def test_itps_requires_batched_tensor_energy() -> None:
    policy = _tiny_policy()
    policy.model = _RecordingZeroModel()
    condition = torch.zeros((2, 2, 7))
    mask = torch.zeros_like(condition, dtype=torch.bool)

    with pytest.raises(TypeError, match="torch.Tensor"):
        policy.stochastic_sample(
            condition,
            mask,
            guidance_fn=lambda _sample: 1.0,  # type: ignore[return-value]
            mcmc_steps=1,
        )
    with pytest.raises(ValueError, match="one energy per batch item"):
        policy.stochastic_sample(
            condition,
            mask,
            guidance_fn=lambda sample: sample.sum().reshape(()),
            mcmc_steps=1,
        )


def test_itps_uses_isolated_ddim_scheduler() -> None:
    policy = _tiny_policy()

    scheduler = policy._make_itps_scheduler()

    assert isinstance(scheduler, DDIMScheduler)
    assert scheduler is not policy.noise_scheduler


def test_itps_is_reproducible_and_matches_one_step_ddim() -> None:
    policy = _tiny_policy()
    policy.set_ddim_eta(1.0)
    policy.model = _RecordingZeroModel()
    condition = torch.zeros((1, 2, 7))
    mask = torch.zeros_like(condition, dtype=torch.bool)

    actual = policy.stochastic_sample(
        condition,
        mask,
        generator=torch.Generator().manual_seed(19),
        guidance_fn=None,
        mcmc_steps=4,
    )
    repeated = policy.stochastic_sample(
        condition,
        mask,
        generator=torch.Generator().manual_seed(19),
        guidance_fn=None,
        mcmc_steps=1,
    )

    generator = torch.Generator().manual_seed(19)
    expected = torch.randn(condition.shape, generator=generator)
    scheduler = policy._make_itps_scheduler()
    scheduler.set_timesteps(policy.num_inference_steps)
    for timestep in scheduler.timesteps:
        expected = scheduler.step(
            torch.zeros_like(expected),
            timestep,
            expected,
            generator=generator,
        ).prev_sample

    torch.testing.assert_close(actual, repeated)
    torch.testing.assert_close(actual, expected)


def test_explicit_itps_noise_lineage_replays_every_stochastic_draw() -> None:
    policy = _tiny_policy()
    policy.model = _RecordingZeroModel()
    condition = torch.zeros((1, 2, 7))
    mask = torch.zeros_like(condition, dtype=torch.bool)
    scheduler = policy._make_itps_scheduler()
    scheduler.set_timesteps(policy.num_inference_steps)
    lineage = ITPSNoiseLineage.derive(
        candidate_seed=19,
        diffusion_timesteps=(int(timestep) for timestep in scheduler.timesteps),
        inner_steps=2,
        root_identity="test",
    )

    def energy(sample: torch.Tensor) -> torch.Tensor:
        return sample[..., 1].sum(dim=1)

    first = policy.stochastic_sample(
        condition,
        mask,
        guidance_fn=energy,
        mcmc_steps=2,
        noise_lineage=lineage,
    )
    second = policy.stochastic_sample(
        condition,
        mask,
        guidance_fn=energy,
        mcmc_steps=2,
        noise_lineage=lineage,
    )

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert lineage.initial_noise.seed != lineage.candidate_seed
    assert len(lineage.inner_renoising) == policy.num_inference_steps
    assert lineage.to_json()["schema_version"] == "pg3d.itps_noise_lineage.v1"


def test_itps_noise_lineage_cannot_share_a_stateful_generator() -> None:
    policy = _tiny_policy()
    condition = torch.zeros((1, 2, 7))
    mask = torch.zeros_like(condition, dtype=torch.bool)
    lineage = policy.make_itps_noise_lineage(candidate_seed=7, mcmc_steps=1)

    with pytest.raises(ValueError, match="mutually exclusive"):
        policy.stochastic_sample(
            condition,
            mask,
            generator=torch.Generator().manual_seed(7),
            noise_lineage=lineage,
        )


def test_ordinary_ddim_eta_one_is_reproducible_and_changes_sample() -> None:
    policy = _tiny_policy()
    policy.model = _RecordingZeroModel()
    condition = torch.zeros((1, 2, 7))
    mask = torch.zeros_like(condition, dtype=torch.bool)

    policy.set_ddim_eta(0.0)
    eta_zero = policy.conditional_sample(
        condition,
        mask,
        generator=torch.Generator().manual_seed(23),
    )
    policy.set_ddim_eta(1.0)
    eta_one = policy.conditional_sample(
        condition,
        mask,
        generator=torch.Generator().manual_seed(23),
    )
    repeated = policy.conditional_sample(
        condition,
        mask,
        generator=torch.Generator().manual_seed(23),
    )

    torch.testing.assert_close(eta_one, repeated)
    assert not torch.equal(eta_zero, eta_one)


def test_ddim_eta_validation() -> None:
    policy = _tiny_policy()

    for value in (-0.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="DDIM eta"):
            policy.set_ddim_eta(value)


def test_predict_action_itps_returns_standard_slice(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = _tiny_policy()
    policy.set_normalizer(
        LinearNormalizer(
            {
                "action": SingleFieldLinearNormalizer.create_manual(
                    scale=torch.full((7,), 2.0),
                    offset=torch.ones(7),
                )
            }
        )
    )
    normalized = torch.arange(28, dtype=torch.float32).reshape(1, 4, 7)
    condition = torch.zeros_like(normalized)
    mask = torch.zeros_like(condition, dtype=torch.bool)
    monkeypatch.setattr(policy, "_build_conditioning", lambda _obs: (condition, mask, None))
    monkeypatch.setattr(policy, "stochastic_sample", lambda *_args, **_kwargs: normalized)

    output = policy.predict_action_itps({}, guidance_fn=lambda sample: sample[:, 0, 0])
    expected = (normalized - 1.0) / 2.0

    torch.testing.assert_close(output["action_pred"], expected)
    torch.testing.assert_close(output["action"], expected[:, 1:2])


def test_itps_validates_configuration() -> None:
    policy = _tiny_policy()
    condition = torch.zeros((1, 2, 7))
    mask = torch.zeros_like(condition, dtype=torch.bool)

    with pytest.raises(ValueError, match="mcmc_steps"):
        policy.stochastic_sample(condition, mask, mcmc_steps=0)
    with pytest.raises(ValueError, match="guide_ratio"):
        policy.stochastic_sample(condition, mask, guide_ratio=-1.0)

    policy.noise_scheduler.config.prediction_type = "sample"
    with pytest.raises(ValueError, match="prediction_type='epsilon'"):
        policy.stochastic_sample(condition, mask)


def _tiny_policy() -> SimpleDP3:
    return SimpleDP3(
        shape_meta=reach_shape_meta(num_points=4, state_dim=9, action_dim=7),
        horizon=4,
        n_obs_steps=2,
        n_action_steps=1,
        num_train_timesteps=4,
        num_inference_steps=2,
        encoder_output_dim=16,
        diffusion_step_embed_dim=32,
        down_dims=(32, 64),
        kernel_size=3,
        n_groups=8,
        pointcloud_encoder_cfg={
            "out_channels": 16,
            "use_layernorm": True,
            "final_norm": "layernorm",
        },
    )
