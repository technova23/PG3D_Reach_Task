from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import TypedDict

import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from pg3d.policies.dp3.goal_markers import DEFAULT_GOAL_MARKER_RADIUS
from pg3d.policies.dp3.modules import (
    ConditionalUnet1D,
    DP3Encoder,
    LowdimMaskGenerator,
    ModuleAttrMixin,
)
from pg3d.policies.dp3.normalizer import LinearNormalizer
from pg3d.policies.dp3.utils import dict_apply

ObsDict = Mapping[str, torch.Tensor]
PolicyOutput = dict[str, torch.Tensor]


class DP3Batch(TypedDict):
    """Minimal batch contract for pg3d-native DP3 smoke training."""

    obs: ObsDict
    action: torch.Tensor


class BasePolicy(ModuleAttrMixin):
    """Common policy interface used by pg3d policy adapters."""

    def predict_action(self, obs_dict: ObsDict) -> PolicyOutput:
        """Return a receding-horizon action chunk for normalized observations."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset policy state for stateful policies; stateless policies do nothing."""
        return None


class SimpleDP3(BasePolicy):
    """Simulation-free DP3 policy core for point-cloud action chunks.

    This class keeps only the model path needed by pg3d: a PointNet-style
    observation encoder, a conditional 1D diffusion U-Net, and DDIM sampling for
    action chunks. It intentionally does not import simulator, benchmark, or
    environment wrappers from upstream DP3.
    """

    def __init__(
        self,
        shape_meta: Mapping[str, Mapping[str, Mapping[str, Sequence[int]]]],
        noise_scheduler: DDIMScheduler | None = None,
        horizon: int = 16,
        n_action_steps: int = 8,
        n_obs_steps: int = 2,
        num_inference_steps: int | None = None,
        obs_as_global_cond: bool = True,
        diffusion_step_embed_dim: int = 128,
        down_dims: Sequence[int] = (128, 256, 384),
        kernel_size: int = 5,
        n_groups: int = 8,
        condition_type: str = "film",
        encoder_output_dim: int = 64,
        use_pc_color: bool = False,
        pointcloud_encoder_cfg: Mapping[str, object] | None = None,
        goal_marker_points: int = 0,
        goal_marker_radius: float = DEFAULT_GOAL_MARKER_RADIUS,
        goal_marker_feature_dim: int = 32,
    ) -> None:
        super().__init__()
        self.condition_type = condition_type
        if use_pc_color:
            raise NotImplementedError("pg3d-native DP3 currently supports XYZ point clouds only")
        self.use_pc_color = use_pc_color
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.goal_marker_points = int(goal_marker_points)
        self.goal_marker_radius = float(goal_marker_radius)
        if self.goal_marker_points < 0:
            raise ValueError("goal_marker_points must be non-negative")
        if self.goal_marker_radius < 0:
            raise ValueError("goal_marker_radius must be non-negative")

        action_shape = tuple(shape_meta["action"]["shape"])
        if len(action_shape) == 1:
            self.action_dim = action_shape[0]
        elif len(action_shape) == 2:
            self.action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")

        obs_shape_meta = shape_meta["obs"]
        obs_shapes = {key: tuple(value["shape"]) for key, value in obs_shape_meta.items()}
        self.obs_encoder = DP3Encoder(
            observation_space=obs_shapes,
            out_channel=encoder_output_dim,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_pc_color=use_pc_color,
            goal_marker_points=self.goal_marker_points,
            goal_marker_feature_dim=goal_marker_feature_dim,
        )
        self.obs_feature_dim = self.obs_encoder.output_shape()
        input_dim = self.action_dim
        global_cond_dim = self.obs_feature_dim * n_obs_steps if obs_as_global_cond else None
        if not obs_as_global_cond:
            input_dim = self.action_dim + self.obs_feature_dim

        self.model = ConditionalUnet1D(
            input_dim=input_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
        )
        self.noise_scheduler = noise_scheduler or DDIMScheduler(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="sample",
        )
        self.noise_scheduler_pc = copy.deepcopy(self.noise_scheduler)
        self.mask_generator = LowdimMaskGenerator(
            action_dim=self.action_dim,
            obs_dim=0 if obs_as_global_cond else self.obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer.identity_for_keys(
            ["action", "point_cloud", "agent_pos", "imagin_robot"]
        )
        self.num_inference_steps = (
            num_inference_steps
            if num_inference_steps is not None
            else self.noise_scheduler.config.num_train_timesteps
        )

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        """Replace fitted normalization statistics without rebuilding the policy."""
        self.normalizer = copy.deepcopy(normalizer)

    def conditional_sample(
        self,
        condition_data: torch.Tensor,
        condition_mask: torch.Tensor,
        global_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run DDIM denoising while clamping known observation/action slots."""
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
        )
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for timestep in self.noise_scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = self.model(
                sample=trajectory,
                timestep=timestep,
                global_cond=global_cond,
            )
            trajectory = self.noise_scheduler.step(model_output, timestep, trajectory).prev_sample
        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def predict_action(self, obs_dict: ObsDict) -> PolicyOutput:
        """Sample an action sequence and return the chunk used by receding horizon."""
        nobs = self.normalizer.normalize(obs_dict)
        assert isinstance(nobs, dict)
        if not self.use_pc_color:
            nobs["point_cloud"] = nobs["point_cloud"][..., :3]

        value = next(iter(nobs.values()))
        batch_size = value.shape[0]
        horizon = self.horizon
        action_dim = self.action_dim
        obs_steps = self.n_obs_steps
        device = self.device
        dtype = self.dtype

        global_cond = None
        if self.obs_as_global_cond:
            # Collapse the observed time window into a single conditioning vector.
            this_nobs = dict_apply(
                nobs,
                lambda x: x[:, :obs_steps, ...].reshape(-1, *x.shape[2:]),
            )
            nobs_features = self.obs_encoder(this_nobs)
            global_cond = nobs_features.reshape(batch_size, -1)
            cond_data = torch.zeros(
                size=(batch_size, horizon, action_dim),
                device=device,
                dtype=dtype,
            )
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # Keep encoded observations inside the denoised trajectory itself.
            this_nobs = dict_apply(nobs, lambda x: x[:, :horizon, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs).reshape(batch_size, horizon, -1)
            cond_data = torch.zeros(
                size=(batch_size, horizon, action_dim + self.obs_feature_dim),
                device=device,
                dtype=dtype,
            )
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :obs_steps, action_dim:] = nobs_features[:, :obs_steps]
            cond_mask[:, :obs_steps, action_dim:] = True

        nsample = self.conditional_sample(cond_data, cond_mask, global_cond=global_cond)
        naction_pred = nsample[..., :action_dim]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)
        start = obs_steps - 1
        end = start + self.n_action_steps
        return {
            "action": action_pred[:, start:end],
            "action_pred": action_pred,
        }

    def compute_loss(self, batch: DP3Batch) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute one diffusion behavior-cloning loss on a batch of action chunks."""
        obs = batch["obs"]
        actions = batch["action"]
        if not isinstance(obs, Mapping) or not isinstance(actions, torch.Tensor):
            raise TypeError("batch must contain obs mapping and action tensor")
        nobs = self.normalizer.normalize(obs)
        assert isinstance(nobs, dict)
        nactions = self.normalizer["action"].normalize(actions)
        if not self.use_pc_color:
            nobs["point_cloud"] = nobs["point_cloud"][..., :3]

        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        trajectory = nactions
        cond_data = trajectory
        global_cond = None

        if self.obs_as_global_cond:
            this_nobs = dict_apply(
                nobs,
                lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:]),
            )
            nobs_features = self.obs_encoder(this_nobs)
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            this_nobs = dict_apply(nobs, lambda x: x[:, :horizon, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs).reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # Mask out fields that are known to the denoiser, then train only on the
        # unobserved action dimensions.
        condition_mask = self.mask_generator(tuple(trajectory.shape))
        noise = torch.randn(trajectory.shape, device=trajectory.device, dtype=trajectory.dtype)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        loss_mask = ~condition_mask
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        pred = self.model(
            sample=noisy_trajectory,
            timestep=timesteps,
            global_cond=global_cond,
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction_type {pred_type!r}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = loss.reshape(loss.shape[0], -1).mean(dim=1).mean()
        return loss, {"bc_loss": float(loss.detach().cpu())}


class DP3(SimpleDP3):
    """Compatibility alias for the full DP3 policy class name.

    The first pg3d slice uses the simple DP3 architecture. We keep the DP3 name
    so callers do not need to know which upstream variant provided the core.
    """
