from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence

import einops
import torch
from torch import nn

logger = logging.getLogger(__name__)


class ModuleAttrMixin(nn.Module):
    """Expose module device and dtype from parameters.

    DP3 utilities frequently need to allocate tensors next to a module. The
    zero-sized parameter keeps the properties valid even for modules with no
    trainable layers of their own.
    """

    def __init__(self) -> None:
        super().__init__()
        self._dummy_variable = nn.Parameter(torch.empty(0))

    @property
    def device(self) -> torch.device:
        return next(iter(self.parameters())).device

    @property
    def dtype(self) -> torch.dtype:
        return next(iter(self.parameters())).dtype


class SinusoidalPosEmb(nn.Module):
    """Diffusion timestep embedding used by the conditional U-Net."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed a batch of scalar timesteps into sinusoidal features."""
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Downsample1d(nn.Module):
    """Halve the temporal resolution of a 1D feature sequence."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    """Double the temporal resolution of a 1D feature sequence."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """Conv1d, GroupNorm, and Mish block used throughout the diffusion U-Net."""

    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        kernel_size: int,
        n_groups: int = 8,
    ) -> None:
        super().__init__()
        # Tiny smoke models can have fewer channels than the upstream group
        # count, so choose the largest divisor that keeps GroupNorm valid.
        groups = min(n_groups, out_channels)
        while out_channels % groups != 0:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """Residual 1D block conditioned by FiLM timestep/global features."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        condition_type: str = "film",
    ) -> None:
        super().__init__()
        self.condition_type = condition_type
        self.out_channels = out_channels
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )
        if condition_type != "film":
            raise NotImplementedError(
                f"condition_type {condition_type!r} is not in the pg3d DP3 slice"
            )
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels * 2),
        )
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the residual block to ``[batch, channels, horizon]`` features."""
        out = self.blocks[0](x)
        if cond is not None:
            embed = self.cond_encoder(cond)
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0]
            bias = embed[:, 1]
            out = scale * out + bias
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    """Conditional 1D U-Net that predicts diffusion targets for action chunks."""

    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int | None = None,
        diffusion_step_embed_dim: int = 256,
        down_dims: Sequence[int] = (256, 512, 1024),
        kernel_size: int = 3,
        n_groups: int = 8,
        condition_type: str = "film",
    ) -> None:
        super().__init__()
        all_dims = [input_dim, *list(down_dims)]
        start_dim = down_dims[0]
        cond_dim = diffusion_step_embed_dim + (global_cond_dim or 0)
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )
        in_out = list(zip(all_dims[:-1], all_dims[1:], strict=True))
        self.down_modules = nn.ModuleList()
        for idx, (dim_in, dim_out) in enumerate(in_out):
            is_last = idx >= len(in_out) - 1
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            condition_type=condition_type,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    condition_type=condition_type,
                )
            ]
        )
        self.up_modules = nn.ModuleList()
        for idx, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = idx >= len(in_out) - 1
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            condition_type=condition_type,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )
        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, kernel_size=1),
        )
        logger.info("ConditionalUnet1D parameters: %d", sum(p.numel() for p in self.parameters()))

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float | int,
        global_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict a denoising target for ``sample`` shaped ``[B, H, D]``."""
        x = einops.rearrange(sample, "b h t -> b t h")
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=x.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(x.device)
        timesteps = timesteps.expand(x.shape[0])
        timestep_embed = self.diffusion_step_encoder(timesteps)
        cond = (
            timestep_embed
            if global_cond is None
            else torch.cat([timestep_embed, global_cond], dim=-1)
        )

        h: list[torch.Tensor] = []
        for resnet, downsample in self.down_modules:
            x = resnet(x, cond)
            h.append(x)
            x = downsample(x)
        for mid_module in self.mid_modules:
            x = mid_module(x, cond)
        for resnet, upsample in self.up_modules:
            skip = h.pop()
            if x.shape[-1] != skip.shape[-1]:
                # Odd horizons can round differently across down/up sampling.
                x = torch.nn.functional.interpolate(x, size=skip.shape[-1], mode="nearest")
            x = torch.cat((x, skip), dim=1)
            x = resnet(x, cond)
            x = upsample(x)
        x = self.final_conv(x)
        return einops.rearrange(x, "b t h -> b h t")


def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: Sequence[int],
    activation_fn: type[nn.Module] = nn.ReLU,
) -> list[nn.Module]:
    """Create a small feed-forward MLP as a flat module list."""
    if len(net_arch) > 0:
        modules: list[nn.Module] = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []
    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())
    last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
    modules.append(nn.Linear(last_layer_dim, output_dim))
    return modules


class PointNetEncoderXYZ(nn.Module):
    """Minimal PointNet encoder for XYZ point clouds.

    The initial pg3d DP3 slice deliberately supports only XYZ coordinates. Color
    and richer point features can be added once ManiSkill observation conventions
    are fixed.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1024,
        use_layernorm: bool = False,
        final_norm: str = "none",
    ) -> None:
        super().__init__()
        if in_channels != 3:
            raise ValueError(f"PointNetEncoderXYZ expects 3 channels, got {in_channels}")
        block_channel = [64, 128, 256]
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )
        if final_norm == "layernorm":
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels),
            )
        elif final_norm == "none":
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm {final_norm!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ``[batch, points, 3]`` into one feature vector per batch item."""
        x = self.mlp(x)
        x = torch.max(x, dim=1)[0]
        return self.final_projection(x)


class DP3Encoder(nn.Module):
    """Encode pg3d DP3 observations into a single conditioning vector."""

    def __init__(
        self,
        observation_space: Mapping[str, Sequence[int]],
        out_channel: int = 256,
        state_mlp_size: Sequence[int] = (64, 64),
        use_pc_color: bool = False,
        pointcloud_encoder_cfg: Mapping[str, object] | None = None,
        goal_marker_points: int = 0,
        goal_marker_feature_dim: int = 32,
    ) -> None:
        super().__init__()
        self.point_cloud_key = "point_cloud"
        self.state_key = "agent_pos"
        self.family_key = "trajectory_family_onehot"
        self.use_pc_color = use_pc_color
        self.point_cloud_shape = tuple(observation_space[self.point_cloud_key])
        self.state_shape = tuple(observation_space[self.state_key])
        self.family_shape = tuple(observation_space[self.family_key]) if self.family_key in observation_space else None
        self.goal_marker_points = int(goal_marker_points)
        if self.goal_marker_points < 0:
            raise ValueError("goal_marker_points must be non-negative")
        if self.goal_marker_points and self.goal_marker_points >= self.point_cloud_shape[0]:
            raise ValueError(
                "goal_marker_points must be smaller than the point-cloud point count"
            )
        pointcloud_encoder_cfg = dict(pointcloud_encoder_cfg or {})
        pointcloud_encoder_cfg["in_channels"] = 6 if use_pc_color else 3
        pointcloud_encoder_cfg.setdefault("out_channels", out_channel)
        self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
        if self.goal_marker_points:
            marker_input_dim = self.goal_marker_points * 3
            self.goal_marker_mlp = nn.Sequential(
                nn.Linear(marker_input_dim, goal_marker_feature_dim),
                nn.ReLU(),
                nn.Linear(goal_marker_feature_dim, goal_marker_feature_dim),
                nn.ReLU(),
            )
        else:
            self.goal_marker_mlp = None

        if len(state_mlp_size) == 0:
            raise ValueError("state_mlp_size must not be empty")
        output_dim = state_mlp_size[-1]
        net_arch = state_mlp_size[:-1]
        self.state_mlp = nn.Sequential(
            *create_mlp(self.state_shape[0], output_dim, net_arch, nn.ReLU)
        )
        if self.family_shape is not None:
            if len(self.family_shape) != 1:
                raise ValueError(
                    f"trajectory_family_onehot must have shape [F], got {self.family_shape}"
                )
            self.family_mlp = nn.Sequential(
                nn.Linear(self.family_shape[0], output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim),
                nn.ReLU(),
            )
        else:
            self.family_mlp = None
        self.n_output_channels = (
            out_channel
            + output_dim
            + (output_dim if self.family_shape is not None else 0)
            + (goal_marker_feature_dim if self.goal_marker_points else 0)
        )

    def forward(self, observations: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Encode ``point_cloud`` and ``agent_pos`` observation tensors."""
        points = observations[self.point_cloud_key]
        if not self.use_pc_color:
            points = points[..., :3]
        if len(points.shape) != 3:
            raise ValueError(f"point_cloud must be [B, N, C], got {tuple(points.shape)}")
        features = []
        if self.goal_marker_points:
            scene_points = points[:, : -self.goal_marker_points]
            marker_points = points[:, -self.goal_marker_points :]
            features.append(self.extractor(scene_points))
            assert self.goal_marker_mlp is not None
            features.append(self.goal_marker_mlp(marker_points.reshape(marker_points.shape[0], -1)))
        else:
            features.append(self.extractor(points))
        state_feat = self.state_mlp(observations[self.state_key])
        features.append(state_feat)
        if self.family_mlp is not None:
    		if self.family_key not in observations:
        		raise KeyError(
           		      "observation is missing trajectory_family_onehot required by shape_meta"
       		)

    		family_tensor = observations[self.family_key].to(
        		next(self.family_mlp.parameters()).device
    		)

    		features.append(self.family_mlp(family_tensor))
        return torch.cat(features, dim=-1)

    def output_shape(self) -> int:
        """Return the encoder feature width."""
        return self.n_output_channels


class LowdimMaskGenerator(ModuleAttrMixin):
    """Build diffusion conditioning masks for low-dimensional trajectories."""

    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        max_n_obs_steps: int = 2,
        fix_obs_steps: bool = True,
        action_visible: bool = False,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.max_n_obs_steps = max_n_obs_steps
        self.fix_obs_steps = fix_obs_steps
        self.action_visible = action_visible

    @torch.no_grad()
    def forward(self, shape: tuple[int, int, int], seed: int | None = None) -> torch.Tensor:
        """Return a boolean mask of known fields for ``[B, H, D]`` trajectories."""
        device = self.device
        batch_size, horizon, dim = shape
        if dim != self.action_dim + self.obs_dim:
            raise ValueError(f"Expected dim {self.action_dim + self.obs_dim}, got {dim}")
        rng = torch.Generator(device=device)
        if seed is not None:
            rng.manual_seed(seed)
        dim_mask = torch.zeros(size=shape, dtype=torch.bool, device=device)
        is_action_dim = dim_mask.clone()
        is_action_dim[..., : self.action_dim] = True
        is_obs_dim = ~is_action_dim
        if self.fix_obs_steps:
            obs_steps = torch.full((batch_size,), self.max_n_obs_steps, device=device)
        else:
            obs_steps = torch.randint(
                low=1,
                high=self.max_n_obs_steps + 1,
                size=(batch_size,),
                generator=rng,
                device=device,
            )
        steps = (
            torch.arange(0, horizon, device=device).reshape(1, horizon).expand(batch_size, horizon)
        )
        obs_mask = (
            (steps.T < obs_steps).T.reshape(batch_size, horizon, 1).expand(batch_size, horizon, dim)
        )
        mask = obs_mask & is_obs_dim
        if self.action_visible:
            action_steps = torch.maximum(
                obs_steps - 1,
                torch.tensor(0, dtype=obs_steps.dtype, device=obs_steps.device),
            )
            action_mask = (
                (steps.T < action_steps)
                .T.reshape(batch_size, horizon, 1)
                .expand(batch_size, horizon, dim)
            )
            mask = mask | (action_mask & is_action_dim)
        return mask


class EMAModel:
    """Exponential moving average wrapper for model weights."""

    def __init__(
        self,
        model: nn.Module,
        update_after_step: int = 0,
        inv_gamma: float = 1.0,
        power: float = 0.75,
        min_value: float = 0.0,
        max_value: float = 0.9999,
    ) -> None:
        self.averaged_model = model
        self.averaged_model.eval()
        self.averaged_model.requires_grad_(False)
        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value
        self.decay = 0.0
        self.optimization_step = 0

    def get_decay(self, optimization_step: int) -> float:
        """Return the EMA decay for the given optimizer step."""
        step = max(0, optimization_step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power
        if step <= 0:
            return 0.0
        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, new_model: nn.Module) -> None:
        """Update averaged parameters from ``new_model`` in place."""
        self.decay = self.get_decay(self.optimization_step)
        for param, ema_param in zip(
            new_model.parameters(),
            self.averaged_model.parameters(),
            strict=True,
        ):
            if not param.requires_grad:
                ema_param.copy_(param.to(dtype=ema_param.dtype).data)
            else:
                ema_param.mul_(self.decay)
                ema_param.add_(param.data.to(dtype=ema_param.dtype), alpha=1 - self.decay)
        self.optimization_step += 1
