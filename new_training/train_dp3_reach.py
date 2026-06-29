from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections import deque
from pathlib import Path
from typing import Any

import torch

from pg3d.envs.maniskill_adapter.dataset import load_reach_metadata
from pg3d.policies.dp3 import ReachDatasetConfig, ReachSequenceDataset, SimpleDP3
from pg3d.policies.dp3.checkpoint import (
    checkpoint_path_for_step,
    save_reach_policy_checkpoint,
    should_save_checkpoint,
)
from pg3d.policies.dp3.goal_markers import (
    DEFAULT_GOAL_MARKER_POINTS,
    DEFAULT_GOAL_MARKER_RADIUS,
)
from pg3d.policies.dp3.modules import EMAModel
from pg3d.policies.dp3.reach_dataset import reach_shape_meta
from pg3d.policies.dp3.utils import dict_apply
from pg3d.utils.devices import select_device
from pg3d.utils.serialization import jsonable


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    train_dataset = ReachSequenceDataset(
        ReachDatasetConfig(
            dataset_path=args.dataset,
            horizon=args.horizon,
            n_obs_steps=args.n_obs_steps,
            pad_after=args.pad_after,
            val_ratio=args.val_ratio,
            seed=args.seed,
            max_train_episodes=args.max_train_episodes,
            goal_marker_points=args.goal_marker_points,
            goal_marker_radius=args.goal_marker_radius,
            normalizer_max_steps=args.normalizer_max_steps,
            use_trajectory_family_conditioning=args.use_trajectory_family_conditioning,
        ),
        split="train",
    )
    if len(train_dataset) == 0:
        raise RuntimeError("training dataset has no sequences")
    val_dataset = (
        train_dataset.get_validation_dataset()
        if args.val_ratio > 0.0 and train_dataset.num_episodes > 1
        else None
    )
    if val_dataset is not None and len(val_dataset) == 0:
        val_dataset = None

    pin_memory = args.pin_memory if args.pin_memory is not None else device.type == "cuda"
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = (
        torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
            pin_memory=pin_memory,
            persistent_workers=args.num_workers > 0,
        )
        if val_dataset is not None
        else None
    )

    policy_kwargs = _policy_kwargs(args, shape_meta=train_dataset.shape_meta)
    policy = SimpleDP3(**policy_kwargs)
    policy.set_normalizer(train_dataset.get_normalizer())
    policy.to(device)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    scheduler = _build_lr_scheduler(optimizer, args)
    ema = (
        EMAModel(copy.deepcopy(policy), max_value=args.ema_max_value)
        if args.use_ema
        else None
    )
    run = _init_wandb(
        args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        policy_kwargs=policy_kwargs,
    )

    data_iter = iter(train_loader)
    loss_window: deque[float] = deque(maxlen=args.loss_window)
    best_val_loss: float | None = None
    latest_val_metrics: dict[str, float] = {}
    checkpoint_paths: list[Path] = []
    rollout_attempted_steps: set[int] = set()
    for step in range(1, args.max_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        batch = _batch_to(batch, device)
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        loss, loss_dict = policy.compute_loss(batch)
        loss.backward()
        grad_norm_before = _grad_norm(policy)
        grad_norm_after = _clip_gradients(policy, args.grad_clip_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.step(policy)

        bc_loss = float(loss_dict["bc_loss"])
        loss_window.append(bc_loss)
        metrics = {
            "train/bc_loss": bc_loss,
            "train/bc_loss_rolling": float(sum(loss_window) / len(loss_window)),
            "train/grad_norm_before_clip": grad_norm_before,
            "train/grad_norm_after_clip": grad_norm_after,
            "train/lr": float(optimizer.param_groups[0]["lr"]),
            "train/action_rms": float(batch["action"].detach().pow(2).mean().sqrt().cpu()),
            "train/point_cloud_mean": float(batch["obs"]["point_cloud"].detach().mean().cpu()),
            "train/step": step,
        }

        if val_loader is not None and (step % args.val_every == 0 or step == args.max_steps):
            eval_policy = ema.averaged_model if ema is not None else policy
            latest_val_metrics = _evaluate_policy(
                eval_policy,
                val_loader,
                device=device,
                max_batches=args.max_val_batches,
            )
            metrics.update(latest_val_metrics)
            val_loss = latest_val_metrics.get("val/bc_loss")
            if val_loss is not None and (best_val_loss is None or val_loss < best_val_loss):
                best_val_loss = val_loss

        print(
            f"step={step} bc_loss={metrics['train/bc_loss']:.6f} "
            f"rolling={metrics['train/bc_loss_rolling']:.6f} "
            f"grad={metrics['train/grad_norm_after_clip']:.6f} "
            f"lr={metrics['train/lr']:.2e}"
        )
        if run is not None:
            _wandb_log(
                run,
                metrics,
                batch=batch,
                policy=policy,
                step=step,
                log_histograms=args.log_histograms and step % args.histogram_every == 0,
            )

        if args.checkpoint_dir is not None and should_save_checkpoint(step, args.checkpoint_every):
            checkpoint_path = checkpoint_path_for_step(args.checkpoint_dir, step)
            save_reach_policy_checkpoint(
                checkpoint_path,
                policy=policy,
                ema_policy=ema.averaged_model if ema is not None else None,
                optimizer=optimizer,
                scheduler=scheduler,
                policy_kwargs=policy_kwargs,
                args=args,
                step=step,
                best_val_loss=best_val_loss,
            )
            checkpoint_paths.append(checkpoint_path)
            print(f"saved checkpoint: {checkpoint_path}")
            _maybe_log_checkpoint_rollouts(
                run,
                args,
                train_dataset=train_dataset,
                policy=ema.averaged_model if ema is not None else policy,
                device=device,
                step=step,
                rollout_attempted_steps=rollout_attempted_steps,
            )

    final_checkpoint_path: Path | None = None
    if args.checkpoint_dir is not None:
        final_checkpoint_path = checkpoint_path_for_step(
            args.checkpoint_dir,
            args.max_steps,
            final=True,
        )
        save_reach_policy_checkpoint(
            final_checkpoint_path,
            policy=policy,
            ema_policy=ema.averaged_model if ema is not None else None,
            optimizer=optimizer,
            scheduler=scheduler,
            policy_kwargs=policy_kwargs,
            args=args,
            step=args.max_steps,
            best_val_loss=best_val_loss,
        )
        checkpoint_paths.append(final_checkpoint_path)
        print(f"saved final checkpoint: {final_checkpoint_path}")
        _maybe_log_checkpoint_rollouts(
            run,
            args,
            train_dataset=train_dataset,
            policy=ema.averaged_model if ema is not None else policy,
            device=device,
            step=args.max_steps,
            rollout_attempted_steps=rollout_attempted_steps,
        )
    if run is not None:
        run.finish()
    print(
        "summary: "
        + json.dumps(
            {
                "dataset": str(args.dataset),
                "num_train_sequences": len(train_dataset),
                "num_val_sequences": len(val_dataset) if val_dataset is not None else 0,
                "num_episodes": train_dataset.num_episodes,
                "max_steps": args.max_steps,
                "best_val_loss": best_val_loss,
                "latest_val_metrics": latest_val_metrics,
                "checkpoints": [str(path) for path in checkpoint_paths],
                "final_checkpoint": str(final_checkpoint_path) if final_checkpoint_path else None,
                "device": str(device),
            },
            sort_keys=True,
        )
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a pg3d-native DP3 training loop on a reach Zarr dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("artifacts/reach-dataset-smoke/pg3d-reach-smoke.zarr"),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--n-obs-steps", type=int, default=2)
    parser.add_argument("--n-action-steps", type=int, default=8)
    parser.add_argument("--pad-after", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--max-val-batches", type=int, default=4)
    parser.add_argument("--max-train-episodes", type=int, default=None)
    parser.add_argument("--goal-marker-points", type=int, default=DEFAULT_GOAL_MARKER_POINTS)
    parser.add_argument("--goal-marker-radius", type=float, default=DEFAULT_GOAL_MARKER_RADIUS)
    parser.add_argument(
        "--use-trajectory-family-conditioning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "include /data/trajectory_family_onehot in DP3 observations when the "
            "dataset provides it"
        ),
    )
    parser.add_argument(
        "--normalizer-max-steps",
        type=int,
        default=4096,
        help=(
            "maximum evenly spaced Zarr timesteps used to fit DP3 normalizers; set to 0 "
            "to use all timesteps"
        ),
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.95)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--lr-scheduler", choices=["none", "cosine"], default="cosine")
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--min-lr-scale", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--loss-window", type=int, default=100)
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ema-max-value", type=float, default=0.9999)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--encoder-output-dim", type=int, default=64)
    parser.add_argument("--diffusion-step-embed-dim", type=int, default=128)
    parser.add_argument("--down-dims", type=int, nargs="+", default=[128, 256, 384])
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--n-groups", type=int, default=8)
    parser.add_argument(
        "--wandb-mode",
        choices=["disabled", "offline", "online"],
        default="disabled",
    )
    parser.add_argument("--wandb-project", default="pg3d")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-required", action="store_true")
    parser.add_argument("--log-histograms", action="store_true")
    parser.add_argument("--histogram-every", type=int, default=100)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument(
        "--checkpoint-rollout-videos",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--checkpoint-rollout-count", type=int, default=5)
    parser.add_argument("--checkpoint-rollout-dataset", type=Path, default=None)
    parser.add_argument("--checkpoint-rollout-selection-seed", type=int, default=None)
    parser.add_argument("--checkpoint-rollout-max-steps", type=int, default=50)
    parser.add_argument("--checkpoint-rollout-post-success-steps", type=int, default=8)
    parser.add_argument("--checkpoint-rollout-seed-start", type=int, default=10000)
    parser.add_argument("--checkpoint-rollout-fps", type=int, default=10)
    args = parser.parse_args(argv)
    if args.pad_after is None:
        args.pad_after = args.n_action_steps - 1
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.n_action_steps <= 0:
        raise ValueError("--n-action-steps must be positive")
    if args.pad_after < 0:
        raise ValueError("--pad-after must be non-negative")
    if args.val_every <= 0:
        raise ValueError("--val-every must be positive")
    if args.max_val_batches <= 0:
        raise ValueError("--max-val-batches must be positive")
    if args.goal_marker_points < 0:
        raise ValueError("--goal-marker-points must be non-negative")
    if args.goal_marker_radius < 0:
        raise ValueError("--goal-marker-radius must be non-negative")
    if args.normalizer_max_steps < 0:
        raise ValueError("--normalizer-max-steps must be non-negative")
    if args.normalizer_max_steps == 0:
        args.normalizer_max_steps = None
    if args.loss_window <= 0:
        raise ValueError("--loss-window must be positive")
    if args.histogram_every <= 0:
        raise ValueError("--histogram-every must be positive")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be non-negative")
    if args.checkpoint_rollout_count <= 0:
        raise ValueError("--checkpoint-rollout-count must be positive")
    if args.checkpoint_rollout_selection_seed is None:
        args.checkpoint_rollout_selection_seed = args.seed
    if args.checkpoint_rollout_max_steps <= 0:
        raise ValueError("--checkpoint-rollout-max-steps must be positive")
    if args.checkpoint_rollout_post_success_steps < 0:
        raise ValueError("--checkpoint-rollout-post-success-steps must be non-negative")
    if args.checkpoint_rollout_fps <= 0:
        raise ValueError("--checkpoint-rollout-fps must be positive")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if args.grad_clip_norm < 0:
        raise ValueError("--grad-clip-norm must be non-negative")
    return args


def _policy_kwargs(
    args: argparse.Namespace,
    *,
    shape_meta: dict[str, dict[str, dict[str, list[int]]]] | None = None,
) -> dict[str, Any]:
    shape_meta = shape_meta or reach_shape_meta()
    return {
        "shape_meta": shape_meta,
        "horizon": args.horizon,
        "n_obs_steps": args.n_obs_steps,
        "n_action_steps": args.n_action_steps,
        "num_inference_steps": args.num_inference_steps,
        "encoder_output_dim": args.encoder_output_dim,
        "diffusion_step_embed_dim": args.diffusion_step_embed_dim,
        "down_dims": tuple(args.down_dims),
        "kernel_size": args.kernel_size,
        "n_groups": args.n_groups,
        "goal_marker_points": args.goal_marker_points,
        "goal_marker_radius": args.goal_marker_radius,
        "pointcloud_encoder_cfg": {
            "out_channels": args.encoder_output_dim,
            "use_layernorm": True,
            "final_norm": "layernorm",
        },
    }


def _batch_to(batch: Any, device: torch.device) -> Any:
    return dict_apply(batch, lambda tensor: tensor.to(device=device, dtype=torch.float32))


def _grad_norm(policy: torch.nn.Module) -> float:
    total = 0.0
    for param in policy.parameters():
        if param.grad is None:
            continue
        total += float(param.grad.detach().pow(2).sum().cpu())
    return math.sqrt(total)


def _clip_gradients(policy: torch.nn.Module, max_norm: float) -> float:
    if max_norm > 0:
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=max_norm)
    return _grad_norm(policy)


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> torch.optim.lr_scheduler.LambdaLR | None:
    if args.lr_scheduler == "none":
        return None
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_scale_for_step(
            step,
            warmup_steps=args.warmup_steps,
            total_steps=args.max_steps,
            min_lr_scale=args.min_lr_scale,
        ),
    )


def lr_scale_for_step(
    step: int,
    *,
    warmup_steps: int,
    total_steps: int,
    min_lr_scale: float,
) -> float:
    """Linear warmup followed by cosine decay."""
    if warmup_steps > 0 and step < warmup_steps:
        return max(float(step + 1) / float(warmup_steps), 1e-8)
    decay_steps = max(total_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_scale + (1.0 - min_lr_scale) * cosine


@torch.no_grad()
def _evaluate_policy(
    policy: SimpleDP3,
    dataloader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    max_batches: int,
) -> dict[str, float]:
    was_training = policy.training
    policy.eval()
    total_loss = 0.0
    total_action_mse = 0.0
    action_mse_dim: torch.Tensor | None = None
    batches = 0
    for batch in dataloader:
        batch = _batch_to(batch, device)
        loss, _loss_dict = policy.compute_loss(batch)
        output = policy.predict_action(batch["obs"])
        target = batch["action"][
            :,
            policy.n_obs_steps - 1 : policy.n_obs_steps - 1 + policy.n_action_steps,
        ]
        error = output["action"] - target
        total_loss += float(loss.detach().cpu())
        total_action_mse += float(error.pow(2).mean().detach().cpu())
        per_dim = error.pow(2).mean(dim=(0, 1)).detach().cpu()
        action_mse_dim = per_dim if action_mse_dim is None else action_mse_dim + per_dim
        batches += 1
        if batches >= max_batches:
            break
    if was_training:
        policy.train()
    if batches == 0:
        return {}
    assert action_mse_dim is not None
    metrics = {
        "val/bc_loss": total_loss / batches,
        "val/action_mse": total_action_mse / batches,
    }
    for dim_idx, value in enumerate(action_mse_dim / batches):
        metrics[f"val/action_mse_dim_{dim_idx}"] = float(value)
    return metrics


def _init_wandb(
    args: argparse.Namespace,
    *,
    train_dataset: ReachSequenceDataset,
    val_dataset: ReachSequenceDataset | None,
    policy_kwargs: dict[str, Any],
) -> Any | None:
    if args.wandb_mode == "disabled":
        return None
    import wandb

    try:
        return wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            mode=args.wandb_mode,
            config={
                "dataset": str(args.dataset),
                "num_train_sequences": len(train_dataset),
                "num_val_sequences": len(val_dataset) if val_dataset is not None else 0,
                "num_episodes": train_dataset.num_episodes,
                "dataset_metadata": train_dataset.metadata,
                "policy": jsonable(policy_kwargs),
                "training": jsonable(vars(args)),
                "command": "scripts/train_dp3_reach.py",
            },
        )
    except Exception as exc:
        if args.wandb_required:
            raise
        print(
            f"warning: W&B init failed, continuing without W&B: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return None


def _wandb_log(
    run: Any,
    metrics: dict[str, float],
    *,
    batch: Any,
    policy: SimpleDP3,
    step: int,
    log_histograms: bool,
) -> None:
    if log_histograms:
        import wandb

        was_training = policy.training
        policy.eval()
        with torch.no_grad():
            output = policy.predict_action(batch["obs"])
        if was_training:
            policy.train()

        output_action = output["action"].detach().cpu()
        output_action_pred = output["action_pred"].detach().cpu()
        target_action = batch["action"].detach().cpu()
        metrics = {
            **metrics,
            "viz/action_hist": wandb.Histogram(target_action.numpy()),
            "viz/output_action_chunk_hist": wandb.Histogram(output_action.numpy()),
            "viz/output_action_pred_full_hist": wandb.Histogram(output_action_pred.numpy()),
            "viz/output_action_chunk_std": float(output_action.std(unbiased=False)),
            "viz/output_action_chunk_rms": float(output_action.pow(2).mean().sqrt()),
            "viz/agent_pos_hist": wandb.Histogram(
                batch["obs"]["agent_pos"].detach().cpu().numpy()
            ),
            "viz/point_cloud_hist": wandb.Histogram(
                batch["obs"]["point_cloud"].detach().cpu().numpy()
            ),
        }
        family = batch["obs"].get("trajectory_family_onehot")
        if family is not None:
            metrics["viz/trajectory_family_id_hist"] = wandb.Histogram(
                family.detach().cpu().argmax(dim=-1).numpy()
            )
    run.log(metrics, step=step)


def _maybe_log_checkpoint_rollouts(
    run: Any | None,
    args: argparse.Namespace,
    *,
    train_dataset: ReachSequenceDataset,
    policy: SimpleDP3,
    device: torch.device,
    step: int,
    rollout_attempted_steps: set[int],
) -> None:
    if (
        run is None
        or args.checkpoint_dir is None
        or not args.checkpoint_rollout_videos
        or step in rollout_attempted_steps
    ):
        return
    rollout_attempted_steps.add(step)
    try:
        _log_checkpoint_rollouts(
            run,
            args,
            train_dataset=train_dataset,
            policy=policy,
            device=device,
            step=step,
        )
    except Exception as exc:
        print(
            "warning: checkpoint rollout video logging failed, continuing training: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        try:
            run.log({"rollout/skipped": 1.0}, step=step)
        except Exception:
            # Do not let a secondary W&B reporting failure interrupt training.
            pass


def _log_checkpoint_rollouts(
    run: Any,
    args: argparse.Namespace,
    *,
    train_dataset: ReachSequenceDataset,
    policy: SimpleDP3,
    device: torch.device,
    step: int,
) -> None:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    import wandb

    from pg3d.envs.maniskill_adapter import register_pg3d_reach_envs
    from scripts.rollout_dp3_reach_policy import (
        _action_mode,
        crop_config_from_metadata,
        rollout_spec_video_stem,
        run_policy_rollout,
    )

    metadata, specs, using_validation_dataset = _checkpoint_rollout_metadata_and_specs(
        args,
        train_dataset=train_dataset,
    )
    if not specs:
        return

    register_pg3d_reach_envs()
    crop_config = crop_config_from_metadata(metadata)
    action_mode = _action_mode(str(metadata.get("action_mode", "abs_joint")))
    env_kwargs = dict(metadata.get("env_kwargs", {}))
    env_kwargs["render_mode"] = "rgb_array"
    env_kwargs.setdefault("obs_mode", "pointcloud")
    output_dir = args.checkpoint_dir / "rollout_videos" / f"step_{step:08d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    was_training = policy.training
    policy.eval()
    env: Any | None = None
    summaries: list[dict[str, Any]] = []
    try:
        env = gym.make(str(metadata["env_id"]), **env_kwargs)
        for spec in specs:
            video_path = output_dir / (
                rollout_spec_video_stem(spec, validation=using_validation_dataset) + ".mp4"
            )
            summaries.append(
                run_policy_rollout(
                    env=env,
                    policy=policy,
                    spec=spec,
                    action_mode=action_mode,
                    crop_config=crop_config,
                    output_dir=output_dir,
                    device=device,
                    max_steps=args.checkpoint_rollout_max_steps,
                    replan_stride=int(policy.n_action_steps),
                    post_success_steps=args.checkpoint_rollout_post_success_steps,
                    gripper_open=0.04,
                    video_fps=args.checkpoint_rollout_fps,
                    metrics_file=None,
                    video_path=video_path,
                    write_rerun=False,
                )
            )
    finally:
        if env is not None:
            env.close()
        if was_training:
            policy.train()

    videos = {
        f"rollout/{Path(str(summary['video'])).stem}": wandb.Video(
            summary["video"],
            fps=args.checkpoint_rollout_fps,
            format="mp4",
        )
        for summary in summaries
    }
    final_distances = [
        float(summary["final_distance"])
        for summary in summaries
        if summary["final_distance"] is not None
        and math.isfinite(float(summary["final_distance"]))
    ]
    success_rate = (
        sum(1 if summary["success"] else 0 for summary in summaries) / len(summaries)
        if summaries
        else 0.0
    )
    run.log(
        {
            **videos,
            "rollout/source": (
                "validation_dataset" if using_validation_dataset else "mixed_train_fresh"
            ),
            "rollout/selected_dataset_episode_indices": json.dumps(
                [
                    summary["dataset_episode_index"]
                    for summary in summaries
                    if summary["dataset_episode_index"] is not None
                ]
            ),
            "rollout/selected_seeds": json.dumps([summary["seed"] for summary in summaries]),
            "rollout/video_count": len(videos),
            "rollout/success_rate": success_rate,
            "rollout/final_distance_mean": (
                float(sum(final_distances) / len(final_distances))
                if final_distances
                else float("nan")
            ),
        },
        step=step,
    )


def _checkpoint_rollout_metadata_and_specs(
    args: argparse.Namespace,
    *,
    train_dataset: ReachSequenceDataset,
) -> tuple[dict[str, Any], list[Any], bool]:
    from scripts.rollout_dp3_reach_policy import (
        select_mixed_rollout_specs,
        select_random_dataset_rollout_specs,
    )

    metadata = (
        load_reach_metadata(args.checkpoint_rollout_dataset)
        if args.checkpoint_rollout_dataset is not None
        else train_dataset.metadata
    )
    dataset_episode_seeds = [
        int(episode["seed"]) for episode in metadata.get("episodes", []) if "seed" in episode
    ]
    using_validation_dataset = args.checkpoint_rollout_dataset is not None
    specs = (
        select_random_dataset_rollout_specs(
            dataset_episode_seeds=dataset_episode_seeds,
            total_count=args.checkpoint_rollout_count,
            seed=args.checkpoint_rollout_selection_seed,
        )
        if using_validation_dataset
        else select_mixed_rollout_specs(
            dataset_episode_seeds=dataset_episode_seeds,
            total_count=args.checkpoint_rollout_count,
            seed_start=args.checkpoint_rollout_seed_start,
        )
    )
    return metadata, specs, using_validation_dataset


if __name__ == "__main__":
    raise SystemExit(main())
