"""LeRobot backend — trains IL policies on LeRobotDataset data using lerobot's factory."""

from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.multiprocessing as mp

if TYPE_CHECKING:
    from luckylab.il.config import IlRunnerCfg

logger = logging.getLogger(__name__)

LUCKYROBOTS_DATA_HOME = Path(os.getenv("LUCKYROBOTS_DATA_HOME", Path.home() / ".luckyrobots" / "data"))


def train(il_cfg: IlRunnerCfg, device: str = "cpu") -> None:
    """Train an IL policy on a LeRobotDataset."""
    from lerobot.configs.types import FeatureType
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.utils import dataset_to_policy_features

    from luckylab.il.augment import apply_augmentations, log_active
    from luckylab.utils import video_decode_patch
    from luckylab.utils.logging import print_info

    video_decode_patch.install()

    if not il_cfg.dataset_repo_id:
        raise ValueError("dataset_repo_id is required for IL training")

    # Dataset metadata
    print_info(f"Loading dataset metadata: {il_cfg.dataset_repo_id}")
    resolved_root = _resolve_dataset_root(il_cfg)
    meta_kwargs = {"root": resolved_root} if resolved_root else {}
    ds_meta = LeRobotDatasetMetadata(il_cfg.dataset_repo_id, **meta_kwargs)

    # Policy features
    features = dataset_to_policy_features(ds_meta.features)
    output_features = {k: f for k, f in features.items() if f.type is FeatureType.ACTION}
    input_features = {k: f for k, f in features.items() if k not in output_features}

    # Build policy
    print_info(f"Building {il_cfg.policy.upper()} policy")
    policy, policy_cfg, preprocessor, postprocessor = _build_policy(
        il_cfg, ds_meta, input_features, output_features, device,
    )

    # Dataset
    print_info("Loading dataset...")
    delta_timestamps = _compute_delta_timestamps(il_cfg, policy_cfg, input_features, ds_meta.fps)
    ds_kwargs = {}
    if resolved_root:
        ds_kwargs["root"] = resolved_root
    if delta_timestamps:
        ds_kwargs["delta_timestamps"] = delta_timestamps
    if il_cfg.episodes is not None:
        ds_kwargs["episodes"] = il_cfg.episodes
        print_info(f"Filtering to {len(il_cfg.episodes)} episodes")
    ds_kwargs["video_backend"] = "torchcodec"
    ds_kwargs["tolerance_s"] = 0.04
    dataset = LeRobotDataset(il_cfg.dataset_repo_id, **ds_kwargs)

    # Dataloader
    with contextlib.suppress(RuntimeError):
        mp.set_start_method("spawn")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=il_cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=device != "cpu",
        drop_last=True,
        multiprocessing_context="spawn",
    )

    # Optimizer + scheduler
    optimizer, scheduler, grad_clip_norm = _build_optimizer(policy, policy_cfg, il_cfg)

    # Output directory
    output_dir = Path(il_cfg.directory) / il_cfg.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Logging
    wandb_log = _init_wandb(il_cfg, policy_cfg, grad_clip_norm)
    active = log_active(il_cfg)
    print_info(f"Augmentations: {', '.join(active)}" if active else "No augmentations (baseline run)")

    # Training loop
    print_info(f"Training for {il_cfg.num_train_steps} steps (batch_size={il_cfg.batch_size})")
    global_step = 0
    data_iter = iter(dataloader)

    while global_step < il_cfg.num_train_steps:
        batch = _next_batch(data_iter, dataloader, preprocessor)
        if batch is None:
            data_iter = iter(dataloader)
            continue

        batch = apply_augmentations(batch, il_cfg)
        loss, loss_dict = policy.forward(batch)

        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

        global_step += 1

        if global_step % 100 == 0:
            _log_step(print_info, wandb_log, loss, loss_dict, global_step, il_cfg.num_train_steps)

        if il_cfg.save_freq > 0 and global_step % il_cfg.save_freq == 0:
            ckpt_dir = output_dir / f"checkpoint-{global_step}"
            _save_checkpoint(policy, preprocessor, postprocessor, ckpt_dir)
            print_info(f"Saved checkpoint: {ckpt_dir}")

    final_dir = output_dir / "final"
    _save_checkpoint(policy, preprocessor, postprocessor, final_dir)
    print_info(f"Training complete! Final model saved to {final_dir}")
    print_info(video_decode_patch.get_stats().summary())
    wandb_log.finish()


def load_policy(checkpoint_path: str, il_cfg: IlRunnerCfg, device: str = "cpu"):
    """Load a trained IL policy with its preprocessor and postprocessor."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    policy_cfg = PreTrainedConfig.from_pretrained(checkpoint_path)
    policy_cfg.device = device
    policy_cls = get_policy_class(il_cfg.policy)
    policy = policy_cls.from_pretrained(checkpoint_path, config=policy_cfg)
    policy.eval()
    policy.to(device)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg,
        pretrained_path=checkpoint_path,
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    return policy, preprocessor, postprocessor


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _resolve_dataset_root(il_cfg: IlRunnerCfg) -> str | None:
    if il_cfg.dataset_root:
        return il_cfg.dataset_root
    candidate = LUCKYROBOTS_DATA_HOME / il_cfg.dataset_repo_id
    if candidate.exists():
        return str(candidate)
    return None


def _build_policy(il_cfg, ds_meta, input_features, output_features, device):
    from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors

    policy_cfg = make_policy_config(
        il_cfg.policy, device=device,
        input_features=input_features, output_features=output_features,
    )

    for key, value in il_cfg.policy_overrides.items():
        if hasattr(policy_cfg, key):
            setattr(policy_cfg, key, value)
        else:
            logger.warning(f"Policy config has no attribute '{key}', skipping override")

    if (
        hasattr(policy_cfg, "n_action_steps")
        and hasattr(policy_cfg, "chunk_size")
        and policy_cfg.n_action_steps > policy_cfg.chunk_size
    ):
        policy_cfg.n_action_steps = policy_cfg.chunk_size

    policy_cls = get_policy_class(il_cfg.policy)
    policy = policy_cls(policy_cfg)
    policy.train()
    policy.to(device)

    preprocessor, postprocessor = make_pre_post_processors(policy_cfg, dataset_stats=ds_meta.stats)
    return policy, policy_cfg, preprocessor, postprocessor


def _compute_delta_timestamps(il_cfg, policy_cfg, input_features, fps):
    if il_cfg.delta_timestamps is not None:
        return il_cfg.delta_timestamps
    dt = {}
    if hasattr(policy_cfg, "observation_delta_indices") and policy_cfg.observation_delta_indices:
        for key in input_features:
            dt[key] = [i / fps for i in policy_cfg.observation_delta_indices]
    if hasattr(policy_cfg, "action_delta_indices") and policy_cfg.action_delta_indices:
        dt["action"] = [i / fps for i in policy_cfg.action_delta_indices]
    return dt


def _build_optimizer(policy, policy_cfg, il_cfg):
    optimizer_cfg = policy_cfg.get_optimizer_preset()
    if il_cfg.learning_rate is not None:
        optimizer_cfg.lr = il_cfg.learning_rate
    if il_cfg.weight_decay is not None:
        optimizer_cfg.weight_decay = il_cfg.weight_decay
    if il_cfg.grad_clip_norm is not None:
        optimizer_cfg.grad_clip_norm = il_cfg.grad_clip_norm
    grad_clip_norm = optimizer_cfg.grad_clip_norm

    optimizer = optimizer_cfg.build(policy.get_optim_params())

    scheduler_cfg = policy_cfg.get_scheduler_preset()
    scheduler = scheduler_cfg.build(optimizer, il_cfg.num_train_steps) if scheduler_cfg else None

    return optimizer, scheduler, grad_clip_norm


def _next_batch(data_iter, dataloader, preprocessor):
    for _retry in range(5):
        try:
            try:
                batch = next(data_iter)
            except StopIteration:
                return None
            return preprocessor(batch)
        except (RuntimeError, UnboundLocalError) as e:
            if _retry < 4:
                logger.warning(f"Dataloader error (retry {_retry + 1}/5): {e}")
                continue
            raise
    return None


def _save_checkpoint(policy, preprocessor, postprocessor, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(str(path))
    preprocessor.save_pretrained(str(path))
    postprocessor.save_pretrained(str(path))


class _WandbLogger:
    def __init__(self, run=None):
        self._run = run

    def __call__(self, loss, loss_dict, step):
        if not self._run:
            return
        data = {"loss": loss.item()}
        if loss_dict:
            data.update({k: v.item() if hasattr(v, "item") else v for k, v in loss_dict.items()})
        self._run.log(data, step=step)

    def finish(self):
        if self._run:
            self._run.finish()


def _init_wandb(il_cfg, policy_cfg, grad_clip_norm) -> _WandbLogger:
    if not il_cfg.wandb:
        return _WandbLogger()
    try:
        import wandb

        optimizer_cfg = policy_cfg.get_optimizer_preset()
        scheduler_cfg = policy_cfg.get_scheduler_preset()
        run = wandb.init(
            project=il_cfg.wandb_project,
            entity=il_cfg.wandb_entity,
            name=il_cfg.experiment_name,
            config={
                "policy": il_cfg.policy,
                "dataset": il_cfg.dataset_repo_id,
                "batch_size": il_cfg.batch_size,
                "learning_rate": optimizer_cfg.lr,
                "grad_clip_norm": grad_clip_norm,
                "scheduler": scheduler_cfg.type if scheduler_cfg else None,
                "num_train_steps": il_cfg.num_train_steps,
                "grayscale": il_cfg.grayscale,
                "camera_noise": il_cfg.camera_noise,
                "state_noise_std": il_cfg.state_noise_std,
                "action_noise_std": il_cfg.action_noise_std,
                "episodes": f"{len(il_cfg.episodes)}" if il_cfg.episodes else "all",
            },
        )
        return _WandbLogger(run)
    except Exception:
        logger.warning("Failed to initialize wandb, continuing without it")
        return _WandbLogger()


def _log_step(print_info, wandb_log, loss, loss_dict, step, total):
    msg = f"step {step}/{total}  loss={loss.item():.4f}"
    if loss_dict:
        extras = "  ".join(
            f"{k}={v.item():.4f}" if hasattr(v, "item") else f"{k}={v:.4f}"
            for k, v in loss_dict.items()
        )
        msg += f"  ({extras})"
    print_info(msg)
    wandb_log(loss, loss_dict, step)
