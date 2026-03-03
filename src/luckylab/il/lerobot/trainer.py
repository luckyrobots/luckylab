"""LeRobot backend — trains IL policies on LeRobotDataset data using lerobot's factory."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from luckylab.il.config import IlRunnerCfg

logger = logging.getLogger(__name__)

LUCKYROBOTS_DATA_HOME = Path(os.getenv("LUCKYROBOTS_DATA_HOME", Path.home() / ".luckyrobots" / "data"))


def _resolve_dataset_root(il_cfg: IlRunnerCfg) -> str | None:
    """Return the dataset root directory, checking LUCKYROBOTS_DATA_HOME as a fallback."""
    if il_cfg.dataset_root:
        return il_cfg.dataset_root
    candidate = LUCKYROBOTS_DATA_HOME / il_cfg.dataset_repo_id
    if candidate.exists():
        return str(candidate)
    return None


def _build_policy(il_cfg: IlRunnerCfg, ds_meta, input_features: dict, output_features: dict, device: str):
    """Build a lerobot policy, preprocessor, and postprocessor via lerobot's factory.

    Returns:
        Tuple of (policy, policy_cfg, preprocessor, postprocessor).
    """
    from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors

    # Build lerobot config via factory, then overlay our overrides
    policy_cfg = make_policy_config(
        il_cfg.policy,
        device=device,
        input_features=input_features,
        output_features=output_features,
    )

    # Apply any overrides from our config
    for key, value in il_cfg.policy_overrides.items():
        if hasattr(policy_cfg, key):
            setattr(policy_cfg, key, value)
        else:
            logger.warning(f"Policy config has no attribute '{key}', skipping override")

    # Create policy
    policy_cls = get_policy_class(il_cfg.policy)
    policy = policy_cls(policy_cfg)
    policy.train()
    policy.to(device)

    # Create pre/post processors (normalization)
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg, dataset_stats=ds_meta.stats)

    return policy, policy_cfg, preprocessor, postprocessor


class _WandbLogger:
    """Thin wrapper — no-ops when wandb is disabled or init fails."""

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


def _init_wandb(il_cfg, optimizer_cfg, scheduler_cfg, grad_clip_norm) -> _WandbLogger:
    if not il_cfg.wandb:
        return _WandbLogger()
    try:
        import wandb

        run = wandb.init(
            project=il_cfg.wandb_project,
            entity=il_cfg.wandb_entity,
            name=il_cfg.experiment_name,
            config={
                "policy": il_cfg.policy,
                "dataset": il_cfg.dataset_repo_id,
                "batch_size": il_cfg.batch_size,
                "optimizer": optimizer_cfg.type,
                "learning_rate": optimizer_cfg.lr,
                "weight_decay": optimizer_cfg.weight_decay,
                "grad_clip_norm": grad_clip_norm,
                "scheduler": scheduler_cfg.type if scheduler_cfg else None,
                "num_train_steps": il_cfg.num_train_steps,
            },
        )
        return _WandbLogger(run)
    except Exception:
        logger.warning("Failed to initialize wandb, continuing without it")
        return _WandbLogger()


def train(il_cfg: IlRunnerCfg, device: str = "cpu") -> None:
    """Train an IL policy on a LeRobotDataset.

    Args:
        il_cfg: IL runner configuration.
        device: Torch device string.
    """
    from lerobot.configs.types import FeatureType
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.utils import dataset_to_policy_features

    from luckylab.utils import video_decode_patch

    video_decode_patch.install()

    from luckylab.utils.logging import print_info

    if not il_cfg.dataset_repo_id:
        raise ValueError("dataset_repo_id is required for IL training")

    # 1. Load dataset metadata
    print_info(f"Loading dataset metadata: {il_cfg.dataset_repo_id}")
    resolved_root = _resolve_dataset_root(il_cfg)
    ds_meta_kwargs = {}
    if resolved_root:
        ds_meta_kwargs["root"] = resolved_root
    ds_meta = LeRobotDatasetMetadata(il_cfg.dataset_repo_id, **ds_meta_kwargs)

    # 2. Derive policy features from dataset
    features = dataset_to_policy_features(ds_meta.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    # 3. Build policy via factory
    print_info(f"Building {il_cfg.policy.upper()} policy")
    policy, policy_cfg, preprocessor, postprocessor = _build_policy(
        il_cfg, ds_meta, input_features, output_features, device,
    )

    # 4. Compute delta_timestamps from policy config
    delta_timestamps = il_cfg.delta_timestamps
    if delta_timestamps is None:
        delta_timestamps = {}
        if hasattr(policy_cfg, "observation_delta_indices") and policy_cfg.observation_delta_indices:
            for key in input_features:
                delta_timestamps[key] = [i / ds_meta.fps for i in policy_cfg.observation_delta_indices]
        if hasattr(policy_cfg, "action_delta_indices") and policy_cfg.action_delta_indices:
            delta_timestamps["action"] = [i / ds_meta.fps for i in policy_cfg.action_delta_indices]

    # 5. Load full dataset
    print_info("Loading dataset...")
    ds_kwargs = {}
    if resolved_root:
        ds_kwargs["root"] = resolved_root
    if delta_timestamps:
        ds_kwargs["delta_timestamps"] = delta_timestamps
    dataset = LeRobotDataset(il_cfg.dataset_repo_id, **ds_kwargs)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=il_cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=device != "cpu",
        drop_last=True,
    )

    # 6. Optimizer + scheduler from policy preset, with optional overrides
    optimizer_cfg = policy_cfg.get_optimizer_preset()
    if il_cfg.learning_rate is not None:
        optimizer_cfg.lr = il_cfg.learning_rate
    if il_cfg.weight_decay is not None:
        optimizer_cfg.weight_decay = il_cfg.weight_decay
    if il_cfg.grad_clip_norm is not None:
        optimizer_cfg.grad_clip_norm = il_cfg.grad_clip_norm
    grad_clip_norm = optimizer_cfg.grad_clip_norm

    optim_params = policy.get_optim_params()
    optimizer = optimizer_cfg.build(optim_params)

    scheduler_cfg = policy_cfg.get_scheduler_preset()
    scheduler = None
    if scheduler_cfg is not None:
        scheduler = scheduler_cfg.build(optimizer, il_cfg.num_train_steps)

    # 7. Output directory
    output_dir = Path(il_cfg.directory) / il_cfg.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 8. Optional wandb
    log = _init_wandb(il_cfg, optimizer_cfg, scheduler_cfg, grad_clip_norm)

    # 9. Training loop
    print_info(f"Training for {il_cfg.num_train_steps} steps (batch_size={il_cfg.batch_size})")
    global_step = 0
    data_iter = iter(dataloader)

    while global_step < il_cfg.num_train_steps:
        # Get next batch (cycle through dataset, retry on bad batches)
        batch = None
        for _retry in range(5):
            try:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)
                batch = preprocessor(batch)
                break
            except (RuntimeError, UnboundLocalError) as e:
                if _retry < 4:
                    logger.warning(f"Dataloader error (retry {_retry + 1}/5): {e}")
                    continue
                raise
        loss, loss_dict = policy.forward(batch)

        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

        global_step += 1

        # Logging
        if global_step % 100 == 0:
            log_msg = f"step {global_step}/{il_cfg.num_train_steps}  loss={loss.item():.4f}"
            if loss_dict:
                extras = "  ".join(
                    f"{k}={v.item():.4f}" if hasattr(v, "item") else f"{k}={v:.4f}"
                    for k, v in loss_dict.items()
                )
                log_msg += f"  ({extras})"
            print_info(log_msg)
            log(loss, loss_dict, global_step)

        # Checkpoint
        if il_cfg.save_freq > 0 and global_step % il_cfg.save_freq == 0:
            ckpt_dir = output_dir / f"checkpoint-{global_step}"
            _save_checkpoint(policy, preprocessor, postprocessor, ckpt_dir)
            print_info(f"Saved checkpoint: {ckpt_dir}")

    # Final save
    final_dir = output_dir / "final"
    _save_checkpoint(policy, preprocessor, postprocessor, final_dir)
    print_info(f"Training complete! Final model saved to {final_dir}")
    print_info(video_decode_patch.get_stats().summary())
    log.finish()


def _save_checkpoint(policy, preprocessor, postprocessor, path: Path) -> None:
    """Save policy, preprocessor, and postprocessor to a directory."""
    path.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(str(path))
    preprocessor.save_pretrained(str(path))
    postprocessor.save_pretrained(str(path))


def load_policy(checkpoint_path: str, il_cfg: IlRunnerCfg, device: str = "cpu"):
    """Load a trained IL policy with its preprocessor and postprocessor.

    Args:
        checkpoint_path: Path to the checkpoint directory (contains config.json + model.safetensors).
        il_cfg: IL runner configuration (used to determine policy type).
        device: Torch device string.

    Returns:
        Tuple of (policy, preprocessor, postprocessor).
    """
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    # Load config, then pass it to from_pretrained to avoid loading it twice
    policy_cfg = PreTrainedConfig.from_pretrained(checkpoint_path)
    policy_cfg.device = device
    policy_cls = get_policy_class(il_cfg.policy)
    policy = policy_cls.from_pretrained(checkpoint_path, config=policy_cfg)
    policy.eval()
    policy.to(device)

    # Load preprocessor/postprocessor from checkpoint
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg, pretrained_path=checkpoint_path,
    )

    return policy, preprocessor, postprocessor
