"""Training utilities for Stable Baselines3 backend."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch.nn as nn

from luckylab.rl.common import make_experiment_name, print_config, wrap_env
from luckylab.rl.config import RlRunnerCfg
from luckylab.utils.logging import WandbLogger, print_info
from luckylab.utils.random import seed_rng

if TYPE_CHECKING:
    from luckylab.envs import ManagerBasedRlEnvCfg
    from luckylab.rl.sb3.wrapper import Sb3Wrapper

# Map our activation names to torch classes (SB3 policy_kwargs expects these).
_ACTIVATIONS = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
}


def _wrap_env(env, rl_cfg: RlRunnerCfg) -> Sb3Wrapper:
    from luckylab.rl.sb3.wrapper import Sb3Wrapper
    return wrap_env(env, rl_cfg, Sb3Wrapper)


def _build_policy_kwargs(rl_cfg: RlRunnerCfg) -> dict:
    """Map our ActorCriticCfg to SB3 policy_kwargs."""
    policy = rl_cfg.policy
    kwargs: dict = {
        "net_arch": dict(
            pi=list(policy.actor_hidden_dims),
            qf=list(policy.critic_hidden_dims),
        ),
        "activation_fn": _ACTIVATIONS.get(policy.activation.lower(), nn.ELU),
    }
    if policy.noise_type == "gsde":
        # log_std_init goes in policy_kwargs; use_sde goes in the algorithm kwargs
        # (SB3 passes use_sde explicitly to the policy, so it must NOT be in policy_kwargs).
        kwargs["log_std_init"] = math.log(max(policy.init_noise_std, 1e-6))
    return kwargs


def _make_sac_kwargs(rl_cfg: RlRunnerCfg, device: str) -> dict:
    """Map SacAlgorithmCfg to SB3 SAC constructor kwargs."""
    sac = rl_cfg.sac
    policy_kwargs = _build_policy_kwargs(rl_cfg)

    kwargs = {
        "policy": "MlpPolicy",
        "learning_rate": sac.actor_learning_rate,
        "buffer_size": sac.memory_size,
        "learning_starts": max(sac.random_timesteps, sac.learning_starts),
        "batch_size": sac.batch_size,
        "tau": sac.polyak,
        "gamma": sac.discount_factor,
        "train_freq": 1,
        "gradient_steps": sac.gradient_steps,
        "policy_kwargs": policy_kwargs,
        "device": device,
        "seed": rl_cfg.seed,
    }

    if rl_cfg.policy.noise_type == "gsde":
        kwargs["use_sde"] = True
        kwargs["sde_sample_freq"] = rl_cfg.policy.gsde_resample_interval

    if sac.learn_entropy:
        # SB3 auto_X syntax sets initial entropy coefficient to X.
        kwargs["ent_coef"] = f"auto_{sac.initial_entropy}"
        kwargs["target_entropy"] = sac.target_entropy if sac.target_entropy is not None else "auto"
    else:
        kwargs["ent_coef"] = sac.initial_entropy

    return kwargs


def _make_ppo_kwargs(rl_cfg: RlRunnerCfg, device: str) -> dict:
    """Map PpoAlgorithmCfg to SB3 PPO constructor kwargs."""
    ppo = rl_cfg.ppo
    policy_kwargs = _build_policy_kwargs(rl_cfg)
    policy_kwargs["net_arch"] = list(rl_cfg.policy.actor_hidden_dims)

    return {
        "policy": "MlpPolicy",
        "learning_rate": ppo.learning_rate,
        "n_steps": ppo.num_steps_per_env,
        "batch_size": ppo.num_steps_per_env // ppo.num_mini_batches,
        "n_epochs": ppo.num_learning_epochs,
        "gamma": ppo.gamma,
        "gae_lambda": ppo.lam,
        "clip_range": ppo.clip_param,
        "ent_coef": ppo.entropy_coef,
        "vf_coef": ppo.value_loss_coef,
        "max_grad_norm": ppo.max_grad_norm,
        "target_kl": ppo.desired_kl if ppo.schedule == "adaptive" else None,
        "policy_kwargs": policy_kwargs,
        "device": device,
        "seed": rl_cfg.seed,
    }


def _make_td3_kwargs(rl_cfg: RlRunnerCfg, device: str) -> dict:
    """Map Td3AlgorithmCfg to SB3 TD3 constructor kwargs."""
    td3 = rl_cfg.td3
    return {
        "policy": "MlpPolicy",
        "learning_rate": td3.actor_learning_rate,
        "buffer_size": td3.memory_size,
        "batch_size": td3.batch_size,
        "tau": td3.polyak,
        "gamma": td3.discount_factor,
        "policy_delay": td3.policy_delay,
        "target_policy_noise": td3.smooth_regularization_noise,
        "target_noise_clip": td3.smooth_regularization_clip,
        "policy_kwargs": _build_policy_kwargs(rl_cfg),
        "device": device,
        "seed": rl_cfg.seed,
    }


def _make_ddpg_kwargs(rl_cfg: RlRunnerCfg, device: str) -> dict:
    """Map DdpgAlgorithmCfg to SB3 DDPG constructor kwargs."""
    ddpg = rl_cfg.ddpg
    return {
        "policy": "MlpPolicy",
        "learning_rate": ddpg.actor_learning_rate,
        "buffer_size": ddpg.memory_size,
        "batch_size": ddpg.batch_size,
        "tau": ddpg.polyak,
        "gamma": ddpg.discount_factor,
        "policy_kwargs": _build_policy_kwargs(rl_cfg),
        "device": device,
        "seed": rl_cfg.seed,
    }


def _make_callbacks(rl_cfg: RlRunnerCfg, wrapped: Sb3Wrapper) -> list:
    """Build the list of SB3 callbacks from config."""
    from stable_baselines3.common.callbacks import BaseCallback

    callbacks = []

    if rl_cfg.wandb:

        class WandbCb(BaseCallback):
            def __init__(self):
                super().__init__()
                self._prev_episode_info: dict = {}
                self._log_freq = 100  # Log SAC internals every N steps.

            def _on_step(self) -> bool:
                import wandb

                # Log episode info on episode boundaries.
                info = wrapped._last_episode_info
                if info and info is not self._prev_episode_info:
                    self._prev_episode_info = info
                    metrics = {f"Info/{k}": v for k, v in info.items()
                               if isinstance(v, (int, float))}
                    if metrics:
                        wandb.log(metrics, step=self.num_timesteps)

                # Log SAC internals periodically.
                if self.num_timesteps % self._log_freq == 0:
                    model = self.model
                    sac_metrics: dict = {}
                    if getattr(model, "log_ent_coef", None) is not None:
                        import torch as th
                        sac_metrics["Coefficient/Entropy coefficient"] = th.exp(model.log_ent_coef).item()
                    elif hasattr(model, "ent_coef") and isinstance(model.ent_coef, (int, float)):
                        sac_metrics["Coefficient/Entropy coefficient"] = float(model.ent_coef)
                    if hasattr(model, "logger") and model.logger is not None:
                        name_map = {
                            "train/actor_loss": "Loss/Policy loss",
                            "train/critic_loss": "Loss/Critic loss",
                            "train/ent_coef_loss": "Loss/Entropy loss",
                        }
                        for sb3_key, wb_key in name_map.items():
                            val = model.logger.name_to_value.get(sb3_key)
                            if val is not None:
                                sac_metrics[wb_key] = val
                    if sac_metrics:
                        wandb.log(sac_metrics, step=self.num_timesteps)
                return True

        callbacks.append(WandbCb())

    if rl_cfg.checkpoint_interval > 0:
        from stable_baselines3.common.callbacks import CheckpointCallback

        experiment_name = make_experiment_name(rl_cfg)
        callbacks.append(CheckpointCallback(
            save_freq=rl_cfg.checkpoint_interval,
            save_path=str(Path(rl_cfg.directory, experiment_name, "checkpoints")),
            name_prefix="agent",
        ))

    return callbacks


def _create_sb3_model(env: Sb3Wrapper, rl_cfg: RlRunnerCfg, device: str):
    """Create the SB3 model for the selected algorithm."""
    algo_map = {
        "sac": ("stable_baselines3", "SAC", _make_sac_kwargs),
        "ppo": ("stable_baselines3", "PPO", _make_ppo_kwargs),
        "td3": ("stable_baselines3", "TD3", _make_td3_kwargs),
        "ddpg": ("stable_baselines3", "DDPG", _make_ddpg_kwargs),
    }

    if rl_cfg.algorithm not in algo_map:
        raise ValueError(f"Unknown algorithm: {rl_cfg.algorithm}")

    mod_name, cls_name, kwargs_fn = algo_map[rl_cfg.algorithm]
    import importlib
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name)

    model = cls(env=env, **kwargs_fn(rl_cfg, device))

    if rl_cfg.resume_from:
        checkpoint_path = Path(rl_cfg.resume_from)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        print_info(f"Loading SB3 checkpoint from: {checkpoint_path}")
        loaded = cls.load(str(checkpoint_path), env=env, device=device)
        print_info(f"Resumed from checkpoint: {checkpoint_path}")
        return loaded

    return model


def train(env_cfg: ManagerBasedRlEnvCfg, rl_cfg: RlRunnerCfg, device: str = "cpu") -> None:
    """Train an RL agent using Stable Baselines3."""
    from luckylab.envs import ManagerBasedRlEnv

    seed_rng(rl_cfg.seed)
    experiment_name = make_experiment_name(rl_cfg)
    Path(rl_cfg.directory, experiment_name).mkdir(parents=True, exist_ok=True)

    with WandbLogger(rl_cfg, experiment_name):
        env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
        print_config(env, rl_cfg, experiment_name, device)
        wrapped = _wrap_env(env, rl_cfg)

        model = _create_sb3_model(wrapped, rl_cfg, device)
        callbacks = _make_callbacks(rl_cfg, wrapped)

        if rl_cfg.algorithm == "ppo":
            total_timesteps = rl_cfg.max_iterations * rl_cfg.ppo.num_steps_per_env
        else:
            total_timesteps = rl_cfg.max_iterations

        print_info(f"Starting SB3 training for {total_timesteps:,} timesteps...")
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks if callbacks else None,
            log_interval=rl_cfg.log_interval or None,
            progress_bar=True,
        )

        final_path = Path(rl_cfg.directory, experiment_name, "checkpoints", "final_model")
        model.save(str(final_path))
        print_info(f"Saved final model to {final_path}")

        wrapped.close()


def load_agent(
    checkpoint_path: str,
    env_cfg: ManagerBasedRlEnvCfg,
    rl_cfg: RlRunnerCfg,
    device: str = "cpu",
):
    """Load a trained SB3 agent from checkpoint.

    Returns:
        Tuple of (sb3_model, Sb3Wrapper).
    """
    import importlib

    from luckylab.envs import ManagerBasedRlEnv

    algo_cls_map = {"sac": "SAC", "ppo": "PPO", "td3": "TD3", "ddpg": "DDPG"}
    if rl_cfg.algorithm not in algo_cls_map:
        raise ValueError(f"Unknown algorithm: {rl_cfg.algorithm}")

    mod = importlib.import_module("stable_baselines3")
    cls = getattr(mod, algo_cls_map[rl_cfg.algorithm])

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    wrapped = _wrap_env(env, rl_cfg)
    model = cls.load(checkpoint_path, env=wrapped, device=device)
    return model, wrapped
