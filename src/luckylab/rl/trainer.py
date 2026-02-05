"""Training utilities for skrl."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from luckylab.rl.config import RlRunnerCfg
from luckylab.rl.models import Critic, DeterministicActor, GaussianActor, QCritic
from luckylab.utils.logging import print_info
from luckylab.utils.random import seed_rng

if TYPE_CHECKING:
    from luckylab.envs import ManagerBasedRlEnvCfg


def _get_experiment_cfg(cfg: RlRunnerCfg) -> dict:
    """Build skrl experiment config for logging and checkpointing."""
    experiment_cfg = {
        "directory": cfg.directory,
        "experiment_name": cfg.experiment_name,
        "write_interval": cfg.log_interval,
        "checkpoint_interval": cfg.checkpoint_interval,
        "wandb": cfg.wandb,
    }

    if cfg.wandb:
        wandb_kwargs = {"project": cfg.wandb_project}
        if cfg.wandb_entity:
            wandb_kwargs["entity"] = cfg.wandb_entity
        experiment_cfg["wandb_kwargs"] = wandb_kwargs

    return experiment_cfg


def _create_ppo_agent(env, cfg: RlRunnerCfg, device: str):
    """Create PPO agent with models and memory."""
    from skrl.agents.torch.ppo import PPO
    from skrl.memories.torch import RandomMemory

    obs_space, act_space = env.observation_space, env.action_space
    num_policy_obs = getattr(env, "num_policy_obs", None)
    ppo = cfg.ppo

    models = {
        "policy": GaussianActor(obs_space, act_space, device, cfg.policy, num_policy_obs),
        "value": Critic(obs_space, act_space, device, cfg.policy),
    }

    memory = RandomMemory(
        memory_size=ppo.num_steps_per_env,
        num_envs=env.num_envs,
        device=device,
    )

    agent_cfg = {
        "rollouts": ppo.num_steps_per_env,
        "learning_epochs": ppo.num_learning_epochs,
        "mini_batches": ppo.num_mini_batches,
        "discount_factor": ppo.gamma,
        "lambda": ppo.lam,
        "learning_rate": ppo.learning_rate,
        "ratio_clip": ppo.clip_param,
        "value_clip": ppo.clip_param,
        "clip_predicted_values": ppo.use_clipped_value_loss,
        "value_loss_scale": ppo.value_loss_coef,
        "entropy_loss_scale": ppo.entropy_coef,
        "grad_norm_clip": ppo.max_grad_norm,
        "kl_threshold": ppo.desired_kl,
    }

    if ppo.schedule == "adaptive":
        from skrl.resources.schedulers.torch import KLAdaptiveLR

        agent_cfg["learning_rate_scheduler"] = KLAdaptiveLR
        agent_cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": ppo.desired_kl}

    agent_cfg["experiment"] = _get_experiment_cfg(cfg)

    for m in models.values():
        m.to(device)

    return PPO(
        models=models,
        memory=memory,
        cfg=agent_cfg,
        observation_space=obs_space,
        action_space=act_space,
        device=device,
    )


def _create_sac_agent(env, cfg: RlRunnerCfg, device: str):
    """Create SAC agent with models and memory."""
    from skrl.agents.torch.sac import SAC
    from skrl.memories.torch import RandomMemory

    obs_space, act_space = env.observation_space, env.action_space
    num_policy_obs = getattr(env, "num_policy_obs", None)
    sac = cfg.sac

    models = {
        "policy": GaussianActor(obs_space, act_space, device, cfg.policy, num_policy_obs),
        "critic_1": QCritic(obs_space, act_space, device, cfg.policy),
        "critic_2": QCritic(obs_space, act_space, device, cfg.policy),
        "target_critic_1": QCritic(obs_space, act_space, device, cfg.policy),
        "target_critic_2": QCritic(obs_space, act_space, device, cfg.policy),
    }

    memory = RandomMemory(
        memory_size=cfg.memory_size or 100_000,
        num_envs=env.num_envs,
        device=device,
    )

    agent_cfg = {
        "batch_size": sac.batch_size,
        "discount_factor": sac.discount_factor,
        "polyak": sac.polyak,
        "actor_learning_rate": sac.actor_learning_rate,
        "critic_learning_rate": sac.critic_learning_rate,
        "learn_entropy": sac.learn_entropy,
        "entropy_learning_rate": sac.actor_learning_rate,
        "initial_entropy_value": sac.initial_entropy,
        "grad_norm_clip": sac.grad_norm_clip,
        "random_timesteps": sac.random_timesteps,
        "learning_starts": sac.learning_starts,
    }

    if sac.target_entropy is not None:
        agent_cfg["target_entropy"] = sac.target_entropy

    agent_cfg["experiment"] = _get_experiment_cfg(cfg)

    for m in models.values():
        m.to(device)

    return SAC(
        models=models,
        memory=memory,
        cfg=agent_cfg,
        observation_space=obs_space,
        action_space=act_space,
        device=device,
    )


def _create_td3_agent(env, cfg: RlRunnerCfg, device: str):
    """Create TD3 agent with models and memory."""
    from skrl.agents.torch.td3 import TD3
    from skrl.memories.torch import RandomMemory

    obs_space, act_space = env.observation_space, env.action_space
    num_policy_obs = getattr(env, "num_policy_obs", None)
    td3 = cfg.td3

    models = {
        "policy": DeterministicActor(obs_space, act_space, device, cfg.policy, num_policy_obs),
        "target_policy": DeterministicActor(obs_space, act_space, device, cfg.policy, num_policy_obs),
        "critic_1": QCritic(obs_space, act_space, device, cfg.policy),
        "critic_2": QCritic(obs_space, act_space, device, cfg.policy),
        "target_critic_1": QCritic(obs_space, act_space, device, cfg.policy),
        "target_critic_2": QCritic(obs_space, act_space, device, cfg.policy),
    }

    memory = RandomMemory(
        memory_size=cfg.memory_size or 100_000,
        num_envs=env.num_envs,
        device=device,
    )

    agent_cfg = {
        "batch_size": td3.batch_size,
        "discount_factor": td3.discount_factor,
        "polyak": td3.polyak,
        "actor_learning_rate": td3.actor_learning_rate,
        "critic_learning_rate": td3.critic_learning_rate,
        "policy_delay": td3.policy_delay,
        "smooth_regularization_noise": td3.smooth_regularization_noise,
        "smooth_regularization_clip": td3.smooth_regularization_clip,
    }

    agent_cfg["experiment"] = _get_experiment_cfg(cfg)

    for m in models.values():
        m.to(device)

    return TD3(
        models=models,
        memory=memory,
        cfg=agent_cfg,
        observation_space=obs_space,
        action_space=act_space,
        device=device,
    )


def _create_ddpg_agent(env, cfg: RlRunnerCfg, device: str):
    """Create DDPG agent with models and memory."""
    from skrl.agents.torch.ddpg import DDPG
    from skrl.memories.torch import RandomMemory

    obs_space, act_space = env.observation_space, env.action_space
    num_policy_obs = getattr(env, "num_policy_obs", None)
    ddpg = cfg.ddpg

    models = {
        "policy": DeterministicActor(obs_space, act_space, device, cfg.policy, num_policy_obs),
        "target_policy": DeterministicActor(obs_space, act_space, device, cfg.policy, num_policy_obs),
        "critic": QCritic(obs_space, act_space, device, cfg.policy),
        "target_critic": QCritic(obs_space, act_space, device, cfg.policy),
    }

    memory = RandomMemory(
        memory_size=cfg.memory_size or 100_000,
        num_envs=env.num_envs,
        device=device,
    )

    agent_cfg = {
        "batch_size": ddpg.batch_size,
        "discount_factor": ddpg.discount_factor,
        "polyak": ddpg.polyak,
        "actor_learning_rate": ddpg.actor_learning_rate,
        "critic_learning_rate": ddpg.critic_learning_rate,
    }

    agent_cfg["experiment"] = _get_experiment_cfg(cfg)

    for m in models.values():
        m.to(device)

    return DDPG(
        models=models,
        memory=memory,
        cfg=agent_cfg,
        observation_space=obs_space,
        action_space=act_space,
        device=device,
    )


def create_agent(env, cfg: RlRunnerCfg, device: str):
    """Create agent based on algorithm selection."""
    creators = {
        "ppo": _create_ppo_agent,
        "sac": _create_sac_agent,
        "td3": _create_td3_agent,
        "ddpg": _create_ddpg_agent,
    }

    if cfg.algorithm not in creators:
        raise ValueError(f"Unknown algorithm: {cfg.algorithm}")

    return creators[cfg.algorithm](env, cfg, device)


def _print_model_info(env, cfg: RlRunnerCfg, device: str) -> None:
    """Print model architecture info."""
    print_info("-" * 80)
    print_info("Resolved observation sets: ")
    print_info(f"         policy :  ('policy',)")
    if getattr(env, "num_critic_obs", 0) > 0:
        print_info(f"         critic :  ('critic',)")
    print_info("-" * 80)

    # Print network architecture
    obs_space, act_space = env.observation_space, env.action_space
    num_policy_obs = getattr(env, "num_policy_obs", None)

    actor = GaussianActor(obs_space, act_space, device, cfg.policy, num_policy_obs)
    critic = Critic(obs_space, act_space, device, cfg.policy)

    print_info(f"Actor MLP: {actor}")
    print_info(f"Critic MLP: {critic}")


def train(env_cfg: ManagerBasedRlEnvCfg, rl_cfg: RlRunnerCfg, device: str = "cpu") -> None:
    """Train an RL agent using skrl."""
    from skrl.trainers.torch import SequentialTrainer

    from luckylab.envs import ManagerBasedRlEnv
    from luckylab.rl.skrl_wrapper import SkrlWrapper

    seed_rng(rl_cfg.seed)

    # Setup logging directory
    log_dir = Path(rl_cfg.directory) / rl_cfg.experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    print_info(f"[INFO] Logging experiment in directory: {log_dir}")

    print_info(f"Training {rl_cfg.algorithm.upper()} for {rl_cfg.max_iterations} iterations")
    if rl_cfg.wandb:
        print_info(f"Logging to wandb: {rl_cfg.wandb_project}")

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    wrapped = SkrlWrapper(env, clip_actions=rl_cfg.clip_actions)

    _print_model_info(wrapped, rl_cfg, device)

    agent = create_agent(wrapped, rl_cfg, device)

    # Compute total timesteps based on algorithm
    if rl_cfg.algorithm == "ppo":
        timesteps = rl_cfg.max_iterations * rl_cfg.ppo.num_steps_per_env
    else:
        # Off-policy algorithms: max_iterations is number of gradient steps
        timesteps = rl_cfg.max_iterations

    trainer_cfg = {
        "timesteps": timesteps,
        "headless": True,
        "environment_info": "episode"
    }

    trainer = SequentialTrainer(cfg=trainer_cfg, env=wrapped, agents=agent)
    trainer.train()

    wrapped.close()
    print_info("Training complete")


def load_agent(
    checkpoint_path: str,
    env_cfg: ManagerBasedRlEnvCfg,
    rl_cfg: RlRunnerCfg,
    device: str = "cpu",
):
    """Load a trained agent from checkpoint."""
    from luckylab.envs import ManagerBasedRlEnv
    from luckylab.rl.skrl_wrapper import SkrlWrapper

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    wrapped = SkrlWrapper(env, clip_actions=rl_cfg.clip_actions)

    agent = create_agent(wrapped, rl_cfg, device)
    agent.load(checkpoint_path)

    return agent, wrapped
