"""Training utilities using skrl."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

from .config import ActorCriticCfg, SkrlCfg

if TYPE_CHECKING:
    from ..envs import ManagerBasedRlEnvCfg

logger = logging.getLogger(__name__)


# =============================================================================
# Models
# =============================================================================


def _build_mlp(in_dim: int, out_dim: int, hidden: tuple[int, ...], act: str) -> nn.Sequential:
    """Build MLP with given architecture."""
    activations = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh, "leaky_relu": nn.LeakyReLU}
    layers = []
    prev = in_dim
    for dim in hidden:
        layers.extend([nn.Linear(prev, dim), activations[act.lower()]()])
        prev = dim
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class GaussianPolicy(GaussianMixin, Model):
    """Stochastic policy for PPO."""

    def __init__(self, obs_space, act_space, device, cfg: ActorCriticCfg):
        Model.__init__(self, obs_space, act_space, device)
        GaussianMixin.__init__(self, clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2)
        self.net = _build_mlp(obs_space.shape[0], act_space.shape[0], cfg.actor_hidden_dims, cfg.activation)
        self.log_std = nn.Parameter(torch.full((act_space.shape[0],), cfg.init_noise_std).log())

    def compute(self, inputs, role=""):
        out = self.net(inputs["states"])
        return out, self.log_std.expand_as(out), {}


class DeterministicPolicy(DeterministicMixin, Model):
    """Deterministic policy for SAC/TD3/DDPG."""

    def __init__(self, obs_space, act_space, device, cfg: ActorCriticCfg):
        Model.__init__(self, obs_space, act_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)
        self.net = _build_mlp(obs_space.shape[0], act_space.shape[0], cfg.actor_hidden_dims, cfg.activation)

    def compute(self, inputs, role=""):
        return torch.tanh(self.net(inputs["states"])), {}


class ValueNetwork(DeterministicMixin, Model):
    """Value function for PPO."""

    def __init__(self, obs_space, act_space, device, cfg: ActorCriticCfg):
        Model.__init__(self, obs_space, act_space, device)
        DeterministicMixin.__init__(self)
        self.net = _build_mlp(obs_space.shape[0], 1, cfg.critic_hidden_dims, cfg.activation)

    def compute(self, inputs, role=""):
        return self.net(inputs["states"]), {}


class QNetwork(DeterministicMixin, Model):
    """Q-function for SAC/TD3/DDPG."""

    def __init__(self, obs_space, act_space, device, cfg: ActorCriticCfg):
        Model.__init__(self, obs_space, act_space, device)
        DeterministicMixin.__init__(self)
        self.net = _build_mlp(
            obs_space.shape[0] + act_space.shape[0], 1, cfg.critic_hidden_dims, cfg.activation
        )

    def compute(self, inputs, role=""):
        return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=-1)), {}


# =============================================================================
# Agent Builder
# =============================================================================


def _build_agent(env, cfg: SkrlCfg, device: str, total_timesteps: int):
    """Build skrl agent based on algorithm."""
    from skrl.memories.torch import RandomMemory
    from skrl.resources.preprocessors.torch import RunningStandardScaler

    obs_space, act_space = env.observation_space, env.action_space
    policy_cfg = cfg.policy

    # Setup observation normalization (matches mjlab's actor/critic obs normalization)
    state_preprocessor = None
    state_preprocessor_kwargs = {}
    value_preprocessor = None
    value_preprocessor_kwargs = {}

    if policy_cfg.normalize_actor_obs:
        state_preprocessor = RunningStandardScaler
        state_preprocessor_kwargs = {"size": obs_space.shape[0], "device": device}
        logger.info("Enabled actor observation normalization (RunningStandardScaler)")

    if policy_cfg.normalize_critic_obs:
        value_preprocessor = RunningStandardScaler
        value_preprocessor_kwargs = {"size": obs_space.shape[0], "device": device}
        logger.info("Enabled critic observation normalization (RunningStandardScaler)")

    # Algorithm-specific setup
    if cfg.algorithm == "ppo":
        from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG

        models = {
            "policy": GaussianPolicy(obs_space, act_space, device, policy_cfg),
            "value": ValueNetwork(obs_space, act_space, device, policy_cfg),
        }
        memory = RandomMemory(memory_size=cfg.ppo.rollouts, num_envs=env.num_envs, device=device)
        agent_cfg = PPO_DEFAULT_CONFIG.copy()
        agent_cfg.update(
            {
                "rollouts": cfg.ppo.rollouts,
                "learning_epochs": cfg.ppo.learning_epochs,
                "mini_batches": cfg.ppo.mini_batches,
                "discount_factor": cfg.ppo.discount_factor,
                "lambda": cfg.ppo.lambda_gae,
                "learning_rate": cfg.ppo.learning_rate,
                "ratio_clip": cfg.ppo.ratio_clip,
                "value_loss_scale": cfg.ppo.value_loss_scale,
                "entropy_loss_scale": cfg.ppo.entropy_loss_scale,
                "grad_norm_clip": cfg.ppo.grad_norm_clip,
                "kl_threshold": cfg.ppo.kl_threshold,
                # Observation normalization
                "state_preprocessor": state_preprocessor,
                "state_preprocessor_kwargs": state_preprocessor_kwargs,
                "value_preprocessor": value_preprocessor,
                "value_preprocessor_kwargs": value_preprocessor_kwargs,
            }
        )

        # Learning rate scheduling
        if cfg.ppo.lr_schedule == "linear":
            import torch.optim.lr_scheduler as lr_scheduler
            # Linear decay from learning_rate to 0 (or end_factor * learning_rate)
            end_factor = cfg.ppo.lr_schedule_kwargs.get("end_factor", 0.0)
            agent_cfg["learning_rate_scheduler"] = lr_scheduler.LinearLR
            agent_cfg["learning_rate_scheduler_kwargs"] = {
                "start_factor": 1.0,
                "end_factor": end_factor,
                "total_iters": total_timesteps // cfg.ppo.rollouts,
            }
            logger.info(f"Enabled linear LR schedule (end_factor={end_factor})")
        elif cfg.ppo.lr_schedule == "adaptive":
            # Use skrl's KLAdaptiveLR scheduler
            # Adjusts LR based on KL divergence: decrease if KL too high, increase if too low
            from skrl.resources.schedulers.torch import KLAdaptiveLR
            agent_cfg["learning_rate_scheduler"] = KLAdaptiveLR
            agent_cfg["learning_rate_scheduler_kwargs"] = {
                "kl_threshold": cfg.ppo.kl_threshold,
                "min_lr": cfg.ppo.lr_schedule_kwargs.get("min_lr", 1e-6),
                "max_lr": cfg.ppo.lr_schedule_kwargs.get("max_lr", 1e-2),
            }
            logger.info(f"Enabled adaptive LR schedule (kl_threshold={cfg.ppo.kl_threshold})")
        # else: fixed schedule (no scheduler)

        Agent = PPO

    elif cfg.algorithm == "sac":
        from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG

        # SAC uses stochastic policy (GaussianPolicy), not deterministic
        models = {
            "policy": GaussianPolicy(obs_space, act_space, device, policy_cfg),
            "critic_1": QNetwork(obs_space, act_space, device, policy_cfg),
            "critic_2": QNetwork(obs_space, act_space, device, policy_cfg),
            "target_critic_1": QNetwork(obs_space, act_space, device, policy_cfg),
            "target_critic_2": QNetwork(obs_space, act_space, device, policy_cfg),
        }
        memory = RandomMemory(memory_size=cfg.memory_size or 100_000, num_envs=env.num_envs, device=device)
        agent_cfg = SAC_DEFAULT_CONFIG.copy()
        agent_cfg.update(
            {
                "batch_size": cfg.sac.batch_size,
                "discount_factor": cfg.sac.discount_factor,
                "polyak": cfg.sac.polyak,
                "actor_learning_rate": cfg.sac.actor_learning_rate,
                "critic_learning_rate": cfg.sac.critic_learning_rate,
                "learn_entropy": cfg.sac.learn_entropy,
                "entropy_learning_rate": cfg.sac.actor_learning_rate,
                "initial_entropy_value": cfg.sac.initial_entropy,
                "grad_norm_clip": cfg.sac.grad_norm_clip,
                # Critical for SAC: warmup with random exploration before learning
                "random_timesteps": cfg.sac.random_timesteps,
                "learning_starts": cfg.sac.learning_starts,
            }
        )
        if cfg.sac.target_entropy is not None:
            agent_cfg["target_entropy"] = cfg.sac.target_entropy
        Agent = SAC

    elif cfg.algorithm == "td3":
        from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG

        models = {
            "policy": DeterministicPolicy(obs_space, act_space, device, policy_cfg),
            "target_policy": DeterministicPolicy(obs_space, act_space, device, policy_cfg),
            "critic_1": QNetwork(obs_space, act_space, device, policy_cfg),
            "critic_2": QNetwork(obs_space, act_space, device, policy_cfg),
            "target_critic_1": QNetwork(obs_space, act_space, device, policy_cfg),
            "target_critic_2": QNetwork(obs_space, act_space, device, policy_cfg),
        }
        memory = RandomMemory(memory_size=cfg.memory_size or 100_000, num_envs=env.num_envs, device=device)
        agent_cfg = TD3_DEFAULT_CONFIG.copy()
        agent_cfg.update(
            {
                "batch_size": cfg.td3.batch_size,
                "discount_factor": cfg.td3.discount_factor,
                "polyak": cfg.td3.polyak,
                "actor_learning_rate": cfg.td3.actor_learning_rate,
                "critic_learning_rate": cfg.td3.critic_learning_rate,
                "policy_delay": cfg.td3.policy_delay,
                "smooth_regularization_noise": cfg.td3.smooth_regularization_noise,
                "smooth_regularization_clip": cfg.td3.smooth_regularization_clip,
            }
        )
        Agent = TD3

    elif cfg.algorithm == "ddpg":
        from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG

        models = {
            "policy": DeterministicPolicy(obs_space, act_space, device, policy_cfg),
            "target_policy": DeterministicPolicy(obs_space, act_space, device, policy_cfg),
            "critic": QNetwork(obs_space, act_space, device, policy_cfg),
            "target_critic": QNetwork(obs_space, act_space, device, policy_cfg),
        }
        memory = RandomMemory(memory_size=cfg.memory_size or 100_000, num_envs=env.num_envs, device=device)
        agent_cfg = DDPG_DEFAULT_CONFIG.copy()
        agent_cfg.update(
            {
                "batch_size": cfg.ddpg.batch_size,
                "discount_factor": cfg.ddpg.discount_factor,
                "polyak": cfg.ddpg.polyak,
                "actor_learning_rate": cfg.ddpg.actor_learning_rate,
                "critic_learning_rate": cfg.ddpg.critic_learning_rate,
            }
        )
        Agent = DDPG
    else:
        raise ValueError(f"Unknown algorithm: {cfg.algorithm}")

    # Common experiment config - disable skrl's built-in logging, we use IterationLogger
    agent_cfg["experiment"] = {
        "directory": cfg.directory,
        "experiment_name": cfg.experiment_name,
        "write_interval": 0,  # Disable skrl's logging
        "checkpoint_interval": 0,  # We handle checkpoints ourselves
    }

    # Move models to device
    for m in models.values():
        m.to(device)

    return Agent(
        models=models,
        memory=memory,
        cfg=agent_cfg,
        observation_space=obs_space,
        action_space=act_space,
        device=device,
    )


# =============================================================================
# Training
# =============================================================================


def _cfg_to_dict(cfg) -> dict:
    """Convert dataclass config to dict for logging."""
    from dataclasses import asdict, is_dataclass

    if is_dataclass(cfg):
        return asdict(cfg)
    return dict(cfg) if hasattr(cfg, "__iter__") else {}


def train(env_cfg: ManagerBasedRlEnvCfg, rl_cfg: SkrlCfg, device: str = "cpu") -> None:
    """Train using skrl with iteration logging."""
    from ..envs import ManagerBasedRlEnv
    from ..utils.iteration_logger import IterationLogger, IterationLoggerCfg
    from .wrapper import SkrlWrapper

    torch.manual_seed(rl_cfg.seed)
    if "cuda" in device:
        torch.cuda.manual_seed(rl_cfg.seed)

    logger.info(f"Training with {rl_cfg.algorithm.upper()} for {rl_cfg.timesteps:,} timesteps")

    # Setup
    env = ManagerBasedRlEnv(cfg=env_cfg)
    wrapped = SkrlWrapper(env, device=device)
    agent = _build_agent(wrapped, rl_cfg, device, total_timesteps=rl_cfg.timesteps)

    # Iteration logger
    rollout_steps = rl_cfg.ppo.rollouts if rl_cfg.algorithm == "ppo" else rl_cfg.rollout_steps
    max_iters = max(1, (rl_cfg.timesteps + rollout_steps - 1) // rollout_steps)

    if rl_cfg.timesteps < rollout_steps:
        logger.warning(f"timesteps ({rl_cfg.timesteps}) < rollout_steps ({rollout_steps})")

    iter_logger = IterationLogger(
        cfg=IterationLoggerCfg(
            log_dir=f"{rl_cfg.directory}/{rl_cfg.experiment_name}",
            log_interval=1,  # Always log every iteration to terminal
            logger=rl_cfg.logger or "none",
            wandb_project=rl_cfg.wandb_project,
            wandb_entity=rl_cfg.wandb_entity,
            experiment_name=rl_cfg.experiment_name,
        ),
        max_iterations=max_iters,
    )

    # Log hyperparameters to wandb
    iter_logger.log_config(
        {
            "device": device,
            "algorithm": rl_cfg.algorithm,
            "timesteps": rl_cfg.timesteps,
            "seed": rl_cfg.seed,
            "rollout_steps": rollout_steps,
            # RL config
            "rl": _cfg_to_dict(rl_cfg),
            # Env config (excluding non-serializable fields)
            "env": {
                "robot": env_cfg.robot,
                "scene": env_cfg.scene,
                "task": env_cfg.task,
                "decimation": env_cfg.decimation,
                "episode_length_s": env_cfg.episode_length_s,
                "sim_dt": env_cfg.sim_dt,
                "num_envs": env_cfg.num_envs,
            },
            # Network architecture
            "policy": _cfg_to_dict(rl_cfg.policy),
        }
    )

    # Training loop
    agent.init()
    states, _ = wrapped.reset()
    timestep, iteration = 0, 0
    ep_reward, ep_length = 0.0, 0

    while timestep < rl_cfg.timesteps:
        iteration += 1
        ep_rewards, ep_lengths = [], []
        t0 = time.time()
        steps = 0

        while steps < rollout_steps and timestep < rl_cfg.timesteps:
            agent.pre_interaction(timestep=timestep, timesteps=rl_cfg.timesteps)

            with torch.no_grad():
                actions = agent.act(states, timestep=timestep, timesteps=rl_cfg.timesteps)[0]
                next_states, rewards, terminated, truncated, _ = wrapped.step(actions)

                agent.record_transition(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos={},
                    timestep=timestep,
                    timesteps=rl_cfg.timesteps,
                )

                ep_reward += rewards.item()
                ep_length += 1

                if (terminated | truncated).item():
                    ep_rewards.append(ep_reward)
                    ep_lengths.append(float(ep_length))
                    ep_reward, ep_length = 0.0, 0
                    states, _ = wrapped.reset()
                else:
                    states = next_states

            agent.post_interaction(timestep=timestep, timesteps=rl_cfg.timesteps)
            timestep += 1
            steps += 1

        collection_time = time.time() - t0

        # Learning update
        t1 = time.time()
        # skrl does learning in post_interaction, but we can measure total iteration time
        learn_time = 0.0  # Learning happens during post_interaction calls above

        # Collect all metrics from skrl's tracking_data
        losses = {}
        if hasattr(agent, "tracking_data") and agent.tracking_data:
            for k, v in agent.tracking_data.items():
                # Get the latest value
                val = v[-1] if isinstance(v, list) and v else v
                if val is not None:
                    try:
                        losses[k] = float(val)
                    except (TypeError, ValueError):
                        pass

        # Get action std for exploration tracking
        action_std = None
        if hasattr(agent, "policy") and hasattr(agent.policy, "log_std"):
            action_std = agent.policy.log_std.exp().mean().item()

        iter_logger.log_iteration(
            iteration=iteration,
            losses=losses,
            episode_rewards=ep_rewards,
            episode_lengths=ep_lengths,
            num_steps=steps,
            collection_time=collection_time,
            learn_time=learn_time,
            action_std=action_std,
        )

    iter_logger.close()
    wrapped.close()
    logger.info("Training complete")


def load_agent(checkpoint_path: str, env_cfg: ManagerBasedRlEnvCfg, rl_cfg: SkrlCfg, device: str = "cpu"):
    """Load a trained agent from checkpoint."""
    from ..envs import ManagerBasedRlEnv
    from .wrapper import SkrlWrapper

    env = ManagerBasedRlEnv(cfg=env_cfg)
    wrapped = SkrlWrapper(env, device=device)
    agent = _build_agent(wrapped, rl_cfg, device, total_timesteps=rl_cfg.timesteps)
    agent.load(checkpoint_path)
    return agent, wrapped
