"""Training utilities using skrl's native training loop."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .config import SkrlCfg, ActorCriticCfg

if TYPE_CHECKING:
    from ..envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg

logger = logging.getLogger(__name__)


# =============================================================================
# PyTorch Models
# =============================================================================


def _get_activation_torch(name: str):
    """Get PyTorch activation by name."""
    import torch.nn as nn
    return {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh, "leaky_relu": nn.LeakyReLU}[name.lower()]


def _build_mlp_torch(input_dim: int, output_dim: int, hidden_dims: tuple[int, ...], activation: str):
    """Build PyTorch MLP."""
    import torch.nn as nn
    act = _get_activation_torch(activation)
    layers = []
    prev = input_dim
    for dim in hidden_dims:
        layers.extend([nn.Linear(prev, dim), act()])
        prev = dim
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


def _create_torch_models(env, cfg: ActorCriticCfg, device: str) -> dict:
    """Create PyTorch models for skrl."""
    import torch
    import torch.nn as nn
    from skrl.models.torch import Model, GaussianMixin, DeterministicMixin

    class GaussianPolicy(GaussianMixin, Model):
        def __init__(self, observation_space, action_space, device, cfg: ActorCriticCfg):
            Model.__init__(self, observation_space, action_space, device)
            GaussianMixin.__init__(self, clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2)
            self.net = _build_mlp_torch(observation_space.shape[0], action_space.shape[0], cfg.actor_hidden_dims, cfg.activation)
            self.log_std = nn.Parameter(torch.full((action_space.shape[0],), cfg.init_noise_std).log())

        def compute(self, inputs, role=""):
            return self.net(inputs["states"]), self.log_std.expand_as(self.net(inputs["states"])), {}

    class DeterministicPolicy(DeterministicMixin, Model):
        def __init__(self, observation_space, action_space, device, cfg: ActorCriticCfg):
            Model.__init__(self, observation_space, action_space, device)
            DeterministicMixin.__init__(self, clip_actions=False)
            self.net = _build_mlp_torch(observation_space.shape[0], action_space.shape[0], cfg.actor_hidden_dims, cfg.activation)

        def compute(self, inputs, role=""):
            return torch.tanh(self.net(inputs["states"])), {}

    class ValueNetwork(DeterministicMixin, Model):
        def __init__(self, observation_space, action_space, device, cfg: ActorCriticCfg):
            Model.__init__(self, observation_space, action_space, device)
            DeterministicMixin.__init__(self)
            self.net = _build_mlp_torch(observation_space.shape[0], 1, cfg.critic_hidden_dims, cfg.activation)

        def compute(self, inputs, role=""):
            return self.net(inputs["states"]), {}

    class QNetwork(DeterministicMixin, Model):
        def __init__(self, observation_space, action_space, device, cfg: ActorCriticCfg):
            Model.__init__(self, observation_space, action_space, device)
            DeterministicMixin.__init__(self)
            input_dim = observation_space.shape[0] + action_space.shape[0]
            self.net = _build_mlp_torch(input_dim, 1, cfg.critic_hidden_dims, cfg.activation)

        def compute(self, inputs, role=""):
            x = torch.cat([inputs["states"], inputs["taken_actions"]], dim=-1)
            return self.net(x), {}

    return {
        "GaussianPolicy": GaussianPolicy,
        "DeterministicPolicy": DeterministicPolicy,
        "ValueNetwork": ValueNetwork,
        "QNetwork": QNetwork,
    }


# =============================================================================
# JAX Models
# =============================================================================


def _get_activation_jax(name: str):
    """Get JAX activation by name."""
    import flax.linen as nn
    return {"elu": nn.elu, "relu": nn.relu, "tanh": nn.tanh, "leaky_relu": nn.leaky_relu}[name.lower()]


def _create_jax_models(env, cfg: ActorCriticCfg, device: str) -> dict:
    """Create JAX models for skrl."""
    import jax
    import jax.numpy as jnp
    import flax.linen as nn
    from skrl.models.jax import Model, GaussianMixin, DeterministicMixin

    class GaussianPolicy(GaussianMixin, Model):
        def __init__(self, observation_space, action_space, device, cfg: ActorCriticCfg, **kwargs):
            Model.__init__(self, observation_space, action_space, device, **kwargs)
            GaussianMixin.__init__(self, clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2)
            self.cfg = cfg

        def setup(self):
            act = _get_activation_jax(self.cfg.activation)
            layers = []
            for dim in self.cfg.actor_hidden_dims:
                layers.extend([nn.Dense(dim), act])
            layers.append(nn.Dense(self.num_actions))
            self.net = nn.Sequential(layers)
            self.log_std = self.param("log_std", nn.initializers.constant(jnp.log(self.cfg.init_noise_std)), (self.num_actions,))

        def __call__(self, inputs, role=""):
            mean = self.net(inputs["states"])
            return mean, self.log_std, {}

    class DeterministicPolicy(DeterministicMixin, Model):
        def __init__(self, observation_space, action_space, device, cfg: ActorCriticCfg, **kwargs):
            Model.__init__(self, observation_space, action_space, device, **kwargs)
            DeterministicMixin.__init__(self, clip_actions=False)
            self.cfg = cfg

        def setup(self):
            act = _get_activation_jax(self.cfg.activation)
            layers = []
            for dim in self.cfg.actor_hidden_dims:
                layers.extend([nn.Dense(dim), act])
            layers.append(nn.Dense(self.num_actions))
            self.net = nn.Sequential(layers)

        def __call__(self, inputs, role=""):
            return jnp.tanh(self.net(inputs["states"])), {}

    class ValueNetwork(DeterministicMixin, Model):
        def __init__(self, observation_space, action_space, device, cfg: ActorCriticCfg, **kwargs):
            Model.__init__(self, observation_space, action_space, device, **kwargs)
            DeterministicMixin.__init__(self)
            self.cfg = cfg

        def setup(self):
            act = _get_activation_jax(self.cfg.activation)
            layers = []
            for dim in self.cfg.critic_hidden_dims:
                layers.extend([nn.Dense(dim), act])
            layers.append(nn.Dense(1))
            self.net = nn.Sequential(layers)

        def __call__(self, inputs, role=""):
            return self.net(inputs["states"]), {}

    class QNetwork(DeterministicMixin, Model):
        def __init__(self, observation_space, action_space, device, cfg: ActorCriticCfg, **kwargs):
            Model.__init__(self, observation_space, action_space, device, **kwargs)
            DeterministicMixin.__init__(self)
            self.cfg = cfg

        def setup(self):
            act = _get_activation_jax(self.cfg.activation)
            layers = []
            for dim in self.cfg.critic_hidden_dims:
                layers.extend([nn.Dense(dim), act])
            layers.append(nn.Dense(1))
            self.net = nn.Sequential(layers)

        def __call__(self, inputs, role=""):
            x = jnp.concatenate([inputs["states"], inputs["taken_actions"]], axis=-1)
            return self.net(x), {}

    return {
        "GaussianPolicy": GaussianPolicy,
        "DeterministicPolicy": DeterministicPolicy,
        "ValueNetwork": ValueNetwork,
        "QNetwork": QNetwork,
    }


# =============================================================================
# Agent Builders
# =============================================================================


def _build_ppo_torch(env, cfg: SkrlCfg, device: str):
    """Build PyTorch PPO agent."""
    import torch
    from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
    from skrl.memories.torch import RandomMemory

    models_cls = _create_torch_models(env, cfg.policy, device)
    models = {
        "policy": models_cls["GaussianPolicy"](env.observation_space, env.action_space, device, cfg.policy),
        "value": models_cls["ValueNetwork"](env.observation_space, env.action_space, device, cfg.policy),
    }
    for m in models.values():
        m.to(device)

    memory = RandomMemory(memory_size=cfg.ppo.rollouts, num_envs=env.num_envs, device=device)

    agent_cfg = PPO_DEFAULT_CONFIG.copy()
    agent_cfg.update({
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
        "experiment": _experiment_cfg(cfg),
    })

    return PPO(models=models, memory=memory, cfg=agent_cfg,
               observation_space=env.observation_space, action_space=env.action_space, device=device)


def _build_ppo_jax(env, cfg: SkrlCfg, device: str):
    """Build JAX PPO agent."""
    from skrl.agents.jax.ppo import PPO, PPO_DEFAULT_CONFIG
    from skrl.memories.jax import RandomMemory

    models_cls = _create_jax_models(env, cfg.policy, device)
    models = {
        "policy": models_cls["GaussianPolicy"](env.observation_space, env.action_space, device, cfg.policy),
        "value": models_cls["ValueNetwork"](env.observation_space, env.action_space, device, cfg.policy),
    }

    memory = RandomMemory(memory_size=cfg.ppo.rollouts, num_envs=env.num_envs, device=device)

    agent_cfg = PPO_DEFAULT_CONFIG.copy()
    agent_cfg.update({
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
        "experiment": _experiment_cfg(cfg),
    })

    return PPO(models=models, memory=memory, cfg=agent_cfg,
               observation_space=env.observation_space, action_space=env.action_space, device=device)


def _build_sac_torch(env, cfg: SkrlCfg, device: str):
    """Build PyTorch SAC agent."""
    from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
    from skrl.memories.torch import RandomMemory

    models_cls = _create_torch_models(env, cfg.policy, device)
    models = {
        "policy": models_cls["DeterministicPolicy"](env.observation_space, env.action_space, device, cfg.policy),
        "critic_1": models_cls["QNetwork"](env.observation_space, env.action_space, device, cfg.policy),
        "critic_2": models_cls["QNetwork"](env.observation_space, env.action_space, device, cfg.policy),
        "target_critic_1": models_cls["QNetwork"](env.observation_space, env.action_space, device, cfg.policy),
        "target_critic_2": models_cls["QNetwork"](env.observation_space, env.action_space, device, cfg.policy),
    }
    for m in models.values():
        m.to(device)

    memory_size = cfg.memory_size or 100_000
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=device)

    agent_cfg = SAC_DEFAULT_CONFIG.copy()
    agent_cfg.update({
        "batch_size": cfg.sac.batch_size,
        "discount_factor": cfg.sac.discount_factor,
        "polyak": cfg.sac.polyak,
        "actor_learning_rate": cfg.sac.actor_learning_rate,
        "critic_learning_rate": cfg.sac.critic_learning_rate,
        "learn_entropy": cfg.sac.learn_entropy,
        "entropy_learning_rate": cfg.sac.actor_learning_rate,
        "initial_entropy_value": cfg.sac.initial_entropy,
        "grad_norm_clip": cfg.sac.grad_norm_clip,
        "experiment": _experiment_cfg(cfg),
    })
    if cfg.sac.target_entropy is not None:
        agent_cfg["target_entropy"] = cfg.sac.target_entropy

    return SAC(models=models, memory=memory, cfg=agent_cfg,
               observation_space=env.observation_space, action_space=env.action_space, device=device)


def _build_sac_jax(env, cfg: SkrlCfg, device: str):
    """Build JAX SAC agent."""
    from skrl.agents.jax.sac import SAC, SAC_DEFAULT_CONFIG
    from skrl.memories.jax import RandomMemory

    models_cls = _create_jax_models(env, cfg.policy, device)
    models = {
        "policy": models_cls["DeterministicPolicy"](env.observation_space, env.action_space, device, cfg.policy),
        "critic_1": models_cls["QNetwork"](env.observation_space, env.action_space, device, cfg.policy),
        "critic_2": models_cls["QNetwork"](env.observation_space, env.action_space, device, cfg.policy),
        "target_critic_1": models_cls["QNetwork"](env.observation_space, env.action_space, device, cfg.policy),
        "target_critic_2": models_cls["QNetwork"](env.observation_space, env.action_space, device, cfg.policy),
    }

    memory_size = cfg.memory_size or 100_000
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=device)

    agent_cfg = SAC_DEFAULT_CONFIG.copy()
    agent_cfg.update({
        "batch_size": cfg.sac.batch_size,
        "discount_factor": cfg.sac.discount_factor,
        "polyak": cfg.sac.polyak,
        "actor_learning_rate": cfg.sac.actor_learning_rate,
        "critic_learning_rate": cfg.sac.critic_learning_rate,
        "learn_entropy": cfg.sac.learn_entropy,
        "entropy_learning_rate": cfg.sac.actor_learning_rate,
        "initial_entropy_value": cfg.sac.initial_entropy,
        "grad_norm_clip": cfg.sac.grad_norm_clip,
        "experiment": _experiment_cfg(cfg),
    })
    if cfg.sac.target_entropy is not None:
        agent_cfg["target_entropy"] = cfg.sac.target_entropy

    return SAC(models=models, memory=memory, cfg=agent_cfg,
               observation_space=env.observation_space, action_space=env.action_space, device=device)


def _build_td3_torch(env, cfg: SkrlCfg, device: str):
    """Build PyTorch TD3 agent."""
    from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG
    from skrl.memories.torch import RandomMemory

    models_cls = _create_torch_models(env, cfg.policy, device)
    models = {
        "policy": models_cls["DeterministicPolicy"](env.observation_space, env.action_space, device, cfg.policy),
        "target_policy": models_cls["DeterministicPolicy"](env.observation_space, env.action_space, device, cfg.policy),
        "critic_1": models_cls["QNetwork"](env.observation_space, env.action_space, device, cfg.policy),
        "critic_2": models_cls["QNetwork"](env.observation_space, env.action_space, device, cfg.policy),
        "target_critic_1": models_cls["QNetwork"](env.observation_space, env.action_space, device, cfg.policy),
        "target_critic_2": models_cls["QNetwork"](env.observation_space, env.action_space, device, cfg.policy),
    }
    for m in models.values():
        m.to(device)

    memory_size = cfg.memory_size or 100_000
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=device)

    agent_cfg = TD3_DEFAULT_CONFIG.copy()
    agent_cfg.update({
        "batch_size": cfg.td3.batch_size,
        "discount_factor": cfg.td3.discount_factor,
        "polyak": cfg.td3.polyak,
        "actor_learning_rate": cfg.td3.actor_learning_rate,
        "critic_learning_rate": cfg.td3.critic_learning_rate,
        "policy_delay": cfg.td3.policy_delay,
        "smooth_regularization_noise": cfg.td3.smooth_regularization_noise,
        "smooth_regularization_clip": cfg.td3.smooth_regularization_clip,
        "experiment": _experiment_cfg(cfg),
    })

    return TD3(models=models, memory=memory, cfg=agent_cfg,
               observation_space=env.observation_space, action_space=env.action_space, device=device)


def _build_td3_jax(env, cfg: SkrlCfg, device: str):
    """Build JAX TD3 agent."""
    from skrl.agents.jax.td3 import TD3, TD3_DEFAULT_CONFIG
    from skrl.memories.jax import RandomMemory

    models_cls = _create_jax_models(env, cfg.policy, device)
    models = {
        "policy": models_cls["DeterministicPolicy"](env.observation_space, env.action_space, device, cfg.policy),
        "target_policy": models_cls["DeterministicPolicy"](env.observation_space, env.action_space, device, cfg.policy),
        "critic_1": models_cls["QNetwork"](env.observation_space, env.action_space, device, cfg.policy),
        "critic_2": models_cls["QNetwork"](env.observation_space, env.action_space, device, cfg.policy),
        "target_critic_1": models_cls["QNetwork"](env.observation_space, env.action_space, device, cfg.policy),
        "target_critic_2": models_cls["QNetwork"](env.observation_space, env.action_space, device, cfg.policy),
    }

    memory_size = cfg.memory_size or 100_000
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=device)

    agent_cfg = TD3_DEFAULT_CONFIG.copy()
    agent_cfg.update({
        "batch_size": cfg.td3.batch_size,
        "discount_factor": cfg.td3.discount_factor,
        "polyak": cfg.td3.polyak,
        "actor_learning_rate": cfg.td3.actor_learning_rate,
        "critic_learning_rate": cfg.td3.critic_learning_rate,
        "policy_delay": cfg.td3.policy_delay,
        "smooth_regularization_noise": cfg.td3.smooth_regularization_noise,
        "smooth_regularization_clip": cfg.td3.smooth_regularization_clip,
        "experiment": _experiment_cfg(cfg),
    })

    return TD3(models=models, memory=memory, cfg=agent_cfg,
               observation_space=env.observation_space, action_space=env.action_space, device=device)


def _build_ddpg_torch(env, cfg: SkrlCfg, device: str):
    """Build PyTorch DDPG agent."""
    from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG
    from skrl.memories.torch import RandomMemory

    models_cls = _create_torch_models(env, cfg.policy, device)
    models = {
        "policy": models_cls["DeterministicPolicy"](env.observation_space, env.action_space, device, cfg.policy),
        "target_policy": models_cls["DeterministicPolicy"](env.observation_space, env.action_space, device, cfg.policy),
        "critic": models_cls["QNetwork"](env.observation_space, env.action_space, device, cfg.policy),
        "target_critic": models_cls["QNetwork"](env.observation_space, env.action_space, device, cfg.policy),
    }
    for m in models.values():
        m.to(device)

    memory_size = cfg.memory_size or 100_000
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=device)

    agent_cfg = DDPG_DEFAULT_CONFIG.copy()
    agent_cfg.update({
        "batch_size": cfg.ddpg.batch_size,
        "discount_factor": cfg.ddpg.discount_factor,
        "polyak": cfg.ddpg.polyak,
        "actor_learning_rate": cfg.ddpg.actor_learning_rate,
        "critic_learning_rate": cfg.ddpg.critic_learning_rate,
        "experiment": _experiment_cfg(cfg),
    })

    return DDPG(models=models, memory=memory, cfg=agent_cfg,
                observation_space=env.observation_space, action_space=env.action_space, device=device)


def _build_ddpg_jax(env, cfg: SkrlCfg, device: str):
    """Build JAX DDPG agent."""
    from skrl.agents.jax.ddpg import DDPG, DDPG_DEFAULT_CONFIG
    from skrl.memories.jax import RandomMemory

    models_cls = _create_jax_models(env, cfg.policy, device)
    models = {
        "policy": models_cls["DeterministicPolicy"](env.observation_space, env.action_space, device, cfg.policy),
        "target_policy": models_cls["DeterministicPolicy"](env.observation_space, env.action_space, device, cfg.policy),
        "critic": models_cls["QNetwork"](env.observation_space, env.action_space, device, cfg.policy),
        "target_critic": models_cls["QNetwork"](env.observation_space, env.action_space, device, cfg.policy),
    }

    memory_size = cfg.memory_size or 100_000
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=device)

    agent_cfg = DDPG_DEFAULT_CONFIG.copy()
    agent_cfg.update({
        "batch_size": cfg.ddpg.batch_size,
        "discount_factor": cfg.ddpg.discount_factor,
        "polyak": cfg.ddpg.polyak,
        "actor_learning_rate": cfg.ddpg.actor_learning_rate,
        "critic_learning_rate": cfg.ddpg.critic_learning_rate,
        "experiment": _experiment_cfg(cfg),
    })

    return DDPG(models=models, memory=memory, cfg=agent_cfg,
                observation_space=env.observation_space, action_space=env.action_space, device=device)


def _experiment_cfg(cfg: SkrlCfg) -> dict:
    """Build experiment config for skrl."""
    exp = {
        "directory": cfg.directory,
        "experiment_name": cfg.experiment_name,
        "write_interval": max(1, cfg.checkpoint_interval // 10) if cfg.checkpoint_interval else 100,
        "checkpoint_interval": cfg.checkpoint_interval,
    }
    if cfg.logger == "wandb":
        exp["wandb"] = True
        exp["wandb_kwargs"] = {"project": cfg.wandb_project}
    return exp


_BUILDERS = {
    ("ppo", "torch"): _build_ppo_torch,
    ("ppo", "jax"): _build_ppo_jax,
    ("sac", "torch"): _build_sac_torch,
    ("sac", "jax"): _build_sac_jax,
    ("td3", "torch"): _build_td3_torch,
    ("td3", "jax"): _build_td3_jax,
    ("ddpg", "torch"): _build_ddpg_torch,
    ("ddpg", "jax"): _build_ddpg_jax,
}


# =============================================================================
# Public API
# =============================================================================


def train(
    env_cfg: ManagerBasedRlEnvCfg,
    rl_cfg: SkrlCfg,
    device: str = "cpu",
) -> None:
    """
    Train using skrl.

    Args:
        env_cfg: Environment configuration.
        rl_cfg: RL training configuration.
        device: Device ('cpu', 'cuda', or 'cuda:0' for JAX).

    Example:
        >>> from luckylab.rl import train, SkrlCfg
        >>> from luckylab.tasks.velocity import ManagerBasedRlEnvCfg
        >>>
        >>> # PyTorch PPO training
        >>> train(ManagerBasedRlEnvCfg(), SkrlCfg(backend="torch", algorithm="ppo"))
        >>>
        >>> # JAX SAC training
        >>> train(ManagerBasedRlEnvCfg(), SkrlCfg(backend="jax", algorithm="sac"))
    """
    from ..envs import ManagerBasedRlEnv
    from .wrapper import SkrlWrapper

    # Set random seeds
    if rl_cfg.backend == "torch":
        import torch
        torch.manual_seed(rl_cfg.seed)
        if "cuda" in device:
            torch.cuda.manual_seed(rl_cfg.seed)
    else:
        import jax
        # JAX uses a different seeding mechanism handled by skrl

    logger.info(f"Training with {rl_cfg.algorithm.upper()} ({rl_cfg.backend}) for {rl_cfg.timesteps:,} timesteps")

    # Create env
    env = ManagerBasedRlEnv(cfg=env_cfg)
    wrapped = SkrlWrapper(env, device=device)

    # Build agent
    builder_key = (rl_cfg.algorithm, rl_cfg.backend)
    if builder_key not in _BUILDERS:
        raise ValueError(f"Unsupported algorithm/backend combination: {rl_cfg.algorithm}/{rl_cfg.backend}")
    builder = _BUILDERS[builder_key]
    agent = builder(wrapped, rl_cfg, device)

    # Train using skrl's trainer
    if rl_cfg.backend == "torch":
        from skrl.trainers.torch import SequentialTrainer
    else:
        from skrl.trainers.jax import SequentialTrainer

    trainer = SequentialTrainer(
        env=wrapped,
        agents=agent,
        cfg={"timesteps": rl_cfg.timesteps, "headless": True},
    )
    trainer.train()

    wrapped.close()
    logger.info("Training complete")


def load_agent(
    checkpoint_path: str,
    env_cfg: ManagerBasedRlEnvCfg,
    rl_cfg: SkrlCfg,
    device: str = "cpu",
):
    """Load a trained agent from checkpoint."""
    from ..envs import ManagerBasedRlEnv
    from .wrapper import SkrlWrapper

    env = ManagerBasedRlEnv(cfg=env_cfg)
    wrapped = SkrlWrapper(env, device=device)

    builder_key = (rl_cfg.algorithm, rl_cfg.backend)
    if builder_key not in _BUILDERS:
        raise ValueError(f"Unsupported algorithm/backend combination: {rl_cfg.algorithm}/{rl_cfg.backend}")
    builder = _BUILDERS[builder_key]
    agent = builder(wrapped, rl_cfg, device)
    agent.load(checkpoint_path)

    return agent, wrapped
