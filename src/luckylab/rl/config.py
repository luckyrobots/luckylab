"""skrl configuration - simple configs that map to skrl's native API."""

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ActorCriticCfg:
    """Actor-Critic network configuration."""

    actor_hidden_dims: tuple[int, ...] = (256, 256, 256)
    """Hidden layer dimensions for the actor network."""
    critic_hidden_dims: tuple[int, ...] = (256, 256, 256)
    """Hidden layer dimensions for the critic network."""
    activation: str = "elu"
    """Activation function ('elu', 'relu', 'tanh', 'leaky_relu')."""
    init_noise_std: float = 1.0
    """Initial noise standard deviation for stochastic policies."""

    # Observation normalization (matches mjlab's actor/critic obs normalization)
    normalize_actor_obs: bool = False
    """Whether to normalize observations for actor (running mean/std)."""
    normalize_critic_obs: bool = False
    """Whether to normalize observations for critic (running mean/std)."""


@dataclass
class PpoCfg:
    """PPO algorithm configuration."""

    rollouts: int = 1024
    """Steps per environment before update."""
    learning_epochs: int = 5
    """Training passes per update."""
    mini_batches: int = 4
    """Mini-batch divisions."""
    discount_factor: float = 0.99
    """Gamma."""
    lambda_gae: float = 0.95
    """GAE lambda."""
    learning_rate: float = 1e-3
    """Learning rate."""
    ratio_clip: float = 0.2
    """PPO clip parameter."""
    value_loss_scale: float = 1.0
    """Value loss coefficient."""
    entropy_loss_scale: float = 0.01
    """Entropy bonus coefficient."""
    grad_norm_clip: float = 1.0
    """Max gradient norm."""
    kl_threshold: float = 0.01
    """KL threshold for early stopping (matches mjlab's desired_kl=0.01)."""

    # Learning rate scheduling (matches mjlab's adaptive/fixed schedule)
    lr_schedule: Literal["fixed", "adaptive", "linear"] = "adaptive"
    """Learning rate schedule type:
    - 'fixed': constant learning rate
    - 'adaptive': reduce LR when KL > 2*kl_threshold, increase when KL < 0.5*kl_threshold
    - 'linear': linearly decay to 0 over training
    """
    lr_schedule_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional kwargs for lr scheduler (e.g., {'end_factor': 0.1} for linear)."""


@dataclass
class SacCfg:
    """SAC algorithm configuration."""

    batch_size: int = 256
    """Mini-batch size."""
    discount_factor: float = 0.99
    """Gamma."""
    polyak: float = 0.005
    """Polyak averaging coefficient."""
    actor_learning_rate: float = 3e-4
    """Actor learning rate."""
    critic_learning_rate: float = 3e-4
    """Critic learning rate."""
    learn_entropy: bool = True
    """Whether to learn entropy coefficient."""
    initial_entropy: float = 0.2
    """Initial entropy coefficient."""
    target_entropy: float | None = None
    """Target entropy (None = auto)."""
    grad_norm_clip: float = 0.0
    """Max gradient norm (0 = disabled)."""
    random_timesteps: int = 10000
    """Number of timesteps with random actions for initial exploration."""
    learning_starts: int = 1000
    """Number of timesteps before learning starts (buffer must have this many samples)."""


@dataclass
class Td3Cfg:
    """TD3 algorithm configuration."""

    batch_size: int = 256
    """Mini-batch size."""
    discount_factor: float = 0.99
    """Gamma."""
    polyak: float = 0.005
    """Polyak averaging coefficient."""
    actor_learning_rate: float = 3e-4
    """Actor learning rate."""
    critic_learning_rate: float = 3e-4
    """Critic learning rate."""
    policy_delay: int = 2
    """Delayed policy updates."""
    smooth_regularization_noise: float = 0.1
    """Target policy smoothing noise."""
    smooth_regularization_clip: float = 0.5
    """Noise clip."""


@dataclass
class DdpgCfg:
    """DDPG algorithm configuration."""

    batch_size: int = 256
    """Mini-batch size."""
    discount_factor: float = 0.99
    """Gamma."""
    polyak: float = 0.005
    """Polyak averaging coefficient."""
    actor_learning_rate: float = 3e-4
    """Actor learning rate."""
    critic_learning_rate: float = 3e-4
    """Critic learning rate."""


@dataclass
class SkrlCfg:
    """
    Top-level skrl training configuration.

    Supports multiple algorithms: PPO, SAC, TD3, DDPG.
    Uses PyTorch backend with skrl's training loop.
    """

    # Algorithm selection
    algorithm: Literal["ppo", "sac", "td3", "ddpg"] = "ppo"
    """Algorithm to use."""

    # Training
    seed: int = 42
    """Random seed."""
    timesteps: int = 1_000_000
    """Total training timesteps."""
    rollout_steps: int = 24
    """Steps per iteration (collection phase). For PPO this is steps_per_env."""
    memory_size: int | None = None
    """Replay buffer size (None = auto based on algorithm)."""

    # Network
    policy: ActorCriticCfg = field(default_factory=ActorCriticCfg)
    """Actor-Critic network config."""

    # Algorithm-specific configs (only the selected one is used)
    ppo: PpoCfg = field(default_factory=PpoCfg)
    """PPO config (used if algorithm='ppo')."""
    sac: SacCfg = field(default_factory=SacCfg)
    """SAC config (used if algorithm='sac')."""
    td3: Td3Cfg = field(default_factory=Td3Cfg)
    """TD3 config (used if algorithm='td3')."""
    ddpg: DdpgCfg = field(default_factory=DdpgCfg)
    """DDPG config (used if algorithm='ddpg')."""

    # Experiment
    experiment_name: str = "luckylab"
    """Experiment name for logging."""
    directory: str = "runs"
    """Output directory."""
    checkpoint_interval: int = 1000
    """Timesteps between checkpoints (0 = disabled)."""

    # Logging
    logger: Literal["wandb"] | None = "wandb"
    """Logger backend (None = disabled)."""
    wandb_project: str = "luckylab"
    """W&B project name."""
    wandb_entity: str | None = "mjlab"
    """W&B entity (team/user name)."""

    # Scope for simultaneous training
    scope: str | None = None
    """
    Scope name for simultaneous training.
    Multiple agents with different scopes can train in parallel on the same env.
    """
