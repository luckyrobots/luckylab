"""RL training configuration."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ActorCriticCfg:
    """Actor-Critic network configuration."""

    init_noise_std: float = 1.0
    """The initial noise standard deviation of the policy."""
    noise_type: Literal["gaussian", "gsde"] = "gaussian"
    """Exploration noise type. 'gaussian' = independent per-step noise,
    'gsde' = generalized State-Dependent Exploration (temporally coherent,
    state-dependent noise for discovering coordinated locomotion)."""
    gsde_resample_interval: int = 50
    """How often to resample gSDE exploration direction (in env steps).
    Lower = more variety, higher = more temporal coherence. Only used when noise_type='gsde'."""
    use_delta_actions: bool = False
    """Interpret policy output as deltas accumulated into a running target.
    Required for position-controlled actuators with gSDE exploration."""
    delta_action_scale: float = 0.2
    """Per-step scaling for delta actions. Each step moves joints by
    action * delta_action_scale in normalized [-1, 1] space."""
    actor_hidden_dims: tuple[int, ...] = (128, 128, 128)
    """The hidden dimensions of the actor network."""
    critic_hidden_dims: tuple[int, ...] = (128, 128, 128)
    """The hidden dimensions of the critic network."""
    activation: str = "elu"
    """The activation function to use in the actor and critic networks."""


@dataclass
class PpoAlgorithmCfg:
    """Config for the PPO algorithm."""

    num_learning_epochs: int = 5
    """The number of learning epochs per update."""
    num_mini_batches: int = 4
    """The number of mini-batches per update.
    mini batch size = num_envs * num_steps / num_mini_batches
    """
    learning_rate: float = 1e-3
    """The learning rate."""
    schedule: Literal["adaptive", "fixed"] = "adaptive"
    """The learning rate schedule."""
    gamma: float = 0.99
    """The discount factor."""
    lam: float = 0.95
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""
    entropy_coef: float = 0.005
    """The coefficient for the entropy loss."""
    max_grad_norm: float = 1.0
    """The maximum gradient norm for the policy."""
    value_loss_coef: float = 1.0
    """The coefficient for the value loss."""
    use_clipped_value_loss: bool = True
    """Whether to use clipped value loss."""
    clip_param: float = 0.2
    """The clipping parameter for the policy."""
    desired_kl: float = 0.01
    """The desired KL divergence between the new and old policies."""
    num_steps_per_env: int = 24
    """The number of steps per environment per update."""
    clip_actions: float | None = None
    """Clamp actions to [-clip, +clip] before sending to env. None = no clipping."""


@dataclass
class SacAlgorithmCfg:
    """SAC algorithm configuration."""

    memory_size: int = 1_000_000
    """Replay buffer size."""
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
    random_timesteps: int = 1000
    """Number of timesteps with random actions for initial exploration."""
    learning_starts: int = 1000
    """Number of timesteps before learning starts (buffer must have this many samples).
    Should match random_timesteps to ensure actor and critic start training together."""
    gradient_steps: int = 1
    """Number of gradient updates per environment step. Higher values (e.g., 32)
    improve sample efficiency for single-env training."""
    clip_actions: float | None = None
    """Clamp actions to [-clip, +clip] before sending to env. None = no clipping."""


@dataclass
class Td3AlgorithmCfg:
    """TD3 algorithm configuration."""

    memory_size: int = 1_000_000
    """Replay buffer size."""
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
    clip_actions: float | None = None
    """Clamp actions to [-clip, +clip] before sending to env. None = no clipping."""


@dataclass
class DdpgAlgorithmCfg:
    """DDPG algorithm configuration."""

    memory_size: int = 1_000_000
    """Replay buffer size."""
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
    clip_actions: float | None = None
    """Clamp actions to [-clip, +clip] before sending to env. None = no clipping."""


@dataclass
class RlRunnerCfg:
    """
    Top-level RL training configuration.

    Supports multiple algorithms: PPO, SAC, TD3, DDPG.
    Supports multiple backends: skrl, sb3 (Stable Baselines3).
    """

    # Backend selection
    backend: str = ""
    """RL backend library. Must be 'skrl' or 'sb3' (Stable Baselines3)."""

    # Algorithm selection
    algorithm: Literal["ppo", "sac", "td3", "ddpg"] = "ppo"
    """Algorithm to use."""

    # Training
    seed: int = 42
    """The seed for the experiment. Default is 42."""
    max_iterations: int = 1500
    """The maximum number of iterations."""

    # Network
    policy: ActorCriticCfg = field(default_factory=ActorCriticCfg)
    """Actor-Critic network config."""

    # Algorithm-specific configs (only the selected one is used)
    ppo: PpoAlgorithmCfg = field(default_factory=PpoAlgorithmCfg)
    """PPO config (used if algorithm='ppo')."""
    sac: SacAlgorithmCfg = field(default_factory=SacAlgorithmCfg)
    """SAC config (used if algorithm='sac')."""
    td3: Td3AlgorithmCfg = field(default_factory=Td3AlgorithmCfg)
    """TD3 config (used if algorithm='td3')."""
    ddpg: DdpgAlgorithmCfg = field(default_factory=DdpgAlgorithmCfg)
    """DDPG config (used if algorithm='ddpg')."""

    # Experiment
    experiment_name: str = "luckylab"
    """The experiment name."""
    directory: str = "runs"
    """Output directory."""
    checkpoint_interval: int = 100
    """Save checkpoint every N iterations (0 to disable)."""
    log_interval: int = 10
    """Log metrics every N iterations (0 to disable)."""

    # Resume
    resume_from: str | None = None
    """Path to checkpoint to resume training from."""

    # Logging
    wandb: bool = True
    """Enable wandb logging."""
    wandb_project: str = "luckylab"
    """W&B project name."""
    wandb_entity: str | None = None
    """W&B entity (team/user name)."""
