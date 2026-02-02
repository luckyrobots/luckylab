"""Unitree Go1 RL configuration for velocity task."""

from luckylab.rl.config import ActorCriticCfg, PpoCfg, SacCfg, SkrlCfg

UNITREE_GO1_PPO_RUNNER_CFG = SkrlCfg(
    algorithm="ppo",
    seed=42,
    timesteps=1_000_000,
    policy=ActorCriticCfg(
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation="elu",
        init_noise_std=1.0,
        # Observation normalization (matches mjlab)
        normalize_actor_obs=False,  # mjlab default is False
        normalize_critic_obs=False,  # mjlab default is False
    ),
    ppo=PpoCfg(
        rollouts=1024,
        learning_epochs=5,
        mini_batches=4,
        discount_factor=0.99,
        lambda_gae=0.95,
        learning_rate=1e-3,
        ratio_clip=0.2,
        value_loss_scale=1.0,
        entropy_loss_scale=0.01,
        grad_norm_clip=1.0,
        # KL early stopping and adaptive LR (matches mjlab's desired_kl=0.01)
        kl_threshold=0.01,
        lr_schedule="adaptive",
    ),
    experiment_name="go1_velocity",
    directory="runs",
    checkpoint_interval=10000,
    logger="wandb",
    wandb_project="luckylab",
    wandb_entity="mjlab",
)

# SAC configuration for Go1 velocity task
# SAC is sample-efficient and good for single-env training
UNITREE_GO1_SAC_RUNNER_CFG = SkrlCfg(
    algorithm="sac",
    seed=42,
    timesteps=1_000_000,
    memory_size=1_000_000,  # Large replay buffer for off-policy learning
    rollout_steps=1,  # SAC typically updates every step
    policy=ActorCriticCfg(
        actor_hidden_dims=(256, 256, 256),  # Standard SAC architecture
        critic_hidden_dims=(256, 256, 256),
        activation="relu",  # ReLU is standard for SAC
        init_noise_std=1.0,
    ),
    sac=SacCfg(
        batch_size=256,
        discount_factor=0.99,
        polyak=0.005,  # Soft target update
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        learn_entropy=True,  # Auto-tune entropy coefficient
        initial_entropy=0.2,  # Reasonable starting point
        target_entropy=None,  # Auto-compute as -dim(action)
        grad_norm_clip=1.0,  # Gradient clipping for stability
        random_timesteps=10000,  # Random exploration before learning
        learning_starts=1000,  # Wait for buffer to have some data
    ),
    experiment_name="go1_velocity_sac",
    directory="runs",
    checkpoint_interval=10000,
    logger="wandb",
    wandb_project="luckylab",
    wandb_entity="mjlab",
)
