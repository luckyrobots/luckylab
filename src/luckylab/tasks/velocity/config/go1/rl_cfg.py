"""Unitree Go1 RL configuration for velocity task."""

from luckylab.rl.config import ActorCriticCfg, PpoCfg, SacCfg, SkrlCfg

# Default PPO configuration for Go1 velocity task
GO1_PPO_CFG = SkrlCfg(
    algorithm="ppo",
    seed=42,
    timesteps=1_000_000,
    policy=ActorCriticCfg(
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation="elu",
        init_noise_std=1.0,
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
    ),
    experiment_name="go1_velocity",
    directory="runs",
    checkpoint_interval=1000,
    logger="tensorboard",
)

# SAC configuration for Go1 velocity task
GO1_SAC_CFG = SkrlCfg(
    algorithm="sac",
    seed=42,
    timesteps=1_000_000,
    memory_size=100_000,
    policy=ActorCriticCfg(
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation="elu",
    ),
    sac=SacCfg(
        batch_size=256,
        discount_factor=0.99,
        polyak=0.005,
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        learn_entropy=True,
        initial_entropy=1.0,
    ),
    experiment_name="go1_velocity_sac",
    logger="tensorboard",
)

# Default RL config for Go1
GO1_RL_CFG = GO1_PPO_CFG
