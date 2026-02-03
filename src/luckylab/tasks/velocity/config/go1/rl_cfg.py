"""Unitree Go1 RL configuration for velocity task."""

from luckylab.rl.config import ActorCriticCfg, PpoAlgorithmCfg, RlRunnerCfg, SacAlgorithmCfg

UNITREE_GO1_PPO_RUNNER_CFG = RlRunnerCfg(
    algorithm="ppo",
    seed=42,
    max_iterations=1500,
    policy=ActorCriticCfg(
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation="elu",
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
    ),
    ppo=PpoAlgorithmCfg(
        num_steps_per_env=24,
        num_learning_epochs=5,
        num_mini_batches=4,
        gamma=0.99,
        lam=0.95,
        learning_rate=1e-3,
        clip_param=0.2,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        max_grad_norm=1.0,
        desired_kl=0.01,
        schedule="adaptive",
    ),
    experiment_name="go1_velocity",
    directory="runs",
    save_interval=50,
    logger="wandb",
    wandb_project="luckylab",
    wandb_entity="mjlab",
)

# SAC configuration for Go1 velocity task
# SAC is sample-efficient and good for single-env training
UNITREE_GO1_SAC_RUNNER_CFG = RlRunnerCfg(
    algorithm="sac",
    seed=42,
    max_iterations=50000,
    memory_size=1_000_000,  # Large replay buffer for off-policy learning
    policy=ActorCriticCfg(
        actor_hidden_dims=(256, 256, 256),  # Standard SAC architecture
        critic_hidden_dims=(256, 256, 256),
        activation="relu",  # ReLU is standard for SAC
        init_noise_std=1.0,
    ),
    sac=SacAlgorithmCfg(
        batch_size=256,
        discount_factor=0.99,
        polyak=0.005,  # Soft target update
        actor_learning_rate=1e-4,  # Lower LR for stability
        critic_learning_rate=1e-4,  # Lower LR for stability
        learn_entropy=False,  # Fixed entropy for stability (auto-tune can explode)
        initial_entropy=0.2,  # Fixed entropy coefficient
        target_entropy=-6.0,  # Less aggressive than -dim(action)=-12
        grad_norm_clip=1.0,  # Gradient clipping for stability
        random_timesteps=1000,  # Random exploration before learning
        learning_starts=1000,  # Should match random_timesteps so actor/critic train together
    ),
    experiment_name="go1_velocity_sac",
    directory="runs",
    save_interval=1000,
    logger="wandb",
    wandb_project="luckylab",
    wandb_entity="mjlab",
)

# Convenient aliases for train.py
GO1_PPO_CFG = UNITREE_GO1_PPO_RUNNER_CFG
GO1_SAC_CFG = UNITREE_GO1_SAC_RUNNER_CFG
