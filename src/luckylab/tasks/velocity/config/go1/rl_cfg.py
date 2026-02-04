"""RL configuration for Unitree Go1 velocity task."""

from luckylab.rl.config import (
    RlRunnerCfg, 
    ActorCriticCfg, 
    PpoAlgorithmCfg, 
    SacAlgorithmCfg
)

UNITREE_GO1_PPO_RUNNER_CFG = RlRunnerCfg(
    algorithm="ppo",
    seed=42,
    max_iterations=1500,
    policy=ActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation="elu",
    ),
    ppo=PpoAlgorithmCfg(
        value_loss_coef=1.0,
        clip_param=0.2,
        entropy_coef=0.01,
        num_steps_per_env=24,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    ),
    experiment_name="go1_velocity_ppo",
    directory="runs",
)


UNITREE_GO1_SAC_RUNNER_CFG = RlRunnerCfg(
    algorithm="sac",
    seed=42,
    max_iterations=50000,
    memory_size=1_000_000,
    policy=ActorCriticCfg(
        actor_hidden_dims=(256, 256, 256),
        critic_hidden_dims=(256, 256, 256),
        activation="elu",
        init_noise_std=1.0,
    ),
    sac=SacAlgorithmCfg(
        batch_size=256,
        discount_factor=0.99,
        polyak=0.005, 
        actor_learning_rate=1e-4,
        critic_learning_rate=1e-4,  
        learn_entropy=True,  
        initial_entropy=1.0,  
        target_entropy=-12.0,  
        grad_norm_clip=1.0,  
        random_timesteps=10000,  
        learning_starts=1000,
    ),
    experiment_name="go1_velocity_sac",
    directory="runs",
)

GO1_PPO_CFG = UNITREE_GO1_PPO_RUNNER_CFG
GO1_SAC_CFG = UNITREE_GO1_SAC_RUNNER_CFG
