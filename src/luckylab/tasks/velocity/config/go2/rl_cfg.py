"""RL configuration for Unitree Go2 velocity task."""

from luckylab.rl.config import (
    RlRunnerCfg,
    ActorCriticCfg,
    PpoAlgorithmCfg,
    SacAlgorithmCfg,
)

UNITREE_GO2_PPO_RUNNER_CFG = RlRunnerCfg(
    algorithm="ppo",
    seed=42,
    max_iterations=5000,
    policy=ActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation="elu",
    ),
    ppo=PpoAlgorithmCfg(
        value_loss_coef=1.0,
        clip_param=0.2,
        entropy_coef=0.01,
        num_steps_per_env=2048,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    ),
    checkpoint_interval=2500,
    experiment_name="go2_velocity_ppo",
    directory="runs",
)


UNITREE_GO2_SAC_RUNNER_CFG = RlRunnerCfg(
    algorithm="sac",
    seed=42,
    max_iterations=1_000_000,
    policy=ActorCriticCfg(
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation="elu",
        init_noise_std=0.25,
        noise_type="gsde",
        gsde_resample_interval=10,
        use_delta_actions=False,
    ),
    sac=SacAlgorithmCfg(
        memory_size=1_000_000,
        batch_size=512,
        discount_factor=0.99,
        polyak=0.005,
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        learn_entropy=False,
        initial_entropy=0.01,
        target_entropy=-12.0,
        grad_norm_clip=1.0,
        random_timesteps=0,
        learning_starts=25_000,
        gradient_steps=16,
    ),
    checkpoint_interval=2500,
    experiment_name="go2_velocity_sac",
    directory="runs",
)

GO2_PPO_CFG = UNITREE_GO2_PPO_RUNNER_CFG
GO2_SAC_CFG = UNITREE_GO2_SAC_RUNNER_CFG
