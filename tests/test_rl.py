"""Tests for LuckyLab RL integration."""

import pytest

from luckylab.rl.config import (
    ActorCriticCfg,
    DdpgAlgorithmCfg,
    PpoAlgorithmCfg,
    RlRunnerCfg,
    SacAlgorithmCfg,
    Td3AlgorithmCfg,
)
from luckylab.tasks.velocity.config.go2.rl_cfg import GO2_PPO_CFG, GO2_SAC_CFG


class TestRlRunnerConfig:
    """Tests for RlRunnerCfg and algorithm config dataclasses."""

    def test_rl_runner_cfg_defaults(self):
        cfg = RlRunnerCfg()
        assert cfg.algorithm == "ppo"
        assert cfg.seed == 42
        assert cfg.max_iterations == 1500
        assert cfg.experiment_name == "luckylab"
        assert cfg.directory == "runs"
        assert cfg.checkpoint_interval == 100
        assert cfg.wandb_project == "luckylab"

    def test_rl_runner_cfg_algorithm_selection(self):
        cfg_ppo = RlRunnerCfg(algorithm="ppo")
        assert cfg_ppo.algorithm == "ppo"

        cfg_sac = RlRunnerCfg(algorithm="sac")
        assert cfg_sac.algorithm == "sac"

        cfg_td3 = RlRunnerCfg(algorithm="td3")
        assert cfg_td3.algorithm == "td3"

        cfg_ddpg = RlRunnerCfg(algorithm="ddpg")
        assert cfg_ddpg.algorithm == "ddpg"

    def test_actor_critic_cfg_defaults(self):
        cfg = ActorCriticCfg()
        assert cfg.actor_hidden_dims == (128, 128, 128)
        assert cfg.critic_hidden_dims == (128, 128, 128)
        assert cfg.activation == "elu"
        assert cfg.init_noise_std == 1.0

    def test_ppo_cfg_defaults(self):
        cfg = PpoAlgorithmCfg()
        assert cfg.num_steps_per_env == 24
        assert cfg.num_learning_epochs == 5
        assert cfg.num_mini_batches == 4
        assert cfg.gamma == 0.99
        assert cfg.lam == 0.95
        assert cfg.learning_rate == 1e-3
        assert cfg.clip_param == 0.2
        assert cfg.value_loss_coef == 1.0
        assert cfg.entropy_coef == 0.005
        assert cfg.max_grad_norm == 1.0
        assert cfg.desired_kl == 0.01

    def test_sac_cfg_defaults(self):
        cfg = SacAlgorithmCfg()
        assert cfg.batch_size == 256
        assert cfg.discount_factor == 0.99
        assert cfg.polyak == 0.005
        assert cfg.actor_learning_rate == 3e-4
        assert cfg.critic_learning_rate == 3e-4
        assert cfg.learn_entropy is True
        assert cfg.initial_entropy == 0.2
        assert cfg.target_entropy is None
        assert cfg.grad_norm_clip == 0.0

    def test_td3_cfg_defaults(self):
        cfg = Td3AlgorithmCfg()
        assert cfg.batch_size == 256
        assert cfg.discount_factor == 0.99
        assert cfg.polyak == 0.005
        assert cfg.actor_learning_rate == 3e-4
        assert cfg.critic_learning_rate == 3e-4
        assert cfg.policy_delay == 2
        assert cfg.smooth_regularization_noise == 0.1
        assert cfg.smooth_regularization_clip == 0.5

    def test_ddpg_cfg_defaults(self):
        cfg = DdpgAlgorithmCfg()
        assert cfg.batch_size == 256
        assert cfg.discount_factor == 0.99
        assert cfg.polyak == 0.005
        assert cfg.actor_learning_rate == 3e-4
        assert cfg.critic_learning_rate == 3e-4

    def test_rl_runner_cfg_custom_values(self):
        cfg = RlRunnerCfg(
            algorithm="sac",
            seed=123,
            max_iterations=500,
            policy=ActorCriticCfg(actor_hidden_dims=(512, 256)),
            experiment_name="test_experiment",
        )
        assert cfg.algorithm == "sac"
        assert cfg.seed == 123
        assert cfg.max_iterations == 500
        assert cfg.policy.actor_hidden_dims == (512, 256)
        assert cfg.experiment_name == "test_experiment"

    def test_rl_runner_cfg_contains_nested_algo_configs(self):
        cfg = RlRunnerCfg()
        assert isinstance(cfg.ppo, PpoAlgorithmCfg)
        assert isinstance(cfg.sac, SacAlgorithmCfg)
        assert isinstance(cfg.td3, Td3AlgorithmCfg)
        assert isinstance(cfg.ddpg, DdpgAlgorithmCfg)
        assert isinstance(cfg.policy, ActorCriticCfg)


class TestGo2RlConfig:
    """Tests for Go2 robot RL configs."""

    def test_go2_ppo_cfg_exists(self):
        assert GO2_PPO_CFG is not None
        assert isinstance(GO2_PPO_CFG, RlRunnerCfg)

    def test_go2_ppo_cfg_values(self):
        cfg = GO2_PPO_CFG
        assert cfg.algorithm == "ppo"
        assert cfg.max_iterations == 5000
        assert cfg.policy.actor_hidden_dims == (512, 256, 128)
        assert cfg.experiment_name == "go2_velocity_ppo"

    def test_go2_sac_cfg_exists(self):
        assert GO2_SAC_CFG is not None
        assert isinstance(GO2_SAC_CFG, RlRunnerCfg)

    def test_go2_sac_cfg_values(self):
        cfg = GO2_SAC_CFG
        assert cfg.algorithm == "sac"
        assert cfg.max_iterations == 4_000_000
        assert cfg.sac.memory_size == 1_000_000
        assert cfg.experiment_name == "go2_velocity_sac"


class TestRegistryWithRlConfig:
    """Tests for registry with RL config support."""

    def setup_method(self):
        from luckylab.tasks.registry import clear_registry

        clear_registry()

    def teardown_method(self):
        from luckylab.tasks.registry import clear_registry

        clear_registry()

    def test_register_with_rl_cfgs(self):
        from luckylab.envs import ManagerBasedRlEnvCfg
        from luckylab.tasks.registry import load_env_cfg, load_rl_cfg, register_task

        env_cfg = ManagerBasedRlEnvCfg(decimation=4, observations={}, actions={})
        rl_cfg = RlRunnerCfg(seed=999, algorithm="sac")
        register_task("test_with_rl", env_cfg, rl_cfgs={"sac": rl_cfg})

        loaded_env_cfg = load_env_cfg("test_with_rl")
        loaded_rl_cfg = load_rl_cfg("test_with_rl", algorithm="sac")

        assert isinstance(loaded_env_cfg, ManagerBasedRlEnvCfg)
        assert loaded_rl_cfg is not None
        assert loaded_rl_cfg.seed == 999
        assert loaded_rl_cfg.algorithm == "sac"

    def test_register_without_rl_cfgs(self):
        from luckylab.envs import ManagerBasedRlEnvCfg
        from luckylab.tasks.registry import load_env_cfg, load_rl_cfg, register_task

        env_cfg = ManagerBasedRlEnvCfg(decimation=4, observations={}, actions={})
        register_task("test_no_rl", env_cfg)

        loaded_env_cfg = load_env_cfg("test_no_rl")
        loaded_rl_cfg = load_rl_cfg("test_no_rl", algorithm="ppo")

        assert isinstance(loaded_env_cfg, ManagerBasedRlEnvCfg)
        assert loaded_rl_cfg is None

    def test_load_rl_cfg_not_found(self):
        from luckylab.tasks.registry import load_rl_cfg

        with pytest.raises(KeyError, match="not found"):
            load_rl_cfg("nonexistent", algorithm="ppo")


class TestExports:
    """Tests for module exports."""

    def test_rl_module_exports(self):
        from luckylab.rl import (
            ActorCriticCfg,
            DdpgAlgorithmCfg,
            PpoAlgorithmCfg,
            RlRunnerCfg,
            SacAlgorithmCfg,
            Td3AlgorithmCfg,
            load_agent,
            train,
        )
        from luckylab.rl.skrl.wrapper import SkrlWrapper

        # Verify all exports are available
        assert RlRunnerCfg is not None
        assert ActorCriticCfg is not None
        assert PpoAlgorithmCfg is not None
        assert SacAlgorithmCfg is not None
        assert Td3AlgorithmCfg is not None
        assert DdpgAlgorithmCfg is not None
        assert train is not None
        assert load_agent is not None
        assert SkrlWrapper is not None


class TestMdpFunctions:
    """Tests for MDP functions (direct imports, no registry)."""

    def test_reward_functions_importable(self):
        from luckylab.tasks.velocity.mdp import (
            action_rate_l2,
            ang_vel_xy_l2,
            joint_pos_limits,
            posture,
            track_angular_velocity,
            track_linear_velocity,
        )

        assert callable(track_linear_velocity)
        assert callable(track_angular_velocity)
        assert callable(action_rate_l2)
        assert callable(joint_pos_limits)
        assert ang_vel_xy_l2 is not None
        assert posture is not None

    def test_termination_functions_importable(self):
        from luckylab.tasks.velocity.mdp import (
            bad_orientation,
            time_out,
        )

        assert callable(time_out)
        assert callable(bad_orientation)
