"""Tests for LuckyLab RL integration."""

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces

from luckylab.rl.config import (
    ActorCriticCfg,
    DdpgCfg,
    PpoCfg,
    SacCfg,
    SkrlCfg,
    Td3Cfg,
)
from luckylab.tasks.velocity import GO1_RL_CFG
from luckylab.tasks.velocity.config.go1.rl_cfg import GO1_PPO_CFG, GO1_SAC_CFG


class TestSkrlConfig:
    """Tests for configuration dataclasses."""

    def test_skrl_cfg_defaults(self):
        cfg = SkrlCfg()
        assert cfg.backend == "torch"
        assert cfg.algorithm == "ppo"
        assert cfg.seed == 42
        assert cfg.timesteps == 1_000_000
        assert cfg.memory_size is None
        assert cfg.experiment_name == "luckylab"
        assert cfg.directory == "runs"
        assert cfg.checkpoint_interval == 1000
        assert cfg.logger == "tensorboard"
        assert cfg.wandb_project == "luckylab"
        assert cfg.scope is None

    def test_skrl_cfg_algorithm_selection(self):
        cfg_ppo = SkrlCfg(algorithm="ppo")
        assert cfg_ppo.algorithm == "ppo"

        cfg_sac = SkrlCfg(algorithm="sac")
        assert cfg_sac.algorithm == "sac"

        cfg_td3 = SkrlCfg(algorithm="td3")
        assert cfg_td3.algorithm == "td3"

        cfg_ddpg = SkrlCfg(algorithm="ddpg")
        assert cfg_ddpg.algorithm == "ddpg"

    def test_skrl_cfg_backend_selection(self):
        cfg_torch = SkrlCfg(backend="torch")
        assert cfg_torch.backend == "torch"

        cfg_jax = SkrlCfg(backend="jax")
        assert cfg_jax.backend == "jax"

    def test_actor_critic_cfg_defaults(self):
        cfg = ActorCriticCfg()
        assert cfg.actor_hidden_dims == (256, 256, 256)
        assert cfg.critic_hidden_dims == (256, 256, 256)
        assert cfg.activation == "elu"
        assert cfg.init_noise_std == 1.0

    def test_ppo_cfg_defaults(self):
        cfg = PpoCfg()
        assert cfg.rollouts == 1024
        assert cfg.learning_epochs == 5
        assert cfg.mini_batches == 4
        assert cfg.discount_factor == 0.99
        assert cfg.lambda_gae == 0.95
        assert cfg.learning_rate == 1e-3
        assert cfg.ratio_clip == 0.2
        assert cfg.value_loss_scale == 1.0
        assert cfg.entropy_loss_scale == 0.01
        assert cfg.grad_norm_clip == 1.0
        assert cfg.kl_threshold == 0.0

    def test_sac_cfg_defaults(self):
        cfg = SacCfg()
        assert cfg.batch_size == 256
        assert cfg.discount_factor == 0.99
        assert cfg.polyak == 0.005
        assert cfg.actor_learning_rate == 3e-4
        assert cfg.critic_learning_rate == 3e-4
        assert cfg.learn_entropy is True
        assert cfg.initial_entropy == 1.0
        assert cfg.target_entropy is None
        assert cfg.grad_norm_clip == 0.0

    def test_td3_cfg_defaults(self):
        cfg = Td3Cfg()
        assert cfg.batch_size == 256
        assert cfg.discount_factor == 0.99
        assert cfg.polyak == 0.005
        assert cfg.actor_learning_rate == 3e-4
        assert cfg.critic_learning_rate == 3e-4
        assert cfg.policy_delay == 2
        assert cfg.smooth_regularization_noise == 0.1
        assert cfg.smooth_regularization_clip == 0.5

    def test_ddpg_cfg_defaults(self):
        cfg = DdpgCfg()
        assert cfg.batch_size == 256
        assert cfg.discount_factor == 0.99
        assert cfg.polyak == 0.005
        assert cfg.actor_learning_rate == 3e-4
        assert cfg.critic_learning_rate == 3e-4

    def test_skrl_cfg_custom_values(self):
        cfg = SkrlCfg(
            algorithm="sac",
            seed=123,
            timesteps=500_000,
            memory_size=50_000,
            policy=ActorCriticCfg(actor_hidden_dims=(512, 256)),
            experiment_name="test_experiment",
            logger="wandb",
        )
        assert cfg.algorithm == "sac"
        assert cfg.seed == 123
        assert cfg.timesteps == 500_000
        assert cfg.memory_size == 50_000
        assert cfg.policy.actor_hidden_dims == (512, 256)
        assert cfg.experiment_name == "test_experiment"
        assert cfg.logger == "wandb"


class TestGo1RlConfig:
    """Tests for Go1 robot RL configs."""

    def test_go1_ppo_cfg_exists(self):
        assert GO1_PPO_CFG is not None
        assert isinstance(GO1_PPO_CFG, SkrlCfg)

    def test_go1_ppo_cfg_values(self):
        cfg = GO1_PPO_CFG
        assert cfg.algorithm == "ppo"
        assert cfg.timesteps == 1_000_000
        assert cfg.policy.actor_hidden_dims == (512, 256, 128)
        assert cfg.experiment_name == "go1_velocity"
        assert cfg.logger == "tensorboard"

    def test_go1_sac_cfg_exists(self):
        assert GO1_SAC_CFG is not None
        assert isinstance(GO1_SAC_CFG, SkrlCfg)

    def test_go1_sac_cfg_values(self):
        cfg = GO1_SAC_CFG
        assert cfg.algorithm == "sac"
        assert cfg.timesteps == 1_000_000
        assert cfg.memory_size == 100_000
        assert cfg.experiment_name == "go1_velocity_sac"

    def test_go1_rl_cfg_is_ppo(self):
        # GO1_RL_CFG should be the default (PPO)
        assert GO1_RL_CFG is GO1_PPO_CFG


class TestRegistryWithRlConfig:
    """Tests for registry with RL config support."""

    def setup_method(self):
        from luckylab.tasks.registry import clear_registry

        clear_registry()

    def teardown_method(self):
        from luckylab.tasks.registry import clear_registry

        clear_registry()

    def test_register_with_rl_cfg(self):
        from luckylab.envs import ManagerBasedRlEnvCfg
        from luckylab.tasks.registry import load_env_cfg, load_rl_cfg, register_task

        def create_test_cfg():
            return ManagerBasedRlEnvCfg(decimation=4)

        rl_cfg = SkrlCfg(seed=999, algorithm="sac")
        register_task("test_with_rl", create_test_cfg, rl_cfg=rl_cfg)

        loaded_env_cfg = load_env_cfg("test_with_rl")
        loaded_rl_cfg = load_rl_cfg("test_with_rl")

        assert isinstance(loaded_env_cfg, ManagerBasedRlEnvCfg)
        assert loaded_rl_cfg is not None
        assert loaded_rl_cfg.seed == 999
        assert loaded_rl_cfg.algorithm == "sac"

    def test_register_without_rl_cfg(self):
        from luckylab.envs import ManagerBasedRlEnvCfg
        from luckylab.tasks.registry import load_env_cfg, load_rl_cfg, register_task

        def create_test_cfg():
            return ManagerBasedRlEnvCfg(decimation=4)

        register_task("test_no_rl", create_test_cfg)

        loaded_env_cfg = load_env_cfg("test_no_rl")
        loaded_rl_cfg = load_rl_cfg("test_no_rl")

        assert isinstance(loaded_env_cfg, ManagerBasedRlEnvCfg)
        assert loaded_rl_cfg is None

    def test_load_rl_cfg_not_found(self):
        from luckylab.tasks.registry import load_rl_cfg

        with pytest.raises(KeyError, match="not found"):
            load_rl_cfg("nonexistent")


class MockGymnasiumEnv(gym.Env):
    """Mock Gymnasium environment for testing."""

    def __init__(self, obs_shape=(48,), action_shape=(12,)):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=action_shape, dtype=np.float32)
        self._obs = np.zeros(obs_shape, dtype=np.float32)
        self._reward = 0.0
        self._terminated = False
        self._truncated = False
        self._info = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self._obs.copy(), self._info.copy()

    def step(self, action):
        return self._obs.copy(), self._reward, self._terminated, self._truncated, self._info.copy()


class TestWrapper:
    """Tests for the SkrlWrapper."""

    def test_wrapper_creation(self):
        from luckylab.rl.wrapper import SkrlWrapper

        env = MockGymnasiumEnv()
        wrapped = SkrlWrapper(env, device="cpu")

        assert wrapped.device == torch.device("cpu")
        assert wrapped.num_envs == 1

    def test_wrapper_reset(self):
        from luckylab.rl.wrapper import SkrlWrapper

        env = MockGymnasiumEnv()
        env._obs = np.ones(48, dtype=np.float32)
        env._info = {"info": "test"}
        wrapped = SkrlWrapper(env, device="cpu")

        obs, info = wrapped.reset()
        assert isinstance(obs, torch.Tensor)
        assert obs.shape == (1, 48)
        assert info["info"] == "test"

    def test_wrapper_step(self):
        from luckylab.rl.wrapper import SkrlWrapper

        env = MockGymnasiumEnv()
        env._obs = np.ones(48, dtype=np.float32)
        env._reward = 1.5
        env._terminated = True
        env._truncated = False
        env._info = {"done": True}
        wrapped = SkrlWrapper(env, device="cpu")

        action = torch.zeros(1, 12)
        obs, reward, terminated, truncated, info = wrapped.step(action)

        assert isinstance(obs, torch.Tensor)
        assert obs.shape == (1, 48)
        assert reward.item() == 1.5
        assert terminated.item() is True
        assert truncated.item() is False
        assert info["done"] is True


class TestExports:
    """Tests for module exports."""

    def test_rl_module_exports(self):
        from luckylab.rl import (
            ActorCriticCfg,
            DdpgCfg,
            PpoCfg,
            SacCfg,
            SkrlCfg,
            SkrlWrapper,
            Td3Cfg,
            load_agent,
            train,
            wrap_env,
        )

        # Verify all exports are available
        assert SkrlCfg is not None
        assert ActorCriticCfg is not None
        assert PpoCfg is not None
        assert SacCfg is not None
        assert Td3Cfg is not None
        assert DdpgCfg is not None
        assert train is not None
        assert load_agent is not None
        assert SkrlWrapper is not None
        assert wrap_env is not None


class TestObservationParser:
    """Tests for ObservationParser."""

    def test_parser_creation(self):
        from luckylab.tasks.velocity.mdp import ObservationParser

        parser = ObservationParser(num_joints=12)
        assert parser.num_joints == 12

    def test_parser_parse(self):
        from luckylab.tasks.velocity.mdp import ObservationParser

        parser = ObservationParser(num_joints=12)
        # Create fake observation: 4 commands + 3 lin_vel + 3 ang_vel + 12 joint_pos + 12 joint_vel + 12 last_act = 46
        obs = np.arange(46, dtype=np.float32)
        parsed = parser.parse(obs)

        assert "commands" in parsed
        assert "base_lin_vel" in parsed
        assert "base_ang_vel" in parsed
        assert "joint_pos" in parsed
        assert "joint_vel" in parsed
        assert "last_act" in parsed

        assert parsed["commands"].shape == (4,)
        assert parsed["base_lin_vel"].shape == (3,)
        assert parsed["base_ang_vel"].shape == (3,)
        assert parsed["joint_pos"].shape == (12,)
        assert parsed["joint_vel"].shape == (12,)
        assert parsed["last_act"].shape == (12,)


class TestMdpFunctions:
    """Tests for MDP functions (direct imports, no registry)."""

    def test_reward_functions_importable(self):
        from luckylab.tasks.velocity.mdp import (
            action_rate_l2,
            body_angular_velocity_penalty,
            joint_pos_limits,
            track_angular_velocity,
            track_linear_velocity,
            variable_posture,
        )

        assert callable(track_linear_velocity)
        assert callable(track_angular_velocity)
        assert callable(variable_posture)
        assert callable(body_angular_velocity_penalty)
        assert callable(joint_pos_limits)
        assert callable(action_rate_l2)

    def test_termination_functions_importable(self):
        from luckylab.tasks.velocity.mdp import (
            bad_orientation,
            fall_termination,
            max_steps_termination,
            nan_detection,
            root_height_below_minimum,
            time_out,
        )

        assert callable(time_out)
        assert callable(bad_orientation)
        assert callable(root_height_below_minimum)
        assert callable(nan_detection)
        assert callable(fall_termination)
        assert callable(max_steps_termination)
