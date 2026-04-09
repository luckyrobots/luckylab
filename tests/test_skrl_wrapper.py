"""Tests for SkrlWrapper: observation concatenation, NaN guard, and visualization."""

import torch

from luckylab.utils.nan_guard import NanGuardCfg

# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


class MockObservationManager:
    def __init__(self, policy_dim: int, critic_dim: int = 0):
        self._policy_dim = policy_dim
        self._critic_dim = critic_dim

    @property
    def group_obs_dim(self):
        dims = {"policy": self._policy_dim}
        if self._critic_dim > 0:
            dims["critic"] = self._critic_dim
        return dims

    def compute(self, update_history=False):
        return {}

    def reset(self, env_ids):
        return {}


class MockActionManager:
    def __init__(self, action_dim: int):
        self.total_action_dim = action_dim

    def process_action(self, action):
        pass

    def reset(self, env_ids):
        return {}



class MockTerminationManager:
    def __init__(self, num_envs):
        self.terminated = torch.zeros(num_envs, dtype=torch.bool)
        self.time_outs = torch.zeros(num_envs, dtype=torch.bool)

    def compute(self):
        return self.terminated | self.time_outs

    def reset(self, env_ids):
        return {}


class MockRewardManager:
    def __init__(self, num_envs):
        self._num_envs = num_envs

    def compute(self, dt):
        return torch.zeros(self._num_envs)

    def reset(self, env_ids):
        return {}


class MockCurriculumManager:
    def compute(self, env_ids=None):
        pass

    def reset(self, env_ids=None):
        return {}


class MockEnvCfg:
    def __init__(self):
        self.is_finite_horizon = False
        self.nan_guard = NanGuardCfg(enabled=False)


class MockManagerBasedRlEnv:
    """Minimal env mock for testing SkrlWrapper without gRPC."""

    def __init__(
        self,
        num_envs: int = 1,
        policy_obs_dim: int = 48,
        critic_obs_dim: int = 0,
        action_dim: int = 12,
    ):
        self._num_envs = num_envs
        self._device = torch.device("cpu")
        self.cfg = MockEnvCfg()
        self.observation_manager = MockObservationManager(policy_obs_dim, critic_obs_dim)
        self.action_manager = MockActionManager(action_dim)
        self.termination_manager = MockTerminationManager(num_envs)
        self.reward_manager = MockRewardManager(num_envs)
        self.curriculum_manager = MockCurriculumManager()
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        self.max_episode_length = 250
        self._policy_obs_dim = policy_obs_dim
        self._critic_obs_dim = critic_obs_dim
        self.obs_buf = self._make_obs_buf()
        self.extras = {}

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def device(self):
        return self._device

    def _make_obs_buf(self):
        buf = {"policy": torch.randn(self._num_envs, self._policy_obs_dim)}
        if self._critic_obs_dim > 0:
            buf["critic"] = torch.randn(self._num_envs, self._critic_obs_dim)
        return buf

    def reset(self, **kwargs):
        self.obs_buf = self._make_obs_buf()
        return self.obs_buf, self.extras

    def step(self, actions):
        self.obs_buf = self._make_obs_buf()
        rew = torch.zeros(self._num_envs)
        terminated = torch.zeros(self._num_envs, dtype=torch.bool)
        truncated = torch.zeros(self._num_envs, dtype=torch.bool)
        return self.obs_buf, rew, terminated, truncated, self.extras

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSkrlWrapperInit:
    """Tests for SkrlWrapper initialization and space creation."""

    def test_policy_only_obs(self):
        from luckylab.rl.skrl.wrapper import SkrlWrapper

        env = MockManagerBasedRlEnv(policy_obs_dim=48)
        w = SkrlWrapper(env)
        assert w.num_obs == 48
        assert w.observation_space.shape == (48,)

    def test_different_obs_dim(self):
        from luckylab.rl.skrl.wrapper import SkrlWrapper

        env = MockManagerBasedRlEnv(policy_obs_dim=24)
        w = SkrlWrapper(env)
        assert w.num_obs == 24
        assert w.observation_space.shape == (24,)

    def test_action_space(self):
        from luckylab.rl.skrl.wrapper import SkrlWrapper

        env = MockManagerBasedRlEnv(action_dim=12)
        w = SkrlWrapper(env)
        assert w.action_space.shape == (12,)
        assert w.num_actions == 12

    def test_num_envs_and_agents(self):
        from luckylab.rl.skrl.wrapper import SkrlWrapper

        env = MockManagerBasedRlEnv(num_envs=4)
        w = SkrlWrapper(env)
        assert w.num_envs == 4
        assert w.num_agents == 1


class TestSkrlWrapperObservations:
    """Tests for observation retrieval."""

    def test_get_observations_returns_policy(self):
        from luckylab.rl.skrl.wrapper import SkrlWrapper

        env = MockManagerBasedRlEnv(num_envs=2, policy_obs_dim=10)
        w = SkrlWrapper(env)
        obs = w.get_observations()
        assert obs.shape == (2, 10)


class TestSkrlWrapperStep:
    """Tests for step method behavior."""

    def test_step_returns_correct_shapes(self):
        from luckylab.rl.skrl.wrapper import SkrlWrapper

        env = MockManagerBasedRlEnv(num_envs=2, policy_obs_dim=10, action_dim=4)
        w = SkrlWrapper(env)
        actions = torch.zeros(2, 4)
        obs, rew, terminated, dones, extras = w.step(actions)
        assert obs.shape == (2, 10)
        assert rew.shape == (2,)
        assert terminated.shape == (2,)
        assert dones.shape == (2,)

    def test_clip_actions(self):
        from luckylab.rl.skrl.wrapper import SkrlWrapper

        env = MockManagerBasedRlEnv(action_dim=4)
        w = SkrlWrapper(env, clip_actions=0.5)
        assert w.clip_actions == 0.5


class TestSkrlWrapperNanGuard:
    """Tests for NaN guard integration via env config."""

    def test_nan_guard_disabled_by_default(self):
        from luckylab.rl.skrl.wrapper import SkrlWrapper

        env = MockManagerBasedRlEnv()
        w = SkrlWrapper(env)
        assert w.nan_guard.enabled is False

    def test_nan_guard_enabled_from_config(self):
        from luckylab.rl.skrl.wrapper import SkrlWrapper

        env = MockManagerBasedRlEnv()
        env.cfg.nan_guard = NanGuardCfg(enabled=True, verbose=False)
        w = SkrlWrapper(env)
        assert w.nan_guard.enabled is True

    def test_nan_in_actions_replaced(self):
        from luckylab.rl.skrl.wrapper import SkrlWrapper

        env = MockManagerBasedRlEnv(action_dim=4)
        env.cfg.nan_guard = NanGuardCfg(enabled=True, recovery_mode=True, halt_on_nan=False, verbose=False)
        w = SkrlWrapper(env)
        actions = torch.tensor([[float("nan"), 1.0, 2.0, 3.0]])
        obs, rew, terminated, dones, extras = w.step(actions)
        assert "nan_detected" in extras
        assert extras["nan_detected"]["action"] is True


class TestSkrlWrapperVisualization:
    """Tests for debug visualization integration."""

    def test_visualizer_created(self):
        from luckylab.rl.skrl.wrapper import SkrlWrapper

        env = MockManagerBasedRlEnv()
        w = SkrlWrapper(env)
        assert w.visualizer is not None

    def test_debug_vis_step(self):
        from luckylab.rl.skrl.wrapper import SkrlWrapper

        env = MockManagerBasedRlEnv()
        w = SkrlWrapper(env)
        actions = torch.zeros(1, 12)
        w.step(actions)
        # No crash = pass (no engine client in test)


class TestSkrlWrapperProperties:
    """Tests for wrapper property pass-through."""

    def test_unwrapped(self):
        from luckylab.rl.skrl.wrapper import SkrlWrapper

        env = MockManagerBasedRlEnv()
        w = SkrlWrapper(env)
        assert w.unwrapped is env

    def test_episode_length_buf(self):
        from luckylab.rl.skrl.wrapper import SkrlWrapper

        env = MockManagerBasedRlEnv(num_envs=2)
        w = SkrlWrapper(env)
        assert w.episode_length_buf.shape == (2,)
        new_buf = torch.tensor([10, 20], dtype=torch.long)
        w.episode_length_buf = new_buf
        assert (env.episode_length_buf == new_buf).all()
