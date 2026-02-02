"""Tests for LuckyLab managers."""

import numpy as np

from luckylab.managers import (
    CurriculumManager,
    CurriculumTermCfg,
    EpisodeMetrics,
    NullCurriculumManager,
    RewardManager,
    RewardTermCfg,
    TerminationManager,
    TerminationTermCfg,
)
from luckylab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg


class TestEpisodeMetrics:
    def test_initial_state(self):
        metrics = EpisodeMetrics()
        assert metrics.steps == 0
        assert metrics.total_reward == 0.0
        assert metrics.terminated is False
        assert metrics.truncated is False

    def test_add_step(self):
        metrics = EpisodeMetrics()
        metrics.add_step()
        metrics.add_step()
        assert metrics.steps == 2

    def test_add_reward(self):
        metrics = EpisodeMetrics()
        metrics.add_reward(1.5)
        metrics.add_reward(2.5)
        assert metrics.total_reward == 4.0

    def test_finalize(self):
        metrics = EpisodeMetrics()
        metrics.add_step()
        metrics.add_step()
        metrics.add_reward(5.0)
        metrics.set_truncated()

        result = metrics.finalize()
        assert result["episode_length"] == 2
        assert result["total_reward"] == 5.0
        assert result["survived"] is True  # truncated but not terminated

    def test_reset(self):
        metrics = EpisodeMetrics()
        metrics.add_step()
        metrics.add_reward(5.0)
        metrics.set_terminated("fell_over")
        metrics.reset()

        assert metrics.steps == 0
        assert metrics.total_reward == 0.0
        assert metrics.terminated is False


class TestUniformVelocityCommandCfg:
    """Tests for the new UniformVelocityCommandCfg that inherits from CommandTermCfg."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        cfg = UniformVelocityCommandCfg()
        assert cfg.resampling_time_range == (5.0, 10.0)
        assert cfg.zero_command_prob == 0.1
        assert cfg.class_type == UniformVelocityCommand

    def test_nested_ranges_class(self):
        """Test nested Ranges dataclass."""
        ranges = UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-2.0, 2.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.5, 1.5),
            heading=(-3.0, 3.0),
        )
        assert ranges.lin_vel_x == (-2.0, 2.0)
        assert ranges.lin_vel_y == (-1.0, 1.0)

    def test_custom_ranges(self):
        """Test creating config with custom ranges."""
        cfg = UniformVelocityCommandCfg(
            resampling_time_range=(3.0, 6.0),
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.5, 0.5),
            ),
            zero_command_prob=0.2,
        )
        assert cfg.resampling_time_range == (3.0, 6.0)
        assert cfg.ranges.lin_vel_x == (-0.5, 0.5)
        assert cfg.zero_command_prob == 0.2


class MockEnv:
    """Mock environment for testing managers."""

    def __init__(self, num_envs: int = 1):
        self.num_envs = num_envs
        self._num_envs = num_envs
        self.common_step_counter = 0
        self.step_count = np.zeros(num_envs, dtype=np.int64)
        self.max_episode_length = 1000
        self.max_steps = 1000
        self._device = "cpu"

    @property
    def device(self) -> str:
        return self._device


class TestCurriculumManager:
    """Tests for the new ManagerBase-based CurriculumManager."""

    def test_init_with_empty_cfg(self):
        """Test initialization with empty curriculum dict."""
        env = MockEnv()
        manager = CurriculumManager({}, env)
        assert manager.active_terms == []

    def test_init_with_terms(self):
        """Test initialization with curriculum terms."""

        def dummy_curriculum(env, value: float) -> None:
            env.test_value = value

        cfg = {
            "test_term": CurriculumTermCfg(
                func=dummy_curriculum,
                params={"value": 0.5},
            )
        }
        env = MockEnv()
        manager = CurriculumManager(cfg, env)
        assert "test_term" in manager.active_terms
        # Function-based terms keep their original function reference
        assert manager.cfg["test_term"].func is dummy_curriculum

    def test_compute_calls_curriculum_functions(self):
        """Test that compute calls all curriculum functions with env."""

        calls = []

        def track_curriculum(env, name: str) -> None:
            calls.append((name, env.common_step_counter))

        cfg = {
            "term_a": CurriculumTermCfg(
                func=track_curriculum,
                params={"name": "a"},
            ),
            "term_b": CurriculumTermCfg(
                func=track_curriculum,
                params={"name": "b"},
            ),
        }
        env = MockEnv()
        env.common_step_counter = 100
        manager = CurriculumManager(cfg, env)

        manager.compute()

        # Both functions should have been called
        assert len(calls) == 2
        assert ("a", 100) in calls
        assert ("b", 100) in calls

    def test_curriculum_function_can_modify_env(self):
        """Test that curriculum functions can modify environment state."""

        def set_difficulty(env, stages: list[dict]) -> None:
            step = env.common_step_counter
            for stage in reversed(stages):
                if step >= stage["step"]:
                    env.terrain_difficulty = stage["difficulty"]
                    break

        cfg = {
            "terrain": CurriculumTermCfg(
                func=set_difficulty,
                params={
                    "stages": [
                        {"step": 0, "difficulty": 0.0},
                        {"step": 100, "difficulty": 0.5},
                        {"step": 200, "difficulty": 1.0},
                    ]
                },
            )
        }
        env = MockEnv()
        env.terrain_difficulty = 0.0
        manager = CurriculumManager(cfg, env)

        # At step 0, difficulty should be 0.0
        env.common_step_counter = 0
        manager.compute()
        assert env.terrain_difficulty == 0.0

        # At step 150, difficulty should be 0.5
        env.common_step_counter = 150
        manager.compute()
        assert env.terrain_difficulty == 0.5

        # At step 250, difficulty should be 1.0
        env.common_step_counter = 250
        manager.compute()
        assert env.terrain_difficulty == 1.0

    def test_reset_returns_empty_dict(self):
        """Test that reset returns empty dict (curriculum has no stats)."""
        env = MockEnv()
        manager = CurriculumManager({}, env)
        result = manager.reset()
        assert result == {}

    def test_str_representation(self):
        """Test string representation of manager."""

        def dummy_curriculum(env) -> None:
            pass

        cfg = {"test": CurriculumTermCfg(func=dummy_curriculum)}
        env = MockEnv()
        manager = CurriculumManager(cfg, env)

        s = str(manager)
        assert "<CurriculumManager>" in s
        assert "test" in s


class TestNullCurriculumManager:
    """Tests for NullCurriculumManager."""

    def test_active_terms_empty(self):
        manager = NullCurriculumManager()
        assert manager.active_terms == []

    def test_reset_returns_empty(self):
        manager = NullCurriculumManager()
        assert manager.reset() == {}

    def test_compute_is_noop(self):
        manager = NullCurriculumManager()
        # Should not raise
        manager.compute()
        manager.compute(env_ids=np.array([0, 1]))


class TestRewardManager:
    """Tests for the RewardManager class."""

    def test_init_with_empty_cfg(self):
        """Test initialization with empty rewards dict."""
        env = MockEnv()
        manager = RewardManager({}, env)
        assert manager.active_terms == []

    def test_init_with_terms(self):
        """Test initialization with reward terms."""

        def dummy_reward(obs_parsed, value: float = 1.0) -> float:
            return value

        cfg = {
            "test_reward": RewardTermCfg(
                func=dummy_reward,
                weight=2.0,
                params={"value": 1.0},
            )
        }
        env = MockEnv()
        manager = RewardManager(cfg, env)

        assert "test_reward" in manager.active_terms
        assert len(manager.active_terms) == 1

    def test_compute_reward(self):
        """Test reward computation."""

        def constant_reward(obs_parsed) -> float:
            return 1.0

        def scaled_reward(obs_parsed, scale: float = 1.0) -> float:
            return scale

        cfg = {
            "constant": RewardTermCfg(func=constant_reward, weight=1.0),
            "scaled": RewardTermCfg(func=scaled_reward, weight=2.0, params={"scale": 0.5}),
        }
        env = MockEnv()
        manager = RewardManager(cfg, env)

        context = {"obs_parsed": {"joint_pos": np.zeros(12)}}
        total = manager.compute(context)

        # constant: 1.0 * 1.0 = 1.0
        # scaled: 0.5 * 2.0 = 1.0
        # total = 2.0
        assert total == 2.0

    def test_episode_sums(self):
        """Test that episode sums accumulate correctly."""

        def constant_reward(obs_parsed) -> float:
            return 1.0

        cfg = {"reward": RewardTermCfg(func=constant_reward, weight=1.0)}
        env = MockEnv()
        manager = RewardManager(cfg, env)

        context = {"obs_parsed": {"joint_pos": np.zeros(12)}}

        # Compute multiple steps
        for _ in range(5):
            manager.compute(context)

        assert manager.episode_sums(0)["reward"] == 5.0

    def test_reset_returns_episode_stats(self):
        """Test that reset returns episode statistics."""

        def constant_reward(obs_parsed) -> float:
            return 1.0

        cfg = {"reward": RewardTermCfg(func=constant_reward, weight=2.0)}
        env = MockEnv()
        manager = RewardManager(cfg, env)

        context = {"obs_parsed": {"joint_pos": np.zeros(12)}}

        # Compute some rewards
        for _ in range(3):
            manager.compute(context)

        # Reset and check stats
        stats = manager.reset()
        assert "Episode_Reward/reward" in stats
        assert stats["Episode_Reward/reward"] == 6.0  # 3 * 2.0

        # Episode sums should be reset
        assert manager.episode_sums(0)["reward"] == 0.0

    def test_zero_weight_term_skipped(self):
        """Test that terms with zero weight are skipped."""

        call_count = [0]

        def counting_reward(obs_parsed) -> float:
            call_count[0] += 1
            return 1.0

        cfg = {"reward": RewardTermCfg(func=counting_reward, weight=0.0)}
        env = MockEnv()
        manager = RewardManager(cfg, env)

        context = {"obs_parsed": {"joint_pos": np.zeros(12)}}
        total = manager.compute(context)

        assert total == 0.0
        assert call_count[0] == 0  # Function should not have been called

    def test_str_representation(self):
        """Test string representation of manager."""

        def dummy_reward(obs_parsed) -> float:
            return 0.0

        cfg = {"test": RewardTermCfg(func=dummy_reward, weight=1.5)}
        env = MockEnv()
        manager = RewardManager(cfg, env)

        s = str(manager)
        assert "<RewardManager>" in s
        assert "test" in s
        assert "1.5" in s


class TestTerminationManager:
    """Tests for the TerminationManager class (multi-env support)."""

    def test_init_with_empty_cfg(self):
        """Test initialization with empty terminations dict."""
        env = MockEnv()
        manager = TerminationManager({}, env)
        assert manager.active_terms == []

    def test_init_with_terms(self):
        """Test initialization with termination terms."""

        def dummy_termination(obs_parsed) -> bool:
            return False

        cfg = {"test_term": TerminationTermCfg(func=dummy_termination, time_out=False)}
        env = MockEnv()
        manager = TerminationManager(cfg, env)

        assert "test_term" in manager.active_terms

    def test_terminated_vs_truncated(self):
        """Test distinction between terminated and truncated (multi-env)."""

        def failure_condition(obs_parsed) -> bool:
            return obs_parsed.get("failed", False)

        def timeout_condition(obs_parsed) -> bool:
            return obs_parsed.get("timeout", False)

        cfg = {
            "failure": TerminationTermCfg(func=failure_condition, time_out=False),
            "timeout": TerminationTermCfg(func=timeout_condition, time_out=True),
        }
        env = MockEnv()
        manager = TerminationManager(cfg, env)

        # Test failure termination
        context = {"obs_parsed": {"failed": True, "timeout": False}}
        manager.compute(context)

        # With multi-env, these return arrays
        assert manager.terminated[0] is True or manager.terminated[0] == True  # noqa: E712
        assert manager.truncated[0] is False or manager.truncated[0] == False  # noqa: E712
        assert manager.dones[0] is True or manager.dones[0] == True  # noqa: E712

        manager.reset()

        # Test timeout truncation
        context = {"obs_parsed": {"failed": False, "timeout": True}}
        manager.compute(context)

        assert manager.terminated[0] is False or manager.terminated[0] == False  # noqa: E712
        assert manager.truncated[0] is True or manager.truncated[0] == True  # noqa: E712

    def test_multiple_terminations(self):
        """Test that multiple termination reasons are tracked."""

        def cond_a(obs_parsed) -> bool:
            return True

        def cond_b(obs_parsed) -> bool:
            return True

        cfg = {
            "cond_a": TerminationTermCfg(func=cond_a, time_out=False),
            "cond_b": TerminationTermCfg(func=cond_b, time_out=False),
        }
        env = MockEnv()
        manager = TerminationManager(cfg, env)

        context = {"obs_parsed": {}}
        manager.compute(context)

        reasons = manager.termination_reasons(env_idx=0)
        assert "cond_a" in reasons
        assert "cond_b" in reasons

    def test_reset_clears_state(self):
        """Test that reset clears termination state."""

        def always_terminate(obs_parsed) -> bool:
            return True

        cfg = {"term": TerminationTermCfg(func=always_terminate)}
        env = MockEnv()
        manager = TerminationManager(cfg, env)

        context = {"obs_parsed": {}}
        manager.compute(context)
        assert manager.terminated[0] is True or manager.terminated[0] == True  # noqa: E712

        manager.reset()
        assert manager.terminated[0] is False or manager.terminated[0] == False  # noqa: E712
        assert manager.truncated[0] is False or manager.truncated[0] == False  # noqa: E712

    def test_get_term(self):
        """Test getting individual term state (returns array)."""

        def cond_a(obs_parsed) -> bool:
            return True

        def cond_b(obs_parsed) -> bool:
            return False

        cfg = {
            "cond_a": TerminationTermCfg(func=cond_a),
            "cond_b": TerminationTermCfg(func=cond_b),
        }
        env = MockEnv()
        manager = TerminationManager(cfg, env)

        context = {"obs_parsed": {}}
        manager.compute(context)

        # get_term returns array
        assert manager.get_term("cond_a")[0] is True or manager.get_term("cond_a")[0] == True  # noqa: E712
        assert manager.get_term("cond_b")[0] is False or manager.get_term("cond_b")[0] == False  # noqa: E712
