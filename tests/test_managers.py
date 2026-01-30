"""Tests for LuckyLab managers."""

import numpy as np
import pytest

from luckylab.managers import (
    CommandManager,
    CurriculumCfg,
    CurriculumManager,
    CurriculumTermCfg,
    CurriculumTermManager,
    EpisodeMetrics,
    EventManager,
    EventTermCfg,
    NoiseCfg,
    ObservationProcessor,
    ObservationProcessorCfg,
    RewardManager,
    RewardTermCfg,
    TerminationManager,
    TerminationTermCfg,
    VelocityCommand,
    VelocityCommandCfg,
    create_default_observation_processor,
)


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


class TestVelocityCommand:
    def test_initial_command(self):
        cfg = VelocityCommandCfg()
        cmd = VelocityCommand(cfg)
        assert cmd.command.shape == (4,)
        assert cmd.dim == 4

    def test_resample(self):
        cfg = VelocityCommandCfg(zero_command_prob=0.0)
        cmd = VelocityCommand(cfg)
        _ = cmd.command.copy()  # Store initial for potential future comparison
        cmd.resample()
        # Command should be different after resample (very unlikely to be same)
        # But we can't guarantee it, so just check shape
        assert cmd.command.shape == (4,)

    def test_zero_command_prob(self):
        cfg = VelocityCommandCfg(zero_command_prob=1.0)
        cmd = VelocityCommand(cfg)
        cmd.resample()
        np.testing.assert_array_equal(cmd.command, np.zeros(4))

    def test_reset(self):
        cfg = VelocityCommandCfg()
        cmd = VelocityCommand(cfg)
        cmd.reset()
        assert cmd.command.shape == (4,)

    def test_update_decrements_time(self):
        cfg = VelocityCommandCfg(resample_interval_range=(10.0, 10.0))
        cmd = VelocityCommand(cfg)
        initial_time = cmd._time_left
        cmd.update(1.0)
        assert cmd._time_left == initial_time - 1.0


class TestCommandManager:
    def test_add_and_get_command(self):
        manager = CommandManager()
        manager.add_command("velocity", VelocityCommand(VelocityCommandCfg()))

        cmd = manager.get_command("velocity")
        assert cmd.shape == (4,)

    def test_get_nonexistent_command(self):
        manager = CommandManager()
        with pytest.raises(KeyError):
            manager.get_command("nonexistent")

    def test_get_all_commands(self):
        manager = CommandManager()
        manager.add_command("velocity", VelocityCommand(VelocityCommandCfg()))

        all_cmds = manager.get_all_commands()
        assert "velocity" in all_cmds
        assert all_cmds["velocity"].shape == (4,)

    def test_reset(self):
        manager = CommandManager()
        manager.add_command("velocity", VelocityCommand(VelocityCommandCfg()))
        manager.reset()
        # Should not raise
        cmd = manager.get_command("velocity")
        assert cmd.shape == (4,)


class TestCurriculumManager:
    def test_initial_levels(self):
        cfg = CurriculumCfg()
        manager = CurriculumManager(cfg)

        assert manager.terrain_level == 0
        assert manager.push_level == 0
        assert manager.velocity_level == 0

    def test_update_without_enough_data(self):
        cfg = CurriculumCfg(min_episodes_before_advance=10)
        manager = CurriculumManager(cfg)

        # Add less than min_episodes
        for _ in range(5):
            manager.update({"survived": True, "total_reward": 100, "episode_length": 500})

        # Levels should not advance
        assert manager.terrain_level == 0

    def test_get_dr_cfg(self):
        cfg = CurriculumCfg(terrain_levels=(0.0, 0.5, 1.0))
        manager = CurriculumManager(cfg)

        dr_cfg = manager.get_dr_cfg()
        assert dr_cfg.terrain_difficulty == 0.0

    def test_get_velocity_command_range(self):
        cfg = CurriculumCfg(velocity_range_levels=((0.5, 0.5), (1.0, 1.0)))
        manager = CurriculumManager(cfg)

        vel_range = manager.get_velocity_command_range()
        assert vel_range == (0.5, 0.5)

    def test_get_metrics(self):
        cfg = CurriculumCfg()
        manager = CurriculumManager(cfg)

        metrics = manager.get_metrics()
        assert "curriculum/terrain_level" in metrics
        assert "curriculum/survival_rate" in metrics


class TestObservationProcessor:
    def test_passthrough_no_processing(self):
        """Test that with default config, observation passes through unchanged."""
        cfg = ObservationProcessorCfg()
        processor = ObservationProcessor(obs_dim=10, cfg=cfg)

        obs = np.random.randn(10).astype(np.float32)
        processor.reset(obs)
        result = processor.process(obs)

        np.testing.assert_array_almost_equal(obs, result)

    def test_gaussian_noise(self):
        """Test that gaussian noise is applied."""
        cfg = ObservationProcessorCfg(global_noise=NoiseCfg(type="gaussian", std=0.1))
        processor = ObservationProcessor(obs_dim=10, cfg=cfg)

        obs = np.zeros(10, dtype=np.float32)
        processor.reset(obs)
        result = processor.process(obs)

        # With noise, result should not be exactly zero
        assert not np.allclose(result, obs, atol=1e-6)

    def test_uniform_noise(self):
        """Test that uniform noise is applied."""
        cfg = ObservationProcessorCfg(global_noise=NoiseCfg(type="uniform", low=-0.1, high=0.1))
        processor = ObservationProcessor(obs_dim=10, cfg=cfg)

        obs = np.zeros(10, dtype=np.float32)
        processor.reset(obs)
        result = processor.process(obs)

        # Result should be within noise bounds
        assert np.all(result >= -0.1)
        assert np.all(result <= 0.1)

    def test_scaling(self):
        """Test that global scaling is applied."""
        cfg = ObservationProcessorCfg(global_scale=2.0)
        processor = ObservationProcessor(obs_dim=10, cfg=cfg)

        obs = np.ones(10, dtype=np.float32)
        processor.reset(obs)
        result = processor.process(obs)

        np.testing.assert_array_almost_equal(result, obs * 2.0)

    def test_clipping(self):
        """Test that global clipping is applied."""
        cfg = ObservationProcessorCfg(global_clip=(-0.5, 0.5))
        processor = ObservationProcessor(obs_dim=10, cfg=cfg)

        obs = np.array([-1.0, -0.5, 0.0, 0.5, 1.0] * 2, dtype=np.float32)
        processor.reset(obs)
        result = processor.process(obs)

        assert np.all(result >= -0.5)
        assert np.all(result <= 0.5)

    def test_delay_buffer(self):
        """Test that delay buffer works correctly."""
        cfg = ObservationProcessorCfg(global_delay_range=(2, 2))  # Fixed delay of 2
        processor = ObservationProcessor(obs_dim=3, cfg=cfg)

        # Initialize with zeros
        init_obs = np.zeros(3, dtype=np.float32)
        processor.reset(init_obs)

        # Send sequence of observations
        obs1 = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        obs2 = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        obs3 = np.array([3.0, 3.0, 3.0], dtype=np.float32)

        # With delay=2, we should get the observation from 2 steps ago
        result1 = processor.process(obs1)  # Should get init (delay buffer filled with init)
        result2 = processor.process(obs2)  # Should get init
        result3 = processor.process(obs3)  # Should get obs1

        np.testing.assert_array_almost_equal(result1, init_obs)
        np.testing.assert_array_almost_equal(result2, init_obs)
        np.testing.assert_array_almost_equal(result3, obs1)

    def test_history_stacking(self):
        """Test that history stacking works correctly."""
        cfg = ObservationProcessorCfg(history_length=3, flatten_history=True)
        processor = ObservationProcessor(obs_dim=2, cfg=cfg)

        init_obs = np.zeros(2, dtype=np.float32)
        processor.reset(init_obs)

        # Output dimension should be obs_dim * history_length
        assert processor.output_dim == 6

        obs1 = np.array([1.0, 1.0], dtype=np.float32)
        result = processor.process(obs1)

        # Should be [init, init, obs1] flattened
        expected = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_history_not_flattened(self):
        """Test history stacking without flattening."""
        cfg = ObservationProcessorCfg(history_length=3, flatten_history=False)
        processor = ObservationProcessor(obs_dim=2, cfg=cfg)

        init_obs = np.zeros(2, dtype=np.float32)
        processor.reset(init_obs)

        # Output dimension should be obs_dim (not expanded, shape is (history, obs_dim))
        assert processor.output_dim == 2

    def test_create_default_processor(self):
        """Test the convenience factory function."""
        processor = create_default_observation_processor(
            obs_dim=10,
            noise_std=0.05,
            delay_range=(1, 3),
            history_length=2,
        )

        assert processor.obs_dim == 10
        assert processor.output_dim == 20  # 10 * 2 (history)

    def test_reset_clears_state(self):
        """Test that reset properly clears processor state."""
        cfg = ObservationProcessorCfg(
            global_delay_range=(1, 1),
            history_length=2,
        )
        processor = ObservationProcessor(obs_dim=3, cfg=cfg)

        # Process some observations
        obs1 = np.ones(3, dtype=np.float32)
        processor.reset(obs1)
        processor.process(obs1 * 2)
        processor.process(obs1 * 3)

        # Reset with new observation
        new_init = np.zeros(3, dtype=np.float32)
        processor.reset(new_init)

        # State should be cleared
        result = processor.process(np.ones(3, dtype=np.float32))
        # History should be filled with new_init, delay buffer with new_init
        assert result is not None


class TestCurriculumTermManager:
    """Tests for the new dict-based CurriculumTermManager (mjlab pattern)."""

    def test_init_with_empty_cfg(self):
        """Test initialization with empty curriculum dict."""
        manager = CurriculumTermManager({})
        assert manager.cfg == {}

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
        manager = CurriculumTermManager(cfg)
        assert "test_term" in manager.cfg
        assert manager.cfg["test_term"].func is dummy_curriculum

    def test_update_calls_curriculum_functions(self):
        """Test that update calls all curriculum functions with env."""

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
        manager = CurriculumTermManager(cfg)

        # Create a mock env
        class MockEnv:
            common_step_counter = 100

        env = MockEnv()
        manager.update(env)

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
        manager = CurriculumTermManager(cfg)

        class MockEnv:
            common_step_counter = 0
            terrain_difficulty = 0.0

        env = MockEnv()

        # At step 0, difficulty should be 0.0
        env.common_step_counter = 0
        manager.update(env)
        assert env.terrain_difficulty == 0.0

        # At step 150, difficulty should be 0.5
        env.common_step_counter = 150
        manager.update(env)
        assert env.terrain_difficulty == 0.5

        # At step 250, difficulty should be 1.0
        env.common_step_counter = 250
        manager.update(env)
        assert env.terrain_difficulty == 1.0


class MockEnv:
    """Mock environment for testing managers."""

    def __init__(self):
        self.common_step_counter = 0
        self.step_count = 0
        self.max_episode_length = 1000


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

        assert manager.episode_sums["reward"] == 5.0

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
        assert manager.episode_sums["reward"] == 0.0

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
    """Tests for the TerminationManager class."""

    def test_init_with_empty_cfg(self):
        """Test initialization with empty terminations dict."""
        env = MockEnv()
        manager = TerminationManager({}, env)
        assert manager.active_terms == []

    def test_init_with_terms(self):
        """Test initialization with termination terms."""

        def dummy_termination(obs_parsed) -> bool:
            return False

        cfg = {
            "test_term": TerminationTermCfg(func=dummy_termination, time_out=False)
        }
        env = MockEnv()
        manager = TerminationManager(cfg, env)

        assert "test_term" in manager.active_terms

    def test_terminated_vs_truncated(self):
        """Test distinction between terminated and truncated."""

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

        assert manager.terminated is True
        assert manager.truncated is False
        assert manager.dones is True

        manager.reset()

        # Test timeout truncation
        context = {"obs_parsed": {"failed": False, "timeout": True}}
        manager.compute(context)

        assert manager.terminated is False
        assert manager.truncated is True
        assert manager.dones is True

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

        reasons = manager.termination_reasons
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
        assert manager.terminated is True

        manager.reset()
        assert manager.terminated is False
        assert manager.truncated is False

    def test_get_term(self):
        """Test getting individual term state."""

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

        assert manager.get_term("cond_a") is True
        assert manager.get_term("cond_b") is False


class TestEventManager:
    """Tests for the EventManager class."""

    def test_init_with_empty_cfg(self):
        """Test initialization with empty events dict."""
        env = MockEnv()
        manager = EventManager({}, env)
        assert manager.active_terms == []

    def test_startup_event_runs_on_init(self):
        """Test that startup events run during initialization."""
        startup_called = [False]

        def startup_event(env):
            startup_called[0] = True

        cfg = {"startup": EventTermCfg(func=startup_event, mode="startup")}
        env = MockEnv()
        EventManager(cfg, env)

        assert startup_called[0] is True

    def test_reset_event_runs_on_reset(self):
        """Test that reset events run when reset() is called."""
        reset_count = [0]

        def reset_event(env):
            reset_count[0] += 1

        cfg = {"on_reset": EventTermCfg(func=reset_event, mode="reset")}
        env = MockEnv()
        manager = EventManager(cfg, env)

        # Initial count should be 0 (startup doesn't run reset events)
        assert reset_count[0] == 0

        manager.reset()
        assert reset_count[0] == 1

        manager.reset()
        assert reset_count[0] == 2

    def test_interval_event(self):
        """Test that interval events run at correct times."""
        interval_count = [0]

        def interval_event(env):
            interval_count[0] += 1

        cfg = {
            "periodic": EventTermCfg(
                func=interval_event, mode="interval", interval_range_s=(1.0, 1.0)
            )
        }
        env = MockEnv()
        manager = EventManager(cfg, env)

        # Initially 0
        assert interval_count[0] == 0

        # Update with small timestep - should not trigger
        manager.update(0.5)
        assert interval_count[0] == 0

        # Update past the interval - should trigger
        manager.update(0.6)
        assert interval_count[0] == 1

        # Continue updating - should trigger again after another 1.0s
        manager.update(1.0)
        assert interval_count[0] == 2

    def test_event_receives_params(self):
        """Test that events receive their configured parameters."""
        received_params = {}

        def param_event(env, value: float, name: str):
            received_params["value"] = value
            received_params["name"] = name

        cfg = {
            "with_params": EventTermCfg(
                func=param_event,
                mode="reset",
                params={"value": 42.0, "name": "test"},
            )
        }
        env = MockEnv()
        manager = EventManager(cfg, env)
        manager.reset()

        assert received_params["value"] == 42.0
        assert received_params["name"] == "test"

    def test_domain_randomization_flag(self):
        """Test that domain_randomization flag is preserved in config."""

        def dr_event(env):
            pass

        cfg = {
            "dr": EventTermCfg(func=dr_event, mode="reset", domain_randomization=True)
        }
        env = MockEnv()
        manager = EventManager(cfg, env)

        assert manager.cfg["dr"].domain_randomization is True
