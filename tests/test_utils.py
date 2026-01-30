"""Tests for LuckyLab utilities."""

import numpy as np
import pytest
import torch

from luckylab.utils.buffers import CircularBuffer, DelayBuffer
from luckylab.utils.logging import (
    colorize,
    format_metric,
    format_metrics_table,
    print_episode_summary,
    print_header,
    print_info,
)
from luckylab.utils.iteration_logger import IterationLogger, IterationLoggerCfg
from luckylab.utils.metrics import MetricsLogger, MetricsLoggerCfg, create_logger
from luckylab.utils.nan_guard import NanGuard, NanGuardCfg
from luckylab.utils.random import seed_rng
from luckylab.utils.torch_utils import configure_torch_backends


class TestSeedRng:
    """Tests for seed_rng function."""

    def test_seed_rng_sets_seeds(self):
        seed_rng(42)
        val1_np = np.random.rand()
        val1_torch = torch.rand(1).item()

        seed_rng(42)
        val2_np = np.random.rand()
        val2_torch = torch.rand(1).item()

        assert val1_np == val2_np
        assert val1_torch == val2_torch

    def test_seed_rng_different_seeds(self):
        seed_rng(42)
        val1 = np.random.rand()

        seed_rng(123)
        val2 = np.random.rand()

        assert val1 != val2


class TestConfigureTorchBackends:
    """Tests for configure_torch_backends function."""

    def test_configure_backends_runs(self):
        # Just verify it doesn't raise
        configure_torch_backends(allow_tf32=True, deterministic=False)
        configure_torch_backends(allow_tf32=False, deterministic=True)


class TestNanGuard:
    """Tests for NanGuard class."""

    def test_disabled_guard_is_noop(self):
        cfg = NanGuardCfg(enabled=False)
        guard = NanGuard(cfg)

        obs = np.array([1.0, 2.0, 3.0])
        guard.capture(observation=obs)
        assert not guard.check_and_dump(observation=obs)

    def test_enabled_guard_captures_states(self):
        cfg = NanGuardCfg(enabled=True, buffer_size=10)
        guard = NanGuard(cfg)

        obs = np.array([1.0, 2.0, 3.0])
        guard.capture(observation=obs)
        assert len(guard.buffer) == 1

        guard.capture(observation=obs)
        assert len(guard.buffer) == 2

    def test_detect_nans(self):
        assert NanGuard.detect_nans(np.array([1.0, np.nan, 3.0]))
        assert NanGuard.detect_nans(np.array([1.0, np.inf, 3.0]))
        assert NanGuard.detect_nans(np.array([1.0, -np.inf, 3.0]))
        assert not NanGuard.detect_nans(np.array([1.0, 2.0, 3.0]))

    def test_check_and_dump_detects_nan(self, tmp_path):
        cfg = NanGuardCfg(enabled=True, buffer_size=10, output_dir=str(tmp_path))
        guard = NanGuard(cfg)

        # Normal observation
        obs = np.array([1.0, 2.0, 3.0])
        guard.capture(observation=obs)
        assert not guard.check_and_dump(observation=obs)

        # NaN observation
        obs_nan = np.array([1.0, np.nan, 3.0])
        assert guard.check_and_dump(observation=obs_nan)

        # Should have created dump file
        assert (tmp_path / "nan_dump_latest.npz").exists()

    def test_context_manager(self):
        cfg = NanGuardCfg(enabled=True, buffer_size=10)
        guard = NanGuard(cfg)

        obs = np.array([1.0, 2.0, 3.0])
        action = np.array([0.5, 0.5])

        with guard.watch(obs, action):
            pass

        assert len(guard.buffer) == 1
        assert guard.buffer[0]["observation"] is not None
        assert guard.buffer[0]["action"] is not None


class TestCircularBuffer:
    """Tests for CircularBuffer class."""

    def test_buffer_creation(self):
        buf = CircularBuffer(max_len=3, batch_size=2, device="cpu")
        assert buf.max_length == 3
        assert buf.batch_size == 2
        assert not buf.is_initialized

    def test_append_and_retrieve(self):
        buf = CircularBuffer(max_len=3, batch_size=2, device="cpu")

        data = torch.tensor([[1.0], [10.0]])
        buf.append(data)
        assert buf.is_initialized

        # LIFO retrieval: lag=0 returns most recent
        result = buf[0]
        assert torch.allclose(result, data)

    def test_backfill_on_first_append(self):
        buf = CircularBuffer(max_len=3, batch_size=2, device="cpu")

        data = torch.tensor([[5.0], [10.0]])
        buf.append(data)

        # buffer property should show backfilled values
        full_buffer = buf.buffer
        assert full_buffer.shape == (2, 3, 1)
        # All values should be backfilled with first value
        assert torch.allclose(full_buffer[0], torch.tensor([[5.0], [5.0], [5.0]]))
        assert torch.allclose(full_buffer[1], torch.tensor([[10.0], [10.0], [10.0]]))

    def test_circular_overwrite(self):
        buf = CircularBuffer(max_len=3, batch_size=1, device="cpu")

        for i in range(5):
            buf.append(torch.tensor([[float(i)]]))

        # buffer should contain [2, 3, 4] (oldest overwritten)
        full_buffer = buf.buffer
        expected = torch.tensor([[[2.0], [3.0], [4.0]]])
        assert torch.allclose(full_buffer, expected)

    def test_reset_specific_batch(self):
        buf = CircularBuffer(max_len=3, batch_size=3, device="cpu")

        for i in range(3):
            buf.append(torch.tensor([[1.0], [2.0], [3.0]]))

        buf.reset(batch_ids=[1])

        # Batch 1 should be zeroed, others unchanged
        assert buf.current_length[0].item() == 3
        assert buf.current_length[1].item() == 0
        assert buf.current_length[2].item() == 3

    def test_lifo_retrieval(self):
        buf = CircularBuffer(max_len=3, batch_size=1, device="cpu")

        buf.append(torch.tensor([[1.0]]))
        buf.append(torch.tensor([[2.0]]))
        buf.append(torch.tensor([[3.0]]))

        assert buf[0].item() == 3.0  # Most recent
        assert buf[1].item() == 2.0  # One step back
        assert buf[2].item() == 1.0  # Oldest


class TestDelayBuffer:
    """Tests for DelayBuffer class."""

    def test_buffer_creation(self):
        buf = DelayBuffer(min_lag=0, max_lag=3, batch_size=4, device="cpu")
        assert buf.min_lag == 0
        assert buf.max_lag == 3
        assert buf.batch_size == 4
        assert not buf.is_initialized

    def test_append_and_compute(self):
        buf = DelayBuffer(min_lag=0, max_lag=0, batch_size=1, device="cpu")

        data = torch.tensor([[1.0, 2.0]])
        buf.append(data)
        result = buf.compute()

        assert buf.is_initialized
        assert torch.allclose(result, data)

    def test_constant_delay(self):
        buf = DelayBuffer(min_lag=2, max_lag=2, batch_size=1, device="cpu")

        for i in range(5):
            buf.append(torch.tensor([[float(i)]]))
            result = buf.compute()

            # With lag=2, we get observation from 2 steps ago (clamped to available)
            if i < 2:
                assert result.item() == 0.0  # Not enough history
            else:
                assert result.item() == float(i - 2)

    def test_reset(self):
        buf = DelayBuffer(min_lag=0, max_lag=3, batch_size=2, device="cpu")

        buf.append(torch.tensor([[1.0], [2.0]]))
        buf.reset(batch_ids=[0])

        assert buf._buffer.current_length[0].item() == 0
        assert buf._buffer.current_length[1].item() == 1

    def test_per_env_lag(self):
        # Set seed for reproducibility
        torch.manual_seed(42)

        buf = DelayBuffer(
            min_lag=0, max_lag=5, batch_size=4, device="cpu", per_env=True
        )

        buf.append(torch.randn(4, 10))
        buf.compute()

        # Per-env should have different lags (with high probability)
        lags = buf.current_lags
        assert lags.shape == (4,)

    def test_shared_lag(self):
        torch.manual_seed(42)

        buf = DelayBuffer(
            min_lag=2, max_lag=2, batch_size=4, device="cpu", per_env=False
        )

        buf.append(torch.randn(4, 10))
        buf.compute()

        # Shared lag should be the same for all envs
        lags = buf.current_lags
        assert torch.all(lags == lags[0])


class TestLogging:
    """Tests for logging utilities."""

    def test_colorize_returns_string(self):
        result = colorize("test", "green")
        assert isinstance(result, str)
        assert "test" in result

    def test_format_metric(self):
        result = format_metric("reward", 1.234)
        assert "reward" in result
        assert "1.234" in result

    def test_format_metric_scientific_small(self):
        result = format_metric("tiny", 0.00001)
        assert "e" in result.lower()

    def test_format_metric_scientific_large(self):
        result = format_metric("huge", 100000)
        assert "e" in result.lower()

    def test_format_metrics_table_groups_by_prefix(self):
        metrics = {
            "Episode_Reward/tracking": 1.0,
            "Episode_Reward/posture": 0.5,
            "Episode_Termination/fell_over": 0.0,
        }
        result = format_metrics_table(metrics, group_by_prefix=True)
        assert "Episode_Reward" in result
        assert "Episode_Termination" in result
        assert "tracking" in result

    def test_format_metrics_table_no_grouping(self):
        metrics = {"a": 1.0, "b": 2.0}
        result = format_metrics_table(metrics, group_by_prefix=False)
        assert "a" in result
        assert "b" in result

    def test_print_info_no_error(self, capsys):
        print_info("test message", "green")
        captured = capsys.readouterr()
        assert "test message" in captured.out

    def test_print_header_no_error(self, capsys):
        print_header("Test Header")
        captured = capsys.readouterr()
        assert "Test Header" in captured.out

    def test_print_episode_summary_no_error(self, capsys):
        print_episode_summary(
            episode_num=1,
            total_reward=10.5,
            episode_length=100,
            terminated=False,
            truncated=True,
        )
        captured = capsys.readouterr()
        assert "Episode" in captured.out
        assert "10.5" in captured.out


class TestMetricsLogger:
    """Tests for MetricsLogger class."""

    def test_creation(self):
        cfg = MetricsLoggerCfg(log_to_console=False)
        logger = MetricsLogger(cfg)
        assert logger.episode_count == 0
        assert logger.total_steps == 0

    def test_log_episode_with_dict(self):
        """Test logging with arbitrary dict (task-agnostic)."""
        cfg = MetricsLoggerCfg(log_to_console=False, log_to_file=False)
        logger = MetricsLogger(cfg)

        # Log with arbitrary metrics - no hardcoded fields
        log_dict = {
            "Episode/total_reward": 10.0,
            "Episode/length": 100,
            "Episode_Reward/tracking": 5.0,
            "Episode_Reward/posture": 3.0,
            "Episode_Termination/fell_over": 0,
        }
        logger.log_episode(log_dict)

        assert logger.episode_count == 1
        assert logger.total_steps == 100

    def test_log_episode_custom_task(self):
        """Test logging with completely custom task metrics."""
        cfg = MetricsLoggerCfg(log_to_console=False, log_to_file=False)
        logger = MetricsLogger(cfg)

        # Custom task with different reward structure
        log_dict = {
            "Episode/length": 200,
            "Episode_Reward/grasp_success": 1.0,
            "Episode_Reward/distance_to_target": -0.5,
            "Episode_Reward/collision_penalty": -0.1,
            "Episode_Termination/task_complete": 1,
            "Custom/gripper_force": 5.2,
        }
        logger.log_episode(log_dict)

        assert logger.episode_count == 1
        assert logger.total_steps == 200

        # Check that custom metrics are tracked
        avg = logger.get_rolling_average("Episode_Reward/grasp_success")
        assert avg == 1.0

    def test_multiple_episodes(self):
        cfg = MetricsLoggerCfg(log_to_console=False, log_to_file=False)
        logger = MetricsLogger(cfg)

        for i in range(5):
            log_dict = {
                "Episode/total_reward": 10.0 * (i + 1),
                "Episode/length": 100,
            }
            logger.log_episode(log_dict)

        assert logger.episode_count == 5
        assert logger.total_steps == 500

    def test_get_rolling_average(self):
        cfg = MetricsLoggerCfg(log_to_console=False, log_to_file=False)
        logger = MetricsLogger(cfg)

        for i in range(3):
            log_dict = {
                "Episode/total_reward": 10.0,
                "Episode/length": 100 + i * 10,
            }
            logger.log_episode(log_dict)

        avg = logger.get_rolling_average("Episode/total_reward")
        assert avg == 10.0

    def test_get_metric_stats(self):
        cfg = MetricsLoggerCfg(log_to_console=False, log_to_file=False)
        logger = MetricsLogger(cfg)

        for i in range(5):
            log_dict = {
                "Episode/total_reward": float(i),  # 0, 1, 2, 3, 4
                "Episode/length": 100,
            }
            logger.log_episode(log_dict)

        stats = logger.get_metric_stats("Episode/total_reward")
        assert stats is not None
        assert stats["mean"] == 2.0
        assert stats["min"] == 0.0
        assert stats["max"] == 4.0
        assert stats["count"] == 5

    def test_log_metrics(self):
        cfg = MetricsLoggerCfg(log_to_console=False, log_to_file=False)
        logger = MetricsLogger(cfg)

        logger.log_metrics({"loss": 0.5, "entropy": 1.2}, step=100)

        avg_loss = logger.get_rolling_average("loss")
        assert avg_loss == 0.5

    def test_get_all_metrics(self):
        cfg = MetricsLoggerCfg(log_to_console=False, log_to_file=False)
        logger = MetricsLogger(cfg)

        log_dict = {
            "Episode/reward": 10.0,
            "Episode/length": 100.0,
        }
        logger.log_episode(log_dict)

        all_metrics = logger.get_all_metrics()
        assert "Episode/reward" in all_metrics
        assert "Episode/length" in all_metrics

    def test_file_logging(self, tmp_path):
        cfg = MetricsLoggerCfg(
            log_dir=str(tmp_path),
            log_to_console=False,
            log_to_file=True,
            file_interval=1,
        )
        logger = MetricsLogger(cfg)

        log_dict = {
            "Episode/total_reward": 10.0,
            "Episode/length": 100,
        }
        logger.log_episode(log_dict)

        # Check that log file was created
        log_file = tmp_path / "episodes.jsonl"
        assert log_file.exists()

    def test_create_logger_convenience(self, tmp_path):
        """Test the create_logger convenience function."""
        logger = create_logger(log_dir=str(tmp_path), console_interval=10)
        assert logger.cfg.console_interval == 10
        assert logger.log_dir == tmp_path

    def test_close(self):
        cfg = MetricsLoggerCfg(log_to_console=False, log_to_file=False)
        logger = MetricsLogger(cfg)
        logger.close()  # Should not raise


class TestIterationLogger:
    """Tests for IterationLogger class (Isaac Lab / RSL-RL style)."""

    def test_creation(self):
        cfg = IterationLoggerCfg(log_interval=1000)  # High interval to suppress output
        logger = IterationLogger(cfg, max_iterations=1000)
        assert logger.current_iteration == 0
        assert logger.total_timesteps == 0

    def test_log_iteration(self, capsys):
        cfg = IterationLoggerCfg(log_interval=1)
        logger = IterationLogger(cfg, max_iterations=100)

        logger.log_iteration(
            iteration=1,
            losses={"value_function": 0.5, "surrogate": 0.1, "entropy": -0.02},
            episode_rewards=[10.0, 12.0, 8.0],
            episode_lengths=[100, 120, 80],
            num_steps=1000,
            collection_time=0.1,
            learn_time=0.05,
            action_std=0.5,
        )

        captured = capsys.readouterr()
        assert "Learning iteration 1/100" in captured.out
        assert "value_function" in captured.out
        assert "Mean reward:" in captured.out

    def test_episode_tracking(self):
        cfg = IterationLoggerCfg(log_interval=1000)
        logger = IterationLogger(cfg, max_iterations=100)

        # Log some episodes
        logger.log_episode(reward=10.0, length=100)
        logger.log_episode(reward=20.0, length=200)
        logger.log_episode(reward=30.0, length=300)

        assert logger.get_mean_reward() == 20.0
        assert logger.get_mean_length() == 200.0

    def test_env_metrics(self):
        cfg = IterationLoggerCfg(log_interval=1000)
        logger = IterationLogger(cfg, max_iterations=100)

        # Log episode with env metrics
        logger.log_episode(
            reward=10.0,
            length=100,
            env_log={
                "Episode_Reward/tracking": 5.0,
                "Episode_Reward/posture": 3.0,
                "Locomotion/velocity_error": 0.1,
            },
        )

        # Check that env metrics are tracked
        assert logger.get_env_metric_mean("Episode_Reward/tracking") == 5.0
        assert logger.get_env_metric_mean("Locomotion/velocity_error") == 0.1

    def test_timestep_tracking(self):
        cfg = IterationLoggerCfg(log_interval=1000)
        logger = IterationLogger(cfg, max_iterations=100)

        logger.log_iteration(iteration=1, num_steps=1000)
        logger.log_iteration(iteration=2, num_steps=1000)
        logger.log_iteration(iteration=3, num_steps=1000)

        assert logger.total_timesteps == 3000
        assert logger.current_iteration == 3

    def test_format_time(self):
        cfg = IterationLoggerCfg(log_interval=1000)
        logger = IterationLogger(cfg, max_iterations=100)

        assert logger._format_time(0) == "00:00:00"
        assert logger._format_time(61) == "00:01:01"
        assert logger._format_time(3661) == "01:01:01"

    def test_close(self, capsys):
        cfg = IterationLoggerCfg(log_interval=1000)
        logger = IterationLogger(cfg, max_iterations=100)
        logger.log_episode(reward=10.0, length=100)
        logger.close()

        captured = capsys.readouterr()
        assert "Training Complete" in captured.out
