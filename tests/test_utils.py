"""Tests for LuckyLab utilities."""

import numpy as np
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
from luckylab.utils.nan_guard import NanGuard, NanGuardCfg
from luckylab.utils.random import seed_rng
from luckylab.utils.torch import configure_torch_backends


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

        for _ in range(3):
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

        buf = DelayBuffer(min_lag=0, max_lag=5, batch_size=4, device="cpu", per_env=True)

        buf.append(torch.randn(4, 10))
        buf.compute()

        # Per-env should have different lags (with high probability)
        lags = buf.current_lags
        assert lags.shape == (4,)

    def test_shared_lag(self):
        torch.manual_seed(42)

        buf = DelayBuffer(min_lag=2, max_lag=2, batch_size=4, device="cpu", per_env=False)

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
