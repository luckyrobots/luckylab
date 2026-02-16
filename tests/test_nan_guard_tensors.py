"""Tests for NaN guard tensor-level checks (actions, observations, rewards)."""

import torch

from luckylab.utils.nan_guard import NanGuard, NanGuardCfg


class TestCheckTensor:
    """Tests for the generic check_tensor method."""

    def test_clean_tensor_passes(self):
        guard = NanGuard(NanGuardCfg(enabled=True, verbose=False))
        t = torch.tensor([1.0, 2.0, 3.0])
        result, had_nan = guard.check_tensor(t, "test")
        assert had_nan is False
        assert torch.equal(result, t)

    def test_nan_detected(self):
        guard = NanGuard(NanGuardCfg(enabled=True, verbose=False, recovery_mode=True, halt_on_nan=False))
        t = torch.tensor([1.0, float("nan"), 3.0])
        result, had_nan = guard.check_tensor(t, "test")
        assert had_nan is True
        assert not torch.isnan(result).any()

    def test_inf_detected(self):
        guard = NanGuard(NanGuardCfg(enabled=True, verbose=False, recovery_mode=True, halt_on_nan=False))
        t = torch.tensor([1.0, float("inf"), 3.0])
        result, had_nan = guard.check_tensor(t, "test")
        assert had_nan is True
        assert not torch.isinf(result).any()

    def test_recovery_replaces_with_value(self):
        guard = NanGuard(NanGuardCfg(enabled=True, verbose=False, recovery_mode=True, halt_on_nan=False))
        t = torch.tensor([1.0, float("nan"), 3.0])
        result, _ = guard.check_tensor(t, "test", replace_value=42.0)
        assert result[1].item() == 42.0
        # Non-NaN values preserved
        assert result[0].item() == 1.0
        assert result[2].item() == 3.0

    def test_disabled_guard_skips(self):
        guard = NanGuard(NanGuardCfg(enabled=False))
        t = torch.tensor([float("nan")])
        result, had_nan = guard.check_tensor(t, "test")
        assert had_nan is False
        assert torch.isnan(result).any()  # NaN not replaced

    def test_original_tensor_not_mutated(self):
        guard = NanGuard(NanGuardCfg(enabled=True, verbose=False, recovery_mode=True, halt_on_nan=False))
        t = torch.tensor([1.0, float("nan"), 3.0])
        original = t.clone()
        guard.check_tensor(t, "test")
        assert torch.isnan(t[1])  # original still has NaN


class TestCheckActions:
    """Tests for check_actions method."""

    def test_clean_actions_pass(self):
        guard = NanGuard(NanGuardCfg(enabled=True, verbose=False))
        actions = torch.randn(4, 12)
        result, had_nan = guard.check_actions(actions)
        assert had_nan is False

    def test_nan_actions_detected(self):
        guard = NanGuard(NanGuardCfg(enabled=True, verbose=False, recovery_mode=True, halt_on_nan=False))
        actions = torch.randn(4, 12)
        actions[2, 5] = float("nan")
        result, had_nan = guard.check_actions(actions)
        assert had_nan is True
        assert not torch.isnan(result).any()

    def test_action_nan_stats_tracked(self):
        guard = NanGuard(NanGuardCfg(enabled=True, verbose=False, recovery_mode=True, halt_on_nan=False))
        actions = torch.tensor([[float("nan"), 1.0]])
        guard.check_actions(actions)
        assert guard.stats.action_nans == 1

    def test_halt_on_nan_raises(self):
        import pytest

        guard = NanGuard(
            NanGuardCfg(enabled=True, verbose=False, halt_on_nan=True, output_dir="/tmp/test_nan")
        )
        actions = torch.tensor([[float("nan"), 1.0]])
        with pytest.raises(RuntimeError, match="NaN detected in actions"):
            guard.check_actions(actions)

    def test_disabled_check_actions_flag(self):
        guard = NanGuard(
            NanGuardCfg(enabled=True, verbose=False, check_actions=False)
        )
        actions = torch.tensor([[float("nan"), 1.0]])
        result, had_nan = guard.check_actions(actions)
        assert had_nan is False  # check skipped


class TestCheckObservations:
    """Tests for check_observations method."""

    def test_clean_observations_pass(self):
        guard = NanGuard(NanGuardCfg(enabled=True, verbose=False))
        obs = torch.randn(4, 48)
        result, had_nan = guard.check_observations(obs)
        assert had_nan is False

    def test_nan_observations_detected(self):
        guard = NanGuard(NanGuardCfg(enabled=True, verbose=False, recovery_mode=True, halt_on_nan=False))
        obs = torch.randn(4, 48)
        obs[0, 0] = float("nan")
        result, had_nan = guard.check_observations(obs)
        assert had_nan is True
        assert guard.stats.observation_nans == 1

    def test_disabled_check_observations_flag(self):
        guard = NanGuard(
            NanGuardCfg(enabled=True, verbose=False, check_observations=False)
        )
        obs = torch.tensor([[float("nan")]])
        result, had_nan = guard.check_observations(obs)
        assert had_nan is False


class TestCheckReward:
    """Tests for check_reward method."""

    def test_clean_reward_passes(self):
        guard = NanGuard(NanGuardCfg(enabled=True, verbose=False))
        rew = torch.tensor([1.0, 2.0])
        result, had_nan = guard.check_reward(rew)
        assert had_nan is False

    def test_nan_reward_detected(self):
        guard = NanGuard(NanGuardCfg(enabled=True, verbose=False, recovery_mode=True, halt_on_nan=False))
        rew = torch.tensor([1.0, float("nan")])
        result, had_nan = guard.check_reward(rew)
        assert had_nan is True
        assert guard.stats.reward_nans == 1
        assert not torch.isnan(result).any()

    def test_scalar_nan_reward(self):
        import math

        guard = NanGuard(NanGuardCfg(enabled=True, verbose=False, recovery_mode=True, halt_on_nan=False))
        result, had_nan = guard.check_reward(float("nan"))
        assert had_nan is True
        assert result == 0.0  # default replace_value


class TestNanGuardStats:
    """Tests for NaN statistics tracking."""

    def test_first_and_last_nan_step(self):
        guard = NanGuard(NanGuardCfg(enabled=True, verbose=False, recovery_mode=True, halt_on_nan=False))
        guard.step_counter = 10
        guard.check_actions(torch.tensor([[float("nan")]]))
        assert guard.stats.first_nan_step == 10
        assert guard.stats.last_nan_step == 10

        guard.step_counter = 20
        guard.check_actions(torch.tensor([[float("nan")]]))
        assert guard.stats.first_nan_step == 10  # unchanged
        assert guard.stats.last_nan_step == 20

    def test_stats_summary_when_enabled(self):
        guard = NanGuard(NanGuardCfg(enabled=True, verbose=False))
        summary = guard.get_stats_summary()
        assert "NanGuard Stats" in summary

    def test_stats_summary_when_disabled(self):
        guard = NanGuard(NanGuardCfg(enabled=False))
        summary = guard.get_stats_summary()
        assert "Disabled" in summary

    def test_recovery_count(self):
        guard = NanGuard(NanGuardCfg(enabled=True, verbose=False, recovery_mode=True, halt_on_nan=False))
        guard.check_actions(torch.tensor([[float("nan")]]))
        guard.check_observations(torch.tensor([[float("nan")]]))
        guard.check_reward(torch.tensor([float("nan")]))
        assert guard.stats.recoveries == 3
