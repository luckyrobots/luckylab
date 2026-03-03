"""Tests for LuckyLab managers."""

import numpy as np

from luckylab.managers import (
    NullCurriculumManager,
)
from luckylab.managers.manager_term_config import (
    CurriculumTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)


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


class TestTermConfigDataclasses:
    """Tests for manager term config dataclasses."""

    def test_curriculum_term_cfg(self):
        def dummy(env, env_ids):
            pass

        cfg = CurriculumTermCfg(func=dummy)
        assert cfg.func is dummy
        assert cfg.params == {}

    def test_reward_term_cfg(self):
        def dummy(env):
            return 0.0

        cfg = RewardTermCfg(func=dummy, weight=2.0, params={"std": 0.5})
        assert cfg.func is dummy
        assert cfg.weight == 2.0
        assert cfg.params == {"std": 0.5}

    def test_termination_term_cfg(self):
        def dummy(env):
            return False

        cfg = TerminationTermCfg(func=dummy, time_out=True)
        assert cfg.func is dummy
        assert cfg.time_out is True

    def test_termination_term_cfg_default_time_out(self):
        def dummy(env):
            return False

        cfg = TerminationTermCfg(func=dummy)
        assert cfg.time_out is False
