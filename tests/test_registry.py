"""Tests for LuckyLab task registry."""

import pytest

from luckylab.envs import ManagerBasedRlEnvCfg
from luckylab.tasks.registry import (
    clear_registry,
    is_registered,
    list_tasks,
    load_env_cfg,
    register_task,
    unregister_task,
)
from luckylab.tasks.velocity import create_velocity_env_cfg


def _create_test_cfg():
    """Factory function for tests that creates a valid config."""
    return ManagerBasedRlEnvCfg(decimation=4)


class TestRegistry:
    def setup_method(self):
        """Clear registry before each test."""
        clear_registry()

    def teardown_method(self):
        """Clear registry after each test."""
        clear_registry()

    def test_register_and_list(self):
        register_task("test_task", _create_test_cfg)
        tasks = list_tasks()
        assert "test_task" in tasks

    def test_register_duplicate_raises(self):
        register_task("test_task", _create_test_cfg)
        with pytest.raises(ValueError, match="already registered"):
            register_task("test_task", _create_test_cfg)

    def test_load_env_cfg(self):
        register_task("test_task", _create_test_cfg)
        cfg = load_env_cfg("test_task")
        assert isinstance(cfg, ManagerBasedRlEnvCfg)

    def test_load_nonexistent_raises(self):
        with pytest.raises(KeyError, match="not found"):
            load_env_cfg("nonexistent")

    def test_is_registered(self):
        assert not is_registered("test_task")
        register_task("test_task", _create_test_cfg)
        assert is_registered("test_task")

    def test_unregister(self):
        register_task("test_task", _create_test_cfg)
        assert is_registered("test_task")
        unregister_task("test_task")
        assert not is_registered("test_task")

    def test_clear_registry(self):
        register_task("task1", _create_test_cfg)
        register_task("task2", _create_test_cfg)
        clear_registry()
        assert list_tasks() == []

    def test_list_tasks_sorted(self):
        register_task("zebra", _create_test_cfg)
        register_task("alpha", _create_test_cfg)
        register_task("beta", _create_test_cfg)
        tasks = list_tasks()
        assert tasks == ["alpha", "beta", "zebra"]

    def test_factory_function_support(self):
        """Test that registry supports factory functions like create_velocity_env_cfg."""

        def create_custom_cfg():
            return create_velocity_env_cfg(robot="unitreego1")

        register_task("factory_task", create_custom_cfg)
        cfg = load_env_cfg("factory_task")
        # Check derived property works
        assert cfg.max_episode_length == 250  # 20.0 / (0.02 * 4)
