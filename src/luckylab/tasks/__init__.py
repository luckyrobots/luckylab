"""Task registry for luckylab environments."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from luckylab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
    from luckylab.rl.config import RlRunnerCfg


# Task registry: task_name -> (env_cfg_factory, rl_cfg)
_TASK_REGISTRY: dict[str, tuple[callable, object]] = {}


def register_task(
    task_name: str,
    env_cfg_factory: callable,
    rl_cfg: object | None = None,
) -> None:
    """Register a task configuration.

    Args:
        task_name: Unique task identifier (e.g., "go1_velocity_flat").
        env_cfg_factory: Callable that returns a ManagerBasedRlEnvCfg.
        rl_cfg: Optional RlRunnerCfg for the task.
    """
    _TASK_REGISTRY[task_name] = (env_cfg_factory, rl_cfg)


def load_env_cfg(task_name: str) -> "ManagerBasedRlEnvCfg":
    """Load environment configuration for a task.

    Args:
        task_name: Task identifier.

    Returns:
        ManagerBasedRlEnvCfg for the task.

    Raises:
        KeyError: If task is not registered.
    """
    if task_name not in _TASK_REGISTRY:
        available = ", ".join(sorted(_TASK_REGISTRY.keys())) or "(none)"
        raise KeyError(f"Task '{task_name}' not found. Available tasks: {available}")

    env_cfg_factory, _ = _TASK_REGISTRY[task_name]
    # Call factory if it's callable, otherwise return as-is
    if callable(env_cfg_factory):
        return env_cfg_factory()
    return deepcopy(env_cfg_factory)


def load_rl_cfg(task_name: str) -> "RlRunnerCfg | None":
    """Load RL configuration for a task.

    Args:
        task_name: Task identifier.

    Returns:
        RlRunnerCfg for the task, or None if not specified.

    Raises:
        KeyError: If task is not registered.
    """
    if task_name not in _TASK_REGISTRY:
        available = ", ".join(sorted(_TASK_REGISTRY.keys())) or "(none)"
        raise KeyError(f"Task '{task_name}' not found. Available tasks: {available}")

    _, rl_cfg = _TASK_REGISTRY[task_name]
    if rl_cfg is not None:
        return deepcopy(rl_cfg)
    return None


def list_tasks() -> list[str]:
    """List all registered tasks.

    Returns:
        Sorted list of task names.
    """
    return sorted(_TASK_REGISTRY.keys())


# Register built-in tasks
def _register_builtin_tasks() -> None:
    """Register built-in velocity tasks."""
    try:
        from luckylab.tasks.velocity.config.go1 import (
            UNITREE_GO1_FLAT_ENV_CFG,
            UNITREE_GO1_PPO_RUNNER_CFG,
            UNITREE_GO1_ROUGH_ENV_CFG,
            UNITREE_GO1_SAC_RUNNER_CFG,
        )

        # Flat terrain (recommended for starting)
        register_task(
            "go1_velocity_flat",
            UNITREE_GO1_FLAT_ENV_CFG,
            UNITREE_GO1_PPO_RUNNER_CFG,
        )

        # Rough terrain
        register_task(
            "go1_velocity_rough",
            UNITREE_GO1_ROUGH_ENV_CFG,
            UNITREE_GO1_PPO_RUNNER_CFG,
        )

        # SAC variants
        register_task(
            "go1_velocity_flat_sac",
            UNITREE_GO1_FLAT_ENV_CFG,
            UNITREE_GO1_SAC_RUNNER_CFG,
        )

    except ImportError:
        # Tasks not available (missing dependencies)
        pass


_register_builtin_tasks()
