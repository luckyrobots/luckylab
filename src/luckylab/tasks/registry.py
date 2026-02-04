"""Task registry system for managing environment registration and creation."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from luckylab.rl.config import RlRunnerCfg

T = TypeVar("T")

# Registry stores (env_cfg, rl_cfgs_dict) pairs
# rl_cfgs_dict maps algorithm name -> RlRunnerCfg
# Using Any at runtime to avoid circular import
EnvRlCfgPair = tuple[type | Callable, dict[str, Any]]
_REGISTRY: dict[str, EnvRlCfgPair] = {}


def register_task(
    task_id: str,
    env_cfg: type | Callable,
    rl_cfgs: dict[str, RlRunnerCfg] | None = None,
) -> None:
    """
    Register a task with its environment configuration and RL configs per algorithm.

    Args:
        task_id: Unique identifier for the task (e.g., "go1_velocity_flat")
        env_cfg: The environment configuration class, instance, or factory function
        rl_cfgs: Dict mapping algorithm name to RL config (e.g., {"ppo": ..., "sac": ...})
    """
    if task_id in _REGISTRY:
        raise ValueError(f"Task '{task_id}' is already registered")
    _REGISTRY[task_id] = (env_cfg, rl_cfgs or {})


def list_tasks() -> list[str]:
    """Return a sorted list of all registered task IDs."""
    return sorted(_REGISTRY.keys())


def load_env_cfg(task_id: str) -> object:
    """
    Load the environment configuration for a task.

    Args:
        task_id: The task identifier

    Returns:
        An instance of the environment configuration

    Raises:
        KeyError: If task_id is not registered
    """
    if task_id not in _REGISTRY:
        available = ", ".join(list_tasks()) or "(none)"
        raise KeyError(f"Task '{task_id}' not found. Available tasks: {available}")

    env_cfg, _ = _REGISTRY[task_id]

    # Support both classes and factory functions (but not @retval decorated ones)
    if isinstance(env_cfg, type) and callable(env_cfg):
        return env_cfg()
    return env_cfg


def load_rl_cfg(task_id: str, algorithm: str = "ppo") -> "RlRunnerCfg | None":
    """
    Load the RL training configuration for a task and algorithm.

    Args:
        task_id: The task identifier
        algorithm: The algorithm name (e.g., "ppo", "sac")

    Returns:
        The RL configuration, or None if not registered for this algorithm

    Raises:
        KeyError: If task_id is not registered
    """
    if task_id not in _REGISTRY:
        available = ", ".join(list_tasks()) or "(none)"
        raise KeyError(f"Task '{task_id}' not found. Available tasks: {available}")

    _, rl_cfgs = _REGISTRY[task_id]
    return rl_cfgs.get(algorithm)


def list_algorithms(task_id: str) -> list[str]:
    """Return available algorithms for a task."""
    if task_id not in _REGISTRY:
        return []
    _, rl_cfgs = _REGISTRY[task_id]
    return sorted(rl_cfgs.keys())


def is_registered(task_id: str) -> bool:
    """Check if a task is registered."""
    return task_id in _REGISTRY


def unregister_task(task_id: str) -> None:
    """Remove a task from the registry (mainly for testing)."""
    _REGISTRY.pop(task_id, None)


def clear_registry() -> None:
    """Clear all registered tasks (mainly for testing)."""
    _REGISTRY.clear()
