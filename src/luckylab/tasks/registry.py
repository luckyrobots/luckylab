"""Task registry for luckylab environments."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from ..rl.config import SkrlCfg

T = TypeVar("T")

# Registry stores (env_cfg_class, rl_cfg) pairs
EnvRlCfgPair = tuple[type | Callable, "SkrlCfg | None"]
_REGISTRY: dict[str, EnvRlCfgPair] = {}


def register_task(
    task_id: str,
    env_cfg_class: type | Callable,
    rl_cfg: SkrlCfg | None = None,
) -> None:
    """
    Register a task with its environment configuration class and optional RL config.

    Args:
        task_id: Unique identifier for the task (e.g., "go1_velocity_flat")
        env_cfg_class: The environment configuration class or factory function
        rl_cfg: Optional RL training configuration for this task
    """
    if task_id in _REGISTRY:
        raise ValueError(f"Task '{task_id}' is already registered")
    _REGISTRY[task_id] = (env_cfg_class, rl_cfg)


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

    env_cfg_class, _ = _REGISTRY[task_id]

    # Support both classes and factory functions
    if callable(env_cfg_class):
        return env_cfg_class()
    return env_cfg_class


def load_rl_cfg(task_id: str) -> SkrlCfg | None:
    """
    Load the RL training configuration for a task.

    Args:
        task_id: The task identifier

    Returns:
        The RL configuration, or None if not registered

    Raises:
        KeyError: If task_id is not registered
    """
    if task_id not in _REGISTRY:
        available = ", ".join(list_tasks()) or "(none)"
        raise KeyError(f"Task '{task_id}' not found. Available tasks: {available}")

    _, rl_cfg = _REGISTRY[task_id]
    return rl_cfg


def is_registered(task_id: str) -> bool:
    """Check if a task is registered."""
    return task_id in _REGISTRY


def unregister_task(task_id: str) -> None:
    """Remove a task from the registry (mainly for testing)."""
    _REGISTRY.pop(task_id, None)


def clear_registry() -> None:
    """Clear all registered tasks (mainly for testing)."""
    _REGISTRY.clear()
