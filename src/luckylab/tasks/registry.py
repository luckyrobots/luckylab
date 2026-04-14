"""Task registry system for managing environment registration and creation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from luckylab.contracts.task_contract import TaskContract
    from luckylab.il.config import IlRunnerCfg
    from luckylab.rl.config import RlRunnerCfg


@dataclass
class TaskEntry:
    """A registered task with its configurations.

    Attributes:
        env_cfg: Environment config class, instance, or factory function.
        rl_cfgs: Dict mapping algorithm name to RL config.
        il_cfgs: Dict mapping policy type to IL config.
        task_contract: Optional TaskContract for engine-side MDP computation.
            When set, ManagerBasedRlEnv will negotiate this contract with the
            engine at session start, enabling engine-computed reward signals
            and termination flags.
    """

    env_cfg: type | Callable | None = None
    rl_cfgs: dict[str, Any] = field(default_factory=dict)
    il_cfgs: dict[str, Any] = field(default_factory=dict)
    task_contract: TaskContract | None = None


_REGISTRY: dict[str, TaskEntry] = {}


def register_task(
    task_id: str,
    env_cfg: type | Callable | None = None,
    rl_cfgs: dict[str, RlRunnerCfg] | None = None,
    il_cfgs: dict[str, IlRunnerCfg] | None = None,
    task_contract: TaskContract | None = None,
) -> None:
    """
    Register a task with its environment configuration and training configs.

    Args:
        task_id: Unique identifier for the task (e.g., "go2_velocity_flat")
        env_cfg: The environment configuration class, instance, or factory function.
                 Can be None for IL-only tasks that don't need ManagerBasedRlEnvCfg.
        rl_cfgs: Dict mapping algorithm name to RL config (e.g., {"ppo": ..., "sac": ...})
        il_cfgs: Dict mapping policy type to IL config (e.g., {"act": ..., "diffusion": ...})
        task_contract: Optional TaskContract for engine-side MDP computation.
    """
    if task_id in _REGISTRY:
        raise ValueError(f"Task '{task_id}' is already registered")
    _REGISTRY[task_id] = TaskEntry(
        env_cfg=env_cfg,
        rl_cfgs=rl_cfgs or {},
        il_cfgs=il_cfgs or {},
        task_contract=task_contract,
    )


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
        ValueError: If env_cfg is None (IL-only task with no env config)
    """
    if task_id not in _REGISTRY:
        available = ", ".join(list_tasks()) or "(none)"
        raise KeyError(f"Task '{task_id}' not found. Available tasks: {available}")

    entry = _REGISTRY[task_id]

    if entry.env_cfg is None:
        raise ValueError(f"Task '{task_id}' has no environment config (IL-only task)")

    # Support both classes and factory functions (but not @retval decorated ones)
    env_cfg = entry.env_cfg
    if isinstance(env_cfg, type) and callable(env_cfg):
        return env_cfg()
    return env_cfg


def load_rl_cfg(task_id: str, algorithm: str) -> "RlRunnerCfg | None":
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

    return _REGISTRY[task_id].rl_cfgs.get(algorithm)


def load_il_cfg(task_id: str, policy_type: str) -> "IlRunnerCfg | None":
    """
    Load the IL training configuration for a task and policy type.

    Args:
        task_id: The task identifier
        policy_type: The policy type (e.g., "act", "diffusion")

    Returns:
        The IL configuration, or None if not registered for this policy type

    Raises:
        KeyError: If task_id is not registered
    """
    if task_id not in _REGISTRY:
        available = ", ".join(list_tasks()) or "(none)"
        raise KeyError(f"Task '{task_id}' not found. Available tasks: {available}")

    return _REGISTRY[task_id].il_cfgs.get(policy_type)


def load_task_contract(task_id: str) -> "TaskContract | None":
    """Load the TaskContract for a task, if one was registered.

    Args:
        task_id: The task identifier.

    Returns:
        The TaskContract, or None if the task has no contract.

    Raises:
        KeyError: If task_id is not registered.
    """
    if task_id not in _REGISTRY:
        available = ", ".join(list_tasks()) or "(none)"
        raise KeyError(f"Task '{task_id}' not found. Available tasks: {available}")

    return _REGISTRY[task_id].task_contract


def list_rl_policies(task_id: str) -> list[str]:
    """Return available RL policies for a task."""
    if task_id not in _REGISTRY:
        return []
    return sorted(_REGISTRY[task_id].rl_cfgs.keys())


def list_il_policies(task_id: str) -> list[str]:
    """Return available IL policy types for a task."""
    if task_id not in _REGISTRY:
        return []
    return sorted(_REGISTRY[task_id].il_cfgs.keys())


def is_registered(task_id: str) -> bool:
    """Check if a task is registered."""
    return task_id in _REGISTRY


def unregister_task(task_id: str) -> None:
    """Remove a task from the registry (mainly for testing)."""
    _REGISTRY.pop(task_id, None)


def clear_registry() -> None:
    """Clear all registered tasks (mainly for testing)."""
    _REGISTRY.clear()
