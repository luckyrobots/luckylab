"""Base class for all managers.

This module provides the abstract base class for managers following the mjlab pattern,
adapted for luckylab's single-environment architecture.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..envs.manager_based_rl_env import ManagerBasedRlEnv
    from .manager_term_config import ManagerTermBaseCfg


class ManagerTermBase:
    """Base class for class-based manager terms.

    Unlike function-based terms, class-based terms can maintain state
    across calls and implement reset logic.
    """

    def __init__(self, env: ManagerBasedRlEnv) -> None:
        """Initialize the manager term.

        Args:
            env: The environment instance.
        """
        self._env = env

    @property
    def name(self) -> str:
        """Get the term class name."""
        return self.__class__.__name__

    def reset(self) -> Any:
        """Reset the term state.

        Called when the environment resets. Override to implement
        custom reset logic for stateful terms.
        """
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Compute the term value.

        Override to implement the term computation.
        """
        raise NotImplementedError


class ManagerBase(abc.ABC):
    """Abstract base class for all managers.

    Managers aggregate multiple terms (rewards, terminations, etc.) and provide
    a unified interface for computing their values and resetting state.

    This follows the mjlab pattern but is adapted for luckylab's single-environment
    architecture (no batching, numpy instead of torch).
    """

    def __init__(self, env: ManagerBasedRlEnv) -> None:
        """Initialize the manager.

        Args:
            env: The environment instance.
        """
        self._env = env
        self._prepare_terms()

    @property
    @abc.abstractmethod
    def active_terms(self) -> list[str]:
        """Get list of active term names.

        Returns:
            List of term names that are currently active.
        """
        raise NotImplementedError

    def reset(self) -> dict[str, Any]:
        """Reset the manager state.

        Called when the environment resets. Returns logging info
        about the completed episode.

        Returns:
            Dictionary with episode statistics for logging.
        """
        return {}

    def get_active_iterable_terms(self) -> Sequence[tuple[str, Sequence[float]]]:
        """Get term values for iteration/logging.

        Returns:
            Sequence of (term_name, [value]) tuples.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _prepare_terms(self) -> None:
        """Parse configuration and prepare terms.

        Called during initialization to set up term functions/objects.
        """
        raise NotImplementedError

    def _resolve_common_term_cfg(self, term_name: str, term_cfg: ManagerTermBaseCfg) -> None:
        """Resolve common term configuration.

        Handles class-based terms by instantiating them with the environment.

        Args:
            term_name: Name of the term.
            term_cfg: Term configuration to resolve.
        """
        import inspect

        del term_name  # Unused but kept for API compatibility with mjlab.

        if inspect.isclass(term_cfg.func):
            # Instantiate class-based term with environment.
            term_cfg.func = term_cfg.func(env=self._env)
