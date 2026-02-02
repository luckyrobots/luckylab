"""Base class for all managers.

This module provides the abstract base class for managers following the mjlab pattern,
with multi-environment support using torch tensors.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from ..envs.manager_based_rl_env import ManagerBasedRlEnv
    from .manager_term_config import ManagerTermBaseCfg


class ManagerTermBase:
    """Base class for class-based manager terms.

    Unlike function-based terms, class-based terms can maintain state
    across calls and implement reset logic.
    """

    def __init__(self, cfg: Any, env: ManagerBasedRlEnv) -> None:
        self.cfg = cfg
        self._env = env

    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    @property
    def device(self) -> torch.device:
        return self._env.device

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, float]:
        """Reset the term state for specified environments."""
        return {}

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class ManagerBase(abc.ABC):
    """Abstract base class for all managers.

    Managers aggregate multiple terms (rewards, terminations, etc.) and provide
    a unified interface for computing their values and resetting state.
    """

    def __init__(self, env: ManagerBasedRlEnv) -> None:
        self._env = env
        self._prepare_terms()

    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    @property
    def device(self) -> torch.device:
        return self._env.device

    @property
    @abc.abstractmethod
    def active_terms(self) -> list[str]:
        raise NotImplementedError

    def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, Any]:
        """Reset the manager state for specified environments."""
        return {}

    def get_active_iterable_terms(self, env_idx: int = 0) -> Sequence[tuple[str, Sequence[float]]]:
        """Get term values for iteration/logging."""
        raise NotImplementedError

    @abc.abstractmethod
    def _prepare_terms(self) -> None:
        """Parse configuration and prepare terms."""
        raise NotImplementedError

    def _resolve_common_term_cfg(self, term_name: str, term_cfg: ManagerTermBaseCfg) -> None:
        """Resolve common term configuration."""
        import inspect

        del term_name
        if inspect.isclass(term_cfg.func):
            term_cfg.func = term_cfg.func(cfg=term_cfg, env=self._env)
