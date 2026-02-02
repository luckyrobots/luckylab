"""Command manager for generating and updating commands.

Follows mjlab pattern with class_type-based command terms and multi-env support.
Uses torch tensors for GPU compatibility.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

import torch

from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_config import CommandTermCfg

if TYPE_CHECKING:
    from ..envs.manager_based_rl_env import ManagerBasedRlEnv


class CommandTerm(ManagerTermBase):
    """Base class for command terms.

    Command terms generate target values (e.g., velocity commands) that the
    policy should track. Commands are resampled periodically based on the
    resampling_time_range in the config.
    """

    def __init__(self, cfg: CommandTermCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        self.cfg: CommandTermCfg = cfg
        self.metrics: dict[str, torch.Tensor] = {}

        # Time left until next resample for each env
        self.time_left = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        # Command counter per env
        self.command_counter = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)

    @property
    @abc.abstractmethod
    def command(self) -> torch.Tensor:
        """Current command array with shape (num_envs, command_dim)."""
        raise NotImplementedError

    def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, float]:
        """Reset command state for specified environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        extras = {}
        for metric_name, metric_value in self.metrics.items():
            extras[metric_name] = float(metric_value[env_ids].mean().item())
            metric_value[env_ids] = 0.0

        self.command_counter[env_ids] = 0
        self._resample(env_ids)
        return extras

    def compute(self, dt: float) -> None:
        """Update commands for all environments."""
        self._update_metrics()
        self.time_left -= dt

        resample_env_ids = torch.where(self.time_left <= 0.0)[0]
        if len(resample_env_ids) > 0:
            self._resample(resample_env_ids)

        self._update_command()

    def _resample(self, env_ids: torch.Tensor) -> None:
        """Resample commands for specified environments."""
        if len(env_ids) == 0:
            return

        low, high = self.cfg.resampling_time_range
        self.time_left[env_ids] = torch.rand(len(env_ids), device=self.device) * (high - low) + low

        self._resample_command(env_ids)
        self.command_counter[env_ids] += 1

    @abc.abstractmethod
    def _update_metrics(self) -> None:
        """Update metrics based on current state."""
        raise NotImplementedError

    @abc.abstractmethod
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """Resample commands for specified environments."""
        raise NotImplementedError

    @abc.abstractmethod
    def _update_command(self) -> None:
        """Update commands based on current state (called every step)."""
        raise NotImplementedError


class CommandManager(ManagerBase):
    """Manager for command generation.

    Aggregates multiple command terms and provides unified interface for
    computing and accessing commands across all environments.
    """

    _env: ManagerBasedRlEnv

    def __init__(self, cfg: dict[str, Any], env: ManagerBasedRlEnv) -> None:
        self._terms: dict[str, CommandTerm] = {}
        self.cfg = cfg
        super().__init__(env)

    def __str__(self) -> str:
        msg = f"<CommandManager> contains {len(self._terms)} active terms.\n"
        msg += "Active Command Terms:\n"
        msg += f"{'Index':<8} {'Name':<30} {'Type':<30}\n"
        msg += "-" * 70 + "\n"
        for idx, (name, term) in enumerate(self._terms.items()):
            msg += f"{idx:<8} {name:<30} {term.__class__.__name__:<30}\n"
        return msg

    @property
    def active_terms(self) -> list[str]:
        """Get list of active term names."""
        return list(self._terms.keys())

    def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, float]:
        """Reset commands for specified environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        extras = {}
        for name, term in self._terms.items():
            metrics = term.reset(env_ids)
            for metric_name, metric_value in metrics.items():
                extras[f"Metrics/{name}/{metric_name}"] = metric_value
        return extras

    def compute(self, dt: float) -> None:
        """Update all command terms."""
        for term in self._terms.values():
            term.compute(dt)

    def get_command(self, name: str) -> torch.Tensor:
        """Get command array for a term."""
        return self._terms[name].command

    def get_term(self, name: str) -> CommandTerm:
        """Get a command term by name."""
        return self._terms[name]

    def get_active_iterable_terms(self, env_idx: int = 0) -> list[tuple[str, list[float]]]:
        """Get term values for iteration/logging."""
        terms = []
        for name, term in self._terms.items():
            terms.append((name, term.command[env_idx].tolist()))
        return terms

    def _prepare_terms(self) -> None:
        """Parse configuration and instantiate command terms."""
        for term_name, term_cfg in self.cfg.items():
            if term_cfg is None:
                continue

            term = term_cfg.class_type(term_cfg, self._env)
            if not isinstance(term, CommandTerm):
                raise TypeError(
                    f"Returned object for term '{term_name}' is not a CommandTerm. Got: {type(term).__name__}"
                )
            self._terms[term_name] = term


class NullCommandManager:
    """Placeholder for absent command manager that safely no-ops all operations."""

    def __init__(self) -> None:
        self.active_terms: list[str] = []
        self._terms: dict[str, Any] = {}
        self.cfg = None

    def __str__(self) -> str:
        return "<NullCommandManager> (inactive)"

    def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, float]:
        return {}

    def compute(self, dt: float) -> None:
        pass

    def get_command(self, name: str) -> None:
        return None

    def get_term(self, name: str) -> None:
        return None

    def get_active_iterable_terms(self, env_idx: int = 0) -> list[tuple[str, list[float]]]:
        return []
