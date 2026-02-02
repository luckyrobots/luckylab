"""Curriculum manager for progressive training difficulty.

Follows the mjlab ManagerBase pattern with term-based configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from .manager_base import ManagerBase

if TYPE_CHECKING:
    from ..envs.manager_based_rl_env import ManagerBasedRlEnv
    from .manager_term_config import CurriculumTermCfg


@dataclass
class EpisodeMetrics:
    """Metrics accumulated during a single episode."""

    steps: int = 0
    total_reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    termination_reason: str = ""

    def add_step(self) -> None:
        self.steps += 1

    def add_reward(self, reward: float) -> None:
        self.total_reward += reward

    def set_terminated(self, reason: str = "") -> None:
        self.terminated = True
        self.termination_reason = reason

    def set_truncated(self) -> None:
        self.truncated = True

    def finalize(self) -> dict:
        """Convert to dictionary for curriculum manager."""
        return {
            "episode_length": self.steps,
            "total_reward": self.total_reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "termination_reason": self.termination_reason,
            "survived": self.truncated and not self.terminated,  # Made it to timeout
        }

    def reset(self) -> None:
        """Reset metrics for a new episode."""
        self.steps = 0
        self.total_reward = 0.0
        self.terminated = False
        self.truncated = False
        self.termination_reason = ""


class CurriculumManager(ManagerBase):
    """Manages curriculum terms using the ManagerBase pattern.

    This follows the mjlab pattern where curriculum is a dict of term configs,
    matching how rewards and terminations are configured.

    Each curriculum term is a function that receives the environment and can
    modify its state based on training progress (step count, episode count, etc.).
    """

    _env: ManagerBasedRlEnv

    def __init__(self, cfg: dict[str, CurriculumTermCfg], env: ManagerBasedRlEnv) -> None:
        """Initialize the curriculum manager.

        Args:
            cfg: Dict mapping term names to CurriculumTermCfg instances.
            env: The environment instance.
        """
        self._term_names: list[str] = []
        self._term_cfgs: list[CurriculumTermCfg] = []
        self.cfg = cfg
        super().__init__(env=env)

    def __str__(self) -> str:
        """Get string representation of the manager."""
        lines = [f"<CurriculumManager> contains {len(self._term_names)} active terms."]
        if self._term_names:
            lines.append("Active Curriculum Terms:")
            lines.append(f"{'Index':<8} {'Name':<30}")
            lines.append("-" * 40)
            for idx, name in enumerate(self._term_names):
                lines.append(f"{idx:<8} {name:<30}")
        return "\n".join(lines)

    @property
    def active_terms(self) -> list[str]:
        """Get list of active term names."""
        return self._term_names

    def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, Any]:
        """Reset the manager state for specified environments.

        Args:
            env_ids: Tensor of environment indices to reset. None = all envs.

        Returns:
            Empty dictionary (curriculum doesn't produce reset stats).
        """
        # Curriculum doesn't need per-env reset tracking
        return {}

    def compute(self, env_ids: torch.Tensor | None = None) -> None:
        """Execute all curriculum terms.

        This is typically called at the start of each episode (during reset).
        Terms can modify environment parameters based on training progress.

        Args:
            env_ids: Tensor of environment indices. None = all envs.
        """
        for term_cfg in self._term_cfgs:
            term_cfg.func(self._env, env_ids, **term_cfg.params)

    def get_active_iterable_terms(self, env_idx: int = 0) -> list[tuple[str, list[float]]]:
        """Get term values for iteration/logging.

        Returns:
            List of (term_name, [0.0]) tuples (curriculum terms don't have values).
        """
        return [(name, [0.0]) for name in self._term_names]

    def _prepare_terms(self) -> None:
        """Parse configuration and prepare terms."""
        for term_name, term_cfg in self.cfg.items():
            if term_cfg is None:
                continue

            self._resolve_common_term_cfg(term_name, term_cfg)
            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)


class NullCurriculumManager:
    """Null curriculum manager that does nothing.

    Used when no curriculum is configured, following the mjlab pattern.
    """

    @property
    def active_terms(self) -> list[str]:
        """Get list of active term names (empty)."""
        return []

    def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, Any]:
        """No-op reset."""
        return {}

    def compute(self, env_ids: torch.Tensor | None = None) -> None:
        """No-op compute."""
        pass
