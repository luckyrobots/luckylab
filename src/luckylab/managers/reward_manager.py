"""Reward manager for computing reward signals.

This module provides the RewardManager class which aggregates multiple reward
terms and computes their weighted sum, following the mjlab pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .manager_base import ManagerBase
from .manager_term_config import RewardTermCfg

if TYPE_CHECKING:
    from ..envs.manager_based_rl_env import ManagerBasedRlEnv


class RewardManager(ManagerBase):
    """Manager for computing reward signals.

    Aggregates multiple reward terms, computes their weighted sum,
    and tracks episode sums for logging.

    Attributes:
        cfg: Dictionary mapping term names to RewardTermCfg.
        active_terms: List of active term names.
    """

    _env: ManagerBasedRlEnv

    def __init__(self, cfg: dict[str, RewardTermCfg], env: ManagerBasedRlEnv) -> None:
        """Initialize the reward manager.

        Args:
            cfg: Dictionary mapping term names to RewardTermCfg instances.
            env: The environment instance.
        """
        self._term_names: list[str] = []
        self._term_cfgs: list[RewardTermCfg] = []
        self._class_term_cfgs: list[RewardTermCfg] = []

        self.cfg = cfg
        super().__init__(env=env)

        # Initialize episode sums for each term.
        self._episode_sums: dict[str, float] = {name: 0.0 for name in self._term_names}

        # Current step reward per term (for logging).
        self._step_rewards: dict[str, float] = {name: 0.0 for name in self._term_names}

    def __str__(self) -> str:
        """Get string representation of the manager."""
        lines = [f"<RewardManager> contains {len(self._term_names)} active terms."]
        lines.append("Active Reward Terms:")
        lines.append(f"{'Index':<8} {'Name':<30} {'Weight':>10}")
        lines.append("-" * 50)
        for idx, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            lines.append(f"{idx:<8} {name:<30} {term_cfg.weight:>10.4f}")
        return "\n".join(lines)

    @property
    def active_terms(self) -> list[str]:
        """Get list of active term names."""
        return self._term_names

    @property
    def episode_sums(self) -> dict[str, float]:
        """Get episode sum for each term."""
        return self._episode_sums.copy()

    def reset(self) -> dict[str, float]:
        """Reset the manager and return episode statistics.

        Returns:
            Dictionary with episode reward sums for each term.
        """
        extras: dict[str, float] = {}

        # Store episode sums before resetting.
        for name in self._term_names:
            extras[f"Episode_Reward/{name}"] = self._episode_sums[name]
            self._episode_sums[name] = 0.0
            self._step_rewards[name] = 0.0

        # Reset class-based terms.
        for term_cfg in self._class_term_cfgs:
            term_cfg.func.reset()

        return extras

    def compute(self, context: dict[str, Any]) -> float:
        """Compute total reward for the current step.

        Args:
            context: MDP context dictionary with observations, actions, etc.

        Returns:
            Total weighted reward.
        """
        total_reward = 0.0

        for name, term_cfg in zip(self._term_names, self._term_cfgs):
            if term_cfg.weight == 0.0:
                self._step_rewards[name] = 0.0
                continue

            # Call the reward function.
            value = self._call_term(term_cfg, context)

            # Apply weight.
            weighted_value = value * term_cfg.weight

            total_reward += weighted_value
            self._episode_sums[name] += weighted_value
            self._step_rewards[name] = value

        return total_reward

    def _call_term(self, term_cfg: RewardTermCfg, context: dict[str, Any]) -> float:
        """Call a reward term function.

        Args:
            term_cfg: The term configuration.
            context: MDP context dictionary.

        Returns:
            Raw reward value (before weighting).
        """
        func = term_cfg.func
        kwargs = dict(term_cfg.params)
        func_name = getattr(func, "__name__", str(func))

        obs_parsed = context.get("obs_parsed")

        # Handle special cases for different function signatures.
        if func_name == "action_rate_l2":
            if context.get("current_action") is None:
                return 0.0
            return func(context["current_action"], context["last_action"], **kwargs)
        elif func_name == "joint_pos_limits":
            kwargs.setdefault("action_low", context.get("action_low"))
            kwargs.setdefault("action_high", context.get("action_high"))
            return func(obs_parsed, **kwargs)
        else:
            if obs_parsed is None:
                return 0.0
            return func(obs_parsed, **kwargs)

    def get_active_iterable_terms(self) -> list[tuple[str, list[float]]]:
        """Get term values for iteration/logging.

        Returns:
            List of (term_name, [step_reward]) tuples.
        """
        return [(name, [self._step_rewards[name]]) for name in self._term_names]

    def get_term_cfg(self, term_name: str) -> RewardTermCfg:
        """Get configuration for a specific term.

        Args:
            term_name: Name of the term.

        Returns:
            The term configuration.

        Raises:
            ValueError: If term_name is not found.
        """
        if term_name not in self._term_names:
            raise ValueError(f"Term '{term_name}' not found in active terms.")
        return self._term_cfgs[self._term_names.index(term_name)]

    def _prepare_terms(self) -> None:
        """Parse configuration and prepare terms."""
        for term_name, term_cfg in self.cfg.items():
            if term_cfg is None:
                continue

            self._resolve_common_term_cfg(term_name, term_cfg)
            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)

            # Track class-based terms for reset.
            if hasattr(term_cfg.func, "reset") and callable(term_cfg.func.reset):
                self._class_term_cfgs.append(term_cfg)
