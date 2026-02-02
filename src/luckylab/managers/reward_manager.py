"""Reward manager for computing reward signals.

Matches mjlab pattern where reward functions take `env` as first parameter.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import torch

from .manager_base import ManagerBase
from .manager_term_config import RewardTermCfg

if TYPE_CHECKING:
    from ..envs.manager_based_rl_env import ManagerBasedRlEnv


class RewardManager(ManagerBase):
    """Manager for computing reward signals.

    Aggregates multiple reward terms, computes their weighted sum,
    and tracks episode sums for logging.
    """

    _env: ManagerBasedRlEnv

    def __init__(self, cfg: dict[str, RewardTermCfg], env: ManagerBasedRlEnv) -> None:
        self._term_names: list[str] = []
        self._term_cfgs: list[RewardTermCfg] = []
        self._term_instances: dict[str, object] = {}  # For class-based terms
        self.cfg = cfg
        super().__init__(env=env)

        # Initialize episode sums and step rewards
        self._episode_sums: dict[str, torch.Tensor] = {
            name: torch.zeros(self.num_envs, device=self.device) for name in self._term_names
        }
        self._step_rewards: dict[str, torch.Tensor] = {
            name: torch.zeros(self.num_envs, device=self.device) for name in self._term_names
        }

    def __str__(self) -> str:
        lines = [f"<RewardManager> contains {len(self._term_names)} active terms."]
        for idx, (name, cfg) in enumerate(zip(self._term_names, self._term_cfgs, strict=False)):
            lines.append(f"  {idx}: {name} (weight={cfg.weight:.4f})")
        return "\n".join(lines)

    @property
    def active_terms(self) -> list[str]:
        return self._term_names

    def episode_sums(self, env_idx: int = 0) -> dict[str, float]:
        return {name: self._episode_sums[name][env_idx].item() for name in self._term_names}

    def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        extras: dict[str, float] = {}
        for name in self._term_names:
            extras[f"Episode_Reward/{name}"] = self._episode_sums[name][env_ids].mean().item()
            self._episode_sums[name][env_ids] = 0.0
            self._step_rewards[name][env_ids] = 0.0

        return extras

    def compute(self, dt: float | None = None) -> torch.Tensor:
        """Compute total reward for the current step.

        Args:
            dt: Timestep for reward scaling. If None, uses env.cfg.step_dt.
                Rewards are multiplied by dt to make them timestep-invariant.
                This matches mjlab's pattern where reward = value * weight * dt.
        """
        if dt is None:
            dt = self._env.cfg.step_dt

        total_reward = torch.zeros(self.num_envs, device=self.device)

        for name, term_cfg in zip(self._term_names, self._term_cfgs, strict=False):
            if term_cfg.weight == 0.0:
                self._step_rewards[name].zero_()
                continue

            value = self._call_term(name, term_cfg)

            # Ensure tensor
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, device=self.device)
            if value.dim() == 0:
                value = value.expand(self.num_envs)

            # Apply weight and dt scaling (mjlab pattern)
            weighted = value * term_cfg.weight * dt
            total_reward += weighted
            self._episode_sums[name] += weighted
            # Store unscaled value for logging (matches mjlab: step_reward = value / dt, but we store pre-scaled)
            self._step_rewards[name] = value

        return total_reward

    def _call_term(self, name: str, term_cfg: RewardTermCfg) -> torch.Tensor:
        """Call a reward term function."""
        # Check if it's a class-based term (has an instance)
        if name in self._term_instances:
            instance = self._term_instances[name]
            return instance(self._env, **term_cfg.params)

        # Function-based term - call with env as first arg
        return term_cfg.func(self._env, **term_cfg.params)

    def get_active_iterable_terms(self, env_idx: int = 0) -> list[tuple[str, list[float]]]:
        return [(name, [self._step_rewards[name][env_idx].item()]) for name in self._term_names]

    def get_term_cfg(self, term_name: str) -> RewardTermCfg:
        if term_name not in self._term_names:
            raise ValueError(f"Term '{term_name}' not found.")
        return self._term_cfgs[self._term_names.index(term_name)]

    def _prepare_terms(self) -> None:
        for term_name, term_cfg in self.cfg.items():
            if term_cfg is None:
                continue
            self._resolve_common_term_cfg(term_name, term_cfg)

            # Check if func is a class (has __init__ and __call__)
            if inspect.isclass(term_cfg.func):
                # Instantiate class-based term with (cfg, env)
                instance = term_cfg.func(term_cfg, self._env)
                self._term_instances[term_name] = instance

            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)
