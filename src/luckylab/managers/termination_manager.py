"""Termination manager for computing done signals.

Matches mjlab pattern where termination functions take `env` as first parameter.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

import torch
from prettytable import PrettyTable

from .manager_base import ManagerBase
from .manager_term_config import TerminationTermCfg

if TYPE_CHECKING:
    from ..envs.manager_based_rl_env import ManagerBasedRlEnv


class TerminationManager(ManagerBase):
    """Manager for computing termination signals.

    Aggregates multiple termination terms and distinguishes between:
    - terminated: Episode ended due to failure condition (e.g., fell over).
    - truncated: Episode ended due to time limit (not a failure).
    """

    _env: ManagerBasedRlEnv

    def __init__(self, cfg: dict[str, TerminationTermCfg], env: ManagerBasedRlEnv) -> None:
        self._term_names: list[str] = []
        self._term_cfgs: list[TerminationTermCfg] = []
        self._term_instances: dict[str, object] = {}

        self.cfg = cfg
        super().__init__(env=env)

        # Track which terms triggered for each env
        self._term_dones: dict[str, torch.Tensor] = {
            name: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            for name in self._term_names
        }

        self._truncated_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._terminated_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def __str__(self) -> str:
        msg = f"<TerminationManager> contains {len(self._term_names)} active terms.\n"
        table = PrettyTable()
        table.title = "Active Termination Terms"
        table.field_names = ["Index", "Name", "Time Out"]
        table.align["Name"] = "l"
        for index, (name, term_cfg) in enumerate(
        zip(self._term_names, self._term_cfgs, strict=False)
        ):
            table.add_row([index, name, term_cfg.time_out])
        msg += table.get_string()
        msg += "\n"
        return msg
    
    # Properties

    @property
    def active_terms(self) -> list[str]:
        return self._term_names

    @property
    def dones(self) -> torch.Tensor:
        """Check if episodes are done (terminated OR truncated)."""
        return self._truncated_buf | self._terminated_buf

    @property
    def truncated(self) -> torch.Tensor:
        """Check if episodes were truncated (time out)."""
        return self._truncated_buf

    @property
    def terminated(self) -> torch.Tensor:
        """Check if episodes were terminated (failure)."""
        return self._terminated_buf

    def termination_reasons(self, env_idx: int = 0) -> list[str]:
        return [name for name, done in self._term_dones.items() if done[env_idx].item()]

    def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, Any]:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        extras: dict[str, Any] = {}
        for name in self._term_names:
            extras[f"Episode_Termination/{name}"] = self._term_dones[name][env_ids].sum().item()
            self._term_dones[name][env_ids] = False

        self._truncated_buf[env_ids] = False
        self._terminated_buf[env_ids] = False

        return extras

    def compute(self) -> torch.Tensor:
        """Compute termination signals for the current step."""
        self._truncated_buf.zero_()
        self._terminated_buf.zero_()

        for name, term_cfg in zip(self._term_names, self._term_cfgs, strict=False):
            value = self._call_term(name, term_cfg)

            # Ensure tensor on correct device
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, dtype=torch.bool, device=self.device)
            elif value.device != self.device:
                value = value.to(self.device)
            if value.dim() == 0:
                value = value.expand(self.num_envs)

            self._term_dones[name] = value

            if term_cfg.time_out:
                self._truncated_buf |= value
            else:
                self._terminated_buf |= value

        return self._truncated_buf | self._terminated_buf

    def _call_term(self, name: str, term_cfg: TerminationTermCfg) -> torch.Tensor:
        """Call a termination term function."""
        # Check if it's a class-based term (has an instance)
        if name in self._term_instances:
            instance = self._term_instances[name]
            return instance(self._env, **term_cfg.params)

        # Function-based term - call with env as first arg
        return term_cfg.func(self._env, **term_cfg.params)

    def get_term(self, name: str) -> torch.Tensor:
        return self._term_dones.get(name, torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))

    def get_active_iterable_terms(self, env_idx: int = 0) -> list[tuple[str, list[float]]]:
        return [(name, [float(self._term_dones[name][env_idx].item())]) for name in self._term_names]

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
