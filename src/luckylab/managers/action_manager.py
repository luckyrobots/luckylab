"""Action manager for processing actions sent to the environment.

Follows mjlab's pattern with ActionTerm base class and term-based management.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Sequence

import torch
from prettytable import PrettyTable

from luckylab.managers.manager_base import ManagerBase, ManagerTermBase

if TYPE_CHECKING:
    from luckylab.envs import ManagerBasedRlEnv
    from luckylab.managers.manager_term_config import ActionTermCfg


class ActionTerm(ManagerTermBase):
    """Base class for action terms.

    Action terms process raw policy outputs and convert them to
    actuator commands. Each term handles a specific group of actuators.
    """

    def __init__(self, cfg: "ActionTermCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg=cfg, env=env)
        self._asset = self._env.scene[self.cfg.asset_name]

    @property
    @abc.abstractmethod
    def action_dim(self) -> int:
        """Dimension of the action space for this term."""
        raise NotImplementedError

    @abc.abstractmethod
    def process_actions(self, actions: torch.Tensor) -> None:
        """Process raw actions from the policy.

        Args:
            actions: Raw action tensor, shape (num_envs, action_dim).
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def raw_action(self) -> torch.Tensor:
        """Raw actions before processing."""
        raise NotImplementedError

    @property
    def processed_actions(self) -> torch.Tensor:
        """Processed actions after scale/offset. Override in subclass."""
        return self.raw_action


class ActionManager(ManagerBase):
    """Manages action processing through action terms.

    Follows mjlab's pattern: splits actions across terms, processes them,
    and collects the results for sending to the simulator.
    """

    def __init__(self, cfg: dict[str, "ActionTermCfg"], env: "ManagerBasedRlEnv"):
        self.cfg = cfg
        super().__init__(env=env)

        # Create buffers to store actions
        self._action = torch.zeros(
            (self.num_envs, self.total_action_dim), device=self.device
        )
        self._prev_action = torch.zeros_like(self._action)

    def __str__(self) -> str:
        msg = f"<ActionManager> contains {len(self._term_names)} active terms.\n"
        table = PrettyTable()
        table.title = f"Active Action Terms (shape: {self.total_action_dim})"
        table.field_names = ["Index", "Name", "Dimension"]
        table.align["Name"] = "l"
        table.align["Dimension"] = "r"
        for index, (name, term) in enumerate(self._terms.items()):
            table.add_row([index, name, term.action_dim])
        msg += table.get_string()
        msg += "\n"
        return msg

    # Properties

    @property
    def total_action_dim(self) -> int:
        """Total dimension of all action terms combined."""
        return sum(self.action_term_dim)

    @property
    def action_term_dim(self) -> list[int]:
        """List of action dimensions for each term."""
        return [term.action_dim for term in self._terms.values()]

    @property
    def action(self) -> torch.Tensor:
        """Current raw actions, shape (num_envs, total_action_dim)."""
        return self._action

    @property
    def prev_action(self) -> torch.Tensor:
        """Previous raw actions, shape (num_envs, total_action_dim)."""
        return self._prev_action

    @property
    def last_action(self) -> torch.Tensor:
        """Alias for prev_action (for observation compatibility)."""
        return self._prev_action

    @property
    def active_terms(self) -> list[str]:
        """List of active term names."""
        return self._term_names

    # Methods

    def get_term(self, name: str) -> ActionTerm:
        """Get an action term by name.

        Args:
            name: Term name.

        Returns:
            The action term.
        """
        return self._terms[name]

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> dict[str, float]:
        """Reset action state for specified environments.

        Args:
            env_ids: Environment indices to reset, or None for all.

        Returns:
            Empty dict (no metrics to report).
        """
        if env_ids is None:
            env_ids = slice(None)
        # Reset action history
        self._prev_action[env_ids] = 0.0
        self._action[env_ids] = 0.0
        # Reset action terms
        for term in self._terms.values():
            term.reset(env_ids=env_ids)
        return {}

    def process_action(self, action: torch.Tensor) -> torch.Tensor:
        """Process raw policy action through all terms.

        Args:
            action: Raw action from policy, shape (num_envs, total_action_dim).

        Returns:
            Processed actions ready to send to simulator.
        """
        if action.dim() == 1:
            action = action.unsqueeze(0)

        if self.total_action_dim != action.shape[1]:
            raise ValueError(
                f"Invalid action shape, expected: {self.total_action_dim}, "
                f"received: {action.shape[1]}."
            )

        # Update action history
        self._prev_action[:] = self._action
        self._action[:] = action.to(self.device)

        # Split and process through each term
        idx = 0
        processed_parts = []
        for term in self._terms.values():
            term_actions = action[:, idx : idx + term.action_dim]
            term.process_actions(term_actions)
            processed_parts.append(term.processed_actions)
            idx += term.action_dim

        # Concatenate processed actions
        if processed_parts:
            return torch.cat(processed_parts, dim=1)
        return action

    def get_processed_actions(self) -> torch.Tensor:
        """Get concatenated processed actions from all terms.

        Returns:
            Processed actions, shape (num_envs, total_action_dim).
        """
        parts = [term.processed_actions for term in self._terms.values()]
        if parts:
            return torch.cat(parts, dim=1)
        return self._action

    def get_active_iterable_terms(
        self, env_idx: int
    ) -> Sequence[tuple[str, Sequence[float]]]:
        """Get action values for each term for a specific environment.

        Args:
            env_idx: Environment index.

        Returns:
            List of (term_name, action_values) tuples.
        """
        terms = []
        idx = 0
        for name, term in self._terms.items():
            term_actions = self._action[env_idx, idx : idx + term.action_dim].cpu()
            terms.append((name, term_actions.tolist()))
            idx += term.action_dim
        return terms

    def _prepare_terms(self) -> None:
        """Prepare action terms from configuration."""
        self._term_names: list[str] = []
        self._terms: dict[str, ActionTerm] = {}

        for term_name, term_cfg in self.cfg.items():
            if term_cfg is None:
                continue
            term = term_cfg.class_type(term_cfg, self._env)
            self._term_names.append(term_name)
            self._terms[term_name] = term


class NullActionManager:
    """Null action manager that passes actions through unchanged.

    Used when no action configuration is provided.
    """

    def __init__(self, num_actions: int, num_envs: int, device: torch.device) -> None:
        self._num_actions = num_actions
        self._num_envs = num_envs
        self._device = device
        self._action = torch.zeros(num_envs, num_actions, dtype=torch.float32, device=device)
        self._prev_action = torch.zeros_like(self._action)

    @property
    def total_action_dim(self) -> int:
        return self._num_actions

    @property
    def action(self) -> torch.Tensor:
        return self._action

    @property
    def prev_action(self) -> torch.Tensor:
        return self._prev_action

    @property
    def last_action(self) -> torch.Tensor:
        return self._prev_action

    @property
    def active_terms(self) -> list[str]:
        return []

    def process_action(self, raw_action: torch.Tensor) -> torch.Tensor:
        """Pass action through unchanged."""
        if raw_action.dim() == 1:
            raw_action = raw_action.unsqueeze(0)
        self._prev_action.copy_(self._action)
        self._action[:] = raw_action
        return raw_action

    def get_processed_actions(self) -> torch.Tensor:
        return self._action

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = slice(None)
        self._action[env_ids] = 0.0
        self._prev_action[env_ids] = 0.0
        return {}
