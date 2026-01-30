"""Termination manager for computing done signals.

This module provides the TerminationManager class which aggregates multiple
termination terms and distinguishes between terminated and truncated states.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .manager_base import ManagerBase
from .manager_term_config import TerminationTermCfg

if TYPE_CHECKING:
    from ..envs.manager_based_rl_env import ManagerBasedRlEnv


class TerminationManager(ManagerBase):
    """Manager for computing termination signals.

    Aggregates multiple termination terms and distinguishes between:
    - terminated: Episode ended due to failure condition (e.g., fell over).
    - truncated: Episode ended due to time limit (not a failure).

    This distinction is important for proper value bootstrapping in RL.

    Attributes:
        cfg: Dictionary mapping term names to TerminationTermCfg.
        active_terms: List of active term names.
        terminated: Whether episode terminated due to failure.
        truncated: Whether episode was truncated (time out).
    """

    _env: ManagerBasedRlEnv

    def __init__(self, cfg: dict[str, TerminationTermCfg], env: ManagerBasedRlEnv) -> None:
        """Initialize the termination manager.

        Args:
            cfg: Dictionary mapping term names to TerminationTermCfg instances.
            env: The environment instance.
        """
        self._term_names: list[str] = []
        self._term_cfgs: list[TerminationTermCfg] = []
        self._class_term_cfgs: list[TerminationTermCfg] = []

        self.cfg = cfg
        super().__init__(env=env)

        # Track which terms triggered.
        self._term_dones: dict[str, bool] = {name: False for name in self._term_names}

        # Separate buffers for terminated vs truncated.
        self._truncated: bool = False
        self._terminated: bool = False

    def __str__(self) -> str:
        """Get string representation of the manager."""
        lines = [f"<TerminationManager> contains {len(self._term_names)} active terms."]
        lines.append("Active Termination Terms:")
        lines.append(f"{'Index':<8} {'Name':<30} {'Time Out':>10}")
        lines.append("-" * 50)
        for idx, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            lines.append(f"{idx:<8} {name:<30} {str(term_cfg.time_out):>10}")
        return "\n".join(lines)

    @property
    def active_terms(self) -> list[str]:
        """Get list of active term names."""
        return self._term_names

    @property
    def dones(self) -> bool:
        """Check if episode is done (terminated OR truncated)."""
        return self._truncated or self._terminated

    @property
    def truncated(self) -> bool:
        """Check if episode was truncated (time out)."""
        return self._truncated

    @property
    def terminated(self) -> bool:
        """Check if episode was terminated (failure)."""
        return self._terminated

    @property
    def termination_reasons(self) -> list[str]:
        """Get list of terms that triggered termination."""
        return [name for name, done in self._term_dones.items() if done]

    def reset(self) -> dict[str, int]:
        """Reset the manager and return episode statistics.

        Returns:
            Dictionary with termination counts for each term.
        """
        extras: dict[str, int] = {}

        # Store which terms triggered before resetting.
        for name in self._term_names:
            extras[f"Episode_Termination/{name}"] = int(self._term_dones[name])
            self._term_dones[name] = False

        # Reset state.
        self._truncated = False
        self._terminated = False

        # Reset class-based terms.
        for term_cfg in self._class_term_cfgs:
            term_cfg.func.reset()

        return extras

    def compute(self, context: dict[str, Any]) -> bool:
        """Compute termination signals for the current step.

        Args:
            context: MDP context dictionary with observations, actions, etc.

        Returns:
            True if episode should end (terminated OR truncated).
        """
        self._truncated = False
        self._terminated = False

        for name, term_cfg in zip(self._term_names, self._term_cfgs):
            # Call the termination function.
            value = self._call_term(term_cfg, context)

            if value:
                self._term_dones[name] = True

                if term_cfg.time_out:
                    self._truncated = True
                else:
                    self._terminated = True

        return self._truncated or self._terminated

    def _call_term(self, term_cfg: TerminationTermCfg, context: dict[str, Any]) -> bool:
        """Call a termination term function.

        Args:
            term_cfg: The term configuration.
            context: MDP context dictionary.

        Returns:
            True if termination condition is met.
        """
        func = term_cfg.func
        kwargs = dict(term_cfg.params)
        func_name = getattr(func, "__name__", str(func))

        obs_parsed = context.get("obs_parsed")

        # Handle special cases for different function signatures.
        if func_name in ("time_out", "max_steps_termination"):
            return func(context["step_count"], context["max_steps"], **kwargs)
        elif func_name == "nan_detection":
            return func(obs_parsed)
        else:
            if obs_parsed is None:
                return False
            return func(obs_parsed, **kwargs)

    def get_term(self, name: str) -> bool:
        """Get the done state for a specific term.

        Args:
            name: Term name.

        Returns:
            True if this term triggered.
        """
        return self._term_dones.get(name, False)

    def get_active_iterable_terms(self) -> list[tuple[str, list[float]]]:
        """Get term values for iteration/logging.

        Returns:
            List of (term_name, [done_value]) tuples.
        """
        return [(name, [float(self._term_dones[name])]) for name in self._term_names]

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
