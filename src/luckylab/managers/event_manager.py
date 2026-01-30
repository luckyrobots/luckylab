"""Event manager for domain randomization events.

This module provides the EventManager class which handles startup, reset,
and interval-based events for domain randomization following the mjlab pattern.

NOTE: This manager is currently a PLACEHOLDER. LuckyLab connects to an external
simulation engine (LuckyEngine) via gRPC, which owns the physics state. Domain
randomization of physics parameters (mass, friction, etc.) requires engine-side
support through gRPC endpoints that don't exist yet.

When the engine exposes DR configuration endpoints, this manager can be integrated
to call those endpoints on startup/reset/interval events.

For now, observation-side randomization (noise, delays) is handled by
ObservationProcessor, which works entirely client-side.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

from .manager_base import ManagerBase
from .manager_term_config import EventTermCfg

if TYPE_CHECKING:
    from ..envs.manager_based_rl_env import ManagerBasedRlEnv


class EventManager(ManagerBase):
    """Manager for domain randomization events.

    Handles three types of events:
    - startup: Called once when the environment is created.
    - reset: Called on every environment reset.
    - interval: Called periodically based on time intervals.

    Attributes:
        cfg: Dictionary mapping term names to EventTermCfg.
        active_terms: List of active term names.
    """

    _env: ManagerBasedRlEnv

    def __init__(self, cfg: dict[str, EventTermCfg], env: ManagerBasedRlEnv) -> None:
        """Initialize the event manager.

        Args:
            cfg: Dictionary mapping term names to EventTermCfg instances.
            env: The environment instance.
        """
        self._term_names: list[str] = []
        self._term_cfgs: list[EventTermCfg] = []
        self._class_term_cfgs: list[EventTermCfg] = []

        # Separate lists by event mode.
        self._startup_terms: list[tuple[str, EventTermCfg]] = []
        self._reset_terms: list[tuple[str, EventTermCfg]] = []
        self._interval_terms: list[tuple[str, EventTermCfg, float]] = []  # (name, cfg, next_trigger_time)

        self.cfg = cfg
        super().__init__(env=env)

        # Current simulation time for interval events.
        self._current_time: float = 0.0

        # Run startup events.
        self._run_startup_events()

    def __str__(self) -> str:
        """Get string representation of the manager."""
        lines = [f"<EventManager> contains {len(self._term_names)} active terms."]
        lines.append("Active Event Terms:")
        lines.append(f"{'Index':<8} {'Name':<30} {'Mode':<10} {'DR':>5}")
        lines.append("-" * 55)
        for idx, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs, strict=False)):
            dr = "Yes" if term_cfg.domain_randomization else "No"
            lines.append(f"{idx:<8} {name:<30} {term_cfg.mode:<10} {dr:>5}")
        return "\n".join(lines)

    @property
    def active_terms(self) -> list[str]:
        """Get list of active term names."""
        return self._term_names

    def reset(self) -> dict[str, Any]:
        """Reset the manager and run reset events.

        Returns:
            Empty dictionary (events don't produce logging info).
        """
        # Reset time tracking.
        self._current_time = 0.0

        # Reschedule interval events.
        self._interval_terms = [
            (name, cfg, self._sample_interval(cfg)) for name, cfg, _ in self._interval_terms
        ]

        # Run reset events.
        self._run_reset_events()

        # Reset class-based terms.
        for term_cfg in self._class_term_cfgs:
            term_cfg.func.reset()

        return {}

    def update(self, dt: float) -> None:
        """Update event manager and run interval events.

        Args:
            dt: Time step in seconds.
        """
        self._current_time += dt

        # Check and run interval events.
        new_interval_terms = []
        for name, cfg, next_trigger in self._interval_terms:
            if self._current_time >= next_trigger:
                # Run the event.
                self._call_term(cfg)
                # Schedule next trigger.
                next_trigger = self._current_time + self._sample_interval(cfg)

            new_interval_terms.append((name, cfg, next_trigger))

        self._interval_terms = new_interval_terms

    def _run_startup_events(self) -> None:
        """Run all startup events."""
        for _name, cfg in self._startup_terms:
            self._call_term(cfg)

    def _run_reset_events(self) -> None:
        """Run all reset events."""
        for _name, cfg in self._reset_terms:
            self._call_term(cfg)

    def _call_term(self, term_cfg: EventTermCfg) -> None:
        """Call an event term function.

        Args:
            term_cfg: The term configuration.
        """
        func = term_cfg.func
        kwargs = dict(term_cfg.params)

        # Event functions receive the environment as first argument.
        func(self._env, **kwargs)

    def _sample_interval(self, cfg: EventTermCfg) -> float:
        """Sample next interval time from configured range.

        Args:
            cfg: Event term configuration.

        Returns:
            Time until next trigger in seconds.
        """
        if cfg.interval_range_s is None:
            return float("inf")

        min_interval, max_interval = cfg.interval_range_s
        return random.uniform(min_interval, max_interval)

    def get_active_iterable_terms(self) -> list[tuple[str, list[float]]]:
        """Get term values for iteration/logging.

        Returns:
            List of (term_name, [0.0]) tuples (events don't have values).
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

            # Categorize by mode.
            if term_cfg.mode == "startup":
                self._startup_terms.append((term_name, term_cfg))
            elif term_cfg.mode == "reset":
                self._reset_terms.append((term_name, term_cfg))
            elif term_cfg.mode == "interval":
                initial_delay = self._sample_interval(term_cfg)
                self._interval_terms.append((term_name, term_cfg, initial_delay))
            else:
                raise ValueError(f"Unknown event mode: {term_cfg.mode}")

            # Track class-based terms for reset.
            if hasattr(term_cfg.func, "reset") and callable(term_cfg.func.reset):
                self._class_term_cfgs.append(term_cfg)
