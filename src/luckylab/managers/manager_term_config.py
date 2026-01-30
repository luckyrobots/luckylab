"""Manager term configurations for rewards, terminations, etc.

This module defines configuration dataclasses for all MDP components,
following the mjlab pattern where each manager uses a dict of term configs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ManagerTermBaseCfg:
    """Base configuration for a manager term."""

    func: Callable
    """Function to call for this term."""
    params: dict[str, Any] = field(default_factory=dict)
    """Parameters to pass to the function."""


@dataclass(kw_only=True)
class RewardTermCfg(ManagerTermBaseCfg):
    """Configuration for a reward term."""

    func: Callable[..., float]
    """Reward function to call."""
    weight: float
    """Weight to apply to this reward term."""


@dataclass
class TerminationTermCfg(ManagerTermBaseCfg):
    """Configuration for a termination term."""

    func: Callable[..., bool]
    """Termination function to call."""
    time_out: bool = False
    """Whether the term contributes towards episodic timeouts."""


@dataclass
class CurriculumTermCfg(ManagerTermBaseCfg):
    """Configuration for a curriculum term."""

    func: Callable[..., None]
    """Curriculum function to call."""


@dataclass
class CommandTermCfg(ManagerTermBaseCfg):
    """Configuration for a command term."""

    func: Callable[..., Any]
    """Command generation function to call."""
    resampling_time_range: tuple[float, float] = (0.0, 0.0)
    """Time range for resampling commands (min, max) in seconds."""
    debug_vis: bool = False
    """Whether to enable debug visualization for this command."""


@dataclass
class ActionTermCfg(ManagerTermBaseCfg):
    """Configuration for an action term."""

    func: Callable[..., Any]
    """Action processing function to call."""
    asset_name: str = "robot"
    """Name of the asset this action applies to."""
    scale: float | dict[str, float] = 1.0
    """Scale factor for the action."""


@dataclass
class EventTermCfg(ManagerTermBaseCfg):
    """Configuration for an event term."""

    func: Callable[..., None]
    """Event function to call."""
    mode: str = "reset"
    """When to trigger: 'startup', 'reset', or 'interval'."""
    interval_range_s: tuple[float, float] | None = None
    """Interval range in seconds for 'interval' mode."""
    domain_randomization: bool = False
    """Whether this event performs domain randomization."""


@dataclass
class ObservationFuncTermCfg(ManagerTermBaseCfg):
    """Configuration for an observation function term (mjlab pattern).

    This is for defining observation computation functions, distinct from
    ObservationTermCfg in observation_manager.py which handles DR/processing.
    """

    func: Callable[..., Any]
    """Observation computation function to call."""
    noise: Any | None = None
    """Noise configuration to apply to the observation."""
    clip: tuple[float, float] | None = None
    """Clipping range (min, max) for the observation."""
    scale: float = 1.0
    """Scale factor for the observation."""


@dataclass
class ObservationGroupCfg:
    """Configuration for a group of observation terms."""

    terms: dict[str, ObservationFuncTermCfg] = field(default_factory=dict)
    """Observation terms in this group."""
    concatenate_terms: bool = True
    """Whether to concatenate all terms into a single tensor."""
    enable_corruption: bool = True
    """Whether to apply noise corruption to observations."""
