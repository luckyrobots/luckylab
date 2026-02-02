"""Manager term configurations for rewards, terminations, etc.

This module defines configuration dataclasses for all MDP components,
following the mjlab pattern where each manager uses a dict of term configs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .command_manager import CommandTerm


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


@dataclass(kw_only=True)
class CommandTermCfg:
    """Configuration for a command generator term.

    Uses class_type pattern like mjlab - you specify a CommandTerm subclass
    that will be instantiated by the manager.
    """

    class_type: type[CommandTerm]
    """CommandTerm subclass to instantiate."""
    resampling_time_range: tuple[float, float] = (5.0, 10.0)
    """Time range for resampling commands (min, max) in seconds."""
    debug_vis: bool = False
    """Whether to enable debug visualization for this command."""
    params: dict[str, Any] = field(default_factory=dict)
    """Additional parameters passed to the command term."""


@dataclass(kw_only=True)
class ActionTermCfg:
    """Configuration for an action term.

    Uses class_type pattern like mjlab - you specify an ActionTerm subclass
    that will be instantiated by the manager.
    """

    class_type: type
    """ActionTerm subclass to instantiate."""
    asset_name: str = "robot"
    """Name of the asset this action applies to."""
    params: dict[str, Any] = field(default_factory=dict)
    """Additional parameters passed to the action term."""


@dataclass
class ObservationTermCfg(ManagerTermBaseCfg):
    """Configuration for an observation term (matches mjlab).

    Defines how to compute an observation and what noise to apply.
    """

    func: Callable[..., Any]
    """Observation computation function to call."""
    noise: Any | None = None
    """Noise configuration to apply to the observation (e.g., UniformNoiseCfg)."""
    clip: tuple[float, float] | None = None
    """Clipping range (min, max) for the observation."""
    scale: float = 1.0
    """Scale factor for the observation."""


@dataclass
class ObservationGroupCfg:
    """Configuration for a group of observation terms (matches mjlab).

    Groups observations into policy/critic sets with optional noise corruption,
    delay simulation, and history stacking.
    """

    terms: dict[str, ObservationTermCfg] = field(default_factory=dict)
    """Observation terms in this group."""
    concatenate_terms: bool = True
    """Whether to concatenate all terms into a single tensor."""
    enable_corruption: bool = True
    """Whether to apply noise corruption to observations."""
    delay_range: tuple[int, int] = (0, 0)
    """Delay range in steps (min, max). 0 = no delay. Simulates sensor latency."""
    history_length: int = 1
    """Number of observations to stack. 1 = no history (current only)."""
    flatten_history: bool = True
    """Whether to flatten history [T, obs_dim] to [T * obs_dim]."""


@dataclass
class JointActuatorCfg:
    """Configuration for a single joint actuator.

    Defines how raw actions [-1, 1] are converted to joint positions:
        joint_position = raw_action * action_scale + default_pos
    """

    name: str
    """Name of the joint (e.g., 'FR_hip')."""
    default_pos: float
    """Default joint position (standing pose)."""
    action_scale: float
    """Scale factor for action to joint position conversion."""
    pos_limit_lower: float | None = None
    """Lower position limit for safety clipping (optional)."""
    pos_limit_upper: float | None = None
    """Upper position limit for safety clipping (optional)."""
