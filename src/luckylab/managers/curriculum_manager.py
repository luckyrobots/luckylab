"""Curriculum manager for progressive training difficulty."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..configs.domain_randomization import PhysicsDRCfg

if TYPE_CHECKING:
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


@dataclass
class CurriculumLevelCfg:
    """Configuration for a single curriculum dimension."""

    name: str
    levels: tuple[float, ...]  # Values for each level
    advance_threshold: float = 0.8  # Metric threshold to advance
    metric_key: str = "survived"  # Which metric to track
    metric_type: str = "rate"  # "rate" (fraction) or "mean" (average)


@dataclass
class CurriculumCfg:
    """Configuration for the curriculum manager."""

    # Base DR config (level 0 values)
    base_dr_cfg: PhysicsDRCfg = field(default_factory=PhysicsDRCfg)

    # Curriculum dimensions
    terrain_levels: tuple[float, ...] = (0.0,)  # terrain_difficulty values
    terrain_advance_threshold: float = 0.8

    push_levels: tuple[tuple[float, float], ...] = ((0.0, 0.0),)  # push_velocity_range values
    push_advance_threshold: float = 0.8

    velocity_range_levels: tuple[tuple[float, float], ...] = ((0.5, 0.5),)  # command velocity max
    velocity_advance_threshold: float = 0.8

    # Window size for computing metrics
    history_size: int = 100

    # Minimum episodes before allowing advancement
    min_episodes_before_advance: int = 50


class CurriculumManager:
    """
    Manages curriculum learning with multiple difficulty dimensions.

    Each dimension (terrain, push forces, velocity range) can advance
    independently based on agent performance over a rolling window.
    """

    def __init__(self, cfg: CurriculumCfg):
        self.cfg = cfg

        # Current level for each dimension
        self.terrain_level: int = 0
        self.push_level: int = 0
        self.velocity_level: int = 0

        # Episode history for computing metrics
        self.episode_history: deque[dict] = deque(maxlen=cfg.history_size)

        # Track if config changed (for logging)
        self._config_changed: bool = True
        self._last_dr_cfg: PhysicsDRCfg | None = None

    def update(self, episode_metrics: dict) -> None:
        """
        Update curriculum based on completed episode metrics.

        Args:
            episode_metrics: Dict from EpisodeMetrics.finalize()
        """
        self.episode_history.append(episode_metrics)

        # Need enough data before making decisions
        if len(self.episode_history) < self.cfg.min_episodes_before_advance:
            return

        # Compute aggregate metrics
        survival_rate = self._compute_rate("survived")
        # These are computed for potential future use in curriculum criteria
        _ = self._compute_mean("total_reward")  # avg_reward
        _ = self._compute_mean("episode_length")  # avg_length

        # Check each curriculum dimension
        config_changed = False

        # Terrain curriculum
        if (
            survival_rate >= self.cfg.terrain_advance_threshold
            and self.terrain_level < len(self.cfg.terrain_levels) - 1
        ):
            self.terrain_level += 1
            config_changed = True

        # Push curriculum
        if (
            survival_rate >= self.cfg.push_advance_threshold
            and self.push_level < len(self.cfg.push_levels) - 1
        ):
            self.push_level += 1
            config_changed = True

        # Velocity range curriculum
        if (
            survival_rate >= self.cfg.velocity_advance_threshold
            and self.velocity_level < len(self.cfg.velocity_range_levels) - 1
        ):
            self.velocity_level += 1
            config_changed = True

        if config_changed:
            # Reset history after advancement to get fresh measurements
            self.episode_history.clear()
            self._config_changed = True

    def get_dr_cfg(self) -> PhysicsDRCfg:
        """
        Get the current domain randomization config based on curriculum state.

        Returns:
            PhysicsDRCfg with values set according to current curriculum levels
        """
        cfg = PhysicsDRCfg(
            # Inherit from base config
            pose_position_noise=self.cfg.base_dr_cfg.pose_position_noise,
            pose_orientation_noise=self.cfg.base_dr_cfg.pose_orientation_noise,
            joint_position_noise=self.cfg.base_dr_cfg.joint_position_noise,
            joint_velocity_noise=self.cfg.base_dr_cfg.joint_velocity_noise,
            friction_range=self.cfg.base_dr_cfg.friction_range,
            restitution_range=self.cfg.base_dr_cfg.restitution_range,
            mass_scale_range=self.cfg.base_dr_cfg.mass_scale_range,
            com_offset_range=self.cfg.base_dr_cfg.com_offset_range,
            motor_strength_range=self.cfg.base_dr_cfg.motor_strength_range,
            motor_offset_range=self.cfg.base_dr_cfg.motor_offset_range,
            push_interval_range=self.cfg.base_dr_cfg.push_interval_range,
            # Curriculum-controlled values
            terrain_type=self.cfg.base_dr_cfg.terrain_type,
            terrain_difficulty=self.cfg.terrain_levels[self.terrain_level],
            push_velocity_range=self.cfg.push_levels[self.push_level],
        )

        self._last_dr_cfg = cfg
        return cfg

    def get_velocity_command_range(self) -> tuple[float, float]:
        """Get current velocity command range based on curriculum."""
        return self.cfg.velocity_range_levels[self.velocity_level]

    @property
    def config_changed(self) -> bool:
        """Check if config changed since last check (resets flag)."""
        changed = self._config_changed
        self._config_changed = False
        return changed

    @property
    def current_levels(self) -> dict[str, int]:
        """Get current level for each curriculum dimension."""
        return {
            "terrain": self.terrain_level,
            "push": self.push_level,
            "velocity": self.velocity_level,
        }

    @property
    def max_levels(self) -> dict[str, int]:
        """Get maximum level for each curriculum dimension."""
        return {
            "terrain": len(self.cfg.terrain_levels) - 1,
            "push": len(self.cfg.push_levels) - 1,
            "velocity": len(self.cfg.velocity_range_levels) - 1,
        }

    def _compute_rate(self, key: str) -> float:
        """Compute rate (fraction true) for a boolean metric."""
        if not self.episode_history:
            return 0.0
        return sum(1 for ep in self.episode_history if ep.get(key, False)) / len(self.episode_history)

    def _compute_mean(self, key: str) -> float:
        """Compute mean for a numeric metric."""
        if not self.episode_history:
            return 0.0
        values = [ep.get(key, 0.0) for ep in self.episode_history]
        return sum(values) / len(values)

    def get_metrics(self) -> dict:
        """Get current curriculum metrics for logging."""
        return {
            "curriculum/terrain_level": self.terrain_level,
            "curriculum/push_level": self.push_level,
            "curriculum/velocity_level": self.velocity_level,
            "curriculum/survival_rate": self._compute_rate("survived"),
            "curriculum/avg_reward": self._compute_mean("total_reward"),
            "curriculum/avg_episode_length": self._compute_mean("episode_length"),
            "curriculum/episodes_in_window": len(self.episode_history),
        }


class CurriculumTermManager:
    """Manages curriculum terms using the dict-based CurriculumTermCfg pattern.

    This follows the mjlab pattern where curriculum is a dict of term configs,
    matching how rewards and terminations are configured.
    """

    def __init__(self, cfg: dict[str, CurriculumTermCfg]):
        """Initialize the curriculum term manager.

        Args:
            cfg: Dict mapping term names to CurriculumTermCfg instances.
        """
        self.cfg = cfg

    def update(self, env: Any) -> None:
        """Call all curriculum functions.

        Args:
            env: The environment instance to update.
        """
        for _name, term_cfg in self.cfg.items():
            term_cfg.func(env, **term_cfg.params)
