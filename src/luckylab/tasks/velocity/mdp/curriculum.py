"""Curriculum functions for velocity task.

Matches mjlab's curriculum pattern for terrain and velocity progression.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, TypedDict

import torch

if TYPE_CHECKING:
    from ....envs.manager_based_rl_env import ManagerBasedRlEnv
    from .velocity_command import UniformVelocityCommandCfg

logger = logging.getLogger(__name__)


class VelocityStage(TypedDict, total=False):
    """Velocity curriculum stage definition."""

    step: int
    lin_vel_x: tuple[float, float] | None
    lin_vel_y: tuple[float, float] | None
    ang_vel_z: tuple[float, float] | None


class RewardWeightStage(TypedDict):
    """Reward weight curriculum stage definition."""

    step: int
    weight: float


def terrain_levels_vel(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
) -> torch.Tensor:
    """Update terrain difficulty based on robot walking distance.

    Robots that walk far enough progress to harder terrains.
    Robots that don't walk far enough regress to easier terrains.

    Note: Full terrain curriculum requires LuckyEngine terrain support.
    Returns placeholder metric for now.

    Args:
        env: The environment instance.
        env_ids: Environment indices being reset.
        command_name: Name of the velocity command term.

    Returns:
        Mean terrain level (placeholder returns 0.0).
    """
    # Placeholder - terrain levels require LuckyEngine terrain generator support
    # Full implementation would:
    # 1. Compute distance robot walked from env origin
    # 2. If distance > threshold: move_up terrain level
    # 3. If distance < required: move_down terrain level
    # 4. Call terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.tensor(0.0)


def commands_vel(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
    """Update velocity command ranges based on training progress.

    Progressively expands velocity ranges as training progresses.

    Args:
        env: The environment instance.
        env_ids: Environment indices (unused, curriculum is global).
        command_name: Name of the velocity command term.
        velocity_stages: List of stages with step thresholds and velocity ranges.

    Returns:
        Dict of current velocity range bounds for logging.
    """
    del env_ids  # Unused - curriculum is global

    command_term = env.command_manager.get_term(command_name)
    if command_term is None:
        logger.warning(f"Command term '{command_name}' not found")
        return {}

    cfg: UniformVelocityCommandCfg = command_term.cfg

    for stage in velocity_stages:
        if env.common_step_counter > stage["step"]:
            if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
                cfg.ranges.lin_vel_x = stage["lin_vel_x"]
            if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
                cfg.ranges.lin_vel_y = stage["lin_vel_y"]
            if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
                cfg.ranges.ang_vel_z = stage["ang_vel_z"]

    return {
        "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
        "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
        "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
        "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
        "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
        "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
    }


def reward_weight(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    reward_name: str,
    weight_stages: list[RewardWeightStage],
) -> torch.Tensor:
    """Update a reward term's weight based on training step stages.

    Args:
        env: The environment instance.
        env_ids: Environment indices (unused, curriculum is global).
        reward_name: Name of the reward term to adjust.
        weight_stages: List of stages with step thresholds and weights.

    Returns:
        Current weight value.
    """
    del env_ids

    if env.reward_manager is None:
        return torch.tensor(0.0)

    reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
    if reward_term_cfg is None:
        logger.warning(f"Reward term '{reward_name}' not found")
        return torch.tensor(0.0)

    for stage in weight_stages:
        if env.common_step_counter > stage["step"]:
            reward_term_cfg.weight = stage["weight"]

    return torch.tensor(reward_term_cfg.weight)
