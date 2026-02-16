from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import torch

from luckylab.entity import Entity
from luckylab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from luckylab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


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
    asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> torch.Tensor:
    """Update terrain difficulty based on robot walking distance.

    Robots that walk far enough progress to harder terrains.
    Robots that don't walk far enough regress to easier terrains.

    Note: Full terrain curriculum requires LuckyEngine terrain support.
    Returns placeholder metric for now.

    Args:
        env: The environment instance.
        env_ids: Environment indices being reset.

    Returns:
        Mean terrain level (placeholder returns 0.0).
    """
    # Placeholder - terrain levels require LuckyEngine terrain generator support
    return torch.tensor(0.0)


def commands_vel(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
    """Update velocity command ranges in SimulationContract based on training step.

    Runs inside _reset_idx() -> curriculum_manager.compute(), BEFORE luckyrobots.reset().
    The updated ranges are sent to the engine immediately via the SimulationContract.
    """
    del env_ids  # Unused.
    contract = env.cfg.simulation_contract
    for stage in velocity_stages:
        if env.common_step_counter > stage["step"]:
            if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
                contract.vel_command_x_range = stage["lin_vel_x"]
            if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
                contract.vel_command_y_range = stage["lin_vel_y"]
            if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
                contract.vel_command_yaw_range = stage["ang_vel_z"]
    return {
        "lin_vel_x_min": torch.tensor(contract.vel_command_x_range[0]),
        "lin_vel_x_max": torch.tensor(contract.vel_command_x_range[1]),
        "lin_vel_y_min": torch.tensor(contract.vel_command_y_range[0]),
        "lin_vel_y_max": torch.tensor(contract.vel_command_y_range[1]),
        "ang_vel_z_min": torch.tensor(contract.vel_command_yaw_range[0]),
        "ang_vel_z_max": torch.tensor(contract.vel_command_yaw_range[1]),
    }


def reward_weight(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    reward_name: str,
    weight_stages: list[RewardWeightStage],
) -> torch.Tensor:
    """Update a reward term's weight based on training step stages."""
    del env_ids  # Unused.
    reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
    for stage in weight_stages:
        if env.common_step_counter > stage["step"]:
            reward_term_cfg.weight = stage["weight"]
    return torch.tensor([reward_term_cfg.weight])
