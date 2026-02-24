from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import torch

if TYPE_CHECKING:
    from luckylab.envs.manager_based_rl_env import ManagerBasedRlEnv


class VelocityStage(TypedDict, total=False):
    """Velocity curriculum stage definition."""

    step: int
    lin_vel_x: tuple[float, float] | None
    lin_vel_y: tuple[float, float] | None
    ang_vel_z: tuple[float, float] | None


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
