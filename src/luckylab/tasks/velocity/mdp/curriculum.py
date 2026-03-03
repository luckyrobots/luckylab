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


class CpgStage(TypedDict):
    """CPG amplitude curriculum stage definition."""

    step: int
    blend: float


def cpg_amplitude(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    action_term_name: str,
    stages: list[CpgStage],
) -> dict[str, torch.Tensor]:
    """Step-wise reduction of CPG amplitudes based on training step.

    Each stage sets a blend factor (1.0 = full CPG, 0.0 = no CPG).
    The policy gets a stable plateau at each level before the next drop.

    The gait phase observations (sin/cos) are still produced even at zero
    amplitude, so the policy retains the timing signal to coordinate its
    own gait after the CPG scaffold is removed.
    """
    del env_ids
    step = env.common_step_counter
    cpg = env.action_manager.get_term(action_term_name)

    # Store original amplitudes on first call
    if not hasattr(cpg, "_orig_amplitude_hip"):
        cpg._orig_amplitude_hip = cpg._amplitude_hip
        cpg._orig_amplitude_thigh = cpg._amplitude_thigh
        cpg._orig_amplitude_calf = cpg._amplitude_calf

    blend = 1.0
    for stage in stages:
        if step > stage["step"]:
            blend = stage["blend"]

    cpg._amplitude_hip = cpg._orig_amplitude_hip * blend
    cpg._amplitude_thigh = cpg._orig_amplitude_thigh * blend
    cpg._amplitude_calf = cpg._orig_amplitude_calf * blend

    return {"cpg_blend": torch.tensor(blend)}
