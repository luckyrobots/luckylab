from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from luckylab.entity import Entity
from luckylab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from luckylab.envs.manager_based_rl_env import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_linear_velocity(
    env: ManagerBasedRlEnv,
    sigma: float = 0.25,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward for tracking the commanded base linear velocity.

    The commanded z velocity is assumed to be zero.
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd_xy = asset.data.vel_command[:, :2]
    actual_xy = asset.data.root_link_lin_vel_b[:, :2]
    error_sq = torch.sum(torch.square(actual_xy - cmd_xy), dim=1)
    baseline_sq = torch.sum(torch.square(cmd_xy), dim=1)
    return torch.exp(-error_sq / sigma) - torch.exp(-baseline_sq / sigma)


def track_angular_velocity(
    env: ManagerBasedRlEnv,
    sigma: float = 0.25,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward heading error for heading-controlled envs, angular velocity for others.

    The commanded xy angular velocities are assumed to be zero.
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd_yaw = asset.data.vel_command[:, 2]
    actual_yaw = asset.data.root_link_ang_vel_b[:, 2]
    error_sq = torch.square(actual_yaw - cmd_yaw)
    baseline_sq = torch.square(cmd_yaw)
    return torch.exp(-error_sq / sigma) - torch.exp(-baseline_sq / sigma)


def stand_still(
    env: ManagerBasedRlEnv,
    command_threshold: float = 0.1,
    vel_sigma: float = 0.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward being still when command is near zero.

    Uses exponential decay based on base velocity and joint velocity so that
    any movement is penalized. The policy must learn to cancel the CPG
    (by outputting -cpg_ref) to achieve zero joint velocity. Returns 0
    when a non-zero command is active.
    """
    asset: Entity = env.scene[asset_cfg.name]
    command = asset.data.vel_command
    cmd_norm = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    is_standing = (cmd_norm < command_threshold).float()

    base_vel_sq = torch.sum(torch.square(asset.data.root_link_lin_vel_b[:, :2]), dim=1)
    base_ang_vel_sq = torch.square(asset.data.root_link_ang_vel_b[:, 2])
    joint_vel_sq = torch.mean(torch.square(asset.data.joint_vel), dim=1)

    motion = base_vel_sq + base_ang_vel_sq + 0.1 * joint_vel_sq
    stillness = torch.exp(-motion / vel_sigma)
    return is_standing * stillness


def action_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Penalize action magnitude (L2 of policy residuals).

    Since the policy output is the residual on top of the CPG, penalizing
    its magnitude pushes the policy to stay close to the CPG pattern and
    only deviate when tracking rewards justify it.
    """
    return torch.sum(torch.square(env.action_manager.action), dim=1)


def contact_force_penalty(
    env: ManagerBasedRlEnv,
    threshold: float = 50.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize excessive foot contact forces above a threshold.

    Only penalizes forces that exceed the threshold, allowing normal
    ground contact. Encourages soft landings and smooth gait.
    """
    asset: Entity = env.scene[asset_cfg.name]
    forces = asset.data.foot_contact_forces  # (num_envs, 4)
    excessive = torch.clamp(forces - threshold, min=0.0)
    return torch.sum(torch.square(excessive), dim=1)


def foot_slip_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize horizontal foot velocity when foot is in contact with the ground.

    Only penalizes xy velocity during stance phase — feet should be stationary
    when touching the ground. Zero penalty when feet are in the air.
    """
    asset: Entity = env.scene[asset_cfg.name]
    contact = asset.data.foot_contact  # (num_envs, 4)
    foot_vel = asset.data.foot_velocity  # (num_envs, 4, 3)
    horizontal_vel_sq = torch.sum(torch.square(foot_vel[:, :, :2]), dim=2)  # (num_envs, 4)
    slip = contact * horizontal_vel_sq  # zero when foot is in air
    return torch.sum(slip, dim=1)


def forward_velocity(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward forward velocity up to commanded speed.

    Returns clamp(actual_vx, 0, cmd_vx). Proportional credit for moving
    forward — dense gradient from the first step. Zero when standing or
    moving backward. Naturally zero when cmd_vx=0 (standing command).
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd_vx = asset.data.vel_command[:, 0]
    actual_vx = asset.data.root_link_lin_vel_b[:, 0]
    return torch.clamp(actual_vx, min=torch.zeros_like(cmd_vx), max=cmd_vx)


def feet_air_time(
    env: ManagerBasedRlEnv,
    threshold_min: float = 0.05,
    threshold_max: float = 0.5,
    command_threshold: float = 0.1,
    velocity_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward feet with air time in valid range. Gated by command magnitude and actual velocity."""
    asset: Entity = env.scene[asset_cfg.name]
    air_time = asset.data.foot_air_time
    in_range = (air_time > threshold_min) & (air_time < threshold_max)
    reward = torch.sum(in_range.float(), dim=1)
    # Gate by command magnitude
    command = asset.data.vel_command
    cmd_norm = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    reward *= (cmd_norm > command_threshold).float()
    # Gate by actual velocity magnitude — robot must be moving
    vel_magnitude = torch.norm(asset.data.root_link_lin_vel_b[:, :2], dim=1)
    reward *= (vel_magnitude > velocity_threshold).float()
    return reward
