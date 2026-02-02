"""MDP observation functions for luckylab environments.

Observation functions compute observation values from the environment state.
They access data via env.scene["robot"].data.* (mjlab pattern).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from ...entity import Entity
    from ..manager_based_rl_env import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


##
# Root state.
##


def base_lin_vel(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Get base linear velocity in body frame.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the asset.

    Returns:
        Linear velocity tensor [vx, vy, vz], shape (num_envs, 3).
    """
    asset: Entity = env.scene[asset_cfg.name]
    result = asset.data.root_link_lin_vel_b
    assert result is not None, f"root_link_lin_vel_b is None for asset '{asset_cfg.name}'"
    return result


def base_ang_vel(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Get base angular velocity in body frame.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the asset.

    Returns:
        Angular velocity tensor [wx, wy, wz], shape (num_envs, 3).
    """
    asset: Entity = env.scene[asset_cfg.name]
    result = asset.data.root_link_ang_vel_b
    assert result is not None, f"root_link_ang_vel_b is None for asset '{asset_cfg.name}'"
    return result


def projected_gravity(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Get projected gravity vector in body frame.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the asset.

    Returns:
        Projected gravity tensor [gx, gy, gz], shape (num_envs, 3).
    """
    asset: Entity = env.scene[asset_cfg.name]
    result = asset.data.projected_gravity_b
    assert result is not None, f"projected_gravity_b is None for asset '{asset_cfg.name}'"
    return result


##
# Joint state.
##


def joint_pos(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Get joint positions.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the asset. Use joint_ids to select specific joints.

    Returns:
        Joint positions tensor, shape (num_envs, num_joints).
    """
    asset: Entity = env.scene[asset_cfg.name]
    result = asset.data.joint_pos
    assert result is not None, f"joint_pos is None for asset '{asset_cfg.name}'"
    return result[:, asset_cfg.joint_ids]


def joint_pos_rel(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Get joint positions relative to default.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the asset. Use joint_ids to select specific joints.

    Returns:
        Relative joint positions tensor, shape (num_envs, num_joints).
    """
    asset: Entity = env.scene[asset_cfg.name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None, f"default_joint_pos is None for asset '{asset_cfg.name}'"
    joint_pos = asset.data.joint_pos
    assert joint_pos is not None, f"joint_pos is None for asset '{asset_cfg.name}'"
    jnt_ids = asset_cfg.joint_ids
    return joint_pos[:, jnt_ids] - default_joint_pos[:, jnt_ids]


def joint_vel(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Get joint velocities.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the asset. Use joint_ids to select specific joints.

    Returns:
        Joint velocities tensor, shape (num_envs, num_joints).
    """
    asset: Entity = env.scene[asset_cfg.name]
    result = asset.data.joint_vel
    assert result is not None, f"joint_vel is None for asset '{asset_cfg.name}'"
    return result[:, asset_cfg.joint_ids]


def joint_vel_rel(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Get joint velocities relative to default.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the asset. Use joint_ids to select specific joints.

    Returns:
        Relative joint velocities tensor, shape (num_envs, num_joints).
    """
    asset: Entity = env.scene[asset_cfg.name]
    default_joint_vel = asset.data.default_joint_vel
    assert default_joint_vel is not None, f"default_joint_vel is None for asset '{asset_cfg.name}'"
    joint_vel = asset.data.joint_vel
    assert joint_vel is not None, f"joint_vel is None for asset '{asset_cfg.name}'"
    jnt_ids = asset_cfg.joint_ids
    return joint_vel[:, jnt_ids] - default_joint_vel[:, jnt_ids]


##
# Actions.
##


def last_action(
    env: ManagerBasedRlEnv,
    action_name: str | None = None,
) -> torch.Tensor:
    """Get the last action taken.

    Args:
        env: The environment instance.
        action_name: Optional name of specific action term. If None, returns all actions.

    Returns:
        Last action tensor, shape (num_envs, action_dim).
    """
    if action_name is None:
        return env.action_manager.action
    term = env.action_manager.get_term(action_name)
    assert term is not None, f"Action term '{action_name}' not found"
    return term.raw_action


##
# Commands.
##


def generated_commands(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
) -> torch.Tensor:
    """Get the current command from the command manager.

    Args:
        env: The environment instance.
        command_name: Name of the command to get.

    Returns:
        Command tensor, shape (num_envs, command_dim).
    """
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found in command manager"
    return command


##
# Privileged observations (for asymmetric actor-critic).
# Names match mjlab: foot_contact, foot_height, foot_air_time, foot_contact_forces
##


def foot_contact(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Get foot contact states (binary).

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the asset.

    Returns:
        Foot contact tensor (binary 0/1), shape (num_envs, 4).
        Order: FR, FL, RR, RL.
    """
    asset: Entity = env.scene[asset_cfg.name]
    result = asset.data.foot_contact
    assert result is not None, (
        f"foot_contact is None for asset '{asset_cfg.name}'. "
        "Ensure LuckyEngine is providing foot contact data in the observation schema."
    )
    return result


def foot_height(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Get foot heights above ground.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the asset.

    Returns:
        Foot heights in meters, shape (num_envs, 4).
        Order: FR, FL, RR, RL.
    """
    asset: Entity = env.scene[asset_cfg.name]
    result = asset.data.foot_height
    assert result is not None, (
        f"foot_height is None for asset '{asset_cfg.name}'. "
        "Ensure LuckyEngine is providing foot height data in the observation schema."
    )
    return result


def foot_contact_forces(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Get foot contact force magnitudes.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the asset.

    Returns:
        Foot forces in Newtons, shape (num_envs, 4).
        Order: FR, FL, RR, RL.
    """
    asset: Entity = env.scene[asset_cfg.name]
    result = asset.data.foot_contact_forces
    assert result is not None, (
        f"foot_contact_forces is None for asset '{asset_cfg.name}'. "
        "Ensure LuckyEngine is providing foot force data in the observation schema."
    )
    return result


def foot_air_time(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Get time since last foot contact.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the asset.

    Returns:
        Time since last contact in seconds, shape (num_envs, 4).
        Order: FR, FL, RR, RL.
    """
    asset: Entity = env.scene[asset_cfg.name]
    result = asset.data.foot_air_time
    assert result is not None, (
        f"foot_air_time is None for asset '{asset_cfg.name}'. "
        "Ensure LuckyEngine is providing foot air time data in the observation schema."
    )
    return result
