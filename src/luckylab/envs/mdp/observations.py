"""MDP observation functions for luckylab environments.

Observation functions compute observation values from the environment state.
They are used by the observation manager to build observation tensors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..manager_based_rl_env import ManagerBasedRlEnv

__all__ = [
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
    "generated_commands",
]


def base_lin_vel(env: ManagerBasedRlEnv, **kwargs: Any) -> np.ndarray:
    """Get base linear velocity in body frame.

    Args:
        env: The environment instance.

    Returns:
        Linear velocity array [vx, vy, vz].
    """
    if env.latest_observation and env.latest_observation.observation:
        obs = env.obs_parser.parse(np.array(env.latest_observation.observation))
        return obs.get("base_linear_velocity", np.zeros(3, dtype=np.float32))
    return np.zeros(3, dtype=np.float32)


def base_ang_vel(env: ManagerBasedRlEnv, **kwargs: Any) -> np.ndarray:
    """Get base angular velocity in body frame.

    Args:
        env: The environment instance.

    Returns:
        Angular velocity array [wx, wy, wz].
    """
    if env.latest_observation and env.latest_observation.observation:
        obs = env.obs_parser.parse(np.array(env.latest_observation.observation))
        return obs.get("base_angular_velocity", np.zeros(3, dtype=np.float32))
    return np.zeros(3, dtype=np.float32)


def projected_gravity(env: ManagerBasedRlEnv, **kwargs: Any) -> np.ndarray:
    """Get projected gravity vector in body frame.

    Args:
        env: The environment instance.

    Returns:
        Projected gravity array [gx, gy, gz].
    """
    if env.latest_observation and env.latest_observation.observation:
        obs = env.obs_parser.parse(np.array(env.latest_observation.observation))
        return obs.get("projected_gravity", np.array([0, 0, -1], dtype=np.float32))
    return np.array([0, 0, -1], dtype=np.float32)


def joint_pos_rel(env: ManagerBasedRlEnv, **kwargs: Any) -> np.ndarray:
    """Get joint positions relative to default.

    Args:
        env: The environment instance.

    Returns:
        Relative joint positions array.
    """
    if env.latest_observation and env.latest_observation.observation:
        obs = env.obs_parser.parse(np.array(env.latest_observation.observation))
        return obs.get("joint_positions", np.zeros(env.num_joints, dtype=np.float32))
    return np.zeros(env.num_joints, dtype=np.float32)


def joint_vel_rel(env: ManagerBasedRlEnv, **kwargs: Any) -> np.ndarray:
    """Get joint velocities relative to default.

    Args:
        env: The environment instance.

    Returns:
        Relative joint velocities array.
    """
    if env.latest_observation and env.latest_observation.observation:
        obs = env.obs_parser.parse(np.array(env.latest_observation.observation))
        return obs.get("joint_velocities", np.zeros(env.num_joints, dtype=np.float32))
    return np.zeros(env.num_joints, dtype=np.float32)


def last_action(env: ManagerBasedRlEnv, **kwargs: Any) -> np.ndarray:
    """Get the last action taken.

    Args:
        env: The environment instance.

    Returns:
        Last action array.
    """
    if env.last_action is not None:
        return env.last_action
    return np.zeros(env.num_joints, dtype=np.float32)


def generated_commands(env: ManagerBasedRlEnv, command_name: str = "velocity", **kwargs: Any) -> np.ndarray:
    """Get the current command from the command manager.

    Args:
        env: The environment instance.
        command_name: Name of the command to get.

    Returns:
        Command array.
    """
    return env.command_manager.get_command(command_name)
