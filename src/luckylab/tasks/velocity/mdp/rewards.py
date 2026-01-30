"""Reward functions for locomotion tasks, adapted from Isaac Lab."""

import math

import numpy as np


def track_linear_velocity(
    obs_parsed: dict[str, np.ndarray],
    std: float = math.sqrt(0.25),
) -> float:
    """
    Reward for tracking the commanded base linear velocity.

    The commanded z velocity is assumed to be zero.

    Args:
        obs_parsed: Parsed observation dictionary
        std: Standard deviation for exponential reward

    Returns:
        Reward value (0-1, higher is better)
    """
    commands = obs_parsed["commands"]  # [vx, vy, wz, heading]
    actual = obs_parsed["base_lin_vel"]  # [x, y, z]

    # Track xy velocity, penalize z velocity
    xy_error = np.sum(np.square(commands[:2] - actual[:2]))
    z_error = actual[2] ** 2
    lin_vel_error = xy_error + z_error

    return float(np.exp(-lin_vel_error / (std**2)))


def track_angular_velocity(
    obs_parsed: dict[str, np.ndarray],
    std: float = math.sqrt(0.5),
) -> float:
    """
    Reward for tracking the commanded angular velocity.

    The commanded xy angular velocities are assumed to be zero.

    Args:
        obs_parsed: Parsed observation dictionary
        std: Standard deviation for exponential reward

    Returns:
        Reward value (0-1, higher is better)
    """
    commands = obs_parsed["commands"]  # [vx, vy, wz, heading]
    actual = obs_parsed["base_ang_vel"]  # [x, y, z]

    # Track z angular velocity, penalize xy angular velocity
    z_error = (commands[2] - actual[2]) ** 2
    xy_error = np.sum(np.square(actual[:2]))
    ang_vel_error = z_error + xy_error

    return float(np.exp(-ang_vel_error / (std**2)))


def variable_posture(
    obs_parsed: dict[str, np.ndarray],
    std_standing: float | np.ndarray = 0.1,
    std_walking: float | np.ndarray = 0.2,
    std_running: float | np.ndarray = 0.3,
    walking_threshold: float = 0.05,
    running_threshold: float = 1.5,
) -> float:
    """
    Penalize deviation from default pose, with tighter constraints when standing.

    Joint positions are already relative to default, so we just penalize deviation.

    Args:
        obs_parsed: Parsed observation dictionary
        std_standing: Standard deviation when standing (can be scalar or per-joint)
        std_walking: Standard deviation when walking (can be scalar or per-joint)
        std_running: Standard deviation when running (can be scalar or per-joint)
        walking_threshold: Speed threshold for walking mode
        running_threshold: Speed threshold for running mode

    Returns:
        Reward value (0-1, higher is better)
    """
    commands = obs_parsed["commands"]  # [vx, vy, wz, heading]
    joint_pos = obs_parsed["joint_pos"]  # Already relative to default

    # Compute total command speed
    linear_speed = np.linalg.norm(commands[:2])
    angular_speed = abs(commands[2])
    total_speed = linear_speed + angular_speed

    # Select std based on speed
    if total_speed < walking_threshold:
        std = std_standing
    elif total_speed < running_threshold:
        std = std_walking
    else:
        std = std_running

    # Ensure std is array if joint_pos is array
    if isinstance(std, (int, float)):
        std = np.full_like(joint_pos, std, dtype=np.float32)
    else:
        std = np.asarray(std, dtype=np.float32)

    # Compute error squared
    error_squared = np.square(joint_pos)

    # Exponential reward
    return float(np.exp(-np.mean(error_squared / (std**2))))


def body_angular_velocity_penalty(
    obs_parsed: dict[str, np.ndarray],
) -> float:
    """
    Penalize excessive body angular velocities (xy components only).

    Args:
        obs_parsed: Parsed observation dictionary

    Returns:
        Penalty value (non-negative, lower is better)
    """
    base_ang_vel = obs_parsed["base_ang_vel"]  # [x, y, z]
    ang_vel_xy = base_ang_vel[:2]  # Don't penalize z-angular velocity
    return float(np.sum(np.square(ang_vel_xy)))


def joint_pos_limits(
    obs_parsed: dict[str, np.ndarray],
    action_low: np.ndarray,
    action_high: np.ndarray,
    margin: float = 0.1,
) -> float:
    """
    Penalize joint positions near limits.

    Args:
        obs_parsed: Parsed observation dictionary
        action_low: Lower joint limits
        action_high: Upper joint limits
        margin: Margin from limits to start penalizing (as fraction of range)

    Returns:
        Penalty value (non-negative, lower is better)
    """
    joint_pos = obs_parsed["joint_pos"]
    # Joint positions are relative to default, so we need to add default back
    # For simplicity, assume default is middle of range
    default_pos = (action_low + action_high) / 2.0
    absolute_joint_pos = joint_pos + default_pos

    # Compute distance to limits
    range_size = action_high - action_low
    margin_size = margin * range_size

    dist_to_low = absolute_joint_pos - (action_low + margin_size)
    dist_to_high = (action_high - margin_size) - absolute_joint_pos

    # Penalty when within margin of limits
    penalty_low = np.sum(np.maximum(0.0, -dist_to_low) ** 2)
    penalty_high = np.sum(np.maximum(0.0, -dist_to_high) ** 2)

    return float(penalty_low + penalty_high)


def action_rate_l2(
    current_action: np.ndarray,
    last_action: np.ndarray,
) -> float:
    """
    Penalize large action changes (L2 norm of action difference).

    Args:
        current_action: Current action
        last_action: Previous action

    Returns:
        Penalty value (non-negative, lower is better)
    """
    action_diff = current_action - last_action
    return float(np.sum(np.square(action_diff)))
