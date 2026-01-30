"""Termination conditions for locomotion tasks, adapted from Isaac Lab."""

import math

import numpy as np


def time_out(step_count: int, max_steps: int) -> bool:
    """
    Terminate when the episode length exceeds its maximum.

    Args:
        step_count: Current step count
        max_steps: Maximum steps per episode

    Returns:
        True if max steps reached
    """
    return step_count >= max_steps


def bad_orientation(
    obs_parsed: dict[str, np.ndarray],
    limit_angle: float = math.radians(70.0),
) -> bool:
    """
    Terminate when the robot's orientation exceeds the limit angle.

    Approximates orientation using base angular velocity and linear velocity.
    If the robot is tilting significantly (high angular velocity or falling),
    consider it as bad orientation.

    Args:
        obs_parsed: Parsed observation dictionary
        limit_angle: Maximum allowed tilt angle in radians

    Returns:
        True if orientation exceeds limit
    """
    base_ang_vel = obs_parsed["base_ang_vel"]  # [x, y, z]
    base_lin_vel = obs_parsed["base_lin_vel"]  # [x, y, z]

    # Approximate tilt using angular velocity magnitude
    # High angular velocity in xy suggests tilting/falling
    ang_vel_xy_magnitude = np.linalg.norm(base_ang_vel[:2])

    # Also check if falling rapidly (negative z velocity)
    falling_rapidly = base_lin_vel[2] < -1.5

    # Approximate angle from angular velocity (rough heuristic)
    # Angular velocity magnitude correlates with tilt rate
    # This is an approximation since we don't have direct orientation
    estimated_tilt_rate = ang_vel_xy_magnitude

    # If tilting rapidly or falling, consider it bad orientation
    return bool(estimated_tilt_rate > limit_angle or falling_rapidly)


def root_height_below_minimum(
    obs_parsed: dict[str, np.ndarray],
    minimum_height: float = 0.2,
    falling_velocity_threshold: float = -2.0,
) -> bool:
    """
    Terminate when the robot's root height is below the minimum height.

    Since we don't have direct height in observations, we approximate by:
    - Checking if falling rapidly downward (negative z velocity)
    - Using a velocity threshold as a proxy for height

    Args:
        obs_parsed: Parsed observation dictionary
        minimum_height: Minimum allowed height (not directly used, kept for API consistency)
        falling_velocity_threshold: Velocity threshold for detecting fall

    Returns:
        True if robot appears to be below minimum height
    """
    base_lin_vel = obs_parsed["base_lin_vel"]  # [x, y, z]

    # If falling downward rapidly, consider below minimum height
    return bool(base_lin_vel[2] < falling_velocity_threshold)


def nan_detection(obs_parsed: dict[str, np.ndarray]) -> bool:
    """
    Terminate if NaN or Inf values are detected in observations.

    Args:
        obs_parsed: Parsed observation dictionary

    Returns:
        True if NaN/Inf detected
    """
    return any(np.any(np.isnan(value)) or np.any(np.isinf(value)) for value in obs_parsed.values())


def fall_termination(
    obs_parsed: dict[str, np.ndarray],
    z_threshold: float = -2.0,
) -> bool:
    """
    Terminate if robot falls (falling rapidly downward).

    Args:
        obs_parsed: Parsed observation dictionary
        z_threshold: Velocity threshold for detecting fall (negative value)

    Returns:
        True if robot has fallen
    """
    base_lin_vel = obs_parsed["base_lin_vel"]  # [x, y, z]

    # If falling downward rapidly (negative z velocity), consider fallen
    return bool(base_lin_vel[2] < z_threshold)


def max_steps_termination(
    step_count: int,
    max_steps: int,
) -> bool:
    """
    Terminate if maximum steps reached.

    Note: This is typically handled by gymnasium's TimeLimit wrapper,
    but included here for completeness.

    Args:
        step_count: Current step count
        max_steps: Maximum steps per episode

    Returns:
        True if max steps reached
    """
    return step_count >= max_steps
