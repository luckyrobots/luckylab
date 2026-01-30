"""MDP termination functions for luckylab environments.

Termination functions check if an episode should end.
They return True if the episode should terminate, False otherwise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..manager_based_rl_env import ManagerBasedRlEnv

__all__ = [
    "time_out",
    "bad_orientation",
    "root_height_below_minimum",
    "nan_detection",
]


def time_out(env: ManagerBasedRlEnv, **kwargs: Any) -> bool:
    """Terminate when the episode length exceeds the maximum.

    Args:
        env: The environment instance.

    Returns:
        True if episode timed out.
    """
    return env.step_count >= env.cfg.max_episode_length


def bad_orientation(
    env: ManagerBasedRlEnv,
    limit_angle: float = 1.0,
    **kwargs: Any,
) -> bool:
    """Terminate when the robot's orientation exceeds the limit angle.

    Args:
        env: The environment instance.
        limit_angle: Maximum allowed tilt angle in radians.

    Returns:
        True if orientation exceeds limit.
    """
    if env.latest_observation is None or not env.latest_observation.observation:
        return False

    obs = env.obs_parser.parse(np.array(env.latest_observation.observation))
    projected_gravity = obs.get("projected_gravity", None)
    if projected_gravity is None:
        return False

    # Compute angle from vertical using z component of projected gravity
    # projected_gravity should be [0, 0, -1] when upright
    cos_angle = -projected_gravity[2]  # cos(angle) = -gz
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)

    return bool(angle > limit_angle)


def root_height_below_minimum(
    env: ManagerBasedRlEnv,
    minimum_height: float = 0.1,
    **kwargs: Any,
) -> bool:
    """Terminate when the robot's root height is below the minimum.

    Args:
        env: The environment instance.
        minimum_height: Minimum allowed height in meters.

    Returns:
        True if height is below minimum.
    """
    if env.latest_observation is None or not env.latest_observation.observation:
        return False

    obs = env.obs_parser.parse(np.array(env.latest_observation.observation))
    base_position = obs.get("base_position", None)
    if base_position is None:
        return False

    return bool(base_position[2] < minimum_height)


def nan_detection(env: ManagerBasedRlEnv, **kwargs: Any) -> bool:
    """Terminate if NaN or Inf values are detected in observations.

    Args:
        env: The environment instance.

    Returns:
        True if NaN/Inf detected.
    """
    if env.latest_observation is None or not env.latest_observation.observation:
        return False

    obs_array = np.array(env.latest_observation.observation)
    return bool(np.any(np.isnan(obs_array)) or np.any(np.isinf(obs_array)))
