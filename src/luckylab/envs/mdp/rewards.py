"""MDP reward functions for luckylab environments.

Reward functions compute scalar reward values from the environment state.
They are used by the reward manager to compute the total reward.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..manager_based_rl_env import ManagerBasedRlEnv

__all__ = [
    "is_alive",
    "is_terminated",
    "action_rate_l2",
    "joint_pos_limits",
    "flat_orientation_l2",
]


def is_alive(env: ManagerBasedRlEnv, **kwargs: Any) -> float:
    """Reward for being alive (not terminated).

    Args:
        env: The environment instance.

    Returns:
        1.0 if alive, 0.0 if terminated.
    """
    return 1.0


def is_terminated(env: ManagerBasedRlEnv, **kwargs: Any) -> float:
    """Penalty for being terminated.

    Args:
        env: The environment instance.

    Returns:
        1.0 if terminated, 0.0 otherwise.
    """
    return 0.0


def action_rate_l2(env: ManagerBasedRlEnv, **kwargs: Any) -> float:
    """Penalize the rate of change of actions using L2 squared kernel.

    Args:
        env: The environment instance.

    Returns:
        Sum of squared action differences.
    """
    if env.current_action is None or env.last_action is None:
        return 0.0
    diff = env.current_action - env.last_action
    return float(np.sum(diff ** 2))


def joint_pos_limits(
    env: ManagerBasedRlEnv,
    soft_ratio: float = 0.9,
    **kwargs: Any,
) -> float:
    """Penalize joint positions if they cross soft limits.

    Args:
        env: The environment instance.
        soft_ratio: Ratio of hard limits to use as soft limits.

    Returns:
        Sum of out-of-limit violations.
    """
    if env.latest_observation is None or not env.latest_observation.observation:
        return 0.0

    obs = env.obs_parser.parse(np.array(env.latest_observation.observation))
    joint_pos = obs.get("joint_positions", None)
    if joint_pos is None:
        return 0.0

    # Compute soft limits from action space
    soft_lower = env.action_low * soft_ratio
    soft_upper = env.action_high * soft_ratio

    # Compute violations
    lower_violation = np.clip(soft_lower - joint_pos, 0, None)
    upper_violation = np.clip(joint_pos - soft_upper, 0, None)

    return float(np.sum(lower_violation + upper_violation))


def flat_orientation_l2(env: ManagerBasedRlEnv, **kwargs: Any) -> float:
    """Penalize non-flat base orientation.

    Args:
        env: The environment instance.

    Returns:
        Sum of squared projected gravity x and y components.
    """
    if env.latest_observation is None or not env.latest_observation.observation:
        return 0.0

    obs = env.obs_parser.parse(np.array(env.latest_observation.observation))
    projected_gravity = obs.get("projected_gravity", None)
    if projected_gravity is None:
        return 0.0

    # Penalize x and y components (should be 0 for flat orientation)
    return float(np.sum(projected_gravity[:2] ** 2))
