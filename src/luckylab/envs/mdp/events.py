"""MDP event functions for luckylab environments.

Events are functions that modify the environment state at specific times:
- startup: Called once when the environment is created
- reset: Called when the environment is reset
- interval: Called at regular intervals during episodes
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..manager_based_rl_env import ManagerBasedRlEnv

__all__ = [
    "reset_scene_to_default",
    "reset_robot_state",
]


def reset_scene_to_default(env: ManagerBasedRlEnv, **kwargs: Any) -> None:
    """Reset the scene to its default state.

    This is the default reset event that restores the robot
    to its initial configuration.

    Args:
        env: The environment instance.
        **kwargs: Additional arguments (unused).
    """
    # The actual reset is handled by luckyrobots gRPC call
    # This function is a placeholder for consistency with mjlab
    pass


def reset_robot_state(
    env: ManagerBasedRlEnv,
    position_noise: float = 0.0,
    orientation_noise: float = 0.0,
    joint_position_noise: float = 0.0,
    joint_velocity_noise: float = 0.0,
    **kwargs: Any,
) -> None:
    """Reset robot state with optional noise.

    Args:
        env: The environment instance.
        position_noise: Standard deviation for position noise.
        orientation_noise: Standard deviation for orientation noise.
        joint_position_noise: Standard deviation for joint position noise.
        joint_velocity_noise: Standard deviation for joint velocity noise.
        **kwargs: Additional arguments (unused).
    """
    # Domain randomization is handled by PhysicsDRCfg in luckyrobots
    # This function is a placeholder for consistency with mjlab
    pass
