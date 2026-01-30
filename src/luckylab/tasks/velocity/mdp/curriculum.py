"""Curriculum functions for velocity task."""

from typing import Any


def terrain_curriculum(env: Any, stages: list[dict]) -> None:
    """Adjust terrain difficulty based on training progress.

    Args:
        env: The environment instance.
        stages: List of stage dicts with "step" and "difficulty" keys.
    """
    step = env.common_step_counter
    for stage in reversed(stages):
        if step >= stage["step"]:
            env.terrain_difficulty = stage["difficulty"]
            break


def velocity_curriculum(env: Any, command_name: str, stages: list[dict]) -> None:
    """Adjust command velocity range based on training progress.

    Args:
        env: The environment instance.
        command_name: Name of the command to adjust.
        stages: List of stage dicts with "step" and "max_velocity" keys.
    """
    step = env.common_step_counter
    for stage in reversed(stages):
        if step >= stage["step"]:
            env.command_manager.set_max_velocity(command_name, stage["max_velocity"])
            break
