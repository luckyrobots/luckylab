"""Command functions for velocity task."""

from typing import Any

import numpy as np


def velocity_command(
    env: Any,
    x_range: tuple[float, float] = (-1.0, 1.0),
    y_range: tuple[float, float] = (-0.5, 0.5),
    yaw_range: tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    """Generate a random velocity command.

    Args:
        env: The environment instance.
        x_range: Range for x velocity (forward/backward).
        y_range: Range for y velocity (left/right).
        yaw_range: Range for yaw rate (turning).

    Returns:
        Velocity command array [vx, vy, vyaw].
    """
    vx = np.random.uniform(x_range[0], x_range[1])
    vy = np.random.uniform(y_range[0], y_range[1])
    vyaw = np.random.uniform(yaw_range[0], yaw_range[1])
    return np.array([vx, vy, vyaw], dtype=np.float32)
