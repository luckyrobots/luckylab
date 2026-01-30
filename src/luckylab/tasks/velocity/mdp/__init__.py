"""Velocity task MDP: observations, rewards, terminations, commands, and curriculum."""

from .commands import velocity_command
from .curriculum import terrain_curriculum, velocity_curriculum
from .observations import ObservationParser
from .rewards import (
    action_rate_l2,
    body_angular_velocity_penalty,
    joint_pos_limits,
    track_angular_velocity,
    track_linear_velocity,
    variable_posture,
)
from .terminations import (
    bad_orientation,
    fall_termination,
    max_steps_termination,
    nan_detection,
    root_height_below_minimum,
    time_out,
)

__all__ = [
    # Observations
    "ObservationParser",
    # Rewards
    "track_linear_velocity",
    "track_angular_velocity",
    "variable_posture",
    "body_angular_velocity_penalty",
    "joint_pos_limits",
    "action_rate_l2",
    # Terminations
    "time_out",
    "bad_orientation",
    "root_height_below_minimum",
    "nan_detection",
    "fall_termination",
    "max_steps_termination",
    # Commands
    "velocity_command",
    # Curriculum
    "terrain_curriculum",
    "velocity_curriculum",
]
