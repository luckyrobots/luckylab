"""Simulation contract between luckylab and LuckyEngine.

SimulationContract defines everything both sides agree on: domain randomization,
command ranges, resampling timing, etc. It's sent to the engine at every reset,
so the curriculum can update it before each episode.

The luckyrobots client converts it to the DomainRandomizationConfig proto.
"""

from dataclasses import dataclass


@dataclass
class SimulationContract:
    """
    Simulation contract - sent to LuckyEngine on each reset.

    All domain randomization and command sampling is executed by the engine.
    Ranges are (min, max) tuples. If min == max, no randomization occurs.
    """

    # --- Domain randomization: Initial state ---
    pose_position_noise: tuple[float, float, float] = (0.0, 0.0, 0.0)  # xyz noise
    pose_orientation_noise: float = 0.0  # radians, applied to yaw
    joint_position_noise: float = 0.0  # radians, per joint
    joint_velocity_noise: float = 0.0  # rad/s, per joint

    # --- Domain randomization: Physics parameters ---
    friction_range: tuple[float, float] = (1.0, 1.0)
    restitution_range: tuple[float, float] = (0.0, 0.0)
    mass_scale_range: tuple[float, float] = (1.0, 1.0)  # multiplier
    com_offset_range: tuple[float, float] = (0.0, 0.0)  # center of mass offset

    # --- Domain randomization: Motor/actuator ---
    motor_strength_range: tuple[float, float] = (1.0, 1.0)  # multiplier
    motor_offset_range: tuple[float, float] = (0.0, 0.0)  # position offset

    # --- Domain randomization: External disturbances ---
    push_interval_range: tuple[float, float] | None = None  # seconds, None = disabled
    push_velocity_range: tuple[float, float] = (0.0, 0.0)  # m/s impulse magnitude

    # --- Domain randomization: Terrain ---
    terrain_type: str = "flat"
    terrain_difficulty: float = 0.0  # 0.0 = easiest, 1.0 = hardest

    # --- Command ranges (sampled by engine) ---
    vel_command_x_range: tuple[float, float] = (-1.0, 1.0)
    vel_command_y_range: tuple[float, float] = (-1.0, 1.0)
    vel_command_yaw_range: tuple[float, float] = (-0.5, 0.5)
    vel_command_resampling_time_range: tuple[float, float] = (3.0, 8.0)
    vel_command_standing_probability: float = 0.1
