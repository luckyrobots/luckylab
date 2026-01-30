"""Domain randomization configuration.

Note: PhysicsDRCfg should eventually move to luckyrobots package
so it can be shared between LuckyLab and LuckyEngine.

For observation domain randomization, use ObservationProcessorCfg from
luckylab.managers.observation_manager instead.
"""

from dataclasses import dataclass


@dataclass
class PhysicsDRCfg:
    """
    Physics domain randomization config - executed by LuckyEngine.

    All ranges are (min, max) tuples. If min == max, no randomization occurs.
    """

    # Initial state randomization
    pose_position_noise: tuple[float, float, float] = (0.0, 0.0, 0.0)  # xyz noise
    pose_orientation_noise: float = 0.0  # radians, applied to yaw
    joint_position_noise: float = 0.0  # radians, per joint
    joint_velocity_noise: float = 0.0  # rad/s, per joint

    # Physics parameter randomization
    friction_range: tuple[float, float] = (1.0, 1.0)
    restitution_range: tuple[float, float] = (0.0, 0.0)
    mass_scale_range: tuple[float, float] = (1.0, 1.0)  # multiplier
    com_offset_range: tuple[float, float] = (0.0, 0.0)  # center of mass offset

    # Motor/actuator randomization
    motor_strength_range: tuple[float, float] = (1.0, 1.0)  # multiplier
    motor_offset_range: tuple[float, float] = (0.0, 0.0)  # position offset

    # External disturbances
    push_interval_range: tuple[float, float] | None = None  # seconds, None = disabled
    push_velocity_range: tuple[float, float] = (0.0, 0.0)  # m/s impulse magnitude

    # Terrain (if supported by scene)
    terrain_type: str = "flat"
    terrain_difficulty: float = 0.0  # 0.0 = easiest, 1.0 = hardest

    def __eq__(self, other: object) -> bool:
        """Compare configs for equality (used by LuckyRobots for deduplication)."""
        if not isinstance(other, PhysicsDRCfg):
            return False
        return (
            self.pose_position_noise == other.pose_position_noise
            and self.pose_orientation_noise == other.pose_orientation_noise
            and self.joint_position_noise == other.joint_position_noise
            and self.joint_velocity_noise == other.joint_velocity_noise
            and self.friction_range == other.friction_range
            and self.restitution_range == other.restitution_range
            and self.mass_scale_range == other.mass_scale_range
            and self.com_offset_range == other.com_offset_range
            and self.motor_strength_range == other.motor_strength_range
            and self.motor_offset_range == other.motor_offset_range
            and self.push_interval_range == other.push_interval_range
            and self.push_velocity_range == other.push_velocity_range
            and self.terrain_type == other.terrain_type
            and self.terrain_difficulty == other.terrain_difficulty
        )
