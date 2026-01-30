"""Unitree Go1 environment configuration for velocity task."""

from ...velocity_env_cfg import create_velocity_env_cfg

# Default Go1 velocity task configuration
# Uses default reward weights and termination parameters
GO1_ENV_CFG = create_velocity_env_cfg(
    robot="unitreego1",
    # Can override weights here if needed for Go1-specific tuning:
    # track_linear_velocity_weight=2.0,
    # body_ang_vel_weight=-0.05,
)
