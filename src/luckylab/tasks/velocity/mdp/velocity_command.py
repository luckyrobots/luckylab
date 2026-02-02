"""Velocity command term for locomotion tasks.

Follows the architecture defined in ARCHITECTURE.md with alignment to mjlab patterns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from luckylab.managers.command_manager import CommandTerm
from luckylab.managers.manager_term_config import CommandTermCfg

if TYPE_CHECKING:
    from luckylab.envs.manager_based_rl_env import ManagerBasedRlEnv


def wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-pi, pi]."""
    return torch.atan2(torch.sin(angles), torch.cos(angles))


class UniformVelocityCommand(CommandTerm):
    """Samples velocity commands uniformly from ranges.

    Stores vel_command as (num_envs, 3) for [vx, vy, wz].
    Stores heading_target separately.
    The command property returns (num_envs, 4) for [vx, vy, wz, heading].
    """

    cfg: "UniformVelocityCommandCfg"

    def __init__(self, cfg: "UniformVelocityCommandCfg", env: "ManagerBasedRlEnv") -> None:
        super().__init__(cfg, env)

        # Velocity command buffer: (num_envs, 3) for [vx, vy, wz]
        self.vel_command = torch.zeros(self.num_envs, 3, device=self.device)

        # Heading target stored separately
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.heading_error = torch.zeros(self.num_envs, device=self.device)

        # Per-environment flags (mjlab pattern)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Initialize metrics
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Returns [vx, vy, wz, heading] for each env with shape (num_envs, 4)."""
        return torch.cat([self.vel_command, self.heading_target.unsqueeze(-1)], dim=-1)

    def _update_metrics(self) -> None:
        """Update velocity tracking metrics.

        Called each step. Requires robot velocity data to be provided externally.
        """
        pass

    def update_metrics_with_velocity(
        self,
        lin_vel_b: torch.Tensor,
        ang_vel_b: torch.Tensor,
    ) -> None:
        """Update tracking metrics with actual robot velocity.

        Args:
            lin_vel_b: Body-frame linear velocity [vx, vy, vz], shape (num_envs, 3).
            ang_vel_b: Body-frame angular velocity [wx, wy, wz], shape (num_envs, 3).
        """
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.cfg.step_dt

        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command[:, :2] - lin_vel_b[:, :2], dim=-1)
            / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command[:, 2] - ang_vel_b[:, 2])
            / max_command_step
        )

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """Resample velocity commands for specified environments."""
        n = len(env_ids)
        if n == 0:
            return

        r = self.cfg.ranges
        rand = torch.empty(n, device=self.device)

        # Sample velocity commands
        self.vel_command[env_ids, 0] = rand.uniform_(*r.lin_vel_x)
        self.vel_command[env_ids, 1] = rand.uniform_(*r.lin_vel_y)
        self.vel_command[env_ids, 2] = rand.uniform_(*r.ang_vel_z)

        # Sample heading
        self.heading_target[env_ids] = rand.uniform_(*r.heading)

        # Determine heading control environments
        if self.cfg.heading_command:
            self.is_heading_env[env_ids] = rand.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs

        # Determine standing environments
        self.is_standing_env[env_ids] = rand.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self) -> None:
        """Update commands based on current state."""
        # Heading control: adjust angular velocity to track heading target
        if self.cfg.heading_command:
            current_heading = self._get_robot_heading()
            if current_heading is not None:
                self.heading_error = wrap_to_pi(self.heading_target - current_heading)
                env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
                if len(env_ids) > 0:
                    self.vel_command[env_ids, 2] = torch.clamp(
                        self.cfg.heading_control_stiffness * self.heading_error[env_ids],
                        min=self.cfg.ranges.ang_vel_z[0],
                        max=self.cfg.ranges.ang_vel_z[1],
                    )

        # Zero out commands for standing environments
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        if len(standing_env_ids) > 0:
            self.vel_command[standing_env_ids, :] = 0.0

    def _get_robot_heading(self) -> torch.Tensor | None:
        """Get current robot heading from environment.

        Uses quaternion from Entity data to compute yaw heading.
        """
        asset = self._env.scene.get(self.cfg.asset_name)
        if asset is None:
            return None

        # Get quaternion from entity data
        quat = asset.data.root_link_quat_w
        if quat is None:
            return None

        # Extract yaw from quaternion (w, x, y, z format)
        # yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return torch.atan2(siny_cosp, cosy_cosp)

    def set_velocity_range(self, max_velocity: float) -> None:
        """Set maximum velocity range for curriculum learning."""
        self.cfg.ranges.lin_vel_x = (-max_velocity, max_velocity)
        self.cfg.ranges.lin_vel_y = (-max_velocity * 0.5, max_velocity * 0.5)
        self.cfg.ranges.ang_vel_z = (-max_velocity, max_velocity)


@dataclass(kw_only=True)
class UniformVelocityCommandCfg(CommandTermCfg):
    """Configuration for uniform velocity command generation."""

    # Asset configuration
    asset_name: str = "robot"
    """Name of the robot asset."""

    # Heading control
    heading_command: bool = False
    """Whether to use closed-loop heading control."""
    heading_control_stiffness: float = 1.0
    """P-gain for heading control."""
    rel_heading_envs: float = 1.0
    """Fraction of environments using heading control."""

    # Standing behavior
    rel_standing_envs: float = 0.0
    """Fraction of environments that output zero command."""

    # Class type
    class_type: type[CommandTerm] = UniformVelocityCommand

    @dataclass
    class Ranges:
        """Velocity command ranges."""

        lin_vel_x: tuple[float, float] = (-1.0, 1.0)
        """Linear velocity x range (m/s)."""
        lin_vel_y: tuple[float, float] = (-1.0, 1.0)
        """Linear velocity y range (m/s)."""
        ang_vel_z: tuple[float, float] = (-1.0, 1.0)
        """Angular velocity z range (rad/s)."""
        heading: tuple[float, float] = (-3.14159, 3.14159)
        """Heading range (rad)."""

    ranges: Ranges = field(default_factory=Ranges)
    """Velocity ranges for sampling."""

    @dataclass
    class VizCfg:
        """Debug visualization configuration."""

        z_offset: float = 0.2
        scale: float = 0.5

    viz: VizCfg = field(default_factory=VizCfg)
    """Visualization settings."""
