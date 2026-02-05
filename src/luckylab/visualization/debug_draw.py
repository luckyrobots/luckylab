"""Debug visualization for LuckyEngine.

Provides utilities for drawing debug primitives (lines, arrows, velocity commands)
in the LuckyEngine viewport during training or evaluation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from luckylab.utils.logging import print_info

if TYPE_CHECKING:
    from luckylab.envs import ManagerBasedRlEnv
    from luckyrobots import LuckyEngineClient


class DebugVisualizer:
    """Handles debug visualization in LuckyEngine.

    Usage:
        viz = DebugVisualizer(env)
        viz.draw_velocity_command(env_idx=0)
        viz.draw_arrow(origin, direction, color)
    """

    def __init__(self, env: "ManagerBasedRlEnv") -> None:
        """Initialize the debug visualizer.

        Args:
            env: The ManagerBasedRlEnv instance to visualize.
        """
        self._env = env
        self._warned_no_client = False
        self._warned_no_command = False

    @property
    def client(self) -> "LuckyEngineClient | None":
        """Get the LuckyEngine gRPC client."""
        luckyrobots = getattr(self._env, "luckyrobots", None)
        if luckyrobots is None:
            return None
        return getattr(luckyrobots, "engine_client", None)

    def get_velocity_command(
        self, env_idx: int = 0
    ) -> tuple[float, float, float] | None:
        """Get the current velocity command for an environment.

        Args:
            env_idx: Environment index.

        Returns:
            Tuple of (lin_vel_x, lin_vel_y, ang_vel_z) or None if unavailable.
        """
        cmd_mgr = getattr(self._env, "command_manager", None)
        if cmd_mgr is None:
            return None

        for term in cmd_mgr.terms.values():
            if hasattr(term, "vel_command_b"):
                cmd = term.vel_command_b[env_idx]
                return (cmd[0].item(), cmd[1].item(), cmd[2].item())
        return None

    def get_robot_position(
        self, env_idx: int = 0, z_offset: float = 0.0
    ) -> tuple[float, float, float]:
        """Get the robot's current position.

        Args:
            env_idx: Environment index.
            z_offset: Height offset to add.

        Returns:
            Tuple of (x, y, z) position.
        """
        robot = self._env.scene.get("robot")
        if robot is None:
            return (0.0, 0.0, z_offset)

        pos = robot.data.root_link_pos_w[env_idx]
        return (pos[0].item(), pos[1].item(), pos[2].item() + z_offset)

    def draw_velocity_command(
        self,
        env_idx: int = 0,
        scale: float = 1.0,
        z_offset: float = 0.5,
    ) -> bool:
        """Draw the current velocity command visualization.

        Draws:
        - Green arrow: forward velocity (lin_vel_x)
        - Blue arrow: lateral velocity (lin_vel_y)
        - Red arc: angular velocity (ang_vel_z)

        Args:
            env_idx: Environment index to visualize.
            scale: Scale factor for the visualization.
            z_offset: Height offset above robot position.

        Returns:
            True if draw succeeded, False otherwise.
        """
        client = self.client
        if client is None:
            if not self._warned_no_client:
                print_info("[DebugVisualizer] No engine client available", "yellow")
                self._warned_no_client = True
            return False

        cmd = self.get_velocity_command(env_idx)
        if cmd is None:
            if not self._warned_no_command:
                print_info("[DebugVisualizer] No velocity command available", "yellow")
                self._warned_no_command = True
            return False

        origin = self.get_robot_position(env_idx, z_offset)
        lin_vel_x, lin_vel_y, ang_vel_z = cmd

        return client.draw_velocity_command(
            origin=origin,
            lin_vel_x=lin_vel_x,
            lin_vel_y=lin_vel_y,
            ang_vel_z=ang_vel_z,
            scale=scale,
            clear_previous=True,
        )

    def draw_arrow(
        self,
        origin: tuple[float, float, float],
        direction: tuple[float, float, float],
        color: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0),
        scale: float = 1.0,
        clear_previous: bool = False,
    ) -> bool:
        """Draw a debug arrow.

        Args:
            origin: (x, y, z) start position.
            direction: (x, y, z) direction and magnitude.
            color: (r, g, b, a) color values (0-1 range).
            scale: Scale factor.
            clear_previous: Clear previous debug draws.

        Returns:
            True if draw succeeded, False otherwise.
        """
        client = self.client
        if client is None:
            if not self._warned_no_client:
                print_info("[DebugVisualizer] No engine client available", "yellow")
                self._warned_no_client = True
            return False

        return client.draw_arrow(
            origin=origin,
            direction=direction,
            color=color,
            scale=scale,
            clear_previous=clear_previous,
        )

    def draw_line(
        self,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        clear_previous: bool = False,
    ) -> bool:
        """Draw a debug line.

        Args:
            start: (x, y, z) start position.
            end: (x, y, z) end position.
            color: (r, g, b, a) color values (0-1 range).
            clear_previous: Clear previous debug draws.

        Returns:
            True if draw succeeded, False otherwise.
        """
        client = self.client
        if client is None:
            if not self._warned_no_client:
                print_info("[DebugVisualizer] No engine client available", "yellow")
                self._warned_no_client = True
            return False

        return client.draw_line(
            start=start,
            end=end,
            color=color,
            clear_previous=clear_previous,
        )
