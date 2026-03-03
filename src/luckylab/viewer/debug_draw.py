"""Debug visualization for LuckyEngine.

Provides utilities for drawing debug primitives (lines, arrows)
in the LuckyEngine viewport during training or evaluation.

Arrow drawing is delegated to command terms via their ``_debug_vis_impl``
methods — this class only exposes the low-level drawing primitives and
a frame-rate throttle.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from luckylab.utils.logging import print_info

if TYPE_CHECKING:
    from luckyrobots import LuckyEngineClient

    from luckylab.envs import ManagerBasedRlEnv


class DebugVisualizer:
    """Handles debug visualization in LuckyEngine.

    Usage:
        viz = DebugVisualizer(env)
        if viz.should_draw():
            viz.draw_arrow(origin, direction, color)
    """

    def __init__(self, env: "ManagerBasedRlEnv", draw_interval_ms: float = 50.0) -> None:
        """Initialize the debug visualizer.

        Args:
            env: The ManagerBasedRlEnv instance to visualize.
            draw_interval_ms: Minimum interval between draws in milliseconds.
                              Prevents flickering when physics steps faster than rendering.
        """
        self._env = env
        self._warned_no_client = False
        self._draw_interval_ms = draw_interval_ms
        self._last_draw_time = 0.0

    @property
    def client(self) -> "LuckyEngineClient | None":
        """Get the LuckyEngine gRPC client."""
        luckyrobots = getattr(self._env, "luckyrobots", None)
        if luckyrobots is None:
            return None
        return getattr(luckyrobots, "engine_client", None)

    def should_draw(self) -> bool:
        """Check if enough time has elapsed since the last draw.

        Returns:
            True if a draw should proceed, False if throttled.
        """
        current_time = time.perf_counter() * 1000.0  # ms
        if current_time - self._last_draw_time < self._draw_interval_ms:
            return False
        self._last_draw_time = current_time
        return True

    def draw_velocity_command(self) -> bool:
        """Draw the current velocity command as an arrow from the robot's position.

        Reads vel_command from EntityData and draws it using the engine's
        built-in velocity command visualization.

        Returns:
            True if draw succeeded, False otherwise.
        """
        client = self.client
        if client is None:
            return False

        robot = self._env.scene["robot"]
        cmd = robot.data.vel_command[0]
        pos = robot.data.root_link_pos_w[0]

        return client.draw_velocity_command(
            origin=(pos[0].item(), pos[1].item(), pos[2].item()),
            lin_vel_x=cmd[0].item(),
            lin_vel_y=cmd[1].item(),
            ang_vel_z=cmd[2].item(),
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
