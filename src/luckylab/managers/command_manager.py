"""Command manager for generating and resampling velocity commands."""

import abc
from dataclasses import dataclass

import numpy as np


@dataclass
class VelocityCommandCfg:
    """Configuration for velocity command generation."""

    # Linear velocity ranges (m/s)
    lin_vel_x_range: tuple[float, float] = (-1.0, 1.0)
    lin_vel_y_range: tuple[float, float] = (-0.5, 0.5)

    # Angular velocity range (rad/s)
    ang_vel_z_range: tuple[float, float] = (-1.0, 1.0)

    # Heading range (rad) - optional, for heading-based tasks
    heading_range: tuple[float, float] = (-3.14, 3.14)

    # Resampling interval (seconds)
    resample_interval_range: tuple[float, float] = (5.0, 10.0)

    # Probability of zero command (standing still)
    zero_command_prob: float = 0.1

    # Scale factor (can be adjusted by curriculum)
    scale: float = 1.0


class CommandTerm(abc.ABC):
    """Base class for command generators."""

    def __init__(self, cfg: object):
        self.cfg = cfg
        self._command: np.ndarray | None = None
        self._time_left: float = 0.0

    @property
    @abc.abstractmethod
    def command(self) -> np.ndarray:
        """Current command vector."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def dim(self) -> int:
        """Dimension of command vector."""
        raise NotImplementedError

    @abc.abstractmethod
    def resample(self) -> None:
        """Sample a new command."""
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset command state (called on episode reset)."""
        raise NotImplementedError

    def update(self, dt: float) -> None:
        """Update command state (called every step)."""
        self._time_left -= dt
        if self._time_left <= 0:
            self.resample()


class VelocityCommand(CommandTerm):
    """
    Velocity command generator for locomotion tasks.

    Generates commands: [vx, vy, wz, heading]
    """

    def __init__(self, cfg: VelocityCommandCfg):
        super().__init__(cfg)
        self.cfg: VelocityCommandCfg = cfg
        self._command = np.zeros(4, dtype=np.float32)
        self._resample_interval()
        self.resample()

    @property
    def command(self) -> np.ndarray:
        return self._command

    @property
    def dim(self) -> int:
        return 4

    def resample(self) -> None:
        """Sample a new velocity command."""
        cfg = self.cfg

        # Check for zero command
        if np.random.random() < cfg.zero_command_prob:
            self._command = np.zeros(4, dtype=np.float32)
        else:
            self._command = np.array(
                [
                    np.random.uniform(*cfg.lin_vel_x_range) * cfg.scale,
                    np.random.uniform(*cfg.lin_vel_y_range) * cfg.scale,
                    np.random.uniform(*cfg.ang_vel_z_range) * cfg.scale,
                    np.random.uniform(*cfg.heading_range),
                ],
                dtype=np.float32,
            )

        self._resample_interval()

    def _resample_interval(self) -> None:
        """Sample a new resample interval."""
        self._time_left = np.random.uniform(*self.cfg.resample_interval_range)

    def reset(self) -> None:
        """Reset command on episode start."""
        self.resample()

    def set_velocity_range(self, max_velocity: float) -> None:
        """
        Update velocity ranges (for curriculum learning).

        Args:
            max_velocity: Maximum velocity magnitude
        """
        self.cfg.lin_vel_x_range = (-max_velocity, max_velocity)
        self.cfg.lin_vel_y_range = (-max_velocity * 0.5, max_velocity * 0.5)
        self.cfg.ang_vel_z_range = (-max_velocity, max_velocity)

    def set_scale(self, scale: float) -> None:
        """Set command scale (for curriculum learning)."""
        self.cfg.scale = scale


class CommandManager:
    """
    Manages multiple command generators.

    Provides a unified interface for command generation and updates.
    """

    def __init__(self):
        self._commands: dict[str, CommandTerm] = {}

    def add_command(self, name: str, command: CommandTerm) -> None:
        """Add a command generator."""
        self._commands[name] = command

    def get_command(self, name: str) -> np.ndarray:
        """Get current command by name."""
        if name not in self._commands:
            raise KeyError(f"Command '{name}' not found. Available: {list(self._commands.keys())}")
        return self._commands[name].command

    def get_all_commands(self) -> dict[str, np.ndarray]:
        """Get all current commands."""
        return {name: cmd.command for name, cmd in self._commands.items()}

    def update(self, dt: float) -> None:
        """Update all commands (call every step)."""
        for cmd in self._commands.values():
            cmd.update(dt)

    def reset(self) -> None:
        """Reset all commands (call on episode reset)."""
        for cmd in self._commands.values():
            cmd.reset()

    def resample(self, name: str | None = None) -> None:
        """
        Force resample command(s).

        Args:
            name: Command name to resample, or None for all
        """
        if name is not None:
            self._commands[name].resample()
        else:
            for cmd in self._commands.values():
                cmd.resample()

    @property
    def command_dims(self) -> dict[str, int]:
        """Get dimensions of all commands."""
        return {name: cmd.dim for name, cmd in self._commands.items()}

    def __getitem__(self, name: str) -> CommandTerm:
        """Get command term by name."""
        return self._commands[name]

    def __contains__(self, name: str) -> bool:
        """Check if command exists."""
        return name in self._commands
