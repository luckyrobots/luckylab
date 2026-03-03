"""Logging utilities for colored terminal output and training metrics."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from luckylab.rl.config import RlRunnerCfg

# ANSI color codes
COLORS = {
    "green": "\033[92m",
    "red": "\033[91m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "cyan": "\033[96m",
    "magenta": "\033[95m",
    "white": "\033[97m",
    "gray": "\033[90m",
    "bold": "\033[1m",
    "reset": "\033[0m",
}


def _supports_color() -> bool:
    """Check if terminal supports color output."""
    return sys.stdout.isatty()


def colorize(text: str, color: str) -> str:
    """Colorize text if terminal supports it.

    Args:
        text: Text to colorize.
        color: Color name from COLORS dict.

    Returns:
        Colorized text or original if color not supported.
    """
    if _supports_color() and color in COLORS:
        return f"{COLORS[color]}{text}{COLORS['reset']}"
    return text


def print_info(message: str, color: str = "green") -> None:
    """Print information message with color.

    Args:
        message: The message to print.
        color: Color name ('green', 'red', 'yellow', 'blue', 'cyan', 'magenta').
    """
    print(colorize(message, color))


def print_header(title: str, width: int = 60) -> None:
    """Print a formatted header.

    Args:
        title: Header title.
        width: Total width of header line.
    """
    padding = (width - len(title) - 2) // 2
    line = "=" * padding + f" {title} " + "=" * padding
    if len(line) < width:
        line += "="
    print(colorize(line, "cyan"))


def print_separator(char: str = "-", width: int = 60) -> None:
    """Print a separator line.

    Args:
        char: Character to use for separator.
        width: Width of separator line.
    """
    print(colorize(char * width, "gray"))


def format_metric(name: str, value: float, width: int = 40) -> str:
    """Format a metric for display.

    Args:
        name: Metric name.
        value: Metric value.
        width: Total width for name column.

    Returns:
        Formatted metric string.
    """
    use_scientific = (abs(value) < 0.0001 and value != 0) or abs(value) >= 1000
    value_str = f"{value:.2e}" if use_scientific else f"{value:.4f}"
    return f"  {name:<{width}} {value_str:>12}"


def format_metrics_table(
    metrics: dict[str, float],
    title: str | None = None,
    group_by_prefix: bool = True,
) -> str:
    """Format metrics as a readable table.

    Args:
        metrics: Dictionary of metric names to values.
        title: Optional title for the table.
        group_by_prefix: Whether to group metrics by their prefix (e.g., Episode_Reward/).

    Returns:
        Formatted table string.
    """
    lines = []

    if title:
        lines.append(colorize(f"{'=' * 20} {title} {'=' * 20}", "cyan"))

    if group_by_prefix:
        # Group metrics by prefix
        groups: dict[str, dict[str, float]] = {}
        for name, value in sorted(metrics.items()):
            if "/" in name:
                prefix, suffix = name.split("/", 1)
                if prefix not in groups:
                    groups[prefix] = {}
                groups[prefix][suffix] = value
            else:
                if "_ungrouped" not in groups:
                    groups["_ungrouped"] = {}
                groups["_ungrouped"][name] = value

        # Format each group
        for group_name, group_metrics in sorted(groups.items()):
            if group_name != "_ungrouped":
                lines.append(colorize(f"\n  [{group_name}]", "yellow"))
            for name, value in sorted(group_metrics.items()):
                lines.append(format_metric(name, value))
    else:
        for name, value in sorted(metrics.items()):
            lines.append(format_metric(name, value))

    return "\n".join(lines)


def print_episode_summary(
    episode_num: int,
    total_reward: float,
    episode_length: int,
    terminated: bool,
    truncated: bool,
    termination_reason: str = "",
    extra_metrics: dict[str, float] | None = None,
) -> None:
    """Print a summary of a completed episode.

    Args:
        episode_num: Episode number.
        total_reward: Total episode reward.
        episode_length: Number of steps in episode.
        terminated: Whether episode was terminated (failure).
        truncated: Whether episode was truncated (timeout).
        termination_reason: Reason for termination if applicable.
        extra_metrics: Additional metrics to display.
    """
    status = "TRUNCATED" if truncated else ("TERMINATED" if terminated else "RUNNING")
    status_color = "yellow" if truncated else ("red" if terminated else "green")

    print_separator()
    print(f"  Episode {colorize(str(episode_num), 'bold')}")
    print(f"  Status:  {colorize(status, status_color)}")
    if terminated and termination_reason:
        print(f"  Reason:  {termination_reason}")
    print(f"  Reward:  {colorize(f'{total_reward:.2f}', 'green' if total_reward > 0 else 'red')}")
    print(f"  Length:  {episode_length} steps")

    if extra_metrics:
        print()
        for name, value in extra_metrics.items():
            print(format_metric(name, value))

    print_separator()


def print_training_iteration(
    iteration: int,
    metrics: dict[str, float],
    fps: float | None = None,
    total_steps: int | None = None,
) -> None:
    """Print training iteration summary.

    Args:
        iteration: Current iteration number.
        metrics: Dictionary of metrics to display.
        fps: Frames per second if available.
        total_steps: Total environment steps if available.
    """
    header = f"Iteration {iteration}"
    if total_steps is not None:
        header += f" | Steps: {total_steps:,}"
    if fps is not None:
        header += f" | FPS: {fps:.0f}"

    print_header(header)
    print(format_metrics_table(metrics, group_by_prefix=True))
    print()


class WandbLogger:
    """Context manager for wandb logging that hooks into skrl agents."""

    def __init__(self, cfg: RlRunnerCfg, experiment_name: str):
        self.cfg = cfg
        self.experiment_name = experiment_name
        self._agent = None
        self._original_write = None

    @property
    def enabled(self) -> bool:
        return self.cfg.wandb

    def __enter__(self):
        if not self.enabled:
            return self
        import wandb

        kwargs = {"project": self.cfg.wandb_project, "name": self.experiment_name}
        if self.cfg.wandb_entity:
            kwargs["entity"] = self.cfg.wandb_entity
        wandb.init(**kwargs)
        print_info(f"Logging to wandb: {self.cfg.wandb_project}")
        return self

    def attach(self, agent, env=None):
        """Hook into agent.write_tracking_data to push metrics to wandb.

        Args:
            agent: The skrl agent.
            env: Optional SkrlWrapper to pull episode info from.
        """
        if not self.enabled:
            return
        import numpy as np

        import wandb

        self._agent = agent
        self._env = env
        self._original_write = agent.write_tracking_data

        def _clean_key(k: str) -> str:
            """Strip skrl's ' / ' group prefixes for cleaner wandb metric names."""
            if " / " in k:
                prefix, rest = k.split(" / ", 1)
                return f"{prefix}/{rest}"
            return k

        def _write(timestep: int, timesteps: int) -> None:
            metrics = {}
            for k, v in agent.tracking_data.items():
                clean = _clean_key(k)
                if k.endswith("(min)"):
                    metrics[clean] = np.min(v)
                elif k.endswith("(max)"):
                    metrics[clean] = np.max(v)
                else:
                    metrics[clean] = np.mean(v)
            # Log cached episode info from the environment wrapper.
            if env is not None and hasattr(env, "_last_episode_info"):
                for k, v in env._last_episode_info.items():
                    if isinstance(v, (int, float)):
                        metrics[f"Info/{k}"] = v
            if metrics:
                wandb.log(metrics, step=timestep)
            self._original_write(timestep, timesteps)

        agent.write_tracking_data = _write

    def __exit__(self, *exc):
        if not self.enabled:
            return
        if self._agent and self._original_write:
            self._agent.write_tracking_data = self._original_write
        import wandb

        wandb.finish()
