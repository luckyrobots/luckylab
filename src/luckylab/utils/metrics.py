"""Metrics tracking and logging for training.

This module provides flexible, task-agnostic metrics logging that automatically
adapts to whatever rewards, terminations, and custom metrics are configured.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .logging import (
    colorize,
    format_metrics_table,
    print_header,
    print_separator,
)


@dataclass
class MetricsLoggerCfg:
    """Configuration for the metrics logger."""

    # Output settings
    log_dir: str | Path | None = None
    """Directory for log files. None = no file logging."""

    log_to_console: bool = True
    """Whether to print metrics to console."""

    log_to_file: bool = True
    """Whether to save metrics to JSON file."""

    log_to_wandb: bool = False
    """Whether to log to Weights & Biases."""

    log_to_tensorboard: bool = False
    """Whether to log to TensorBoard."""

    # Logging frequency
    console_interval: int = 1
    """Print to console every N episodes."""

    file_interval: int = 10
    """Save to file every N episodes."""

    # Metric aggregation
    window_size: int = 100
    """Rolling window size for computing averages."""

    # Wandb settings
    wandb_project: str = "luckylab"
    """Wandb project name."""

    wandb_entity: str | None = None
    """Wandb entity (username or team)."""

    wandb_run_name: str | None = None
    """Wandb run name."""

    # Display settings
    group_by_prefix: bool = True
    """Whether to group metrics by prefix (e.g., Episode_Reward/) in console output."""

    show_rolling_stats: bool = True
    """Whether to show rolling statistics (mean, min, max) for key metrics."""


class MetricsLogger:
    """Task-agnostic logger for training metrics.

    Automatically adapts to whatever metrics are provided - no hardcoded
    metric names. Works with any task configuration.

    The logger accepts arbitrary metric dictionaries and:
    - Groups them by prefix (Episode_Reward/, Episode_Termination/, etc.)
    - Tracks rolling averages for numeric metrics
    - Outputs to console, file, Wandb, and/or TensorBoard

    Example:
        >>> cfg = MetricsLoggerCfg(log_dir="logs/run_001")
        >>> logger = MetricsLogger(cfg)
        >>>
        >>> # After each episode, pass the env's log dict directly
        >>> obs, info = env.reset()
        >>> logger.log_episode(info.get("log", {}))
        >>>
        >>> # Or log custom metrics
        >>> logger.log_metrics({"Custom/my_metric": 1.0})
        >>>
        >>> logger.close()
    """

    def __init__(self, cfg: MetricsLoggerCfg) -> None:
        """Initialize the metrics logger.

        Args:
            cfg: Logger configuration.
        """
        self.cfg = cfg

        # Episode tracking
        self.episode_count = 0
        self.total_steps = 0

        # Rolling history for all metrics (metric_name -> deque of values)
        self._metric_history: dict[str, deque[float]] = {}

        # Full episode log history (for computing aggregate stats)
        self._episode_history: deque[dict[str, Any]] = deque(maxlen=cfg.window_size)

        # Timing
        self.start_time = time.time()
        self.last_log_time = self.start_time

        # Setup log directory
        self.log_dir: Path | None = None
        if cfg.log_dir is not None:
            self.log_dir = Path(cfg.log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize backends
        self._wandb_run = None
        self._tb_writer = None

        if cfg.log_to_wandb:
            self._init_wandb()

        if cfg.log_to_tensorboard and self.log_dir:
            self._init_tensorboard()

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases logging."""
        try:
            import wandb

            self._wandb_run = wandb.init(
                project=self.cfg.wandb_project,
                entity=self.cfg.wandb_entity,
                name=self.cfg.wandb_run_name,
                dir=str(self.log_dir) if self.log_dir else None,
            )
        except ImportError:
            print(colorize("Warning: wandb not installed, disabling wandb logging", "yellow"))
            self.cfg.log_to_wandb = False

    def _init_tensorboard(self) -> None:
        """Initialize TensorBoard logging."""
        try:
            from torch.utils.tensorboard import SummaryWriter

            self._tb_writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
        except ImportError:
            print(colorize("Warning: tensorboard not installed, disabling tensorboard logging", "yellow"))
            self.cfg.log_to_tensorboard = False

    def log_episode(self, log_dict: dict[str, Any], episode_length: int | None = None) -> None:
        """Log metrics from a completed episode.

        This is the main logging method. Pass the env's extras["log"] dict
        directly - the logger will automatically handle whatever metrics
        are present.

        Args:
            log_dict: Dictionary of metrics from the episode. Can contain any
                metrics - the logger adapts automatically. Common prefixes:
                - Episode_Reward/<term>: Reward term values
                - Episode_Termination/<term>: Termination term triggers
                - Episode/<metric>: Episode-level metrics
                - Custom/<metric>: Any custom metrics
            episode_length: Optional episode length. If not provided, looks for
                "Episode/length" in log_dict.
        """
        self.episode_count += 1

        # Extract episode length
        if episode_length is not None:
            self.total_steps += episode_length
        elif "Episode/length" in log_dict:
            self.total_steps += int(log_dict["Episode/length"])

        # Store in history
        self._episode_history.append(log_dict.copy())

        # Update rolling history for each metric
        for name, value in log_dict.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if name not in self._metric_history:
                    self._metric_history[name] = deque(maxlen=self.cfg.window_size)
                self._metric_history[name].append(float(value))

        # Add timing metrics
        elapsed = time.time() - self.start_time
        timing_metrics = {
            "Time/total_steps": float(self.total_steps),
            "Time/episodes": float(self.episode_count),
            "Time/fps": self.total_steps / elapsed if elapsed > 0 else 0.0,
            "Time/elapsed_s": elapsed,
        }

        # Combine all metrics for logging
        all_metrics = {**log_dict, **timing_metrics}

        # Add rolling averages if configured
        if self.cfg.show_rolling_stats:
            rolling_metrics = self._compute_rolling_stats()
            all_metrics.update(rolling_metrics)

        # Log to backends
        if self.cfg.log_to_console and self.episode_count % self.cfg.console_interval == 0:
            self._log_console(all_metrics)

        if self.cfg.log_to_file and self.log_dir and self.episode_count % self.cfg.file_interval == 0:
            self._log_file(log_dict)

        if self.cfg.log_to_wandb and self._wandb_run:
            self._log_wandb(all_metrics)

        if self.cfg.log_to_tensorboard and self._tb_writer:
            self._log_tensorboard(all_metrics)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log arbitrary metrics (e.g., training loss, entropy).

        Use this for step-level metrics that aren't tied to episodes.

        Args:
            metrics: Dictionary of metric names to values.
            step: Optional step number for x-axis. Defaults to total_steps.
        """
        step = step or self.total_steps

        for name, value in metrics.items():
            if name not in self._metric_history:
                self._metric_history[name] = deque(maxlen=self.cfg.window_size)
            self._metric_history[name].append(float(value))

        if self.cfg.log_to_wandb and self._wandb_run:
            import wandb
            wandb.log(metrics, step=step)

        if self.cfg.log_to_tensorboard and self._tb_writer:
            for name, value in metrics.items():
                self._tb_writer.add_scalar(name, value, step)

    def _compute_rolling_stats(self) -> dict[str, float]:
        """Compute rolling statistics for key metrics.

        Returns:
            Dictionary with rolling averages, prefixed with "Rolling/".
        """
        stats: dict[str, float] = {}

        # Compute rolling average for each metric
        for name, history in self._metric_history.items():
            if len(history) > 0:
                # Only compute rolling stats for episode-level metrics
                if name.startswith("Episode"):
                    avg = sum(history) / len(history)
                    # Use shorter key for rolling stats
                    short_name = name.split("/")[-1] if "/" in name else name
                    stats[f"Rolling/{short_name}"] = avg

        return stats

    def _log_console(self, metrics: dict[str, float]) -> None:
        """Log metrics to console with nice formatting."""
        elapsed = time.time() - self.start_time
        fps = self.total_steps / elapsed if elapsed > 0 else 0.0

        print_header(f"Episode {self.episode_count} | Steps: {self.total_steps:,} | FPS: {fps:.0f}")
        print(format_metrics_table(metrics, group_by_prefix=self.cfg.group_by_prefix))
        print()

    def _log_file(self, log_dict: dict[str, Any]) -> None:
        """Log episode data to JSON file."""
        if self.log_dir is None:
            return

        # Add metadata
        record = {
            "episode": self.episode_count,
            "total_steps": self.total_steps,
            "timestamp": time.time(),
            **log_dict,
        }

        log_file = self.log_dir / "episodes.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _log_wandb(self, metrics: dict[str, float]) -> None:
        """Log metrics to Weights & Biases."""
        import wandb

        # Filter to only numeric values
        numeric_metrics = {
            k: v for k, v in metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        wandb.log(numeric_metrics, step=self.total_steps)

    def _log_tensorboard(self, metrics: dict[str, float]) -> None:
        """Log metrics to TensorBoard."""
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self._tb_writer.add_scalar(name, value, self.total_steps)

    def get_rolling_average(self, metric_name: str) -> float | None:
        """Get rolling average for a specific metric.

        Args:
            metric_name: Name of the metric.

        Returns:
            Rolling average or None if metric not found.
        """
        if metric_name in self._metric_history and len(self._metric_history[metric_name]) > 0:
            history = self._metric_history[metric_name]
            return sum(history) / len(history)
        return None

    def get_metric_stats(self, metric_name: str) -> dict[str, float] | None:
        """Get statistics for a specific metric.

        Args:
            metric_name: Name of the metric.

        Returns:
            Dict with mean, min, max, std or None if not found.
        """
        if metric_name not in self._metric_history or len(self._metric_history[metric_name]) == 0:
            return None

        history = list(self._metric_history[metric_name])
        mean = sum(history) / len(history)
        min_val = min(history)
        max_val = max(history)

        # Compute std
        variance = sum((x - mean) ** 2 for x in history) / len(history)
        std = variance ** 0.5

        return {
            "mean": mean,
            "min": min_val,
            "max": max_val,
            "std": std,
            "count": len(history),
        }

    def get_all_metrics(self) -> dict[str, float]:
        """Get current rolling averages for all tracked metrics.

        Returns:
            Dictionary of metric names to their rolling averages.
        """
        return {
            name: sum(history) / len(history)
            for name, history in self._metric_history.items()
            if len(history) > 0
        }

    def print_summary(self) -> None:
        """Print a summary of training so far."""
        elapsed = time.time() - self.start_time

        print_header("Training Summary")
        print(f"  Total Episodes: {colorize(str(self.episode_count), 'bold')}")
        print(f"  Total Steps:    {colorize(f'{self.total_steps:,}', 'bold')}")
        print(f"  Elapsed Time:   {elapsed:.1f}s")
        if elapsed > 0:
            print(f"  Average FPS:    {self.total_steps / elapsed:.0f}")

        # Show rolling stats for key metrics
        if len(self._metric_history) > 0:
            print()
            print(colorize("  [Rolling Statistics]", "yellow"))

            # Find episode-level metrics
            episode_metrics = {
                k: v for k, v in self._metric_history.items()
                if k.startswith("Episode") and len(v) > 0
            }

            for name, history in sorted(episode_metrics.items()):
                mean = sum(history) / len(history)
                min_val = min(history)
                max_val = max(history)
                short_name = name.split("/")[-1] if "/" in name else name
                print(f"    {short_name}: mean={mean:.2f}, min={min_val:.2f}, max={max_val:.2f}")

        print_separator()

    def close(self) -> None:
        """Close all logging backends and print summary."""
        self.print_summary()

        if self._wandb_run:
            import wandb
            wandb.finish()

        if self._tb_writer:
            self._tb_writer.close()


# Convenience function for quick logging setup
def create_logger(
    log_dir: str | Path | None = None,
    use_wandb: bool = False,
    use_tensorboard: bool = False,
    wandb_project: str = "luckylab",
    console_interval: int = 1,
) -> MetricsLogger:
    """Create a metrics logger with common settings.

    Args:
        log_dir: Directory for log files.
        use_wandb: Enable Wandb logging.
        use_tensorboard: Enable TensorBoard logging.
        wandb_project: Wandb project name.
        console_interval: Print every N episodes.

    Returns:
        Configured MetricsLogger instance.
    """
    cfg = MetricsLoggerCfg(
        log_dir=log_dir,
        log_to_wandb=use_wandb,
        log_to_tensorboard=use_tensorboard,
        wandb_project=wandb_project,
        console_interval=console_interval,
    )
    return MetricsLogger(cfg)
