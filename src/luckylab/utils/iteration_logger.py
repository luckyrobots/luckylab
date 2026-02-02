"""Iteration-based training logger following Isaac Lab / mjlab pattern.

This module provides logging that matches the RSL-RL / Isaac Lab training output format,
showing per-iteration metrics including losses, rewards, and timing information.
"""

from __future__ import annotations

import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    "dim": "\033[2m",
    "reset": "\033[0m",
}


def _supports_color() -> bool:
    """Check if terminal supports color output."""
    return sys.stdout.isatty()


def _c(text: str, color: str) -> str:
    """Colorize text if terminal supports it."""
    if _supports_color() and color in COLORS:
        return f"{COLORS[color]}{text}{COLORS['reset']}"
    return text


@dataclass
class IterationLoggerCfg:
    """Configuration for the iteration logger."""

    # Output settings
    log_dir: str | Path | None = None
    """Directory for log files."""

    log_interval: int = 10
    """Print to console every N iterations."""

    save_interval: int = 100
    """Save checkpoint every N iterations."""

    # Metric aggregation
    episode_window: int = 100
    """Window size for computing mean episode statistics."""

    # Backend settings
    logger: str = "wandb"
    """Logger backend: 'wandb' or 'none'."""

    wandb_project: str = "luckylab"
    """Wandb project name."""

    wandb_entity: str | None = None
    """Wandb entity."""

    experiment_name: str = "default"
    """Experiment name for organizing logs."""


class IterationLogger:
    """Training logger that outputs per-iteration metrics.

    Follows the Isaac Lab / RSL-RL logging format with:
    - Iteration progress
    - Computation metrics (FPS, timing)
    - Loss metrics (value, policy, entropy)
    - Episode statistics (mean reward, length)
    - Timing (elapsed, ETA)

    Example:
        >>> logger = IterationLogger(cfg, max_iterations=5000)
        >>> for it in range(max_iterations):
        ...     # Training step
        ...     losses = algorithm.update()
        ...     logger.log_iteration(
        ...         iteration=it,
        ...         losses=losses,
        ...         episode_rewards=rewards,
        ...         episode_lengths=lengths,
        ...         collection_time=0.1,
        ...         learn_time=0.05,
        ...     )
        >>> logger.close()
    """

    def __init__(self, cfg: IterationLoggerCfg, max_iterations: int) -> None:
        """Initialize the iteration logger.

        Args:
            cfg: Logger configuration.
            max_iterations: Total number of training iterations.
        """
        self.cfg = cfg
        self.max_iterations = max_iterations

        # Timing
        self.start_time = time.time()
        self.iteration_start_time = self.start_time

        # Episode tracking (rolling window)
        self._episode_rewards: deque[float] = deque(maxlen=cfg.episode_window)
        self._episode_lengths: deque[float] = deque(maxlen=cfg.episode_window)

        # Step tracking
        self.total_timesteps = 0
        self.current_iteration = 0

        # Custom metrics from environment
        self._env_metrics: dict[str, deque[float]] = {}

        # Setup log directory
        self.log_dir: Path | None = None
        if cfg.log_dir is not None:
            self.log_dir = Path(cfg.log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize backends
        self._writer = None
        self._wandb_run = None

        if cfg.logger == "tensorboard" and self.log_dir:
            self._init_tensorboard()
        elif cfg.logger == "wandb":
            self._init_wandb()

    def _init_tensorboard(self) -> None:
        """Initialize TensorBoard writer."""
        try:
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(log_dir=str(self.log_dir))
        except ImportError:
            print("Warning: tensorboard not installed, disabling tensorboard logging")

    def _init_wandb(self, config: dict | None = None) -> None:
        """Initialize Weights & Biases."""
        try:
            import wandb

            self._wandb_run = wandb.init(
                project=self.cfg.wandb_project,
                entity=self.cfg.wandb_entity,
                name=self.cfg.experiment_name,
                dir=str(self.log_dir) if self.log_dir else None,
                config=config,
            )
        except ImportError:
            print("Warning: wandb not installed, disabling wandb logging")

    def log_config(self, config: dict) -> None:
        """Log hyperparameters/config to wandb."""
        if self._wandb_run:
            import wandb

            wandb.config.update(config, allow_val_change=True)

    def log_iteration(
        self,
        iteration: int,
        losses: dict[str, float] | None = None,
        episode_rewards: list[float] | None = None,
        episode_lengths: list[float] | None = None,
        num_steps: int = 0,
        collection_time: float = 0.0,
        learn_time: float = 0.0,
        action_std: float | None = None,
        env_metrics: dict[str, float] | None = None,
    ) -> None:
        """Log metrics for a training iteration.

        Args:
            iteration: Current iteration number.
            losses: Dictionary of loss values (e.g., value_function, surrogate, entropy).
            episode_rewards: List of episode rewards completed this iteration.
            episode_lengths: List of episode lengths completed this iteration.
            num_steps: Number of environment steps taken this iteration.
            collection_time: Time spent collecting rollout data.
            learn_time: Time spent updating policy.
            action_std: Mean action standard deviation (exploration noise).
            env_metrics: Custom metrics from the environment (from extras["log"]).
        """
        self.current_iteration = iteration
        self.total_timesteps += num_steps
        iteration_time = time.time() - self.iteration_start_time
        self.iteration_start_time = time.time()

        # Update episode statistics
        if episode_rewards:
            self._episode_rewards.extend(episode_rewards)
        if episode_lengths:
            self._episode_lengths.extend(episode_lengths)

        # Update env metrics
        if env_metrics:
            for name, value in env_metrics.items():
                if name not in self._env_metrics:
                    self._env_metrics[name] = deque(maxlen=self.cfg.episode_window)
                self._env_metrics[name].append(value)

        # Compute statistics
        mean_reward = (
            sum(self._episode_rewards) / len(self._episode_rewards) if self._episode_rewards else 0.0
        )
        mean_length = (
            sum(self._episode_lengths) / len(self._episode_lengths) if self._episode_lengths else 0.0
        )

        # Compute FPS
        total_time = collection_time + learn_time
        fps = num_steps / total_time if total_time > 0 else 0.0

        # Prepare metrics for logging backends
        metrics: dict[str, float] = {}

        # Loss metrics
        if losses:
            for name, value in losses.items():
                metrics[f"Loss/{name}"] = value

        # Training metrics
        metrics["Train/mean_reward"] = mean_reward
        metrics["Train/mean_episode_length"] = mean_length
        if action_std is not None:
            metrics["Train/action_std"] = action_std

        # Performance metrics
        metrics["Perf/total_fps"] = fps
        metrics["Perf/collection_time"] = collection_time
        metrics["Perf/learning_time"] = learn_time

        # Env metrics (grouped by prefix)
        if env_metrics:
            for name, value in env_metrics.items():
                metrics[name] = value

        # Log to backends
        self._log_backends(metrics, iteration)

        # Print to console
        if iteration % self.cfg.log_interval == 0:
            self._print_iteration(
                iteration=iteration,
                fps=fps,
                collection_time=collection_time,
                learn_time=learn_time,
                losses=losses or {},
                mean_reward=mean_reward,
                mean_length=mean_length,
                action_std=action_std,
                iteration_time=iteration_time,
                env_metrics=env_metrics,
            )

    def _log_backends(self, metrics: dict[str, float], step: int) -> None:
        """Log metrics to TensorBoard and/or Wandb."""
        if self._writer:
            for name, value in metrics.items():
                self._writer.add_scalar(name, value, step)

        if self._wandb_run:
            import wandb

            wandb.log(metrics, step=step)

    def _print_iteration(
        self,
        iteration: int,
        fps: float,
        collection_time: float,
        learn_time: float,
        losses: dict[str, float],
        mean_reward: float,
        mean_length: float,
        action_std: float | None,
        iteration_time: float,
        env_metrics: dict[str, float] | None,
    ) -> None:
        """Print iteration summary to console with colors."""
        elapsed = time.time() - self.start_time
        remaining_iterations = self.max_iterations - iteration
        eta = (elapsed / max(iteration, 1)) * remaining_iterations if iteration > 0 else 0
        progress_pct = (iteration / self.max_iterations) * 100

        # Header
        print()
        print(_c("=" * 80, "blue"))
        progress_str = f"{progress_pct:.1f}%"
        header = f"Learning iteration {_c(str(iteration), 'bold')}/{self.max_iterations} [{_c(progress_str, 'cyan')}]"
        # Center the header
        padding = (80 - len(f"Learning iteration {iteration}/{self.max_iterations} [{progress_str}]")) // 2
        print(" " * padding + header)
        print(_c("=" * 80, "blue"))

        # Computation section
        print()
        print(_c("  Computation", "bold"))
        fps_color = "green" if fps > 100 else "yellow" if fps > 50 else "red"
        print(f"    Steps/sec:                       {_c(f'{fps:.0f}', fps_color)}")
        print(f"    Collection time:                 {_c(f'{collection_time:.3f}s', 'dim')}")
        if learn_time > 0:
            print(f"    Learning time:                   {_c(f'{learn_time:.3f}s', 'dim')}")

        # Action std
        if action_std is not None:
            print(f"    Action noise std:                {_c(f'{action_std:.4f}', 'cyan')}")

        # Losses section
        if losses:
            print()
            print(_c("  Losses", "bold"))
            for name, value in sorted(losses.items()):
                # Shorten the name for cleaner display
                short_name = name.replace("Loss / ", "").replace(" loss", "")
                print(f"    {short_name + ':':<32} {_c(f'{value:.6f}', 'magenta')}")

        # Episode stats section
        print()
        print(_c("  Episode Stats", "bold"))
        reward_color = "green" if mean_reward > 0 else "red"
        print(f"    Mean reward:                     {_c(f'{mean_reward:.2f}', reward_color)}")
        print(f"    Mean episode length:             {_c(f'{mean_length:.1f}', 'cyan')}")

        # Environment metrics (grouped by prefix)
        if env_metrics:
            groups: dict[str, list[tuple[str, float]]] = {}
            for name, value in sorted(env_metrics.items()):
                if "/" in name:
                    prefix, suffix = name.split("/", 1)
                    if prefix not in groups:
                        groups[prefix] = []
                    groups[prefix].append((suffix, value))

            for group_name, group_metrics in sorted(groups.items()):
                if group_name not in ("Episode", "Time", "Rolling"):
                    print()
                    print(_c(f"  {group_name}", "bold"))
                    for name, value in group_metrics:
                        print(f"    {name + ':':<32} {_c(f'{value:.4f}', 'cyan')}")

        # Footer with timing
        print()
        print(_c("-" * 80, "dim"))
        print(f"    Total timesteps:                 {_c(f'{self.total_timesteps:,}', 'bold')}")
        print(f"    Iteration time:                  {_c(f'{iteration_time:.2f}s', 'dim')}")
        print(f"    Elapsed:                         {_c(self._format_time(elapsed), 'cyan')}")
        if eta >= 0:
            print(f"    ETA:                             {_c(self._format_time(eta), 'green')}")

    def _format_time(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def log_episode(self, reward: float, length: int, env_log: dict[str, float] | None = None) -> None:
        """Log a completed episode.

        Call this when an episode ends to track episode statistics.

        Args:
            reward: Total episode reward.
            length: Episode length in steps.
            env_log: Environment log dict (from info["log"]).
        """
        self._episode_rewards.append(reward)
        self._episode_lengths.append(float(length))

        if env_log:
            for name, value in env_log.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if name not in self._env_metrics:
                        self._env_metrics[name] = deque(maxlen=self.cfg.episode_window)
                    self._env_metrics[name].append(float(value))

    def get_mean_reward(self) -> float:
        """Get mean reward over the episode window."""
        if not self._episode_rewards:
            return 0.0
        return sum(self._episode_rewards) / len(self._episode_rewards)

    def get_mean_length(self) -> float:
        """Get mean episode length over the window."""
        if not self._episode_lengths:
            return 0.0
        return sum(self._episode_lengths) / len(self._episode_lengths)

    def get_env_metric_mean(self, name: str) -> float | None:
        """Get mean value for an environment metric."""
        if name in self._env_metrics and len(self._env_metrics[name]) > 0:
            return sum(self._env_metrics[name]) / len(self._env_metrics[name])
        return None

    def save_checkpoint(self, model_state: Any, filename: str | None = None) -> Path | None:
        """Save a checkpoint file.

        Args:
            model_state: Model state dict to save.
            filename: Optional filename. Defaults to model_{iteration}.pt.

        Returns:
            Path to saved checkpoint or None if no log_dir.
        """
        if self.log_dir is None:
            return None

        import torch

        if filename is None:
            filename = f"model_{self.current_iteration}.pt"

        checkpoint_path = self.log_dir / filename
        torch.save(model_state, checkpoint_path)
        return checkpoint_path

    def close(self) -> None:
        """Close all logging backends."""
        if self._writer:
            self._writer.close()

        if self._wandb_run:
            import wandb

            wandb.finish()

        # Print final summary with colors
        elapsed = time.time() - self.start_time
        avg_steps_per_iter = self.total_timesteps / max(self.current_iteration, 1)

        print()
        print(_c("=" * 80, "green"))
        complete_text = "Training Complete"
        padding = (80 - len(complete_text)) // 2
        print(" " * padding + _c(complete_text, "green"))
        print(_c("=" * 80, "green"))
        print()
        print(_c("  Summary", "bold"))
        print(f"    Total iterations:                {_c(f'{self.current_iteration:,}', 'bold')}")
        print(f"    Total timesteps:                 {_c(f'{self.total_timesteps:,}', 'bold')}")
        print(f"    Avg steps/iteration:             {_c(f'{avg_steps_per_iter:.1f}', 'cyan')}")
        print(f"    Total time:                      {_c(self._format_time(elapsed), 'cyan')}")
        if elapsed > 0:
            print(
                f"    Average FPS:                     {_c(f'{self.total_timesteps / elapsed:.0f}', 'green')}"
            )

        if self._episode_rewards:
            print()
            print(_c("  Final Episode Stats", "bold"))
            reward_color = "green" if self.get_mean_reward() > 0 else "red"
            print(f"    Mean reward:                     {_c(f'{self.get_mean_reward():.2f}', reward_color)}")
            print(f"    Mean episode length:             {_c(f'{self.get_mean_length():.1f}', 'cyan')}")
            print(f"    Total episodes:                  {_c(f'{len(self._episode_rewards)}', 'cyan')}")

        print()
        print(_c("=" * 80, "green"))
