"""NaN guard for detecting and debugging NaN/Inf in RL training.

Provides comprehensive NaN detection at multiple points:
- Policy outputs (actions from neural network)
- Processed actions (after scaling/offset)
- Observations from environment
- Rewards

When NaN is detected, can either:
- Replace with safe values (recovery mode)
- Halt training and dump diagnostics
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    pass


@dataclass
class NanGuardCfg:
    """Configuration for NaN guard.

    Attributes:
        enabled: Whether to enable NaN detection.
        buffer_size: Number of states to keep in rolling buffer.
        output_dir: Directory to write dumps to.
        recovery_mode: If True, replace NaNs with safe values instead of halting.
        halt_on_nan: If True, raise exception when NaN detected (after dumping).
        check_actions: Check policy outputs for NaN.
        check_observations: Check observations for NaN.
        check_rewards: Check rewards for NaN.
        verbose: Print warnings when NaN detected.
    """

    enabled: bool = False
    buffer_size: int = 100
    output_dir: str = "/tmp/luckylab/nan_dumps"
    recovery_mode: bool = True
    halt_on_nan: bool = False
    check_actions: bool = True
    check_observations: bool = True
    check_rewards: bool = True
    verbose: bool = True


@dataclass
class NanStats:
    """Statistics about NaN occurrences."""
    total_checks: int = 0
    action_nans: int = 0
    observation_nans: int = 0
    reward_nans: int = 0
    recoveries: int = 0
    first_nan_step: int | None = None
    last_nan_step: int | None = None


class NanGuard:
    """Guards against NaN/Inf by buffering states and dumping on detection.

    When enabled, maintains a rolling buffer of observations and actions,
    and writes them to disk when NaN or Inf is detected. When disabled,
    all operations are no-ops with minimal overhead.

    Example:
        >>> from luckylab.utils import NanGuard, NanGuardCfg
        >>>
        >>> cfg = NanGuardCfg(enabled=True, buffer_size=50)
        >>> guard = NanGuard(cfg)
        >>>
        >>> # During training loop:
        >>> with guard.watch(obs, action):
        ...     obs, reward, terminated, truncated, info = env.step(action)
        ...     guard.check_and_dump(obs)
    """

    def __init__(self, cfg: NanGuardCfg | None = None) -> None:
        if cfg is None:
            cfg = NanGuardCfg()

        self.cfg = cfg
        self.enabled = cfg.enabled
        self.step_counter = 0

        if not self.enabled:
            return

        self.buffer_size = cfg.buffer_size
        self.output_dir = Path(cfg.output_dir)
        self.buffer: deque[dict[str, Any]] = deque(maxlen=self.buffer_size)
        self._dumped = False
        self.stats = NanStats()

    def capture(
        self,
        observation: np.ndarray | None = None,
        action: np.ndarray | None = None,
        reward: float | None = None,
        info: dict[str, Any] | None = None,
    ) -> None:
        """Capture current state to buffer.

        Args:
            observation: Current observation array.
            action: Current action array.
            reward: Current reward value.
            info: Additional info dictionary.
        """
        if not self.enabled:
            return

        state = {
            "step": self.step_counter,
            "observation": observation.copy() if observation is not None else None,
            "action": action.copy() if action is not None else None,
            "reward": reward,
            "info": info.copy() if info is not None else None,
        }
        self.buffer.append(state)
        self.step_counter += 1

    @contextmanager
    def watch(
        self,
        observation: np.ndarray | None = None,
        action: np.ndarray | None = None,
    ) -> Iterator[None]:
        """Context manager that captures state before and checks for NaN/Inf after.

        Usage:
            with nan_guard.watch(obs, action):
                obs, reward, terminated, truncated, info = env.step(action)
        """
        self.capture(observation=observation, action=action)
        yield

    @staticmethod
    def detect_nans(data: np.ndarray) -> bool:
        """Detect NaN/Inf values in an array.

        Args:
            data: NumPy array to check.

        Returns:
            True if NaN or Inf values are present.
        """
        return bool(np.any(np.isnan(data)) or np.any(np.isinf(data)))

    def check_and_dump(
        self,
        observation: np.ndarray | None = None,
        reward: float | None = None,
    ) -> bool:
        """Check for NaN/Inf and dump buffer if detected.

        Args:
            observation: Observation to check for NaN/Inf.
            reward: Reward to check for NaN/Inf.

        Returns:
            True if NaN/Inf detected and dump occurred, False otherwise.
        """
        if not self.enabled or self._dumped:
            return False

        has_nan = False

        if observation is not None and self.detect_nans(observation):
            has_nan = True

        if reward is not None and (np.isnan(reward) or np.isinf(reward)):
            has_nan = True

        if has_nan:
            self._dump_buffer()
            self._dumped = True
            return True

        return False

    def _dump_buffer(self) -> None:
        """Write buffered states to disk."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"nan_dump_{timestamp}.npz"

        data = {}
        for item in self.buffer:
            step = item["step"]
            if item["observation"] is not None:
                data[f"obs_step_{step:06d}"] = item["observation"]
            if item["action"] is not None:
                data[f"action_step_{step:06d}"] = item["action"]
            if item["reward"] is not None:
                data[f"reward_step_{step:06d}"] = np.array([item["reward"]])

        data["_metadata"] = np.array(
            {
                "buffer_size": len(self.buffer),
                "detection_step": self.step_counter,
                "timestamp": timestamp,
            },
            dtype=object,
        )

        np.savez_compressed(filename, **data)

        # Create symlink to latest dump.
        latest_dump = self.output_dir / "nan_dump_latest.npz"
        latest_dump.unlink(missing_ok=True)
        latest_dump.symlink_to(filename.name)

        print(f"[NanGuard] Detected NaN/Inf at step {self.step_counter}")
        print(f"[NanGuard] Dumped {len(self.buffer)} states to: {filename}")
        print(f"[NanGuard] Latest dump symlinked at: {latest_dump}")

    def reset(self) -> None:
        """Reset the guard state (allows dumping again after reset)."""
        self._dumped = False
        self.buffer.clear()
        self.step_counter = 0

    def check_tensor(
        self,
        tensor: torch.Tensor | np.ndarray,
        name: str = "tensor",
        replace_value: float = 0.0,
    ) -> tuple[torch.Tensor | np.ndarray, bool]:
        """Check a tensor for NaN/Inf and optionally replace.

        Args:
            tensor: Tensor to check.
            name: Name for logging.
            replace_value: Value to replace NaN/Inf with in recovery mode.

        Returns:
            Tuple of (sanitized tensor, had_nan).
        """
        if not self.enabled:
            return tensor, False

        is_torch = isinstance(tensor, torch.Tensor)

        if is_torch:
            has_nan = torch.isnan(tensor).any().item()
            has_inf = torch.isinf(tensor).any().item()
        else:
            has_nan = bool(np.any(np.isnan(tensor)))
            has_inf = bool(np.any(np.isinf(tensor)))

        if not (has_nan or has_inf):
            return tensor, False

        # Record stats
        if self.stats.first_nan_step is None:
            self.stats.first_nan_step = self.step_counter
        self.stats.last_nan_step = self.step_counter

        if self.cfg.verbose:
            if is_torch:
                nan_count = torch.isnan(tensor).sum().item()
                inf_count = torch.isinf(tensor).sum().item()
            else:
                nan_count = np.sum(np.isnan(tensor))
                inf_count = np.sum(np.isinf(tensor))
            print(f"[NanGuard] {name}: {nan_count} NaN, {inf_count} Inf at step {self.step_counter}")

        if self.cfg.recovery_mode:
            # Replace NaN/Inf with safe values
            self.stats.recoveries += 1
            if is_torch:
                tensor = tensor.clone()
                tensor = torch.where(torch.isnan(tensor) | torch.isinf(tensor),
                                     torch.full_like(tensor, replace_value), tensor)
            else:
                tensor = tensor.copy()
                tensor = np.where(np.isnan(tensor) | np.isinf(tensor), replace_value, tensor)

        return tensor, True

    def check_actions(
        self,
        actions: torch.Tensor,
        replace_value: float = 0.0,
    ) -> tuple[torch.Tensor, bool]:
        """Check actions for NaN/Inf.

        Args:
            actions: Action tensor from policy.
            replace_value: Value to use for recovery.

        Returns:
            Tuple of (sanitized actions, had_nan).
        """
        if not self.enabled or not self.cfg.check_actions:
            return actions, False

        self.stats.total_checks += 1
        actions, had_nan = self.check_tensor(actions, "actions", replace_value)

        if had_nan:
            self.stats.action_nans += 1
            self.capture(action=actions.detach().cpu().numpy() if isinstance(actions, torch.Tensor) else actions)

            if self.cfg.halt_on_nan and not self._dumped:
                self._dump_buffer()
                self._dumped = True
                raise RuntimeError(f"[NanGuard] NaN detected in actions at step {self.step_counter}")

        return actions, had_nan

    def check_observations(
        self,
        observations: torch.Tensor | np.ndarray,
        replace_value: float = 0.0,
    ) -> tuple[torch.Tensor | np.ndarray, bool]:
        """Check observations for NaN/Inf.

        Args:
            observations: Observation tensor/array.
            replace_value: Value to use for recovery.

        Returns:
            Tuple of (sanitized observations, had_nan).
        """
        if not self.enabled or not self.cfg.check_observations:
            return observations, False

        self.stats.total_checks += 1
        observations, had_nan = self.check_tensor(observations, "observations", replace_value)

        if had_nan:
            self.stats.observation_nans += 1

            if self.cfg.halt_on_nan and not self._dumped:
                self._dump_buffer()
                self._dumped = True
                raise RuntimeError(f"[NanGuard] NaN detected in observations at step {self.step_counter}")

        return observations, had_nan

    def check_reward(
        self,
        reward: torch.Tensor | float,
        replace_value: float = 0.0,
    ) -> tuple[torch.Tensor | float, bool]:
        """Check reward for NaN/Inf.

        Args:
            reward: Reward value.
            replace_value: Value to use for recovery.

        Returns:
            Tuple of (sanitized reward, had_nan).
        """
        if not self.enabled or not self.cfg.check_rewards:
            return reward, False

        self.stats.total_checks += 1

        if isinstance(reward, torch.Tensor):
            has_nan = torch.isnan(reward).any().item() or torch.isinf(reward).any().item()
        else:
            has_nan = np.isnan(reward) or np.isinf(reward)

        if not has_nan:
            return reward, False

        self.stats.reward_nans += 1

        if self.stats.first_nan_step is None:
            self.stats.first_nan_step = self.step_counter
        self.stats.last_nan_step = self.step_counter

        if self.cfg.verbose:
            print(f"[NanGuard] reward: NaN/Inf at step {self.step_counter}")

        if self.cfg.recovery_mode:
            self.stats.recoveries += 1
            if isinstance(reward, torch.Tensor):
                reward = torch.where(torch.isnan(reward) | torch.isinf(reward),
                                     torch.full_like(reward, replace_value), reward)
            else:
                reward = replace_value

        if self.cfg.halt_on_nan and not self._dumped:
            self._dump_buffer()
            self._dumped = True
            raise RuntimeError(f"[NanGuard] NaN detected in reward at step {self.step_counter}")

        return reward, True

    def get_stats_summary(self) -> str:
        """Get a summary of NaN statistics."""
        if not self.enabled:
            return "[NanGuard] Disabled"

        return (
            f"[NanGuard Stats] "
            f"checks={self.stats.total_checks}, "
            f"action_nans={self.stats.action_nans}, "
            f"obs_nans={self.stats.observation_nans}, "
            f"reward_nans={self.stats.reward_nans}, "
            f"recoveries={self.stats.recoveries}, "
            f"first_nan={self.stats.first_nan_step}, "
            f"last_nan={self.stats.last_nan_step}"
        )
