"""NaN guard for detecting and debugging NaN/Inf in observations."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass


@dataclass
class NanGuardCfg:
    """Configuration for NaN guard.

    Attributes:
        enabled: Whether to enable NaN detection.
        buffer_size: Number of states to keep in rolling buffer.
        output_dir: Directory to write dumps to.
    """

    enabled: bool = False
    buffer_size: int = 100
    output_dir: str = "/tmp/luckylab/nan_dumps"


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

    def __init__(self, cfg: NanGuardCfg) -> None:
        self.enabled = cfg.enabled

        if not self.enabled:
            return

        self.buffer_size = cfg.buffer_size
        self.output_dir = Path(cfg.output_dir)
        self.buffer: deque[dict[str, Any]] = deque(maxlen=self.buffer_size)
        self.step_counter = 0
        self._dumped = False

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
