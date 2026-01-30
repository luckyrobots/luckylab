"""Observation processing with domain randomization (noise, delay, history)."""

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class NoiseCfg:
    """Noise configuration for observations."""

    type: str = "gaussian"  # "gaussian", "uniform", "none"
    mean: float = 0.0  # For gaussian
    std: float = 0.0  # For gaussian
    low: float = 0.0  # For uniform
    high: float = 0.0  # For uniform


@dataclass
class ObservationTermCfg:
    """Configuration for processing a single observation term."""

    # Slice indices for this term in the observation vector
    start_idx: int = 0
    end_idx: int = -1  # -1 means end of observation

    # Processing pipeline (applied in order)
    noise: NoiseCfg | None = None
    scale: float = 1.0
    clip: tuple[float, float] | None = None  # (low, high)

    # Delay (in steps) - samples uniformly from range
    delay_range: tuple[int, int] = (0, 0)  # (min_steps, max_steps)


@dataclass
class ObservationProcessorCfg:
    """Configuration for the observation processor."""

    # Per-term configurations (applied to slices of observation)
    terms: dict[str, ObservationTermCfg] = field(default_factory=dict)

    # Global noise (applied to entire observation if no per-term config)
    global_noise: NoiseCfg | None = None

    # Global scale
    global_scale: float = 1.0

    # Global clip
    global_clip: tuple[float, float] | None = None

    # Global delay range (in steps)
    global_delay_range: tuple[int, int] = (0, 0)

    # History stacking
    history_length: int = 1  # 1 = no history, just current observation
    flatten_history: bool = True  # Flatten [T, obs_dim] to [T * obs_dim]


class DelayBuffer:
    """
    Circular buffer for simulating observation delay.

    Stores observations and returns them after a configurable delay.
    """

    def __init__(self, obs_dim: int, max_delay: int):
        """
        Initialize delay buffer.

        Args:
            obs_dim: Dimension of observation vector
            max_delay: Maximum delay in steps
        """
        self.obs_dim = obs_dim
        self.max_delay = max_delay
        self.buffer_size = max_delay + 1  # Need +1 to store current + delayed

        # Circular buffer
        self.buffer = np.zeros((self.buffer_size, obs_dim), dtype=np.float32)
        self.write_idx = 0

        # Current delay (can vary per reset)
        self.current_delay = 0

    def set_delay(self, delay: int) -> None:
        """Set the current delay (call on reset)."""
        self.current_delay = min(delay, self.max_delay)

    def reset(self, initial_obs: np.ndarray) -> None:
        """Reset buffer with initial observation."""
        self.buffer.fill(0)
        # Fill entire buffer with initial observation
        for i in range(self.buffer_size):
            self.buffer[i] = initial_obs
        self.write_idx = 0

    def push_and_get(self, obs: np.ndarray) -> np.ndarray:
        """
        Push new observation and get delayed observation.

        Args:
            obs: Current observation

        Returns:
            Delayed observation
        """
        # Write current observation
        self.buffer[self.write_idx] = obs

        # Calculate read index (delay steps behind write)
        read_idx = (self.write_idx - self.current_delay) % self.buffer_size

        # Advance write index
        self.write_idx = (self.write_idx + 1) % self.buffer_size

        return self.buffer[read_idx].copy()


class HistoryBuffer:
    """
    Buffer for stacking observation history.

    Maintains a sliding window of past observations.
    """

    def __init__(self, obs_dim: int, history_length: int, flatten: bool = True):
        """
        Initialize history buffer.

        Args:
            obs_dim: Dimension of single observation
            history_length: Number of observations to stack
            flatten: Whether to flatten output
        """
        self.obs_dim = obs_dim
        self.history_length = history_length
        self.flatten = flatten

        # Use deque for efficient sliding window
        self.history: deque[np.ndarray] = deque(maxlen=history_length)

    def reset(self, initial_obs: np.ndarray) -> None:
        """Reset history with initial observation."""
        self.history.clear()
        # Fill history with initial observation
        for _ in range(self.history_length):
            self.history.append(initial_obs.copy())

    def push_and_get(self, obs: np.ndarray) -> np.ndarray:
        """
        Push new observation and get stacked history.

        Args:
            obs: Current observation

        Returns:
            Stacked observations [history_length, obs_dim] or flattened
        """
        self.history.append(obs.copy())

        # Stack: oldest to newest
        stacked = np.array(list(self.history), dtype=np.float32)

        if self.flatten:
            return stacked.flatten()
        return stacked

    @property
    def output_dim(self) -> int:
        """Get output dimension."""
        if self.flatten:
            return self.obs_dim * self.history_length
        return self.obs_dim  # Shape would be (history_length, obs_dim)


class ObservationProcessor:
    """
    Processes observations with domain randomization.

    Pipeline: Raw Obs → Per-term Processing → Global Processing → Delay → History

    Per-term processing (if configured):
        - Noise injection
        - Scaling
        - Clipping

    Global processing (applied after per-term):
        - Global noise
        - Global scaling
        - Global clipping

    Temporal processing:
        - Delay buffer (simulates sensor latency)
        - History stacking (for temporal context)
    """

    def __init__(self, obs_dim: int, cfg: ObservationProcessorCfg):
        """
        Initialize observation processor.

        Args:
            obs_dim: Dimension of raw observation
            cfg: Processor configuration
        """
        self.obs_dim = obs_dim
        self.cfg = cfg

        # Random state for noise generation
        self._rng = np.random.default_rng()

        # Initialize delay buffer if delay is configured
        max_delay = max(cfg.global_delay_range[1], 0)
        for term_cfg in cfg.terms.values():
            max_delay = max(max_delay, term_cfg.delay_range[1])

        self.delay_buffer: DelayBuffer | None = None
        if max_delay > 0:
            self.delay_buffer = DelayBuffer(obs_dim, max_delay)

        # Initialize history buffer if history is configured
        self.history_buffer: HistoryBuffer | None = None
        if cfg.history_length > 1:
            self.history_buffer = HistoryBuffer(obs_dim, cfg.history_length, cfg.flatten_history)

        # Current delay (sampled on reset)
        self._current_delay = 0

    def reset(self, initial_obs: np.ndarray | None = None) -> None:
        """
        Reset processor state (call on episode reset).

        Args:
            initial_obs: Initial observation to fill buffers (optional)
        """
        # Sample new delay for this episode
        self._current_delay = self._rng.integers(
            self.cfg.global_delay_range[0], self.cfg.global_delay_range[1] + 1
        )

        if initial_obs is not None:
            # Process initial observation (without delay/history effects)
            processed_init = self._apply_noise_scale_clip(initial_obs)

            if self.delay_buffer is not None:
                self.delay_buffer.set_delay(self._current_delay)
                self.delay_buffer.reset(processed_init)

            if self.history_buffer is not None:
                self.history_buffer.reset(processed_init)

    def process(self, obs: np.ndarray) -> np.ndarray:
        """
        Process observation through the full pipeline.

        Args:
            obs: Raw observation from environment

        Returns:
            Processed observation
        """
        # Step 1: Apply noise, scale, clip
        processed = self._apply_noise_scale_clip(obs)

        # Step 2: Apply delay
        if self.delay_buffer is not None:
            processed = self.delay_buffer.push_and_get(processed)

        # Step 3: Stack history
        if self.history_buffer is not None:
            processed = self.history_buffer.push_and_get(processed)

        return processed

    def _apply_noise_scale_clip(self, obs: np.ndarray) -> np.ndarray:
        """Apply noise, scaling, and clipping to observation."""
        result = obs.copy()

        # Apply per-term processing
        for _term_name, term_cfg in self.cfg.terms.items():
            start = term_cfg.start_idx
            end = term_cfg.end_idx if term_cfg.end_idx != -1 else len(result)

            # Extract term slice
            term_obs = result[start:end]

            # Apply term-specific noise
            if term_cfg.noise is not None:
                term_obs = self._apply_noise(term_obs, term_cfg.noise)

            # Apply term-specific scale
            if term_cfg.scale != 1.0:
                term_obs = term_obs * term_cfg.scale

            # Apply term-specific clip
            if term_cfg.clip is not None:
                term_obs = np.clip(term_obs, term_cfg.clip[0], term_cfg.clip[1])

            # Write back
            result[start:end] = term_obs

        # Apply global processing
        if self.cfg.global_noise is not None:
            result = self._apply_noise(result, self.cfg.global_noise)

        if self.cfg.global_scale != 1.0:
            result = result * self.cfg.global_scale

        if self.cfg.global_clip is not None:
            result = np.clip(result, self.cfg.global_clip[0], self.cfg.global_clip[1])

        return result.astype(np.float32)

    def _apply_noise(self, obs: np.ndarray, noise_cfg: NoiseCfg) -> np.ndarray:
        """Apply noise to observation."""
        if noise_cfg.type == "none" or (noise_cfg.std == 0.0 and noise_cfg.type == "gaussian"):
            return obs

        if noise_cfg.type == "gaussian":
            noise = self._rng.normal(noise_cfg.mean, noise_cfg.std, size=obs.shape)
            return obs + noise.astype(np.float32)

        elif noise_cfg.type == "uniform":
            noise = self._rng.uniform(noise_cfg.low, noise_cfg.high, size=obs.shape)
            return obs + noise.astype(np.float32)

        return obs

    @property
    def output_dim(self) -> int:
        """Get output dimension after processing."""
        if self.history_buffer is not None:
            return self.history_buffer.output_dim
        return self.obs_dim

    @property
    def current_delay(self) -> int:
        """Get current delay setting."""
        return self._current_delay


def create_default_observation_processor(
    obs_dim: int,
    noise_std: float = 0.0,
    delay_range: tuple[int, int] = (0, 0),
    history_length: int = 1,
    clip_range: tuple[float, float] | None = None,
) -> ObservationProcessor:
    """
    Create an observation processor with common defaults.

    Args:
        obs_dim: Observation dimension
        noise_std: Gaussian noise standard deviation (0 = no noise)
        delay_range: (min, max) delay in steps
        history_length: Number of observations to stack (1 = no history)
        clip_range: Optional (low, high) clipping bounds

    Returns:
        Configured ObservationProcessor
    """
    cfg = ObservationProcessorCfg(
        global_noise=NoiseCfg(type="gaussian", std=noise_std) if noise_std > 0 else None,
        global_delay_range=delay_range,
        global_clip=clip_range,
        history_length=history_length,
        flatten_history=True,
    )
    return ObservationProcessor(obs_dim, cfg)
