"""Observation manager for computing observations with noise, delay, and history.

Follows mjlab pattern where observations are computed by calling functions
and noise is applied based on the term configuration.

Pipeline: Term Functions → Noise (per-term) → Concatenate → Delay → History
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import torch

from .manager_term_config import ObservationGroupCfg, ObservationTermCfg

if TYPE_CHECKING:
    from ..envs.manager_based_rl_env import ManagerBasedRlEnv


class DelayBuffer:
    """Circular buffer for simulating observation delay (torch-based)."""

    def __init__(self, num_envs: int, obs_dim: int, max_delay: int, device: torch.device):
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.max_delay = max_delay
        self.buffer_size = max_delay + 1
        self.device = device

        # Buffer shape: (buffer_size, num_envs, obs_dim)
        self.buffer = torch.zeros((self.buffer_size, num_envs, obs_dim), dtype=torch.float32, device=device)
        self.write_idx = 0
        # Per-env delay (can vary per environment)
        self.current_delay = torch.zeros(num_envs, dtype=torch.long, device=device)

    def set_delay(self, delay: int | torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        """Set the current delay for specified environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        if isinstance(delay, int):
            delay = torch.full((len(env_ids),), delay, dtype=torch.long, device=self.device)

        self.current_delay[env_ids] = torch.clamp(delay, 0, self.max_delay)

    def reset(self, initial_obs: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        """Reset buffer with initial observation for specified environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Fill entire buffer with initial observation for reset envs
        for i in range(self.buffer_size):
            self.buffer[i, env_ids] = initial_obs[env_ids] if initial_obs.dim() == 2 else initial_obs

    def push_and_get(self, obs: torch.Tensor) -> torch.Tensor:
        """Push new observation and get delayed observation."""
        # Write current observation
        self.buffer[self.write_idx] = obs

        # Compute read indices per environment
        read_idx = (self.write_idx - self.current_delay) % self.buffer_size

        # Gather delayed observations
        result = self.buffer[read_idx, torch.arange(self.num_envs, device=self.device)]

        # Advance write pointer
        self.write_idx = (self.write_idx + 1) % self.buffer_size

        return result


class HistoryBuffer:
    """Buffer for stacking observation history (torch-based)."""

    def __init__(
        self,
        num_envs: int,
        obs_dim: int,
        history_length: int,
        device: torch.device,
        flatten: bool = True,
    ):
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.history_length = history_length
        self.flatten = flatten
        self.device = device

        # Use deque per environment for simplicity
        # Shape after stacking: (num_envs, history_length, obs_dim) or (num_envs, history_length * obs_dim)
        self._history: list[deque[torch.Tensor]] = [deque(maxlen=history_length) for _ in range(num_envs)]

    def reset(self, initial_obs: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        """Reset history with initial observation for specified environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        for env_id in env_ids.tolist():
            self._history[env_id].clear()
            obs = initial_obs[env_id] if initial_obs.dim() == 2 else initial_obs
            for _ in range(self.history_length):
                self._history[env_id].append(obs.clone())

    def push_and_get(self, obs: torch.Tensor) -> torch.Tensor:
        """Push new observation and get stacked history."""
        results = []
        for env_id in range(self.num_envs):
            self._history[env_id].append(obs[env_id].clone())
            stacked = torch.stack(list(self._history[env_id]), dim=0)
            if self.flatten:
                stacked = stacked.flatten()
            results.append(stacked)

        return torch.stack(results, dim=0)

    @property
    def output_dim(self) -> int:
        """Get output dimension."""
        if self.flatten:
            return self.obs_dim * self.history_length
        return self.obs_dim


class ObservationManager:
    """Manages observation computation, noise, delay, and history.

    Pipeline: Term Functions → Noise (per-term) → Concatenate → Delay → History

    Computes observations by calling term functions and applies:
    - Per-term noise (when enable_corruption=True)
    - Per-term scaling and clipping
    - Optional delay simulation (sensor latency)
    - Optional history stacking (temporal context)
    """

    def __init__(
        self,
        groups: dict[str, ObservationGroupCfg],
        env: ManagerBasedRlEnv,
    ) -> None:
        """Initialize the observation manager.

        Args:
            groups: Dict mapping group names to their configurations.
            env: The environment instance.
        """
        self._env = env
        self._groups = groups
        self._device = env.device
        self._num_envs = env.num_envs

        # Pre-instantiate class-based terms (those with __init__ that takes cfg, env)
        self._term_instances: dict[str, dict[str, object]] = {}
        for group_name, group_cfg in groups.items():
            self._term_instances[group_name] = {}
            for term_name, term_cfg in group_cfg.terms.items():
                if isinstance(term_cfg.func, type):
                    # Class-based term: instantiate it
                    self._term_instances[group_name][term_name] = term_cfg.func(term_cfg, env)
                else:
                    # Function-based term: store None
                    self._term_instances[group_name][term_name] = None

        # Track observation dimensions per group (computed lazily)
        self._group_dims: dict[str, int] = {}

        # Delay and history buffers per group (initialized lazily)
        self._delay_buffers: dict[str, DelayBuffer | None] = {}
        self._history_buffers: dict[str, HistoryBuffer | None] = {}
        self._buffers_initialized: dict[str, bool] = {name: False for name in groups}

    @property
    def active_groups(self) -> list[str]:
        """Get list of active group names."""
        return list(self._groups.keys())

    def get_group_cfg(self, group_name: str) -> ObservationGroupCfg | None:
        """Get configuration for a specific group."""
        return self._groups.get(group_name)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset observation buffers for specified environments.

        Args:
            env_ids: Environment indices to reset. None = all environments.
        """
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)

        # Reset delay and history buffers for each group
        for group_name, group_cfg in self._groups.items():
            if not self._buffers_initialized.get(group_name, False):
                continue

            # Compute initial observation for buffer reset
            obs = self._compute_raw(group_name)

            # Reset delay buffer
            delay_buffer = self._delay_buffers.get(group_name)
            if delay_buffer is not None:
                # Randomize delay per environment
                delay_range = group_cfg.delay_range
                if delay_range[1] > 0:
                    delays = torch.randint(
                        delay_range[0], delay_range[1] + 1, (len(env_ids),), device=self._device
                    )
                    delay_buffer.set_delay(delays, env_ids)
                delay_buffer.reset(obs, env_ids)

            # Reset history buffer
            history_buffer = self._history_buffers.get(group_name)
            if history_buffer is not None:
                history_buffer.reset(obs, env_ids)

    def compute(self, group_name: str) -> torch.Tensor | dict[str, torch.Tensor]:
        """Compute observations for a specific group.

        Args:
            group_name: Name of the observation group (e.g., "policy", "critic").

        Returns:
            If concatenate_terms is True: concatenated tensor of shape (num_envs, total_dim).
            If concatenate_terms is False: dict mapping term names to tensors.
        """
        group_cfg = self._groups.get(group_name)
        if group_cfg is None:
            raise ValueError(f"Unknown observation group: {group_name}")

        # Compute raw observations (with noise, scale, clip)
        obs = self._compute_raw(group_name)

        # If not concatenated, skip delay/history (they expect flat tensor)
        if not group_cfg.concatenate_terms:
            return obs

        # Initialize buffers on first compute (need obs_dim)
        if not self._buffers_initialized.get(group_name, False):
            self._init_buffers(group_name, obs.shape[-1], group_cfg)
            self._buffers_initialized[group_name] = True

        # Apply delay
        delay_buffer = self._delay_buffers.get(group_name)
        if delay_buffer is not None:
            obs = delay_buffer.push_and_get(obs)

        # Apply history stacking
        history_buffer = self._history_buffers.get(group_name)
        if history_buffer is not None:
            obs = history_buffer.push_and_get(obs)

        return obs

    def _compute_raw(self, group_name: str) -> torch.Tensor | dict[str, torch.Tensor]:
        """Compute raw observations with noise, scale, clip (no delay/history)."""
        group_cfg = self._groups[group_name]
        observations: dict[str, torch.Tensor] = {}

        for term_name, term_cfg in group_cfg.terms.items():
            # Compute the observation term
            obs = self._compute_term(group_name, term_name, term_cfg)

            # Apply noise if corruption is enabled and term has noise config
            if group_cfg.enable_corruption and term_cfg.noise is not None:
                obs = term_cfg.noise.apply(obs)

            # Apply scaling
            if term_cfg.scale != 1.0:
                obs = obs * term_cfg.scale

            # Apply clipping
            if term_cfg.clip is not None:
                obs = torch.clamp(obs, term_cfg.clip[0], term_cfg.clip[1])

            # Ensure 2D shape (num_envs, dim)
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)

            observations[term_name] = obs

        if group_cfg.concatenate_terms:
            return torch.cat(list(observations.values()), dim=-1)
        return observations

    def _compute_term(
        self,
        group_name: str,
        term_name: str,
        term_cfg: ObservationTermCfg,
    ) -> torch.Tensor:
        """Compute a single observation term."""
        instance = self._term_instances[group_name].get(term_name)

        if instance is not None:
            # Class-based term: call the instance
            obs = instance(self._env, **term_cfg.params)
        else:
            # Function-based term: call the function directly
            obs = term_cfg.func(self._env, **term_cfg.params)

        return obs

    def _init_buffers(self, group_name: str, obs_dim: int, group_cfg: ObservationGroupCfg) -> None:
        """Initialize delay and history buffers for a group."""
        # Initialize delay buffer if configured
        delay_range = group_cfg.delay_range
        if delay_range[1] > 0:
            self._delay_buffers[group_name] = DelayBuffer(
                self._num_envs, obs_dim, delay_range[1], self._device
            )
        else:
            self._delay_buffers[group_name] = None

        # Initialize history buffer if configured
        history_length = group_cfg.history_length
        if history_length > 1:
            self._history_buffers[group_name] = HistoryBuffer(
                self._num_envs, obs_dim, history_length, self._device, group_cfg.flatten_history
            )
        else:
            self._history_buffers[group_name] = None

    def get_observation_dim(self, group_name: str) -> int:
        """Get total observation dimension for a group (computes if not cached).

        Args:
            group_name: Name of the observation group.

        Returns:
            Total observation dimension (including history if configured).
        """
        if group_name in self._group_dims:
            return self._group_dims[group_name]

        group_cfg = self._groups.get(group_name)
        if group_cfg is None:
            return 0

        # Compute observations to determine base dimension
        obs = self._compute_raw(group_name)
        if isinstance(obs, torch.Tensor):
            base_dim = obs.shape[-1]
        else:
            base_dim = sum(v.shape[-1] for v in obs.values())

        # Account for history stacking
        history_length = group_cfg.history_length
        if history_length > 1 and group_cfg.flatten_history:
            dim = base_dim * history_length
        else:
            dim = base_dim

        self._group_dims[group_name] = dim
        return dim


class NullObservationManager:
    """Null observation manager when no observations are configured."""

    @property
    def active_groups(self) -> list[str]:
        return []

    def get_group_cfg(self, group_name: str) -> ObservationGroupCfg | None:
        return None

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        pass

    def compute(self, group_name: str) -> torch.Tensor:
        raise ValueError(f"NullObservationManager cannot compute observations for {group_name}")

    def get_observation_dim(self, group_name: str) -> int:
        return 0
