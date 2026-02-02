"""Environment wrapper for skrl compatibility.

Since ManagerBasedRlEnv is torch-native (not gymnasium), this wrapper
handles shape conversion and provides the interface skrl expects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..envs import ManagerBasedRlEnv


class SkrlWrapper:
    """
    Wraps ManagerBasedRlEnv for skrl compatibility.

    Handles:
    - Device placement (ensure outputs are on correct device)
    - Exposes observation_space/action_space for skrl

    Note: The underlying env already returns batched tensors with shape
    (num_envs, ...) so minimal conversion is needed.
    """

    def __init__(self, env: ManagerBasedRlEnv, device: str = "cpu"):
        self.env = env
        self._device = torch.device(device)
        self._num_envs = env.num_envs

        # Get observation/action dimensions
        obs_dim = env.observation_space_shape[0]

        # Pre-allocate output tensors for consistent memory
        self._obs_buffer = torch.empty((self._num_envs, obs_dim), dtype=torch.float32, device=self._device)
        self._reward_buffer = torch.empty(self._num_envs, dtype=torch.float32, device=self._device)
        self._terminated_buffer = torch.empty(self._num_envs, dtype=torch.bool, device=self._device)
        self._truncated_buffer = torch.empty(self._num_envs, dtype=torch.bool, device=self._device)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def num_agents(self) -> int:
        """Number of agents (1 for single-agent RL)."""
        return 1

    @property
    def observation_space(self):
        """Observation space (gymnasium-compatible for skrl)."""
        return self.env.observation_space

    @property
    def action_space(self):
        """Action space (gymnasium-compatible for skrl)."""
        return self.env.action_space

    def reset(self, seed: int | None = None, **kwargs) -> tuple[torch.Tensor, dict]:
        obs, info = self.env.reset(seed=seed, **kwargs)
        # Ensure batch dimension and correct device
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if obs.device != self._device:
            obs = obs.to(self._device)
        self._obs_buffer.copy_(obs)
        return self._obs_buffer, info

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # Move to env device if needed
        env_device = self.env.device
        if actions.device != env_device:
            actions = actions.to(env_device)

        # Step environment (returns batched torch tensors)
        obs, reward, terminated, truncated, info = self.env.step(actions)

        # Ensure correct shapes and device
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if obs.device != self._device:
            obs = obs.to(self._device, non_blocking=True)
        self._obs_buffer.copy_(obs)

        # Copy reward/terminated/truncated to buffers on correct device
        if reward.device != self._device:
            reward = reward.to(self._device, non_blocking=True)
        self._reward_buffer.copy_(reward)

        if terminated.device != self._device:
            terminated = terminated.to(self._device, non_blocking=True)
        self._terminated_buffer.copy_(terminated)

        if truncated.device != self._device:
            truncated = truncated.to(self._device, non_blocking=True)
        self._truncated_buffer.copy_(truncated)

        return (
            self._obs_buffer,
            self._reward_buffer,
            self._terminated_buffer,
            self._truncated_buffer,
            info,
        )

    def close(self):
        """Close the environment."""
        self.env.close()


def wrap_env(env: ManagerBasedRlEnv, device: str = "cpu") -> SkrlWrapper:
    """Wrap a ManagerBasedRlEnv environment for skrl."""
    return SkrlWrapper(env, device=device)
