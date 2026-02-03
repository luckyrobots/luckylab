from typing import Any, Cast

import torch

from luckylab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg


class SkrlWrapper:
    """
    Wraps ManagerBasedRlEnv for skrl compatibility.

    Handles:
    - Device placement (ensure outputs are on correct device)
    - Action clipping (optional)
    - Episode length tracking
    - Time outs handling for infinite horizon tasks
    - Exposes observation_space/action_space for skrl

    Note: The underlying env already returns batched tensors with shape
    (num_envs, ...) so minimal conversion is needed.
    """

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        device: str = "cpu",
        clip_actions: float | None = None,
    ):
        self.env = env
        self._device = torch.device(device)
        self._num_envs = env.num_envs
        self.clip_actions = clip_actions

        # Get observation/action dimensions
        obs_dim = env.observation_space_shape[0]

        # Pre-allocate output tensors for consistent memory
        self._obs_buffer = torch.empty(
            (self._num_envs, obs_dim), dtype=torch.float32, device=self._device
        )
        self._reward_buffer = torch.empty(
            self._num_envs, dtype=torch.float32, device=self._device
        )
        self._terminated_buffer = torch.empty(
            self._num_envs, dtype=torch.bool, device=self._device
        )
        self._truncated_buffer = torch.empty(
            self._num_envs, dtype=torch.bool, device=self._device
        )

        # Modify action space if clipping is enabled
        if self.clip_actions is not None:
            self._modify_action_space()

    # Properties

    @property
    def unwrapped(self) -> ManagerBasedRlEnv:
        """Access the underlying unwrapped environment."""
        return self.env

    @property
    def cfg(self) -> ManagerBasedRlEnvCfg:
        """Access the environment configuration."""
        return self.unwrapped.cfg

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
    def num_actions(self) -> int:
        """Total action dimension."""
        return self.unwrapped.action_manager.total_action_dim

    @property
    def max_episode_length(self) -> int:
        """Maximum episode length in steps."""
        return self.unwrapped.max_episode_length

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """Buffer tracking current episode length for each environment."""
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.unwrapped.episode_length_buf = value

    @property
    def observation_space(self):
        """Observation space (gymnasium-compatible for skrl)."""
        return self.env.observation_space

    @property
    def action_space(self):
        """Action space (gymnasium-compatible for skrl)."""
        return self.env.action_space

    # Methods

    def reset(self, seed: int | None = None, **kwargs) -> tuple[torch.Tensor, dict]:
        """Reset the environment."""
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
        """Step the environment with actions."""
        # Move to env device if needed
        env_device = self.env.device
        if actions.device != env_device:
            actions = actions.to(env_device)

        # Clip actions if enabled
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        # Step environment (returns batched torch tensors)
        obs, reward, terminated, truncated, info = self.env.step(actions)

        # Handle time outs for infinite horizon tasks
        if not self.cfg.is_finite_horizon:
            info["time_outs"] = truncated

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

    def get_observations(self) -> torch.Tensor:
        """Get current observations without stepping.

        Useful for getting observations after reset or for logging.
        """
        obs = self.unwrapped.observation_manager.compute()
        if isinstance(obs, dict):
            # Flatten observation dict to single tensor if needed
            obs = torch.cat(list(obs.values()), dim=-1)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if obs.device != self._device:
            obs = obs.to(self._device)
        return obs

    def close(self):
        """Close the environment."""
        self.env.close()

    # Private methods

    def _modify_action_space(self) -> None:
        """Modify action space bounds based on clip_actions."""
        if self.clip_actions is None:
            return

        from gymnasium.spaces import Box

        # Update action space with clipped bounds
        self.unwrapped.action_space = Box(
            low=-self.clip_actions,
            high=self.clip_actions,
            shape=(self.num_actions,),
            dtype=float,
        )


def wrap_env(
    env: ManagerBasedRlEnv,
    device: str = "cpu",
    clip_actions: float | None = None,
) -> SkrlWrapper:
    """Wrap a ManagerBasedRlEnv environment for skrl.

    Args:
        env: The environment to wrap.
        device: Device to place output tensors on.
        clip_actions: Optional action clipping range. If provided, actions will be
            clipped to [-clip_actions, clip_actions].

    Returns:
        Wrapped environment compatible with skrl.
    """
    return SkrlWrapper(env, device=device, clip_actions=clip_actions)
