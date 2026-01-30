"""Environment wrapper for skrl compatibility."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
import torch

if TYPE_CHECKING:
    from ..envs import ManagerBasedRlEnv


class SkrlWrapper(gym.Wrapper):
    """
    Wraps ManagerBasedRlEnv for skrl compatibility.

    Handles tensor conversion and exposes properties expected by skrl.
    """

    def __init__(self, env: ManagerBasedRlEnv, device: str = "cpu"):
        super().__init__(env)
        self._device = torch.device(device)
        self._num_envs = 1  # Single env for now

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def reset(self, seed: int | None = None, **kwargs) -> tuple[torch.Tensor, dict]:
        obs, info = self.env.reset(seed=seed, **kwargs)
        return self._to_tensor(obs), info

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        action_np = actions.squeeze(0).cpu().numpy()
        obs, reward, terminated, truncated, info = self.env.step(action_np)

        return (
            self._to_tensor(obs),
            torch.tensor([[reward]], dtype=torch.float32, device=self._device),
            torch.tensor([[terminated]], dtype=torch.bool, device=self._device),
            torch.tensor([[truncated]], dtype=torch.bool, device=self._device),
            info,
        )

    def _to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(obs).float().unsqueeze(0).to(self._device)


def wrap_env(env: ManagerBasedRlEnv, device: str = "cpu") -> SkrlWrapper:
    """Wrap a ManagerBasedRlEnv environment for skrl."""
    return SkrlWrapper(env, device=device)
