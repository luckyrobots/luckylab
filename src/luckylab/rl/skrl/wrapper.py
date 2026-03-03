"""Environment wrapper for skrl compatibility."""

from __future__ import annotations

import logging

import numpy as np
import torch
from gymnasium.spaces import Box

from luckylab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from luckylab.utils.nan_guard import NanGuard
from luckylab.viewer import DebugVisualizer

logger = logging.getLogger(__name__)


class SkrlWrapper:
    """Wraps ManagerBasedRlEnv for skrl compatibility."""

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        clip_actions: float | None = None,
        **kwargs,
    ):
        self.env = env
        self.clip_actions = clip_actions
        self.nan_guard = NanGuard(env.cfg.nan_guard)
        self._last_episode_info: dict[str, float] = {}

        self.visualizer = DebugVisualizer(env)

        self.num_envs = self.unwrapped.num_envs
        self.num_agents = 1  # Single agent environment
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length
        self.num_actions = self.unwrapped.action_manager.total_action_dim

        obs_dims = self.unwrapped.observation_manager.group_obs_dim
        self.num_obs = self._get_obs_dim(obs_dims["policy"])

        self._observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32
        )

        # Create action space with bounded [-1, 1] for normalized actions
        self._action_space = Box(
            low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32
        )

    def _get_obs_dim(self, dim) -> int:
        """Extract observation dimension from various formats."""
        if isinstance(dim, tuple):
            return dim[0]
        elif isinstance(dim, list):
            # Sum dimensions from list of tuples
            return sum(d[0] if isinstance(d, tuple) else d for d in dim)
        else:
            return dim

    @property
    def cfg(self) -> ManagerBasedRlEnvCfg:
        return self.unwrapped.cfg

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        # skrl expects gymnasium Box spaces
        return self._action_space

    @classmethod
    def class_name(cls) -> str:
        return cls.__name__

    @property
    def unwrapped(self) -> ManagerBasedRlEnv:
        return self.env

    @property
    def is_realtime(self) -> bool:
        """Check if running in realtime mode."""
        return getattr(self.cfg, "simulation_mode", "fast") == "realtime"

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.unwrapped.episode_length_buf = value

    def seed(self, seed: int) -> None:
        self.unwrapped.seed(seed)

    def get_observations(self) -> torch.Tensor:
        return self.unwrapped.obs_buf["policy"]

    def reset(self) -> tuple[torch.Tensor, dict]:
        obs_dict, extras = self.env.reset()
        return obs_dict["policy"], extras

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # Check actions for NaN before processing
        actions, action_nan = self.nan_guard.check_actions(actions, replace_value=0.0)

        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)

        # Check observations for NaN
        obs = obs_dict["policy"]
        obs, obs_nan = self.nan_guard.check_observations(obs, replace_value=0.0)

        # Check reward for NaN
        rew, rew_nan = self.nan_guard.check_reward(rew, replace_value=0.0)

        # Increment step counter
        self.nan_guard.step_counter += 1

        # Track NaN occurrences
        if action_nan or obs_nan or rew_nan:
            extras["nan_detected"] = {
                "action": action_nan,
                "observation": obs_nan,
                "reward": rew_nan,
                "step": self.nan_guard.step_counter,
            }

        # Cache episode info for wandb logging (only present on reset steps).
        if "episode" in extras:
            self._last_episode_info = {
                k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in extras["episode"].items()
            }

        if not self.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        return obs, rew, terminated, truncated, extras

    def close(self) -> None:
        if self.nan_guard.enabled:
            logger.info(self.nan_guard.get_stats_summary())
        self.env.close()
