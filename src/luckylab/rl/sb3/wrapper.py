"""Gymnasium environment wrapper for Stable Baselines3 compatibility.

Bridges LuckyLab's ManagerBasedRlEnv (torch tensors, batch dim) to SB3's
expected interface (numpy arrays, no batch dim, standard gymnasium API).
"""

from __future__ import annotations

import logging
from typing import Any

import gymnasium
import numpy as np
import torch
from gymnasium.spaces import Box

from luckylab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from luckylab.utils.nan_guard import NanGuard
from luckylab.viewer import DebugVisualizer

logger = logging.getLogger(__name__)


class Sb3Wrapper(gymnasium.Env):
    """Wraps ManagerBasedRlEnv for Stable Baselines3 compatibility.

    Handles:
    - torch ↔ numpy conversion (SB3 expects numpy arrays)
    - Squeeze/unsqueeze of batch dimension (our env is batched, SB3 is single-env)
    - NaN guard integration
    - Episode tracking (cumulative reward, length) in SB3's Monitor format
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        clip_actions: float | None = None,
        use_delta_actions: bool = False,
        delta_action_scale: float = 0.2,
    ):
        super().__init__()
        self.env = env
        self.clip_actions = clip_actions
        self.nan_guard = NanGuard(env.cfg.nan_guard)
        self._last_episode_info: dict[str, float] = {}

        self.visualizer = DebugVisualizer(env)

        # Delta action accumulation for position-controlled actuators.
        self.use_delta_actions = use_delta_actions
        self.delta_action_scale = delta_action_scale
        self._action_target: np.ndarray | None = None

        self.device = self.env.device
        num_obs = self._get_obs_dim(env.observation_manager.group_obs_dim["policy"])
        num_actions = env.action_manager.total_action_dim

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(num_obs,), dtype=np.float32
        )
        self.action_space = Box(
            low=-1.0, high=1.0, shape=(num_actions,), dtype=np.float32
        )

    @property
    def cfg(self) -> ManagerBasedRlEnvCfg:
        return self.env.cfg

    @property
    def unwrapped_env(self) -> ManagerBasedRlEnv:
        """Access the underlying LuckyLab environment."""
        return self.env

    def _get_obs_dim(self, dim) -> int:
        if isinstance(dim, tuple):
            return dim[0]
        elif isinstance(dim, list):
            return sum(d[0] if isinstance(d, tuple) else d for d in dim)
        return dim

    @property
    def is_realtime(self) -> bool:
        """Check if running in realtime mode."""
        return getattr(self.cfg, "simulation_mode", "fast") == "realtime"

    def _obs_to_numpy(self, obs_dict: dict[str, torch.Tensor]) -> np.ndarray:
        """Extract policy obs, squeeze batch dim, convert to numpy."""
        obs = obs_dict["policy"]
        obs, _ = self.nan_guard.check_observations(obs, replace_value=0.0)
        return obs.squeeze(0).cpu().numpy()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.env.seed(seed)
        obs_dict, extras = self.env.reset()
        if self.use_delta_actions:
            self._action_target = np.zeros(self.action_space.shape, dtype=np.float32)
        # Strip non-standard "episode" key from reset info.
        extras.pop("episode", None)
        return self._obs_to_numpy(obs_dict), extras

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        # Delta action accumulation: policy output is a delta, accumulated into a running target.
        if self.use_delta_actions:
            self._action_target += action * self.delta_action_scale
            self._action_target = np.clip(self._action_target, -1.0, 1.0)
            action = self._action_target

        # Convert numpy action → torch with batch dim.
        action_t = torch.from_numpy(action).unsqueeze(0).to(dtype=torch.float32, device=self.device)

        # NaN guard on actions.
        action_t, action_nan = self.nan_guard.check_actions(action_t, replace_value=0.0)

        if self.clip_actions is not None:
            action_t = torch.clamp(action_t, -self.clip_actions, self.clip_actions)

        obs_dict, rew, terminated, truncated, extras = self.env.step(action_t)

        # NaN guard on reward.
        rew, rew_nan = self.nan_guard.check_reward(rew, replace_value=0.0)
        self.nan_guard.step_counter += 1

        if action_nan or rew_nan:
            extras["nan_detected"] = {
                "action": action_nan,
                "reward": rew_nan,
                "step": self.nan_guard.step_counter,
            }

        # Cache episode info for wandb logging before stripping.
        episode = extras.pop("episode", None)
        if episode:
            self._last_episode_info = {
                k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in episode.items()
            }

        obs = self._obs_to_numpy(obs_dict)

        return obs, rew.item(), terminated.item(), truncated.item(), extras

    def close(self) -> None:
        if self.nan_guard.enabled:
            logger.info(self.nan_guard.get_stats_summary())
        self.env.close()
