"""Environment wrapper for skrl compatibility.

Asymmetric Actor-Critic
-----------------------
skrl doesn't natively support asymmetric actor-critic (separate observations for
policy vs critic). We work around this by concatenating observations:

- Observations are returned as [policy_obs | critic_obs]
- Actor models slice to use only [:num_policy_obs]
- Critic models use the full observation tensor

To enable: define both "policy" and "critic" observation groups in your env config.
If only "policy" is defined, falls back to symmetric (same obs for both).
"""

from __future__ import annotations

import numpy as np
import torch
from gymnasium.spaces import Box

from luckylab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from luckylab.envs.types import VecEnvObs
from luckylab.utils.nan_guard import NanGuard, NanGuardCfg
from luckylab.visualization import DebugVisualizer


class SkrlWrapper:
    """Wraps ManagerBasedRlEnv for skrl compatibility."""

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        clip_actions: float | None = None,
        nan_guard_cfg: NanGuardCfg | None = None,
    ):
        self.env = env
        self.clip_actions = clip_actions

        if nan_guard_cfg is None and getattr(env.cfg, "enable_nan_guard", False):
            nan_guard_cfg = NanGuardCfg(
                enabled=True,
                recovery_mode=True,
                halt_on_nan=False,
                verbose=True,
            )
        self.nan_guard = NanGuard(nan_guard_cfg)

        # Debug visualization (auto-enabled in realtime mode)
        self.visualizer = DebugVisualizer(env)

        self.num_envs = self.unwrapped.num_envs
        self.num_agents = 1  # Single agent environment
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length
        self.num_actions = self.unwrapped.action_manager.total_action_dim

        obs_dims = self.unwrapped.observation_manager.group_obs_dim
        self.num_policy_obs = self._get_obs_dim(obs_dims["policy"])

        if "critic" in obs_dims:
            self.num_critic_obs = self._get_obs_dim(obs_dims["critic"])
        else:
            self.num_critic_obs = 0

        self.num_obs = self.num_policy_obs + self.num_critic_obs

        # Create observation space for concatenated [policy | critic] observations
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
        # Returns concatenated [policy | critic] observation space
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

    def seed(self, seed: int = -1) -> int:
        return self.unwrapped.seed(seed)

    def _extract_obs(self, obs_dict: VecEnvObs, group: str) -> torch.Tensor:
        obs = obs_dict[group]
        if isinstance(obs, dict):
            obs = torch.cat(list(obs.values()), dim=-1)
        return obs

    def _concat_obs(self, obs_dict: VecEnvObs) -> torch.Tensor:
        policy_obs = self._extract_obs(obs_dict, "policy")

        if self.num_critic_obs > 0 and "critic" in obs_dict:
            critic_obs = self._extract_obs(obs_dict, "critic")
            return torch.cat([policy_obs, critic_obs], dim=-1)

        return policy_obs

    def get_observations(self) -> torch.Tensor:
        return self._concat_obs(self.unwrapped.obs_buf)

    def reset(self) -> tuple[torch.Tensor, dict]:
        obs_dict, extras = self.env.reset()
        return self._concat_obs(obs_dict), extras

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # Check actions for NaN before processing
        actions, action_nan = self.nan_guard.check_actions(actions, replace_value=0.0)

        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)

        # Check observations for NaN
        obs = self._concat_obs(obs_dict)
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

        dones = (terminated | truncated).to(dtype=torch.long)

        if not self.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        # Auto-draw velocity command in realtime mode
        if self.is_realtime:
            self.visualizer.draw_velocity_command(env_idx=0)

        return obs, rew, terminated, dones, extras

    def close(self) -> None:
        if self.nan_guard.enabled:
            print(self.nan_guard.get_stats_summary())
        self.env.close()
