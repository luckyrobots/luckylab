"""Useful methods for MDP terminations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from luckylab.managers.scene_entity_config import SceneEntityCfg
from luckylab.utils.nan_guard import NanGuard

if TYPE_CHECKING:
    from luckylab.entity import Entity
    from luckylab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def time_out(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Terminate when the episode length exceeds its maximum."""
    return env.episode_length_buf >= env.max_episode_length


def bad_orientation(
  env: ManagerBasedRlEnv,
  limit_angle: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
  """Terminate when the asset's orientation exceeds the limit angle."""
  asset: Entity = env.scene[asset_cfg.name]
  projected_gravity = asset.data.projected_gravity_b
  return torch.acos(-projected_gravity[:, 2]).abs() > limit_angle


# TODO: Missing root_height_below_minimum

# TODO: Missing nan_detection
