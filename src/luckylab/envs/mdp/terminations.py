"""MDP termination functions for luckylab environments.

Termination functions check if an episode should end.
They access data via env.scene["robot"].data.* (mjlab pattern).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from ...managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from ...entity import Entity
    from ..manager_based_rl_env import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def time_out(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Terminate when the episode length exceeds the maximum.

    Args:
        env: The environment instance.

    Returns:
        Boolean tensor indicating timeout per env.
    """
    return env.step_count >= env.cfg.max_episode_length


def bad_orientation(
    env: ManagerBasedRlEnv,
    limit_angle: float = math.radians(70.0),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate when the robot's orientation exceeds the limit angle.

    Args:
        env: The environment instance.
        limit_angle: Maximum allowed tilt angle in radians.
        asset_cfg: Configuration for the asset.

    Returns:
        Boolean tensor indicating bad orientation per env.
    """
    asset: Entity = env.scene[asset_cfg.name]
    gz = asset.data.projected_gravity_b[:, 2]
    gz_clamped = torch.clamp(-gz, -1.0, 1.0)
    tilt_angle = torch.abs(torch.acos(gz_clamped))
    return tilt_angle > limit_angle


def root_height_below_minimum(
    env: ManagerBasedRlEnv,
    minimum_height: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate when the robot's root height is below the minimum.

    Note: This requires root position data which may not be available
    in all observation schemas. Returns False if data is unavailable.

    Args:
        env: The environment instance.
        minimum_height: Minimum allowed height in meters.
        asset_cfg: Configuration for the asset.

    Returns:
        Boolean tensor indicating low height per env.
    """
    # Root position requires additional data not typically in standard observations
    # For now, return False (no termination)
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def nan_detection(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Terminate if NaN or Inf values are detected in observations.

    Args:
        env: The environment instance.

    Returns:
        Boolean tensor indicating NaN/Inf detection per env.
    """
    if env.latest_observation is None or not env.latest_observation.observation:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    obs_tensor = torch.tensor(
        env.latest_observation.observation,
        dtype=torch.float32,
        device=env.device,
    )
    has_nan = torch.any(torch.isnan(obs_tensor))
    has_inf = torch.any(torch.isinf(obs_tensor))

    # Broadcast to all envs (single observation applies to all for single-env)
    if has_nan or has_inf:
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
