"""Termination conditions for locomotion tasks.

Matches mjlab/tasks/velocity/mdp/terminations.py.
All functions take `env` as first parameter and access data via env.scene[asset_cfg.name].data.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from luckylab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from luckylab.entity import Entity
    from luckylab.envs.manager_based_rl_env import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def time_out(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Terminate when the episode length exceeds its maximum."""
    return env.step_count >= env.cfg.max_episode_length


def bad_orientation(
    env: ManagerBasedRlEnv,
    limit_angle: float = math.radians(70.0),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate when the robot's orientation exceeds the limit angle."""
    asset: Entity = env.scene[asset_cfg.name]
    gz = asset.data.projected_gravity_b[:, 2]
    gz_clamped = torch.clamp(-gz, -1.0, 1.0)
    tilt_angle = torch.abs(torch.acos(gz_clamped))
    return tilt_angle > limit_angle


def illegal_contact(
    env: ManagerBasedRlEnv,
    z_threshold: float = -2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate if robot is falling rapidly (proxy for illegal contact/fall)."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_lin_vel_b[:, 2] < z_threshold


def joint_velocity_limit(
    env: ManagerBasedRlEnv,
    limit: float = 100.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate if any joint velocity exceeds the limit."""
    asset: Entity = env.scene[asset_cfg.name]
    max_vel = torch.max(torch.abs(asset.data.joint_vel), dim=1).values
    return max_vel > limit
