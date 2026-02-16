from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from luckylab.entity import Entity
from luckylab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from luckylab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def foot_height(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.foot_height


def foot_air_time(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.foot_air_time


def foot_contact(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.foot_contact


def foot_contact_forces(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Foot contact forces with log transform for network input.

    Matches mjlab: applies sign(f) * log1p(|f|) to compress the force
    range for better network training stability.
    """
    asset: Entity = env.scene[asset_cfg.name]
    forces = asset.data.foot_contact_forces
    return torch.sign(forces) * torch.log1p(torch.abs(forces))
