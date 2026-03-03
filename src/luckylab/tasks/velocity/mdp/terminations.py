from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from luckylab.entity import Entity
from luckylab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from luckylab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def illegal_contact(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate if any non-foot body touches the ground.

    Uses the illegal_contact flag from LuckyEngine which is 1.0 when
    any non-foot collision geom contacts the terrain.
    """
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.illegal_contact > 0.5
