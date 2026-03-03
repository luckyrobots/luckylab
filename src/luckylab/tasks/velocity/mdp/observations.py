from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from luckylab.entity import Entity
from luckylab.envs.mdp.actions.cpg_action import CPGAction
from luckylab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from luckylab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def foot_contact(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Foot contact binary observation. Returns (num_envs, 4)."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.foot_contact


def gait_phase(env: ManagerBasedRlEnv) -> torch.Tensor:
    """CPG gait phase observation: sin/cos per leg. Returns (num_envs, 8)."""
    action_term = env.action_manager.get_term("joint_pos")
    assert isinstance(action_term, CPGAction), (
        "gait_phase observation requires CPGAction as the 'joint_pos' action term"
    )
    return action_term.get_leg_phase_obs()
