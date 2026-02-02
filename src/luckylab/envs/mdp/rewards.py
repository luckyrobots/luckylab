"""MDP reward functions for luckylab environments.

General reward functions that can be used across different tasks.
Task-specific rewards should be in the task's mdp/rewards.py.

Matches mjlab/envs/mdp/rewards.py structure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...managers.manager_term_config import RewardTermCfg
from ...managers.scene_entity_config import SceneEntityCfg
from ...utils.string import resolve_matching_names_values

if TYPE_CHECKING:
    from ...entity import Entity
    from ..manager_based_rl_env import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


##
# Alive/terminated rewards.
##


def is_alive(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Reward for being alive."""
    if env.termination_manager is None:
        return torch.ones(env.num_envs, device=env.device)
    return (~env.termination_manager.terminated).float()


def is_terminated(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    if env.termination_manager is None:
        return torch.zeros(env.num_envs, device=env.device)
    return env.termination_manager.terminated.float()


##
# Action penalties.
##


def action_rate_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
    )


##
# Joint penalties.
##


def joint_pos_limits(
    env: ManagerBasedRlEnv,
    soft_ratio: float = 0.9,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize joint positions if they cross soft limits.

    Uses a soft_ratio of the action space limits as the soft limits.

    Args:
        env: The environment instance.
        soft_ratio: Ratio of hard limits to use as soft limits (default 0.9).
        asset_cfg: Configuration for the asset.

    Returns:
        Sum of out-of-limit violations per environment.
    """
    asset: Entity = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos

    # Compute soft limits from action space
    soft_lower = env.action_low * soft_ratio
    soft_upper = env.action_high * soft_ratio

    # Compute violations (how much past the soft limit)
    lower_violation = (soft_lower - joint_pos).clamp(min=0.0)
    upper_violation = (joint_pos - soft_upper).clamp(min=0.0)

    return torch.sum(lower_violation + upper_violation, dim=1)


class posture:
    """Penalize the deviation of the joint positions from the default positions.

    Note: This is implemented as a class so that we can resolve the standard deviation
    dictionary into a tensor and thereafter use it in the __call__ method.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        asset_cfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
        asset: Entity = env.scene[asset_cfg.name]
        default_joint_pos = asset.data.default_joint_pos
        assert default_joint_pos is not None
        self.default_joint_pos = default_joint_pos

        joint_names = asset.data.joint_names

        _, _, std = resolve_matching_names_values(
            data=cfg.params["std"],
            list_of_strings=joint_names,
        )
        self.std = torch.tensor(std, device=env.device, dtype=torch.float32)

    def __call__(
        self, env: ManagerBasedRlEnv, std, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
    ) -> torch.Tensor:
        del std  # Unused - resolved in __init__.
        asset: Entity = env.scene[asset_cfg.name]
        current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
        desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
        error_squared = torch.square(current_joint_pos - desired_joint_pos)
        return torch.exp(-torch.mean(error_squared / (self.std**2), dim=1))


##
# Orientation penalties.
##


def flat_orientation_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
