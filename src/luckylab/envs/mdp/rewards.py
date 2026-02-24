"""Useful methods for MDP rewards."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from luckylab.entity import Entity
from luckylab.managers.manager_term_config import RewardTermCfg
from luckylab.managers.scene_entity_config import SceneEntityCfg
from luckylab.utils.string import resolve_matching_names_values

if TYPE_CHECKING:
    from luckylab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def is_alive(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Reward for being alive."""
  return (~env.termination_manager.terminated).float()


def is_terminated(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalize terminated episodes that don't correspond to episodic timeouts."""
  return env.termination_manager.terminated.float()


def action_rate_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalize the rate of change of the actions using L2 squared kernel."""
  return torch.sum(
    torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
  )


class action_acc_l2:
    """Penalize action acceleration (second derivative of actions).

    Returns sum((a - 2*a_prev + a_prev_prev)^2).
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self._prev_prev_action = torch.zeros(
            env.num_envs, env.action_manager.total_action_dim, device=env.device
        )
        self._prev_action = torch.zeros_like(self._prev_prev_action)

    def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        action = env.action_manager.action
        acc = action - 2.0 * self._prev_action + self._prev_prev_action
        self._prev_prev_action[:] = self._prev_action
        self._prev_action[:] = action
        return torch.sum(torch.square(acc), dim=1)


class joint_acc_l2:
    """Penalize joint acceleration ((joint_vel - prev_joint_vel) / dt)^2.

    Returns sum(((joint_vel - prev_joint_vel) / dt)^2).
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self._prev_joint_vel = torch.zeros(
            env.num_envs, env.scene["robot"].num_joints, device=env.device
        )
        self._dt = env.step_dt

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        joint_vel = asset.data.joint_vel
        joint_acc = (joint_vel - self._prev_joint_vel) / self._dt
        self._prev_joint_vel[:] = joint_vel
        return torch.sum(torch.square(joint_acc), dim=1)


def joint_pos_limits(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits."""
    asset: Entity = env.scene[asset_cfg.name]
    soft_joint_pos_limits = asset.data.soft_joint_pos_limits
    assert soft_joint_pos_limits is not None
    out_of_limits = -(
      asset.data.joint_pos[:, asset_cfg.joint_ids]
      - soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
      asset.data.joint_pos[:, asset_cfg.joint_ids]
      - soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


class posture:
    """Penalize the deviation of the joint positions from the default positions.

    Note: This is implemented as a class so that we can resolve the standard deviation
    dictionary into a tensor and thereafter use it in the __call__ method.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        asset: Entity = env.scene[cfg.params["asset_cfg"].name]
        default_joint_pos = asset.data.default_joint_pos
        assert default_joint_pos is not None
        self.default_joint_pos = default_joint_pos

        _, joint_names = asset.find_joints(
        cfg.params["asset_cfg"].joint_names,
        )
        _, _, std = resolve_matching_names_values(
            data=cfg.params["std"],
            list_of_strings=joint_names,
        )
        self.std = torch.tensor(std, device=env.device, dtype=torch.float32)

    def __call__(
        self, env: ManagerBasedRlEnv, std, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
    ) -> torch.Tensor:
        del std  # Unused
        asset: Entity = env.scene[asset_cfg.name]
        current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
        desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
        error_squared = torch.square(current_joint_pos - desired_joint_pos)
        return torch.exp(-torch.mean(error_squared / (self.std**2), dim=1))


def flat_orientation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize non-flat base orientation (grav_x^2 + grav_y^2)."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def lin_vel_z_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize vertical linear velocity (vz^2)."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_link_lin_vel_b[:, 2])


def ang_vel_xy_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize roll/pitch angular velocity (wx^2 + wy^2) in body frame."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)


def illegal_contact(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize non-foot body parts touching the ground."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.illegal_contact.float()


def foot_clearance(
    env: ManagerBasedRlEnv,
    max_height: float = 0.15,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward foot height during swing phase (clamped). Gated by command magnitude.

    Sums clamped foot heights across all 4 feet. Provides continuous gradient
    for lifting feet higher, unlike binary feet_air_time. Returns zero when
    command is near-zero so the robot learns to stand still.
    """
    asset: Entity = env.scene[asset_cfg.name]
    height = asset.data.foot_height  # (num_envs, 4)
    clamped = torch.clamp(height, min=0.0, max=max_height)
    reward = torch.sum(clamped, dim=1)
    # Gate by command magnitude
    command = asset.data.vel_command  # (num_envs, 3)
    cmd_norm = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    gate = (cmd_norm > command_threshold).float()
    return reward * gate
