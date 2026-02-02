"""Joint action terms for luckylab.

Follows mjlab's pattern with ActionTerm base class and specific implementations
for position, velocity, and effort control.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from luckylab.managers.action_manager import ActionTerm
from luckylab.utils.string import resolve_matching_names_values

if TYPE_CHECKING:
    from luckylab.entity import Entity
    from luckylab.envs import ManagerBasedRlEnv
    from luckylab.envs.mdp.actions import actions_config


class JointAction(ActionTerm):
    """Base class for joint actions.

    Handles finding joints by actuator names, applying scale/offset,
    and storing raw/processed actions.
    """

    _asset: "Entity"

    def __init__(self, cfg: "actions_config.JointActionCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg=cfg, env=env)

        # Find joints matching the actuator names
        joint_ids, joint_names = self._asset.find_joints(cfg.actuator_names)
        self._joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)
        self._joint_names = joint_names

        self._num_joints = len(joint_ids)
        self._action_dim = len(joint_ids)

        # Action buffers
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        # Parse scale (float or dict)
        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        elif isinstance(cfg.scale, dict):
            self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
            index_list, _, value_list = resolve_matching_names_values(
                cfg.scale, self._joint_names
            )
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(
                f"Unsupported scale type: {type(cfg.scale)}. "
                "Supported types are float and dict."
            )

        # Parse offset (float or dict)
        if isinstance(cfg.offset, (float, int)):
            self._offset = float(cfg.offset)
        elif isinstance(cfg.offset, dict):
            self._offset = torch.zeros_like(self._raw_actions)
            index_list, _, value_list = resolve_matching_names_values(
                cfg.offset, self._joint_names
            )
            self._offset[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(
                f"Unsupported offset type: {type(cfg.offset)}. "
                "Supported types are float and dict."
            )

    # Properties

    @property
    def joint_ids(self) -> torch.Tensor:
        """Joint indices being controlled."""
        return self._joint_ids

    @property
    def joint_names(self) -> list[str]:
        """Joint names being controlled."""
        return self._joint_names

    @property
    def scale(self) -> torch.Tensor | float:
        """Scale factor for actions."""
        return self._scale

    @property
    def offset(self) -> torch.Tensor | float:
        """Offset for actions."""
        return self._offset

    @property
    def raw_action(self) -> torch.Tensor:
        """Raw actions before processing."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Processed actions after scale/offset."""
        return self._processed_actions

    @property
    def action_dim(self) -> int:
        """Dimension of the action space for this term."""
        return self._action_dim

    # Methods

    def process_actions(self, actions: torch.Tensor) -> None:
        """Process raw actions with scale and offset.

        Args:
            actions: Raw actions from policy, shape (num_envs, action_dim).
        """
        self._raw_actions[:] = actions
        self._processed_actions = self._raw_actions * self._scale + self._offset

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Reset action state for specified environments.

        Args:
            env_ids: Environment indices to reset, or None for all.
        """
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = self._offset if isinstance(self._offset, torch.Tensor) else 0.0


class JointPositionAction(JointAction):
    """Joint position action term.

    Converts raw actions to joint position targets using:
        joint_position = raw_action * scale + offset

    When use_default_offset=True, offset is initialized from the robot's
    default joint positions.
    """

    def __init__(
        self, cfg: "actions_config.JointPositionActionCfg", env: "ManagerBasedRlEnv"
    ):
        super().__init__(cfg=cfg, env=env)

        # Use default joint positions as offset if configured
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()


class JointVelocityAction(JointAction):
    """Joint velocity action term.

    Converts raw actions to joint velocity targets using:
        joint_velocity = raw_action * scale + offset
    """

    def __init__(
        self, cfg: "actions_config.JointVelocityActionCfg", env: "ManagerBasedRlEnv"
    ):
        super().__init__(cfg=cfg, env=env)

        # Use default joint velocities as offset if configured
        if cfg.use_default_offset:
            # Default velocities are typically zero
            self._offset = torch.zeros(self.num_envs, self.action_dim, device=self.device)


class JointEffortAction(JointAction):
    """Joint effort/torque action term.

    Converts raw actions to joint effort targets using:
        joint_effort = raw_action * scale + offset
    """

    def __init__(
        self, cfg: "actions_config.JointEffortActionCfg", env: "ManagerBasedRlEnv"
    ):
        super().__init__(cfg=cfg, env=env)
