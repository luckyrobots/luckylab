"""Action configurations for luckylab.

Follows mjlab's pattern with dataclass-based config that references
the action term class to instantiate.
"""

from __future__ import annotations

from dataclasses import dataclass

from luckylab.envs.mdp.actions import joint_actions
from luckylab.managers.action_manager import ActionTerm
from luckylab.managers.manager_term_config import ActionTermCfg


@dataclass(kw_only=True)
class JointActionCfg(ActionTermCfg):
    """Base configuration for joint action terms.

    Attributes:
        actuator_names: Tuple of actuator/joint names or regex expressions.
        scale: Scale factor for actions (float or dict of regex->float).
        offset: Offset factor for actions (float or dict of regex->float).
        preserve_order: Whether to preserve order of joint names in output.
    """

    actuator_names: tuple[str, ...] = (".*",)
    """Tuple of actuator names or regex expressions that the action will be mapped to."""
    scale: float | dict[str, float] = 1.0
    """Scale factor for the action (float or dict of regex expressions). Defaults to 1.0."""
    offset: float | dict[str, float] = 0.0
    """Offset factor for the action (float or dict of regex expressions). Defaults to 0.0."""
    preserve_order: bool = False
    """Whether to preserve the order of the joint names in the action output. Defaults to False."""


@dataclass(kw_only=True)
class JointPositionActionCfg(JointActionCfg):
    """Configuration for joint position action term.

    Converts raw actions [-1, 1] to joint positions:
        joint_position = raw_action * scale + offset

    When use_default_offset=True, offset is taken from robot's default joint positions.
    """

    class_type: type[ActionTerm] = joint_actions.JointPositionAction
    use_default_offset: bool = True
    """Whether to use default joint positions as offset. Defaults to True."""


@dataclass(kw_only=True)
class JointVelocityActionCfg(JointActionCfg):
    """Configuration for joint velocity action term.

    Converts raw actions to joint velocities:
        joint_velocity = raw_action * scale + offset
    """

    class_type: type[ActionTerm] = joint_actions.JointVelocityAction
    use_default_offset: bool = False
    """Whether to use default joint velocities as offset. Defaults to False."""


@dataclass(kw_only=True)
class JointEffortActionCfg(JointActionCfg):
    """Configuration for joint effort/torque action term.

    Converts raw actions to joint efforts:
        joint_effort = raw_action * scale + offset
    """

    class_type: type[ActionTerm] = joint_actions.JointEffortAction
