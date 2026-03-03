import math
from dataclasses import dataclass

from luckylab.envs.mdp.actions import cpg_action, joint_actions
from luckylab.managers.action_manager import ActionTerm
from luckylab.managers.manager_term_config import ActionTermCfg


@dataclass(kw_only=True)
class JointActionCfg(ActionTermCfg):
  actuator_names: tuple[str, ...]
  """Tuple of actuator names or regex expressions that the action will be mapped to."""
  scale: float | dict[str, float] = 1.0
  """Scale factor for the action (float or dict of regex expressions). Defaults to 1.0."""
  offset: float | dict[str, float] = 0.0
  """Offset factor for the action (float or dict of regex expressions). Defaults to 0.0."""
  preserve_order: bool = False
  """Whether to preserve the order of the joint names in the action output. Defaults to False."""


@dataclass(kw_only=True)
class JointPositionActionCfg(JointActionCfg):
  class_type: type[ActionTerm] = joint_actions.JointPositionAction
  use_default_offset: bool = True


@dataclass(kw_only=True)
class JointVelocityActionCfg(JointActionCfg):
  class_type: type[ActionTerm] = joint_actions.JointVelocityAction
  use_default_offset: bool = True


@dataclass(kw_only=True)
class JointEffortActionCfg(JointActionCfg):
  class_type: type[ActionTerm] = joint_actions.JointEffortAction


@dataclass(kw_only=True)
class CPGActionCfg(JointPositionActionCfg):
  """Joint position action with CPG gait scaffold."""
  class_type: type[ActionTerm] = cpg_action.CPGAction
  frequency: float = 2.0
  """CPG oscillation frequency in Hz."""
  amplitude_hip: float = 0.12
  """Normalized amplitude for hip joints."""
  amplitude_thigh: float = 0.50
  """Normalized amplitude for thigh joints."""
  amplitude_calf: float = 0.50
  """Normalized amplitude for calf joints."""
  calf_phase_offset: float = math.pi / 4.0
  """Phase offset for calf relative to thigh."""
