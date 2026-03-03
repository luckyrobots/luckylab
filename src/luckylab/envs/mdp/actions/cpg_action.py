from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from luckylab.envs.mdp.actions.joint_actions import JointPositionAction

if TYPE_CHECKING:
    from luckylab.envs import ManagerBasedRlEnv
    from luckylab.envs.mdp.actions import actions_config


# Trot phase offsets: FL=0, FR=pi, RL=pi, RR=0
_TROT_PHASES = [0.0, math.pi, math.pi, 0.0]


class CPGAction(JointPositionAction):
    """CPG (Central Pattern Generator) action for trot gait scaffolding.

    Extends JointPositionAction with a sinusoidal trot reference that the policy
    learns to correct: action = clamp(policy + cpg_ref, -1, 1).

    The RL policy adds corrections on top of the fixed open-loop CPG pattern.
    Generates a sinusoidal trot reference and adds it to the policy output
    before applying scale + offset. The CPG keeps the robot's legs moving
    in a trot pattern; the policy learns corrections on top.

    Leg order: FL, FR, RL, RR (3 joints each: hip, thigh, calf).
    """

    def __init__(
        self, cfg: actions_config.CPGActionCfg, env: ManagerBasedRlEnv
    ):
        super().__init__(cfg=cfg, env=env)

        self._frequency = cfg.frequency
        self._amplitude_hip = cfg.amplitude_hip
        self._amplitude_thigh = cfg.amplitude_thigh
        self._amplitude_calf = cfg.amplitude_calf
        self._calf_phase_offset = cfg.calf_phase_offset

        # Phase per leg: (num_envs, 4)
        self._leg_phases = torch.zeros(self.num_envs, 4, device=self.device)

        # Trot offsets: (4,) constant
        self._trot_offsets = torch.tensor(_TROT_PHASES, device=self.device, dtype=torch.float32)

        self._step_dt = env.step_dt

    def process_actions(self, actions: torch.Tensor) -> None:
        """Add CPG reference to policy output, then apply scale + offset."""
        self._raw_actions[:] = actions

        # Advance phase: phase += 2*pi*freq*dt
        self._leg_phases += 2.0 * math.pi * self._frequency * self._step_dt
        self._leg_phases = self._leg_phases % (2.0 * math.pi)

        cpg_ref = self._compute_cpg_reference()
        combined = torch.clamp(actions + cpg_ref, -1.0, 1.0)

        self._processed_actions = combined * self._scale + self._offset

    def _compute_cpg_reference(self) -> torch.Tensor:
        """Generate normalized [-1, 1] CPG reference for all 12 joints.

        Returns:
            Tensor of shape (num_envs, 12) with CPG joint targets.
        """
        # Phase with trot offsets: (num_envs, 4)
        phase = self._leg_phases + self._trot_offsets.unsqueeze(0)

        # Per-joint sinusoids (normalized to [-1, 1] via amplitude)
        hip = self._amplitude_hip * torch.sin(phase) # (num_envs, 4)
        thigh = self._amplitude_thigh * torch.sin(phase) # (num_envs, 4)
        calf = self._amplitude_calf * torch.sin(phase + self._calf_phase_offset) # (num_envs, 4)

        # Interleave: for each leg [hip, thigh, calf]
        # Final order: FL_hip, FL_thigh, FL_calf, FR_hip, ...
        ref = torch.stack([hip, thigh, calf], dim=2)  # (num_envs, 4, 3)
        return ref.reshape(self.num_envs, 12)

    def get_leg_phase_obs(self) -> torch.Tensor:
        """Get sin/cos phase observation for each leg.

        Returns:
            Tensor of shape (num_envs, 8): [sin_FL, cos_FL, sin_FR, cos_FR, ...]
        """
        phase = self._leg_phases + self._trot_offsets.unsqueeze(0)
        sin_phase, cos_phase = torch.sin(phase), torch.cos(phase)  # (num_envs, 4)
        # Interleave sin/cos: (num_envs, 4, 2) -> (num_envs, 8)
        return torch.stack([sin_phase, cos_phase], dim=2).reshape(
            self.num_envs, 8
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        if env_ids is not None:
            self._leg_phases[env_ids] = 0.0