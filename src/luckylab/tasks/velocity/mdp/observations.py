"""Observation functions and utilities for velocity task.

Re-exports observation functions from envs.mdp.observations for use in
velocity_env_cfg. Also provides observation clipping ranges.
"""

# Re-export observation functions from envs.mdp
from luckylab.envs.mdp.observations import base_ang_vel as base_ang_vel
from luckylab.envs.mdp.observations import base_lin_vel as base_lin_vel
from luckylab.envs.mdp.observations import foot_air_time as foot_air_time
from luckylab.envs.mdp.observations import foot_contact as foot_contact
from luckylab.envs.mdp.observations import foot_contact_forces as foot_contact_forces
from luckylab.envs.mdp.observations import foot_height as foot_height
from luckylab.envs.mdp.observations import generated_commands as generated_commands
from luckylab.envs.mdp.observations import joint_pos as joint_pos
from luckylab.envs.mdp.observations import joint_pos_rel as joint_pos_rel
from luckylab.envs.mdp.observations import joint_vel as joint_vel
from luckylab.envs.mdp.observations import joint_vel_rel as joint_vel_rel
from luckylab.envs.mdp.observations import last_action as last_action
from luckylab.envs.mdp.observations import projected_gravity as projected_gravity

OBSERVATION_CLIP_RANGES: dict[str, tuple[float, float]] = {
    "base_lin_vel": (-10.0, 10.0),
    "base_ang_vel": (-20.0, 20.0),
    "projected_gravity": (-1.0, 1.0),
    "joint_pos": (-3.14159, 3.14159),
    "joint_vel": (-30.0, 30.0),
    # Privileged observations (singular names matching mjlab)
    "foot_contact": (0.0, 1.0),
    "foot_height": (-1.0, 1.0),
    "foot_contact_forces": (0.0, 500.0),
    "foot_air_time": (0.0, 2.0),
}
