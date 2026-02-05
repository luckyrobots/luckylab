"""Velocity tracking task configuration.

This module defines the configuration for velocity tracking tasks.
Robot-specific configurations are located in the config/ directory.
"""

import math

from luckylab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from luckylab.envs.mdp.actions import JointPositionActionCfg
from luckylab.managers.manager_term_config import (
    ActionTermCfg,
    CommandTermCfg,
    CurriculumTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from luckylab.managers.scene_entity_config import SceneEntityCfg
from luckylab.utils.noise import UniformNoiseCfg as Unoise
from luckylab.tasks.velocity import mdp
from luckylab.tasks.velocity.mdp import UniformVelocityCommandCfg


def create_velocity_env_cfg(
    robot: str,
    action_scale: float | dict[str, float],
    trunk_body_name: str,
    posture_std_standing: float | dict[str, float],
    posture_std_walking: float | dict[str, float],
    posture_std_running: float | dict[str, float],
    body_ang_vel_weight: float,
) -> ManagerBasedRlEnvCfg:
    """Create a velocity locomotion task configuration.

    Matches mjlab's create_velocity_env_cfg() as closely as possible.

    Args:
        robot: Robot name (e.g., "unitreego1").
        action_scale: Scale factor for actions (float or dict of joint scales).
        trunk_body_name: Name of the trunk/torso body for orientation rewards.
        posture_std_standing: Joint std for standing posture reward.
        posture_std_walking: Joint std for walking posture reward.
        posture_std_running: Joint std for running posture reward.
        body_ang_vel_weight: Weight for body angular velocity penalty.

    Returns:
        Complete ManagerBasedRlEnvCfg for velocity task.
    """

    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            asset_name="robot",
            actuator_names=(".*",),
            scale=action_scale,
            use_default_offset=True,
        ),
    }

    commands: dict[str, CommandTermCfg] = {
        "twist": UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(3.0, 8.0),
            rel_standing_envs=0.1,
            rel_heading_envs=0.3,
            heading_command=True,
            heading_control_stiffness=0.5,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0),
                lin_vel_y=(-1.0, 1.0),
                ang_vel_z=(-0.5, 0.5),
                heading=(-math.pi, math.pi),
            ),
        ),
    }

    policy_terms: dict[str, ObservationTermCfg] = {
        "base_lin_vel": ObservationTermCfg(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.5, n_max=0.5),
        ),
        "base_ang_vel": ObservationTermCfg(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
        "command": ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "twist"},
        ),
    }

    critic_terms: dict[str, ObservationTermCfg] = {
        **policy_terms,
        "foot_height": ObservationTermCfg(
            func=mdp.foot_height
        ),
        "foot_air_time": ObservationTermCfg(
            func=mdp.foot_air_time
        ),
        "foot_contact": ObservationTermCfg(
            func=mdp.foot_contact
        ),
        "foot_contact_forces": ObservationTermCfg(
            func=mdp.foot_contact_forces
        ),
    }

    observations: dict[str, ObservationGroupCfg] = {
        "policy": ObservationGroupCfg(
            terms=policy_terms,
            concatenate_terms=True,
            enable_corruption=True,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    rewards: dict[str, RewardTermCfg] = {
        "track_linear_velocity": RewardTermCfg(
            func=mdp.track_linear_velocity,
            weight=2.0,
            params={"command_name": "twist", "std": math.sqrt(0.25)},
        ),
        "track_angular_velocity": RewardTermCfg(
            func=mdp.track_angular_velocity,
            weight=2.0,
            params={"command_name": "twist", "std": math.sqrt(0.5)},
        ),
        "upright": RewardTermCfg(
            func=mdp.flat_orientation,
            weight=1.0,
            params={
                "std": math.sqrt(0.2),
                "asset_cfg": SceneEntityCfg("robot", body_names=(trunk_body_name,)),
            },
        ),
        "pose": RewardTermCfg(
            func=mdp.variable_posture,
            weight=1.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                "command_name": "twist",
                "std_standing": posture_std_standing,
                "std_walking": posture_std_walking,
                "std_running": posture_std_running,
                "walking_threshold": 0.05,
                "running_threshold": 1.5,
            },
        ),
        "is_alive": RewardTermCfg(func=mdp.is_alive, weight=0.5),
        "body_ang_vel": RewardTermCfg(
            func=mdp.body_angular_velocity_penalty,
            weight=body_ang_vel_weight,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=(trunk_body_name,))},
        ),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
    }

    terminations: dict[str, TerminationTermCfg] = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(
            func=mdp.bad_orientation,
            params={"limit_angle": math.radians(70.0)},
        ),
    }

    curriculum: dict[str, CurriculumTermCfg] = {
        "terrain_levels": CurriculumTermCfg(
            func=mdp.terrain_levels_vel,
            params={"command_name": "twist"},
        ),
        "command_vel": CurriculumTermCfg(
            func=mdp.commands_vel,
            params={
                "command_name": "twist",
                "velocity_stages": [
                    {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
                    {"step": 5000 * 24, "lin_vel_x": (-1.5, 2.0), "ang_vel_z": (-0.7, 0.7)},
                    {"step": 10000 * 24, "lin_vel_x": (-2.0, 3.0)},
                ],
            },
        ),
    }

    return ManagerBasedRlEnvCfg(
        decimation=4,
        episode_length_s=20.0,
        sim_dt=0.02,
        robot=robot,
        scene="velocity",
        task="locomotion",
        actions=actions,
        commands=commands,
        observations=observations,
        rewards=rewards,
        terminations=terminations,
        curriculum=curriculum,
    )
