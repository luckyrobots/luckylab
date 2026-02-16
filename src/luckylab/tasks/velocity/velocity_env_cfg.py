"""Velocity tracking task configuration.

This module defines the configuration for velocity tracking tasks.
Robot-specific configurations are located in the config/ directory.
"""

import math

from luckylab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from luckylab.envs.mdp.actions import JointPositionActionCfg
from luckylab.managers.manager_term_config import (
    ActionTermCfg,
    CurriculumTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from luckylab.managers.scene_entity_config import SceneEntityCfg
from luckylab.utils.noise import UniformNoiseCfg as Unoise
from luckylab.tasks.velocity import mdp


def create_velocity_env_cfg(
    robot: str,
    action_scale: float | dict[str, float],
    trunk_body_name: str,
) -> ManagerBasedRlEnvCfg:
    """Create a velocity locomotion task configuration.

    Args:
        robot: Robot name (e.g., "unitreego2").
        action_scale: Scale factor for actions (float or dict of joint scales).
        trunk_body_name: Name of the trunk/torso body for orientation rewards.
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
        "command": ObservationTermCfg(func=mdp.generated_commands),
        "foot_height": ObservationTermCfg(func=mdp.foot_height),
        "foot_air_time": ObservationTermCfg(func=mdp.foot_air_time),
        "foot_contact": ObservationTermCfg(func=mdp.foot_contact),
        "foot_contact_forces": ObservationTermCfg(func=mdp.foot_contact_forces),
    }

    observations: dict[str, ObservationGroupCfg] = {
        "policy": ObservationGroupCfg(
            terms=policy_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    rewards: dict[str, RewardTermCfg] = {
        # -- Core tracking (baselined exp kernel, wider std for discovery) --
        "track_linear_velocity": RewardTermCfg(
            func=mdp.track_linear_velocity,
            weight=5.0,
            params={"std": 0.7},
        ),
        "track_angular_velocity": RewardTermCfg(
            func=mdp.track_angular_velocity,
            weight=2.0,
            params={"std": 0.7},
        ),
        # -- Shaping (bridge from standing to walking) --
        "move_in_command_direction": RewardTermCfg(
            func=mdp.move_in_command_direction,
            weight=4.0,
        ),
        "upright": RewardTermCfg(
            func=mdp.flat_orientation,
            weight=1.0,
            params={
                "std": 0.7,
                "asset_cfg": SceneEntityCfg("robot", body_names=(trunk_body_name,)),
            },
        ),
        # -- Regularization --
        "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
        # -- Gait quality (ramped by curriculum, start at 0) --
        "foot_clearance": RewardTermCfg(
            func=mdp.feet_clearance,
            weight=0.0,
            params={
                "target_height": 0.1,
                "command_threshold": 0.05,
            },
        ),
        "foot_swing_height": RewardTermCfg(
            func=mdp.feet_swing_height,
            weight=0.0,
            params={
                "target_height": 0.1,
                "command_threshold": 0.05,
            },
        ),
        "foot_slip": RewardTermCfg(
            func=mdp.feet_slip,
            weight=0.0,
            params={
                "command_threshold": 0.05,
            },
        ),
        "soft_landing": RewardTermCfg(
            func=mdp.soft_landing,
            weight=0.0,
            params={
                "command_threshold": 0.05,
            },
        ),
        # -- Gait nudge (speed-gated, 0 when standing) --
        "trot_contact": RewardTermCfg(
            func=mdp.trot_contact,
            weight=0.5,
        ),
        # "posture": RewardTermCfg(
        #     func=mdp.variable_posture,
        #     weight=0.0,
        #     params={
        #         "std_standing": {
        #             ".*_hip_joint": 0.15,
        #             ".*_thigh_joint": 0.15,
        #             ".*_calf_joint": 0.15,
        #         },
        #         "std_walking": {
        #             ".*_hip_joint": 0.5,
        #             ".*_thigh_joint": 0.8,
        #             ".*_calf_joint": 0.8,
        #         },
        #         "std_running": {
        #             ".*_hip_joint": 0.8,
        #             ".*_thigh_joint": 1.2,
        #             ".*_calf_joint": 1.2,
        #         },
        #         "asset_cfg": SceneEntityCfg("robot"),
        #         "walking_threshold": 1.0,
        #         "running_threshold": 3.0,
        #     },
        # ),
    }

    terminations: dict[str, TerminationTermCfg] = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(
            func=mdp.bad_orientation,
            params={"limit_angle": math.radians(70.0)},
        ),
    }

    curriculum: dict[str, CurriculumTermCfg] = {
        "command_vel": CurriculumTermCfg(
            func=mdp.commands_vel,
            params={
                "velocity_stages": [
                    {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-1.0, 1.0)},
                    {"step": 300_000, "lin_vel_x": (-1.5, 2.0), "ang_vel_z": (-1.5, 1.5)},
                    {"step": 600_000, "lin_vel_x": (-2.0, 3.0), "ang_vel_z": (-2.0, 2.0)},
                ],
            },
        ),
        "foot_clearance_weight": CurriculumTermCfg(
            func=mdp.reward_weight,
            params={
                "reward_name": "foot_clearance",
                "weight_stages": [
                    {"step": 0, "weight": 0.0},
                    {"step": 500_000, "weight": -0.1},
                    {"step": 800_000, "weight": -0.5},
                ],
            },
        ),
        "foot_swing_height_weight": CurriculumTermCfg(
            func=mdp.reward_weight,
            params={
                "reward_name": "foot_swing_height",
                "weight_stages": [
                    {"step": 0, "weight": 0.0},
                    {"step": 500_000, "weight": -0.05},
                    {"step": 800_000, "weight": -0.25},
                ],
            },
        ),
        "foot_slip_weight": CurriculumTermCfg(
            func=mdp.reward_weight,
            params={
                "reward_name": "foot_slip",
                "weight_stages": [
                    {"step": 0, "weight": 0.0},
                    {"step": 500_000, "weight": -0.025},
                    {"step": 800_000, "weight": -0.1},
                ],
            },
        ),
        "soft_landing_weight": CurriculumTermCfg(
            func=mdp.reward_weight,
            params={
                "reward_name": "soft_landing",
                "weight_stages": [
                    {"step": 0, "weight": 0.0},
                    {"step": 500_000, "weight": -2.5e-5},
                    {"step": 800_000, "weight": -1e-4},
                ],
            },
        ),
        # "posture_weight": CurriculumTermCfg(
        #     func=mdp.reward_weight,
        #     params={
        #         "reward_name": "posture",
        #         "weight_stages": [
        #             {"step": 0, "weight": 0.0},
        #         ],
        #     },
        # ),
    }

    return ManagerBasedRlEnvCfg(
        robot=robot,
        scene="velocity",
        task="locomotion",
        actions=actions,
        observations=observations,
        rewards=rewards,
        terminations=terminations,
        curriculum=curriculum,
        sim_dt=0.005,
        decimation=4,
        episode_length_s=20.0,
    )
