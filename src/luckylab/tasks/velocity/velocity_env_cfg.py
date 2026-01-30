"""Velocity tracking task configuration.

This module defines the configuration for velocity tracking tasks.
Robot-specific configurations are located in the config/ directory.

This follows the mjlab pattern where the factory function returns
a ManagerBasedRlEnvCfg directly with all MDP components configured.
"""

import math

from ...envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from ...managers import (
    CommandTermCfg,
    CurriculumTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from . import mdp


def create_velocity_env_cfg(
    robot: str,
    decimation: int = 4,
    episode_length_s: float = 20.0,
    sim_dt: float = 0.02,
    posture_std_standing: float | dict[str, float] = 0.1,
    posture_std_walking: float | dict[str, float] = 0.2,
    posture_std_running: float | dict[str, float] = 0.3,
    body_ang_vel_weight: float = -0.05,
    action_rate_l2_weight: float = -0.1,
) -> ManagerBasedRlEnvCfg:
    """Create a velocity locomotion task configuration.

    Args:
        robot: Robot name (e.g., "unitreego1").
        decimation: Number of sim steps per env step.
        episode_length_s: Maximum episode length in seconds.
        sim_dt: Simulation timestep in seconds.
        posture_std_standing: Joint std devs for standing posture reward.
        posture_std_walking: Joint std devs for walking posture reward.
        posture_std_running: Joint std devs for running posture reward.
        body_ang_vel_weight: Weight for body angular velocity penalty.
        action_rate_l2_weight: Weight for action rate L2 penalty.

    Returns:
        Complete ManagerBasedRlEnvCfg for velocity task.
    """
    rewards: dict[str, RewardTermCfg] = {
        "track_linear_velocity": RewardTermCfg(
            func=mdp.track_linear_velocity,
            weight=2.0,
            params={"std": math.sqrt(0.25)},
        ),
        "track_angular_velocity": RewardTermCfg(
            func=mdp.track_angular_velocity,
            weight=2.0,
            params={"std": math.sqrt(0.5)},
        ),
        "variable_posture": RewardTermCfg(
            func=mdp.variable_posture,
            weight=1.0,
            params={
                "std_standing": posture_std_standing,
                "std_walking": posture_std_walking,
                "std_running": posture_std_running,
                "walking_threshold": 0.05,
                "running_threshold": 1.5,
            },
        ),
        "body_angular_velocity_penalty": RewardTermCfg(
            func=mdp.body_angular_velocity_penalty,
            weight=body_ang_vel_weight,
        ),
        "joint_pos_limits": RewardTermCfg(
            func=mdp.joint_pos_limits,
            weight=-1.0,
        ),
        "action_rate_l2": RewardTermCfg(
            func=mdp.action_rate_l2,
            weight=action_rate_l2_weight,
        ),
    }

    terminations: dict[str, TerminationTermCfg] = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(
            func=mdp.bad_orientation,
            params={"limit_angle": math.radians(70.0)},
        ),
    }

    commands: dict[str, CommandTermCfg] = {
        "velocity": CommandTermCfg(
            func=mdp.velocity_command,
            params={
                "x_range": (-1.0, 1.0),
                "y_range": (-0.5, 0.5),
                "yaw_range": (-1.0, 1.0),
            },
            resampling_time_range=(5.0, 10.0),
        ),
    }

    curriculum: dict[str, CurriculumTermCfg] = {
        "terrain_difficulty": CurriculumTermCfg(
            func=mdp.terrain_curriculum,
            params={
                "stages": [
                    {"step": 0, "difficulty": 0.0},
                    {"step": 50000, "difficulty": 0.5},
                    {"step": 100000, "difficulty": 1.0},
                ],
            },
        ),
        "command_velocity": CurriculumTermCfg(
            func=mdp.velocity_curriculum,
            params={
                "command_name": "velocity",
                "stages": [
                    {"step": 0, "max_velocity": 0.5},
                    {"step": 50000, "max_velocity": 1.0},
                ],
            },
        ),
    }

    return ManagerBasedRlEnvCfg(
        decimation=decimation,
        episode_length_s=episode_length_s,
        sim_dt=sim_dt,
        robot=robot,
        scene="velocity",
        task="locomotion",
        rewards=rewards,
        terminations=terminations,
        commands=commands,
        curriculum=curriculum,
    )
