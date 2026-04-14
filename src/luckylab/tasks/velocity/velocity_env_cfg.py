"""Velocity tracking task configuration.

This module defines the configuration for velocity tracking tasks.
Robot-specific configurations are located in the config/ directory.
"""

import math

from luckylab.configs.simulation_contract import SimulationContract
from luckylab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from luckylab.envs.mdp.actions import CPGActionCfg
from luckylab.managers.manager_term_config import (
    ActionTermCfg,
    CurriculumTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from luckylab.tasks.velocity import mdp
from luckylab.utils.noise import UniformNoiseCfg as Unoise


def create_velocity_env_cfg(
    robot: str,
    action_scale: float | dict[str, float],
) -> ManagerBasedRlEnvCfg:
    """Create a velocity locomotion task configuration.

    Args:
        robot: Robot name (e.g., "unitreego2").
        action_scale: Scale factor for actions (float or dict of joint scales).
    Returns:
        Complete ManagerBasedRlEnvCfg for velocity task.
    """

    actions: dict[str, ActionTermCfg] = {
        "joint_pos": CPGActionCfg(
            asset_name="robot",
            actuator_names=(".*",),
            scale=action_scale,
            use_default_offset=True,
            frequency=2.0,
            amplitude_hip=0.12,
            amplitude_thigh=0.50,
            amplitude_calf=0.50,
            calf_phase_offset=math.pi / 4.0,
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
        "gait_phase": ObservationTermCfg(func=mdp.gait_phase),
    }

    observations: dict[str, ObservationGroupCfg] = {
        "policy": ObservationGroupCfg(
            terms=policy_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    rewards: dict[str, RewardTermCfg] = {
        # --- Primary tracking (dominant signal) ---
        "track_linear_velocity": RewardTermCfg(
            func=mdp.track_linear_velocity,
            weight=2.0,
            params={"sigma": 0.25},
        ),
        "track_angular_velocity": RewardTermCfg(
            func=mdp.track_angular_velocity,
            weight=1.5,
            params={"sigma": 0.25},
        ),
        # --- Gait shaping (secondary, NOT dominant) ---
        "feet_air_time": RewardTermCfg(
            func=mdp.feet_air_time,
            weight=0.2,
            params={"threshold_min": 0.05, "threshold_max": 0.5},
        ),
        "foot_clearance": RewardTermCfg(
            func=mdp.foot_clearance,
            weight=0.25,
            params={"max_height": 0.10},
        ),
        "foot_slip": RewardTermCfg(
            func=mdp.foot_slip_l2,
            weight=-0.25,
        ),
        "stand_still": RewardTermCfg(
            func=mdp.stand_still,
            weight=2.0,
        ),
        # --- Stability penalties ---
        "lin_vel_z": RewardTermCfg(
            func=mdp.lin_vel_z_l2,
            weight=-2.0,
        ),
        "ang_vel_xy": RewardTermCfg(
            func=mdp.ang_vel_xy_l2,
            weight=-0.05,
        ),
        "orientation": RewardTermCfg(
            func=mdp.flat_orientation_l2,
            weight=-1.0,
        ),
        # --- Smoothness penalties ---
        "action_rate": RewardTermCfg(
            func=mdp.action_rate_l2,
            weight=-0.01,
        ),
        "action_magnitude": RewardTermCfg(
            func=mdp.action_l2,
            weight=-0.01,
        ),
    }

    terminations: dict[str, TerminationTermCfg] = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(
            func=mdp.bad_orientation,
            params={"limit_angle": math.radians(70.0)},
        ),
    }

    curriculum: dict[str, CurriculumTermCfg] = {
        "commands_vel": CurriculumTermCfg(
            func=mdp.commands_vel,
            params={
                "velocity_stages": [
                    # Phase 1 (0-500k): Forward/backward only
                    {"step": 0, "lin_vel_x": (-0.5, 0.5), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (0.0, 0.0)},
                    # Phase 2 (500k-1M): Add yaw turning
                    {"step": 500_000, "lin_vel_x": (-0.8, 0.8), "ang_vel_z": (-0.5, 0.5)},
                    # Phase 3 (1M-1.5M): Add lateral movement
                    {"step": 1_000_000, "lin_vel_x": (-0.8, 0.8), "lin_vel_y": (-0.3, 0.3), "ang_vel_z": (-0.8, 0.8)},
                    # Phase 4 (1.5M+): Full range
                    {"step": 1_500_000, "lin_vel_x": (-1.0, 1.0), "lin_vel_y": (-0.6, 0.6), "ang_vel_z": (-1.0, 1.0)},
                ],
            },
        ),
        "cpg_amplitude": CurriculumTermCfg(
            func=mdp.cpg_amplitude,
            params={
                "action_term_name": "joint_pos",
                "stages": [
                    {"step": 2_000_000, "blend": 0.75},
                    {"step": 2_250_000, "blend": 0.50},
                    {"step": 2_500_000, "blend": 0.25},
                    {"step": 2_750_000, "blend": 0.0},
                ],
            },
        ),
    }

    contract = SimulationContract(
        vel_command_x_range=(-0.5, 0.5),
        vel_command_y_range=(0.0, 0.0),
        vel_command_yaw_range=(0.0, 0.0),
        vel_command_standing_probability=0.2,
        vel_command_resampling_time_range=(5.0, 10.0),
    )

    # Task contract for engine-side MDP computation.
    # When set on ManagerBasedRlEnvCfg, the engine computes these reward signals
    # and termination flags alongside observations, reducing Python overhead and
    # enabling high-frequency signals (foot contact, slip) from MuJoCo data.
    from luckylab.contracts.task_contract import (
        ObservationContract,
        ObservationTermRequest,
        RewardContract,
        RewardTermRequest,
        TaskContract,
        TerminationContract,
        TerminationTermRequest,
    )

    task_contract = TaskContract(
        task_id=f"{robot}_velocity_flat",
        robot=robot,
        scene="velocity",
        observations=ObservationContract(
            required=[
                ObservationTermRequest("base_lin_vel"),
                ObservationTermRequest("base_ang_vel"),
                ObservationTermRequest("projected_gravity"),
                ObservationTermRequest("joint_pos"),
                ObservationTermRequest("joint_vel"),
                ObservationTermRequest("vel_command"),
                ObservationTermRequest("actions"),
            ],
            optional=[
                ObservationTermRequest("foot_contact",
                    params={"foot_geom_names": "FL_foot,FR_foot,RL_foot,RR_foot"}),
                ObservationTermRequest("foot_heights",
                    params={"foot_site_names": "FL_foot_site,FR_foot_site,RL_foot_site,RR_foot_site"}),
            ],
        ),
        rewards=RewardContract(
            engine_terms=[
                RewardTermRequest("track_linear_velocity", weight=2.0, params={"scale": "3.0"}),
                RewardTermRequest("track_angular_velocity", weight=1.5, params={"scale": "1.5"}),
                RewardTermRequest("feet_air_time", weight=0.2, params={"target_air_time": "0.25"}),
                RewardTermRequest("foot_slip_penalty", weight=-0.25),
                RewardTermRequest("stand_still", weight=2.0),
                RewardTermRequest("lin_vel_z_penalty", weight=-2.0),
                RewardTermRequest("ang_vel_xy_penalty", weight=-0.05),
                RewardTermRequest("orientation_error", weight=-1.0),
                RewardTermRequest("action_rate", weight=-0.01),
                RewardTermRequest("action_magnitude", weight=-0.01),
            ],
            python_terms=["foot_clearance"],  # Computed Python-side (needs CPG gait info)
        ),
        terminations=TerminationContract(
            terms=[
                TerminationTermRequest("time_out", is_timeout=True),
                TerminationTermRequest("fell_over", params={"threshold": "-0.5"}),
            ],
        ),
        max_episode_length_s=20.0,
    )

    return ManagerBasedRlEnvCfg(
        robot=robot,
        scene="velocity",
        task="locomotion",
        actions=actions,
        observations=observations,
        rewards=rewards,
        terminations=terminations,
        curriculum=curriculum,
        simulation_contract=contract,
        task_contract=task_contract,
        sim_dt=0.005,
        decimation=4,
        episode_length_s=20.0,
    )
