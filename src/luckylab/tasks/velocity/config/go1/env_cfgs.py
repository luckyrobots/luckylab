"""Unitree Go1 velocity tracking environment configurations."""

from copy import deepcopy

from luckylab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from luckylab.tasks.velocity.velocity_env_cfg import create_velocity_env_cfg
from luckylab.utils.retval import retval


GO1_ACTION_SCALE = {
    ".*_hip_joint": 0.3727530387083568,
    ".*_thigh_joint": 0.3727530387083568,
    ".*_calf_joint": 0.24850202580557115,
}

@retval
def UNITREE_GO1_ROUGH_ENV_CFG() -> ManagerBasedRlEnvCfg:
    """Create Unitree Go1 rough terrain velocity tracking configuration.

    Note: LuckyLab doesn't have terrain system yet, but this matches
    mjlab's UNITREE_GO1_ROUGH_ENV_CFG structure.
    """
    cfg = create_velocity_env_cfg(
        robot="unitreego1",
        action_scale=GO1_ACTION_SCALE,
        trunk_body_name="trunk",
        # Per-joint posture std using regex patterns (matches mjlab exactly)
        posture_std_standing={
            r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.05,
            r".*(FR|FL|RR|RL)_calf_joint.*": 0.1,
        },
        posture_std_walking={
            r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
            r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
        },
        posture_std_running={
            r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
            r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
        },
        body_ang_vel_weight=0.0,
    )

    # TODO: Add illegal_contact termination when contact sensors are available
    # cfg.terminations["illegal_contact"] = TerminationTermCfg(
    #     func=mdp.illegal_contact,
    #     params={"sensor_name": "nonfoot_ground_touch"},
    # )

    return cfg


@retval
def UNITREE_GO1_FLAT_ENV_CFG() -> ManagerBasedRlEnvCfg:
    """Create Unitree Go1 flat terrain velocity tracking configuration."""
    # Start with rough terrain config
    cfg = deepcopy(UNITREE_GO1_ROUGH_ENV_CFG)

    # Disable terrain curriculum.
    assert cfg.curriculum is not None
    del cfg.curriculum["terrain_levels"]

    return cfg


# Default export - flat terrain config (recommended for LuckyLab)
GO1_ENV_CFG = UNITREE_GO1_FLAT_ENV_CFG
