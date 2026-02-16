"""Unitree Go2 velocity tracking environment configurations."""

from copy import deepcopy

from luckylab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from luckylab.managers.manager_term_config import TerminationTermCfg
from luckylab.tasks.velocity import mdp
from luckylab.tasks.velocity.velocity_env_cfg import create_velocity_env_cfg
from luckylab.utils.retval import retval


GO2_ACTION_SCALE = {
    ".*_hip_joint": 0.3727530387083568,
    ".*_thigh_joint": 0.3727530387083568,
    ".*_calf_joint": 0.24850202580557115,
}

@retval
def UNITREE_GO2_ROUGH_ENV_CFG() -> ManagerBasedRlEnvCfg:
    """Create Unitree Go2 rough terrain velocity tracking configuration."""
    cfg = create_velocity_env_cfg(
        robot="unitreego2",
        action_scale=GO2_ACTION_SCALE,
        trunk_body_name="base",
    )

    assert cfg.terminations is not None
    cfg.terminations["illegal_contact"] = TerminationTermCfg(
        func=mdp.illegal_contact,
    )

    return cfg


@retval
def UNITREE_GO2_FLAT_ENV_CFG() -> ManagerBasedRlEnvCfg:
    """Create Unitree Go2 flat terrain velocity tracking configuration."""
    # Start with rough terrain config
    cfg = deepcopy(UNITREE_GO2_ROUGH_ENV_CFG)

    return cfg


# Default export - flat terrain config (recommended for LuckyLab)
GO2_ENV_CFG = UNITREE_GO2_FLAT_ENV_CFG
