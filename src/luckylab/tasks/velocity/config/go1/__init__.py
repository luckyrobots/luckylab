"""Unitree Go1 configuration for velocity task."""

from luckylab.tasks.velocity.config.go1.env_cfg import (
    GO1_ACTION_SCALE as GO1_ACTION_SCALE,
)
from luckylab.tasks.velocity.config.go1.env_cfg import GO1_ENV_CFG as GO1_ENV_CFG
from luckylab.tasks.velocity.config.go1.env_cfg import (
    UNITREE_GO1_FLAT_ENV_CFG as UNITREE_GO1_FLAT_ENV_CFG,
)
from luckylab.tasks.velocity.config.go1.env_cfg import (
    UNITREE_GO1_ROUGH_ENV_CFG as UNITREE_GO1_ROUGH_ENV_CFG,
)
from luckylab.tasks.velocity.config.go1.rl_cfg import (
    UNITREE_GO1_PPO_RUNNER_CFG as UNITREE_GO1_PPO_RUNNER_CFG,
)
from luckylab.tasks.velocity.config.go1.rl_cfg import (
    UNITREE_GO1_SAC_RUNNER_CFG as UNITREE_GO1_SAC_RUNNER_CFG,
)

# Convenient aliases
GO1_RL_CFG = UNITREE_GO1_PPO_RUNNER_CFG
GO1_PPO_CFG = UNITREE_GO1_PPO_RUNNER_CFG
GO1_SAC_CFG = UNITREE_GO1_SAC_RUNNER_CFG
