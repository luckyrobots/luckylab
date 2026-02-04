from luckylab.tasks.registry import register_task

from .env_cfgs import (
    UNITREE_GO1_FLAT_ENV_CFG,
    UNITREE_GO1_ROUGH_ENV_CFG,
)
from .rl_cfg import (
    UNITREE_GO1_PPO_RUNNER_CFG,
    UNITREE_GO1_SAC_RUNNER_CFG,
)

register_task(
    task_id="go1_velocity_rough",
    env_cfg=UNITREE_GO1_ROUGH_ENV_CFG,
    rl_cfgs={
        "ppo": UNITREE_GO1_PPO_RUNNER_CFG,
        "sac": UNITREE_GO1_SAC_RUNNER_CFG,
    },
)

register_task(
    task_id="go1_velocity_flat",
    env_cfg=UNITREE_GO1_FLAT_ENV_CFG,
    rl_cfgs={
        "ppo": UNITREE_GO1_PPO_RUNNER_CFG,
        "sac": UNITREE_GO1_SAC_RUNNER_CFG,
    },
)
