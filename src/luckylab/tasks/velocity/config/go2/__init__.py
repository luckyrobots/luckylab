from luckylab.tasks.registry import register_task

from .env_cfgs import (
    UNITREE_GO2_FLAT_ENV_CFG,
    UNITREE_GO2_ROUGH_ENV_CFG,
)
from .rl_cfg import (
    UNITREE_GO2_PPO_RUNNER_CFG,
    UNITREE_GO2_SAC_RUNNER_CFG,
)

register_task(
    task_id="go2_velocity_rough",
    env_cfg=UNITREE_GO2_ROUGH_ENV_CFG,
    rl_cfgs={
        "ppo": UNITREE_GO2_PPO_RUNNER_CFG,
        "sac": UNITREE_GO2_SAC_RUNNER_CFG,
    },
)

register_task(
    task_id="go2_velocity_flat",
    env_cfg=UNITREE_GO2_FLAT_ENV_CFG,
    rl_cfgs={
        "ppo": UNITREE_GO2_PPO_RUNNER_CFG,
        "sac": UNITREE_GO2_SAC_RUNNER_CFG,
    },
)
