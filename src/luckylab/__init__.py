"""LuckyLab - RL training framework for LuckyRobots."""

from gymnasium.envs.registration import register

from .envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from .tasks.velocity import GO1_ENV_CFG

# Register Gymnasium environment
register(
    id="luckylab/Go1-Velocity-v0",
    entry_point="luckylab.envs:ManagerBasedRlEnv",
    max_episode_steps=GO1_ENV_CFG.max_episode_length,
    kwargs={"cfg": GO1_ENV_CFG},
)

__all__ = [
    "ManagerBasedRlEnv",
    "ManagerBasedRlEnvCfg",
    "GO1_ENV_CFG",
]
