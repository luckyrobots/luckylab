"""Velocity tracking task for legged robots."""

from .config import GO1_ENV_CFG, GO1_RL_CFG
from .velocity_env_cfg import create_velocity_env_cfg

# Re-export ManagerBasedRlEnvCfg for convenience
from ...envs.manager_based_rl_env import ManagerBasedRlEnvCfg

# Self-register when imported
from ..registry import register_task

register_task("go1_velocity_flat", GO1_ENV_CFG, rl_cfg=GO1_RL_CFG)

__all__ = [
    # Base config (mjlab pattern)
    "ManagerBasedRlEnvCfg",
    # Factory function
    "create_velocity_env_cfg",
    # Robot-specific configs
    "GO1_ENV_CFG",
    "GO1_RL_CFG",
]
