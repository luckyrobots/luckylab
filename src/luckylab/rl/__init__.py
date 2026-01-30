"""
skrl reinforcement learning integration.

Supports PPO, SAC, TD3, and DDPG with minimal configuration.
Uses skrl's native training loop, logging, and checkpointing.

Example:
    >>> from luckylab.rl import train, SkrlCfg
    >>> from luckylab.tasks.velocity import GO1_ENV_CFG
    >>>
    >>> # Train with PPO (default)
    >>> train(GO1_ENV_CFG, SkrlCfg(timesteps=100_000))
    >>>
    >>> # Train with SAC
    >>> train(GO1_ENV_CFG, SkrlCfg(algorithm="sac", timesteps=100_000))
    >>>
    >>> # Train with TD3 and W&B logging
    >>> train(GO1_ENV_CFG, SkrlCfg(algorithm="td3", logger="wandb"))
"""

from .config import ActorCriticCfg, DdpgCfg, PpoCfg, SacCfg, SkrlCfg, Td3Cfg
from .trainer import load_agent, train
from .wrapper import SkrlWrapper, wrap_env

__all__ = [
    # Config
    "SkrlCfg",
    "ActorCriticCfg",
    "PpoCfg",
    "SacCfg",
    "Td3Cfg",
    "DdpgCfg",
    # Training
    "train",
    "load_agent",
    # Wrapper
    "SkrlWrapper",
    "wrap_env",
]
