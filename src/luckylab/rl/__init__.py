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

from luckylab.rl.config import ActorCriticCfg as ActorCriticCfg
from luckylab.rl.config import DdpgCfg as DdpgCfg
from luckylab.rl.config import PpoCfg as PpoCfg
from luckylab.rl.config import SacCfg as SacCfg
from luckylab.rl.config import SkrlCfg as SkrlCfg
from luckylab.rl.config import Td3Cfg as Td3Cfg
from luckylab.rl.trainer import load_agent as load_agent
from luckylab.rl.trainer import train as train
from luckylab.rl.wrapper import SkrlWrapper as SkrlWrapper
from luckylab.rl.wrapper import wrap_env as wrap_env
