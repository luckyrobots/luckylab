"""RL training dispatcher — routes to the configured backend (skrl or sb3)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from luckylab.envs import ManagerBasedRlEnvCfg
    from luckylab.rl.config import RlRunnerCfg

_VALID_BACKENDS = ("skrl", "sb3")


def _check_backend(backend: str) -> None:
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"Invalid backend: {backend!r}. Must be one of {_VALID_BACKENDS}."
        )


def train(env_cfg: ManagerBasedRlEnvCfg, rl_cfg: RlRunnerCfg, device: str = "cpu") -> None:
    """Train an RL agent using the configured backend.

    Args:
        env_cfg: Environment configuration.
        rl_cfg: RL runner configuration (includes ``backend`` field).
        device: Torch device string.
    """
    _check_backend(rl_cfg.backend)

    if rl_cfg.backend == "sb3":
        from luckylab.rl.sb3.trainer import train as sb3_train
        sb3_train(env_cfg, rl_cfg, device)
    elif rl_cfg.backend == "skrl":
        from luckylab.rl.skrl.trainer import train as skrl_train
        skrl_train(env_cfg, rl_cfg, device)


def load_agent(
    checkpoint_path: str,
    env_cfg: ManagerBasedRlEnvCfg,
    rl_cfg: RlRunnerCfg,
    device: str = "cpu",
):
    """Load a trained agent from checkpoint using the configured backend.

    Returns:
        Tuple of (agent_or_model, wrapped_env).
    """
    _check_backend(rl_cfg.backend)

    if rl_cfg.backend == "sb3":
        from luckylab.rl.sb3.trainer import load_agent as sb3_load
        return sb3_load(checkpoint_path, env_cfg, rl_cfg, device)
    elif rl_cfg.backend == "skrl":
        from luckylab.rl.skrl.trainer import load_agent as skrl_load
        return skrl_load(checkpoint_path, env_cfg, rl_cfg, device)
