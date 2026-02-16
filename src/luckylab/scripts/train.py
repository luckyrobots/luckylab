#!/usr/bin/env python3
"""
Train a policy using skrl.

Usage:
    python -m luckylab.scripts.train go2_velocity_flat
    python -m luckylab.scripts.train go2_velocity_flat --agent.algorithm sac
    python -m luckylab.scripts.train go2_velocity_flat --device cuda
    python -m luckylab.scripts.train go2_velocity_flat --agent.max-iterations 500
    python -m luckylab.scripts.train go2_velocity_flat --env.simulation-mode realtime
    python -m luckylab.scripts.train go2_velocity_flat --agent.clip-actions 1.0
"""

import sys
from dataclasses import dataclass
from typing import Any

import tyro

from luckylab.utils.logging import print_info
from luckylab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class TrainConfig:
    """Training configuration."""

    env: Any
    """Environment configuration."""
    agent: Any
    """RL agent configuration."""
    device: str = "cpu"
    """Device to run training on."""


def run_train(cfg: TrainConfig) -> int:
    """Run training with the given configuration."""
    from luckylab.rl import train

    if cfg.device.startswith("cuda"):
        configure_torch_backends()

    try:
        train(env_cfg=cfg.env, rl_cfg=cfg.agent, device=cfg.device)
        print_info("Training complete!")
        return 0
    except KeyboardInterrupt:
        print_info("Training interrupted by user", color="yellow")
        return 130
    except Exception as e:
        print_info(f"Training failed: {e}", color="red")
        raise


def _extract_arg(args: list[str], name: str, default: str = "") -> str:
    """Extract a --name value from an arg list (supports --name val and --name=val)."""
    flag = f"--{name}"
    flag_alt = f"--{name.replace('.', '-')}"
    for i, arg in enumerate(args):
        if arg in (flag, flag_alt) and i + 1 < len(args):
            return args[i + 1].lower()
        for prefix in (f"{flag}=", f"{flag_alt}="):
            if arg.startswith(prefix):
                return arg.split("=", 1)[1].lower()
    return default


def main() -> int:
    # Import tasks to populate the registry
    import luckylab.tasks  # noqa: F401
    from luckylab.rl import RlRunnerCfg
    from luckylab.tasks import list_algorithms, list_tasks, load_env_cfg, load_rl_cfg

    all_tasks = list_tasks()
    if not all_tasks:
        print_info("No tasks registered!", color="red")
        return 1

    # Parse first argument to choose the task
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
    )

    # Load default configs for the chosen task
    try:
        env_cfg = load_env_cfg(chosen_task)
    except KeyError as e:
        print_info(str(e), color="red")
        return 1

    # Extract algorithm from args (required).
    algorithm = _extract_arg(remaining_args, "agent.algorithm")
    if not algorithm:
        available = ", ".join(list_algorithms(chosen_task)) or "ppo, sac, td3, ddpg"
        print_info(f"--agent.algorithm is required. Available: {available}", color="red")
        return 1

    # Extract backend from args (required).
    backend = _extract_arg(remaining_args, "agent.backend")
    if not backend:
        print_info("--agent.backend is required. Available: skrl, sb3", color="red")
        return 1

    agent_cfg = load_rl_cfg(chosen_task, algorithm) or RlRunnerCfg(algorithm=algorithm)

    args = tyro.cli(
        TrainConfig,
        args=remaining_args,
        default=TrainConfig(env=env_cfg, agent=agent_cfg),
        prog=sys.argv[0] + f" {chosen_task}",
        config=(
            tyro.conf.AvoidSubcommands,
            tyro.conf.FlagConversionOff,
        ),
    )

    return run_train(args)


if __name__ == "__main__":
    sys.exit(main())
