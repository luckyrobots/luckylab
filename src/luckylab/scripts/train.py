#!/usr/bin/env python3
"""
Train a policy — supports both RL and IL modes.

The mode is inferred from the task registry: tasks with rl_cfgs route to RL,
tasks with il_cfgs route to IL. If a task has both, the CLI args determine the mode.

RL usage:
    python -m luckylab.scripts.train go2_velocity_flat --agent.algorithm sac --agent.backend skrl
    python -m luckylab.scripts.train go2_velocity_flat --device cuda

IL usage:
    python -m luckylab.scripts.train so100_pickplace --il.policy act --il.dataset-repo-id lerobot/so100_pickplace
"""

import sys
from dataclasses import dataclass
from typing import Any

import tyro

from luckylab.utils.logging import print_info
from luckylab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class TrainRlConfig:
    """RL training configuration."""

    env: Any
    """Environment configuration."""
    agent: Any
    """RL agent configuration."""
    device: str = "cpu"
    """Device to run training on."""


@dataclass(frozen=True)
class TrainIlConfig:
    """IL training configuration."""

    il: Any
    """IL runner configuration."""
    device: str = "cpu"
    """Device to run training on."""


def run_train_rl(cfg: TrainRlConfig) -> int:
    """Run RL training with the given configuration."""
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


def run_train_il(cfg: TrainIlConfig) -> int:
    """Run IL training with the given configuration."""
    from luckylab.il import train

    if cfg.device.startswith("cuda"):
        configure_torch_backends()

    try:
        train(il_cfg=cfg.il, device=cfg.device)
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


def _has_il_args(args: list[str]) -> bool:
    """Check if any --il.* args are present."""
    return any(arg.startswith("--il.") or arg.startswith("--il-") for arg in args)


def _has_rl_args(args: list[str]) -> bool:
    """Check if any --agent.* args are present."""
    return any(arg.startswith("--agent.") or arg.startswith("--agent-") for arg in args)


def main() -> int:
    # Import tasks to populate the registry
    import luckylab.tasks  # noqa: F401
    from luckylab.tasks import (
        list_il_policies,
        list_rl_policies,
        list_tasks,
        load_env_cfg,
        load_il_cfg,
        load_rl_cfg,
    )

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

    has_rl = bool(list_rl_policies(chosen_task))
    has_il = bool(list_il_policies(chosen_task))

    # Determine mode: RL vs IL
    if has_rl and has_il:
        # Both available — infer from args
        if _has_il_args(remaining_args):
            mode = "il"
        elif _has_rl_args(remaining_args):
            mode = "rl"
        else:
            print_info(
                f"Task '{chosen_task}' supports both RL and IL. "
                "Use --agent.* args for RL or --il.* args for IL.",
                color="red",
            )
            return 1
    elif has_il:
        mode = "il"
    elif has_rl:
        mode = "rl"
    else:
        print_info(
            f"Task '{chosen_task}' has no registered RL or IL configs. "
            "Use --agent.* args for RL or --il.* args for IL.",
            color="red",
        )
        return 1

    if mode == "il":
        # IL training path
        policy_type = _extract_arg(remaining_args, "il.policy", default="act")
        il_cfg = load_il_cfg(chosen_task, policy_type)

        if il_cfg is None:
            from luckylab.il import IlRunnerCfg

            il_cfg = IlRunnerCfg(policy=policy_type)
            print_info(f"Using default IL configuration for {policy_type.upper()}")

        args = tyro.cli(
            TrainIlConfig,
            args=remaining_args,
            default=TrainIlConfig(il=il_cfg),
            prog=sys.argv[0] + f" {chosen_task}",
            config=(
                tyro.conf.AvoidSubcommands,
                tyro.conf.FlagConversionOff,
            ),
        )
        return run_train_il(args)

    else:
        # RL training path (existing flow)
        from luckylab.rl import RlRunnerCfg

        try:
            env_cfg = load_env_cfg(chosen_task)
        except (KeyError, ValueError) as e:
            print_info(str(e), color="red")
            return 1

        # Extract algorithm from args (required).
        algorithm = _extract_arg(remaining_args, "agent.algorithm")
        if not algorithm:
            available = ", ".join(list_rl_policies(chosen_task)) or "ppo, sac, td3, ddpg"
            print_info(f"--agent.algorithm is required. Available: {available}", color="red")
            return 1

        # Extract backend from args (required).
        backend = _extract_arg(remaining_args, "agent.backend")
        if not backend:
            print_info("--agent.backend is required. Available: skrl, sb3", color="red")
            return 1

        agent_cfg = load_rl_cfg(chosen_task, algorithm) or RlRunnerCfg(algorithm=algorithm)

        args = tyro.cli(
            TrainRlConfig,
            args=remaining_args,
            default=TrainRlConfig(env=env_cfg, agent=agent_cfg),
            prog=sys.argv[0] + f" {chosen_task}",
            config=(
                tyro.conf.AvoidSubcommands,
                tyro.conf.FlagConversionOff,
            ),
        )

        return run_train_rl(args)


if __name__ == "__main__":
    sys.exit(main())
