#!/usr/bin/env python3
"""
Train a policy using skrl.

Usage:
    python -m luckylab.scripts.train go1_velocity_flat
    python -m luckylab.scripts.train go1_velocity_flat --agent.algorithm sac
    python -m luckylab.scripts.train go1_velocity_flat --device cuda
    python -m luckylab.scripts.train go1_velocity_flat --agent.max-iterations 500
    python -m luckylab.scripts.train go1_velocity_flat --env.simulation-mode realtime
    python -m luckylab.scripts.train go1_velocity_flat --agent.clip-actions 1.0
"""

import sys
from dataclasses import dataclass
from typing import Any

import tyro

from luckylab.utils.logging import print_header, print_info


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

    # Log configuration
    print_header("Training Configuration")
    print_info(f"  Algorithm:      {cfg.agent.algorithm.upper()}")
    print_info(f"  Device:         {cfg.device}")
    print_info(f"  Max Iterations: {cfg.agent.max_iterations:,}")
    print_info(f"  Seed:           {cfg.agent.seed}")
    print_info(f"  LuckyEngine:    {cfg.env.host}:{cfg.env.port}")
    print_info(f"  Sim Mode:       {cfg.env.simulation_mode}")
    print_info(f"  Clip Actions:   {cfg.agent.clip_actions}")
    print_info(f"  Wandb:          {cfg.agent.wandb}")
    if cfg.agent.wandb:
        print_info(f"  Wandb Project:  {cfg.agent.wandb_project}")

    # Train
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


def main() -> int:
    # Import tasks to populate the registry
    import luckylab.tasks  # noqa: F401
    from luckylab.rl import RlRunnerCfg
    from luckylab.tasks import list_tasks, load_env_cfg, load_rl_cfg

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

    print_info(f"Loading task: {chosen_task}")

    # Load default configs for the chosen task
    try:
        env_cfg = load_env_cfg(chosen_task)
    except KeyError as e:
        print_info(str(e), color="red")
        return 1

    # Try to load RL config, fall back to defaults
    rl_cfg = load_rl_cfg(chosen_task, "ppo")
    if rl_cfg is None:
        rl_cfg = RlRunnerCfg()

    # Parse the rest of the arguments, allowing overrides of env and agent configs
    args = tyro.cli(
        TrainConfig,
        args=remaining_args,
        default=TrainConfig(env=env_cfg, agent=rl_cfg),
        prog=sys.argv[0] + f" {chosen_task}",
        config=(
            tyro.conf.AvoidSubcommands,
            tyro.conf.FlagConversionOff,
        ),
    )

    return run_train(args)


if __name__ == "__main__":
    sys.exit(main())
