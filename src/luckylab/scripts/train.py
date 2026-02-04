#!/usr/bin/env python3
"""
Train a policy using skrl.

Usage:
    python -m luckylab.scripts.train --task go1_velocity_flat
    python -m luckylab.scripts.train --task go1_velocity_flat --algorithm sac
    python -m luckylab.scripts.train --task go1_velocity_flat --device cuda
    python -m luckylab.scripts.train --task go1_velocity_flat --max-iterations 500 --seed 123
"""

import argparse
import copy
import sys

from luckylab.utils.logging import print_header, print_info


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a reinforcement learning policy using skrl.",
    )

    parser.add_argument(
        "--task",
        type=str,
        default="go1_velocity_flat",
        help="Task identifier (default: go1_velocity_flat)",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        choices=["ppo", "sac", "td3", "ddpg"],
        help="RL algorithm (default: from task config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run training on (default: cpu)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum number of training iterations",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="LuckyEngine gRPC host address",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="LuckyEngine gRPC port",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=None,
        help="Enable wandb logging",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="W&B entity (team/user)",
    )

    args = parser.parse_args()

    from luckylab.rl import RlRunnerCfg, train
    from luckylab.tasks import load_env_cfg, load_rl_cfg

    # Load environment config
    print_info(f"Loading task: {args.task}")
    try:
        env_cfg = load_env_cfg(args.task)
    except KeyError as e:
        print_info(str(e), color="red")
        return 1

    # Load RL config for specified algorithm
    algorithm = args.algorithm or "ppo"
    rl_cfg = load_rl_cfg(args.task, algorithm)
    if rl_cfg is None:
        print_info(f"No RL config found for algorithm '{algorithm}', using defaults", color="yellow")
        rl_cfg = RlRunnerCfg(algorithm=algorithm)
    else:
        rl_cfg = copy.deepcopy(rl_cfg)

    if args.max_iterations is not None:
        rl_cfg.max_iterations = args.max_iterations

    if args.seed is not None:
        rl_cfg.seed = args.seed

    if args.host is not None:
        env_cfg.host = args.host

    if args.port is not None:
        env_cfg.port = args.port

    if args.wandb:
        rl_cfg.wandb = True
    elif args.no_wandb:
        rl_cfg.wandb = False

    if args.wandb_project is not None:
        rl_cfg.wandb_project = args.wandb_project

    if args.wandb_entity is not None:
        rl_cfg.wandb_entity = args.wandb_entity

    # Log configuration
    print_header("Training Configuration")
    print_info(f"  Task:           {args.task}")
    print_info(f"  Algorithm:      {rl_cfg.algorithm.upper()}")
    print_info(f"  Device:         {args.device}")
    print_info(f"  Max Iterations: {rl_cfg.max_iterations:,}")
    print_info(f"  Seed:           {rl_cfg.seed}")
    print_info(f"  LuckyEngine:    {env_cfg.host}:{env_cfg.port}")
    print_info(f"  Wandb:          {rl_cfg.wandb}")
    if rl_cfg.wandb:
        print_info(f"  Wandb Project:  {rl_cfg.wandb_project}")

    # Train
    try:
        train(env_cfg=env_cfg, rl_cfg=rl_cfg, device=args.device)
        print_info("Training complete!")
        return 0
    except KeyboardInterrupt:
        print_info("Training interrupted by user", color="yellow")
        return 130
    except Exception as e:
        print_info(f"Training failed: {e}", color="red")
        raise


if __name__ == "__main__":
    sys.exit(main())
