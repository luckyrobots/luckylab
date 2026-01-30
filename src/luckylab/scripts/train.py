#!/usr/bin/env python3
"""
Train a policy using skrl.

Usage:
    # PPO training (default, PyTorch backend)
    python -m luckylab.scripts.train --task go1_velocity_flat

    # SAC training
    python -m luckylab.scripts.train --task go1_velocity_flat --algorithm sac

    # JAX backend
    python -m luckylab.scripts.train --task go1_velocity_flat --backend jax

    # With GPU
    python -m luckylab.scripts.train --task go1_velocity_flat --device cuda

    # With W&B logging
    python -m luckylab.scripts.train --task go1_velocity_flat --logger wandb

    # Custom timesteps and seed
    python -m luckylab.scripts.train --task go1_velocity_flat --timesteps 500000 --seed 123

Examples:
    # Basic PPO training (PyTorch)
    python -m luckylab.scripts.train --task go1_velocity_flat

    # PPO with JAX backend
    python -m luckylab.scripts.train --task go1_velocity_flat --backend jax --device cuda:0

    # SAC training on GPU with W&B
    python -m luckylab.scripts.train --task go1_velocity_flat --algorithm sac --device cuda --logger wandb

    # Quick test run
    python -m luckylab.scripts.train --task go1_velocity_flat --timesteps 1000
"""

import argparse
import copy
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a reinforcement learning policy using skrl.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Task and algorithm
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
        "--backend",
        type=str,
        default=None,
        choices=["torch", "jax"],
        help="Deep learning backend (default: torch)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run training on (default: cpu)",
    )

    # Training parameters
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override total training timesteps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed for reproducibility",
    )

    # Experiment naming
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Override experiment name for logging",
    )

    # Logging options
    parser.add_argument(
        "--logger",
        type=str,
        default=None,
        choices=["tensorboard", "wandb"],
        help="Logger backend (default: tensorboard)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project name (default: luckylab)",
    )

    # Checkpoint options
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help="Override checkpoint save interval (timesteps)",
    )

    args = parser.parse_args()

    # Import here to avoid slow imports when just checking --help
    from luckylab.rl import train
    from luckylab.tasks import load_env_cfg, load_rl_cfg

    # Load configurations
    logger.info(f"Loading task: {args.task}")
    try:
        env_cfg = load_env_cfg(args.task)
    except KeyError as e:
        logger.error(str(e))
        return 1

    rl_cfg = load_rl_cfg(args.task)
    if rl_cfg is None:
        # Use default config
        from luckylab.rl import SkrlCfg

        rl_cfg = SkrlCfg()
        logger.info("Using default RL configuration")
    else:
        # Make a copy to avoid modifying the original
        rl_cfg = copy.deepcopy(rl_cfg)

    # Apply CLI overrides
    if args.backend is not None:
        rl_cfg.backend = args.backend
        logger.info(f"Using backend: {args.backend}")

    if args.algorithm is not None:
        rl_cfg.algorithm = args.algorithm
        logger.info(f"Using algorithm: {args.algorithm}")

    if args.timesteps is not None:
        rl_cfg.timesteps = args.timesteps
        logger.info(f"Overriding timesteps: {args.timesteps}")

    if args.seed is not None:
        rl_cfg.seed = args.seed
        logger.info(f"Overriding seed: {args.seed}")

    if args.experiment_name is not None:
        rl_cfg.experiment_name = args.experiment_name
        logger.info(f"Overriding experiment name: {args.experiment_name}")

    if args.logger is not None:
        rl_cfg.logger = args.logger
        logger.info(f"Using logger: {args.logger}")

    if args.wandb_project is not None:
        rl_cfg.wandb_project = args.wandb_project

    if args.checkpoint_interval is not None:
        rl_cfg.checkpoint_interval = args.checkpoint_interval

    # Print configuration summary
    logger.info("=" * 60)
    logger.info("Training Configuration")
    logger.info("=" * 60)
    logger.info(f"  Task: {args.task}")
    logger.info(f"  Algorithm: {rl_cfg.algorithm.upper()}")
    logger.info(f"  Backend: {rl_cfg.backend}")
    logger.info(f"  Device: {args.device}")
    logger.info(f"  Timesteps: {rl_cfg.timesteps:,}")
    logger.info(f"  Seed: {rl_cfg.seed}")
    logger.info(f"  Experiment: {rl_cfg.experiment_name}")
    logger.info(f"  Logger: {rl_cfg.logger}")
    logger.info("=" * 60)

    # Run training
    try:
        train(env_cfg=env_cfg, rl_cfg=rl_cfg, device=args.device)
        logger.info("Training complete!")
        return 0
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
