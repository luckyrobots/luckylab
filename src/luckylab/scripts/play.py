#!/usr/bin/env python3
"""
Run a trained policy for inference/evaluation.

Usage:
    python -m luckylab.scripts.play --checkpoint runs/go1_velocity/checkpoints/agent_1000000.pt
    python -m luckylab.scripts.play --checkpoint runs/go1_velocity/checkpoints/agent.pt --episodes 20
    python -m luckylab.scripts.play --checkpoint runs/go1_velocity/checkpoints/agent.pt --device cuda

Examples:
    # Basic evaluation with 10 episodes
    python -m luckylab.scripts.play --checkpoint runs/experiment/checkpoints/agent.pt

    # Extended evaluation
    python -m luckylab.scripts.play --checkpoint runs/experiment/checkpoints/agent.pt --episodes 50

    # Evaluation on GPU
    python -m luckylab.scripts.play --checkpoint runs/experiment/checkpoints/agent.pt --device cuda
"""

import argparse
import copy
import logging
import sys

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a trained policy for inference/evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint file",
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
        help="RL algorithm used for training",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run inference on (default: cpu)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes (default: 10)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions (no sampling)",
    )

    args = parser.parse_args()

    # Import here to avoid slow imports when just checking --help
    from luckylab.rl import load_agent
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
        from luckylab.rl import SkrlCfg

        rl_cfg = SkrlCfg()
        logger.info("Using default RL configuration")
    else:
        rl_cfg = copy.deepcopy(rl_cfg)

    # Apply CLI overrides
    if args.algorithm is not None:
        rl_cfg.algorithm = args.algorithm
        logger.info(f"Using algorithm: {args.algorithm}")

    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    try:
        agent, wrapped_env = load_agent(
            checkpoint_path=args.checkpoint,
            env_cfg=env_cfg,
            rl_cfg=rl_cfg,
            device=args.device,
        )
    except FileNotFoundError:
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        return 1

    # Run evaluation
    logger.info(f"Running evaluation for {args.episodes} episodes...")
    episode_rewards = []
    episode_lengths = []

    try:
        for ep in range(args.episodes):
            obs, _ = wrapped_env.reset()
            done = False
            total_reward = 0.0
            steps = 0

            while not done:
                with agent.eval():
                    action = agent.act(obs, timestep=0, timesteps=0)
                obs, reward, terminated, truncated, _ = wrapped_env.step(action)
                total_reward += reward.item()
                steps += 1
                done = terminated.item() or truncated.item()

            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            logger.info(f"Episode {ep + 1}: reward={total_reward:.2f}, length={steps}")

    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        wrapped_env.close()
        return 130

    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Algorithm:     {rl_cfg.algorithm.upper()}")
    print(f"Episodes:      {args.episodes}")
    print(f"Mean Reward:   {np.mean(episode_rewards):.2f}")
    print(f"Std Reward:    {np.std(episode_rewards):.2f}")
    print(f"Min Reward:    {np.min(episode_rewards):.2f}")
    print(f"Max Reward:    {np.max(episode_rewards):.2f}")
    print(f"Mean Length:   {np.mean(episode_lengths):.1f}")
    print("=" * 50)

    # Cleanup
    wrapped_env.close()
    logger.info("Evaluation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
