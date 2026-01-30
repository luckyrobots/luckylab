#!/usr/bin/env python3
"""Demo script for LuckyLab environments.

Usage:
    python scripts/demo.py                      # Run with defaults
    python scripts/demo.py --task go1_velocity_flat
    python scripts/demo.py --episodes 5 --steps 200
    python scripts/demo.py --host 127.0.0.1 --port 50051
"""

import argparse
import time

import gymnasium as gym
import numpy as np

import luckylab  # noqa: F401 - registers environments
from luckylab.tasks import list_tasks, load_env_cfg


def parse_args():
    parser = argparse.ArgumentParser(description="LuckyLab environment demo")
    parser.add_argument(
        "--task",
        type=str,
        default="go1_velocity_flat",
        choices=list_tasks(),
        help="Task to run",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="LuckyEngine host (overrides config)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="LuckyEngine port (overrides config)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable rendering (if supported)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed step information",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load environment config
    env_cfg = load_env_cfg(args.task)

    # Override connection settings if provided
    if args.host:
        env_cfg.host = args.host
    if args.port:
        env_cfg.port = args.port

    env_cfg.max_episode_length = args.steps

    print(f"Running demo for task: {args.task}")
    print(f"  Robot: {env_cfg.robot}")
    print(f"  Host: {env_cfg.host}:{env_cfg.port}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Max steps per episode: {args.steps}")
    print()

    # Create environment
    env = gym.make("luckylab/UnitreeGo1-Locomotion-v0", env_cfg=env_cfg)

    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print()

    total_rewards = []
    episode_lengths = []

    for episode in range(args.episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        step_count = 0

        print(f"Episode {episode + 1}/{args.episodes}")
        if "command" in info:
            cmd = info["command"]
            print(f"  Initial command: vx={cmd[0]:.2f}, vy={cmd[1]:.2f}, wz={cmd[2]:.2f}")

        start_time = time.time()

        for step in range(args.steps):
            # Sample random action
            action = env.action_space.sample()

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1

            if args.verbose and step % 50 == 0:
                print(f"    Step {step}: reward={reward:.3f}, obs_mean={obs.mean():.3f}")

            if terminated or truncated:
                reason = info.get("termination_reason", "truncated" if truncated else "unknown")
                print(f"  Ended at step {step_count}: {reason}")
                break

        elapsed = time.time() - start_time
        fps = step_count / elapsed

        print(f"  Total reward: {episode_reward:.2f}")
        print(f"  Steps: {step_count}, FPS: {fps:.1f}")
        print()

        total_rewards.append(episode_reward)
        episode_lengths.append(step_count)

    # Summary
    print("=" * 40)
    print("Summary")
    print("=" * 40)
    print(f"  Episodes: {args.episodes}")
    print(f"  Avg reward: {np.mean(total_rewards):.2f} (+/- {np.std(total_rewards):.2f})")
    print(f"  Avg length: {np.mean(episode_lengths):.1f} (+/- {np.std(episode_lengths):.1f})")

    env.close()
    print("\nDemo complete.")


if __name__ == "__main__":
    main()
