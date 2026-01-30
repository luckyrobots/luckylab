"""
LuckyLab Example: Unitree Go1 Locomotion

This example demonstrates how to use the luckylab environment
to control a Unitree Go1 quadruped robot in the LuckyRobots simulator.

Requirements:
    - LuckyEngine simulator running
    - pip install luckylab

Usage:
    python example.py
"""

import gymnasium as gym
import numpy as np

import luckylab  # noqa: F401 - registers environments

# Create the Unitree Go1 locomotion environment
env = gym.make("luckylab/UnitreeGo1-Locomotion-v0")

print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")

# Reset the environment
observation, info = env.reset()
print(f"Initial observation shape: {observation.shape}")

# Run a simple control loop
total_reward = 0
step_count = 0

for i in range(500):
    # Sample random action within joint limits
    action = env.action_space.sample()

    # Step the environment
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    step_count += 1

    # Print progress every 100 steps
    if (i + 1) % 100 == 0:
        print(f"Step {i + 1}: obs_mean={observation.mean():.3f}, reward={reward:.3f}")

    if terminated or truncated:
        print(f"Episode ended at step {i + 1}")
        observation, info = env.reset()

print(f"\nCompleted {step_count} steps, total reward: {total_reward:.3f}")

# Clean up
env.close()
