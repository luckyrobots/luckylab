# LuckyLab

RL training environments for LuckyRobots.

## Installation

Create a virtual environment with Python 3.10+ and activate it:

```bash
uv venv && source .venv/bin/activate
```

Install luckylab:

```bash
pip install luckylab
```

## Quickstart

```python
import gymnasium as gym
import luckylab  # noqa: F401 - registers environments

# Create the Unitree Go1 locomotion environment
env = gym.make("luckylab/UnitreeGo1-Locomotion-v0")

print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")

observation, info = env.reset()

for _ in range(500):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

## Environments

### Unitree Go1 Locomotion

Velocity tracking task for the Unitree Go1 quadruped robot.

**Environment ID:** `luckylab/UnitreeGo1-Locomotion-v0`

#### Action Space

12-dimensional continuous action space for joint position commands:
- 4 legs x 3 joints (hip, thigh, calf)

#### Observation Space

Flat observation vector containing:
- Commands: [vx, vy, wz, heading] (4D)
- Base linear velocity: [x, y, z] (3D)
- Base angular velocity: [x, y, z] (3D)
- Joint positions relative to default (12D)
- Joint velocities (12D)
- Last actions (12D)

#### Rewards

Configurable reward function combining:
- Linear velocity tracking
- Angular velocity tracking
- Posture maintenance
- Body angular velocity penalty
- Joint position limits penalty
- Action rate penalty

#### Termination Conditions

Episodes terminate on:
- Timeout (max steps reached)
- Fall detection (rapid downward velocity)
- Bad orientation (excessive tilt rate)
- Root height below minimum
- NaN detection (optional)

## Configuration

Environment configuration uses dataclasses (RSL-RL style):

```python
from luckylab.tasks.velocity import VelocityEnvCfg, VelocityRewardCfg

# Create custom config
cfg = VelocityEnvCfg(
    host="127.0.0.1",
    port=50051,
    skip_launch=True,  # Connect to existing LuckyEngine
    max_episode_length=500,
)

# Customize rewards
cfg.reward.track_linear_velocity_weight = 3.0
cfg.reward.action_rate_l2_weight = -0.05

# Create environment with custom config
env = gym.make("luckylab/UnitreeGo1-Locomotion-v0", env_cfg=cfg)
```

## Development

Install with dev dependencies:

```bash
uv sync --group dev
```

Run tests:

```bash
uv run pytest tests/ -v
```

Apply linting:

```bash
pre-commit install
pre-commit run --all-files
```

## Requirements

- Python 3.10+
- LuckyEngine simulator running (or set `skip_launch=False`)
- luckyrobots >= 0.1.70

## Acknowledgment

LuckyLab is built on top of [LuckyRobots](https://github.com/luckyrobots/luckyrobots) and inspired by [Isaac Lab](https://github.com/isaac-sim/IsaacLab) and [RSL-RL](https://github.com/leggedrobotics/rsl_rl).
