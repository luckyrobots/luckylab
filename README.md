# LuckyLab

Reinforcement learning framework for robotics simulation with LuckyEngine.

## Features

- **Manager-based RL environments** following Isaac Lab / mjlab patterns
- **Config-driven MDP components** with direct function references
- **Multiple RL algorithms** via skrl: PPO, SAC, TD3, DDPG
- **PyTorch and JAX backends** for training
- **Task-agnostic logging** with TensorBoard and Weights & Biases support

## Installation

```bash
# With uv (recommended)
uv pip install luckylab

# With pip
pip install luckylab
```

### Optional Dependencies

```bash
# JAX backend support
uv pip install "luckylab[jax]"

# All logging backends
uv pip install "luckylab[logging]"

# Everything
uv pip install "luckylab[all]"
```

## Quickstart

### Basic Environment Usage

```python
from luckylab.envs import ManagerBasedRlEnv
from luckylab.tasks.velocity import create_velocity_env_cfg

# Create environment config
cfg = create_velocity_env_cfg(robot="unitreego1")

# Create environment
env = ManagerBasedRlEnv(cfg=cfg)

# Standard Gymnasium interface
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### Training with CLI

```bash
# Train with PPO (default)
python -m luckylab.scripts.train --task go1_velocity_flat

# Train with SAC on GPU
python -m luckylab.scripts.train --task go1_velocity_flat --algorithm sac --device cuda

# Train with JAX backend
python -m luckylab.scripts.train --task go1_velocity_flat --backend jax

# With Weights & Biases logging
python -m luckylab.scripts.train --task go1_velocity_flat --logger wandb
```

### Training Programmatically

```python
from luckylab.rl import train, SkrlCfg
from luckylab.tasks import load_env_cfg

# Load task configuration
env_cfg = load_env_cfg("go1_velocity_flat")

# Configure training
rl_cfg = SkrlCfg(
    algorithm="ppo",
    backend="torch",
    timesteps=1_000_000,
    seed=42,
)

# Train
train(env_cfg=env_cfg, rl_cfg=rl_cfg, device="cuda")
```

## Architecture

LuckyLab uses a manager-based architecture where each MDP component is handled by a dedicated manager:

```
ManagerBasedRlEnv
├── RewardManager        # Config-driven reward computation
├── TerminationManager   # Config-driven termination detection
├── CommandManager       # Velocity command generation
├── CurriculumManager    # Progressive difficulty adjustment
└── ObservationManager   # Observation processing (noise, delay, history)
```

### Config-Driven MDP Components

All MDP components use direct function references in configs:

```python
from luckylab.managers import RewardTermCfg, TerminationTermCfg
from luckylab.tasks.velocity import mdp

rewards = {
    "track_velocity": RewardTermCfg(
        func=mdp.track_linear_velocity,
        weight=2.0,
        params={"std": 0.5},
    ),
    "action_rate": RewardTermCfg(
        func=mdp.action_rate_l2,
        weight=-0.1,
    ),
}

terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
        func=mdp.bad_orientation,
        params={"limit_angle": 1.2},
    ),
}
```

## Available Tasks

| Task ID | Description | Robot |
|---------|-------------|-------|
| `go1_velocity_flat` | Velocity tracking on flat terrain | Unitree Go1 |

## Project Structure

```
src/luckylab/
├── configs/          # Configuration dataclasses
├── envs/             # Environment implementations
│   └── mdp/          # Common MDP functions
├── managers/         # Manager classes
├── rl/               # RL training (skrl integration)
├── scripts/          # CLI tools (train, play, etc.)
├── tasks/            # Task definitions
│   └── velocity/     # Velocity tracking task
└── utils/            # Utilities (logging, buffers, etc.)
```

## Development

```bash
# Clone and install
git clone https://github.com/luckyrobots/luckylab.git
cd luckylab
uv sync --all-groups

# Install pre-commit hooks
uv run pre-commit install

# Run tests
uv run pytest tests -v

# Run linting
uv run ruff check src tests
uv run ruff format src tests
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed development guidelines.

## Requirements

- Python 3.10+
- LuckyEngine simulator
- luckyrobots >= 0.1.70

## Acknowledgments

LuckyLab is inspired by:
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
- [mjlab](https://github.com/google-deepmind/mujoco_playground)
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl)

## License

MIT License - see [LICENSE](LICENSE) for details.
