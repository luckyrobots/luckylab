# LuckyLab

Reinforcement learning framework for robotics simulation with LuckyEngine.

## Features

- **Manager-based RL environments** following Isaac Lab / mjlab patterns
- **Config-driven MDP components** with direct function references
- **Dual RL backends** — [skrl](https://github.com/Toni-SM/skrl) and [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
- **Multiple RL algorithms** — PPO, SAC, TD3, DDPG
- **Keyboard control** for interactive evaluation (WASD + QE)
- **NaN guard** with recovery mode for safe training
- **Debug visualization** via LuckyEngine viewer
- **Logging** with TensorBoard and Weights & Biases

## Installation

```bash
# With uv (recommended)
uv pip install luckylab

# With pip
pip install luckylab
```

## Quickstart

### Training with CLI

LuckyLab uses [tyro](https://github.com/brentyi/tyro) for CLI parsing. The task name is a positional argument, followed by `--agent.algorithm` and `--agent.backend`:

```bash
# Train with SAC on the skrl backend
python -m luckylab.scripts.train go2_velocity_flat --agent.algorithm sac --agent.backend skrl

# Train with SAC on Stable Baselines3
python -m luckylab.scripts.train go2_velocity_flat --agent.algorithm sac --agent.backend sb3

# Train with PPO on GPU
python -m luckylab.scripts.train go2_velocity_flat --agent.algorithm ppo --agent.backend skrl --device cuda

# Override training hyperparameters
python -m luckylab.scripts.train go2_velocity_flat --agent.algorithm sac --agent.backend skrl --agent.max-iterations 500

# Run in realtime mode (for visual debugging)
python -m luckylab.scripts.train go2_velocity_flat --agent.algorithm sac --agent.backend skrl --env.simulation-mode realtime
```

### Playing a Trained Policy

```bash
# Evaluate a checkpoint
python -m luckylab.scripts.play go2_velocity_flat --algorithm sac \
    --checkpoint runs/go2_velocity_sac/checkpoints/best_agent.pt

# Interactive keyboard control (WASD + QE for velocity commands)
python -m luckylab.scripts.play go2_velocity_flat --algorithm sac \
    --checkpoint runs/go2_velocity_sac/checkpoints/best_agent.pt --keyboard
```

Keyboard controls (`--keyboard` mode):
- **W/S** forward/backward, **A/D** strafe left/right, **Q/E** turn left/right
- **Space** zero all commands, **Esc** quit

### Training Programmatically

```python
from luckylab.rl import train, RlRunnerCfg
from luckylab.tasks import load_env_cfg

# Load task configuration
env_cfg = load_env_cfg("go2_velocity_flat")

# Configure training
rl_cfg = RlRunnerCfg(
    algorithm="sac",
    backend="skrl",
    max_iterations=1_000,
    seed=42,
)

# Train
train(env_cfg=env_cfg, rl_cfg=rl_cfg, device="cuda")
```

### Basic Environment Usage

```python
from luckylab.envs import ManagerBasedRlEnv
from luckylab.tasks.velocity import create_velocity_env_cfg

# Create environment config
cfg = create_velocity_env_cfg(robot="unitreego2")

# Create environment (standard Gymnasium interface)
env = ManagerBasedRlEnv(cfg=cfg)

obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

## Architecture

LuckyLab uses a manager-based architecture where each MDP component is handled by a dedicated manager:

```
ManagerBasedRlEnv
├── RewardManager        # Config-driven reward computation
├── TerminationManager   # Config-driven termination detection
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
| `go2_velocity_flat` | Velocity tracking on flat terrain | Unitree Go2 |
| `go2_velocity_rough` | Velocity tracking on rough terrain | Unitree Go2 |

## Project Structure

```
src/luckylab/
├── configs/          # Configuration dataclasses
├── entity/           # Entity data (observations, state)
├── envs/             # Environment implementations
│   └── mdp/          # Common MDP functions
├── managers/         # Manager classes (reward, termination, etc.)
├── rl/               # RL training
│   ├── skrl/         # skrl backend (models, trainer, wrapper)
│   ├── sb3/          # Stable Baselines3 backend (trainer, wrapper)
│   ├── config.py     # RlRunnerCfg and algorithm configs
│   ├── common.py     # Shared utilities (print_config, wrap_env)
│   └── trainer.py    # Backend-agnostic train/load entry point
├── scene/            # Scene management
├── scripts/          # CLI tools (train, play, list_envs)
├── tasks/            # Task definitions
│   └── velocity/     # Velocity tracking task
│       └── config/   # Per-robot configs (go2/)
├── utils/            # Utilities (logging, NaN guard, keyboard, math, etc.)
└── viewer/           # Debug visualization (debug_draw, run_policy)
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
- luckyrobots >= 0.1.77

## Acknowledgments

LuckyLab is inspired by:
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
- [mjlab](https://github.com/google-deepmind/mujoco_playground)
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl)

## License

MIT License - see [LICENSE](LICENSE) for details.
