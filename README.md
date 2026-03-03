<p align="center">
  <h1 align="center">LuckyLab</h1>
  <p align="center">
    <strong>A unified robot learning framework powered by <a href="https://github.com/luckyrobots/luckyrobots">LuckyEngine</a></strong>
  </p>
  <p align="center">
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
    <a href="https://docs.astral.sh/ruff/"><img src="https://img.shields.io/badge/code%20style-ruff-000000.svg" alt="Ruff"></a>
  </p>
</p>

LuckyLab is a modular, config-driven framework that brings reinforcement learning, imitation learning, and real-time visualization together in one place. It communicates with LuckyEngine through [luckyrobots](https://github.com/luckyrobots/luckyrobots) and runs on both CPU and GPU.

The framework ships with locomotion and manipulation tasks but is easily extensible to any robot or task. It supports all imitation learning algorithms in [LeRobot](https://github.com/huggingface/lerobot) and multiple RL algorithms via [skrl](https://github.com/Toni-SM/skrl) and [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3). Live inspection is available through [Rerun](https://rerun.io) and [Viser](https://github.com/nerfstudio-project/viser).

| Robot | Task | Learning |
|-------|------|----------|
| Unitree Go2 | Velocity tracking | RL (PPO, SAC, TD3, DDPG) |
| Piper | Pick-and-place | IL (via LeRobot) |

---

## Requirements

- Python 3.10+
- [LuckyEngine](https://luckyrobots.com) executable
- [luckyrobots](https://github.com/luckyrobots/luckyrobots) >= 0.1.81
- PyTorch >= 2.0

## Installation

```bash
git clone https://github.com/luckyrobots/luckylab.git
cd luckylab

# Core + RL
uv sync --group rl

# Core + IL (LeRobot)
uv sync --group il

# Everything (RL + IL + Rerun + dev tools)
uv sync --all-groups
```

---

## Quick Start

### Train

```bash
# RL — train SAC on the Go2
python -m luckylab.scripts.train go2_velocity_flat \
    --agent.algorithm sac --agent.backend skrl --device cuda

# IL — train ACT on a local dataset
python -m luckylab.scripts.train piper_pickandplace \
    --il.policy act --il.dataset-repo-id piper/pickandplace --device cuda
```

### Evaluate

```bash
# RL — with keyboard control
python -m luckylab.scripts.play go2_velocity_flat \
    --algorithm sac --checkpoint runs/go2_velocity_sac/checkpoints/best_agent.pt \
    --keyboard

# IL
python -m luckylab.scripts.play piper_pickandplace \
    --policy act --checkpoint runs/luckylab_il/final
```

Keyboard controls: **W/S** forward/back, **A/D** strafe, **Q/E** turn, **Space** zero, **Esc** quit.

### Visualize

```bash
# Browse a dataset in Rerun (opens in browser)
python -m luckylab.scripts.visualize_dataset \
    --repo-id piper/pickandplace --episode-index 0 --web

# List all registered tasks
python -m luckylab.scripts.list_envs
```

---

## Reinforcement Learning

Four algorithms across two backends, all configurable via CLI or Python:

| Algorithm | Type | Backends |
|-----------|------|----------|
| **PPO** | On-policy | skrl, sb3 |
| **SAC** | Off-policy | skrl, sb3 |
| **TD3** | Off-policy | skrl, sb3 |
| **DDPG** | Off-policy | skrl, sb3 |

```bash
python -m luckylab.scripts.train go2_velocity_flat \
    --agent.algorithm sac --agent.backend skrl \
    --agent.max-iterations 5000 \
    --env.num-envs 4096 \
    --device cuda
```

```python
from luckylab.rl import train, RlRunnerCfg
from luckylab.tasks import load_env_cfg

env_cfg = load_env_cfg("go2_velocity_flat")
rl_cfg = RlRunnerCfg(algorithm="sac", backend="skrl", max_iterations=5000)
train(env_cfg=env_cfg, rl_cfg=rl_cfg, device="cuda")
```

> **Note:** LuckyEngine does not currently support environment parallelization, so on-policy algorithms like PPO that depend on large batch collection are not recommended. Off-policy algorithms like SAC are the best fit for now. Parallelization support is actively being worked on.

> **Backend recommendation:** Stable Baselines3 is not designed for GPU training. If you want to train on GPU, use the skrl backend (`--agent.backend skrl`).

---

## Imitation Learning

LuckyLab integrates with [LeRobot](https://github.com/huggingface/lerobot) for imitation learning. ACT and Diffusion Policy are ready to use out of the box. Other LeRobot policies (Pi0, SmolVLA, etc.) are supported but require registering a task config for them first, similar to how the ACT and Diffusion configs are set up.

```bash
python -m luckylab.scripts.train piper_pickandplace \
    --il.policy act \
    --il.dataset-repo-id piper/pickandplace \
    --il.batch-size 8 \
    --il.num-train-steps 100000 \
    --device cuda
```

Datasets are loaded from the [HuggingFace Hub](https://huggingface.co/datasets) or from a local directory at `~/.luckyrobots/data/` (configurable via `LUCKYROBOTS_DATA_HOME`).

---

## Tasks

Tasks bundle an environment config with RL and/or IL configs. The registry makes it easy to add new ones:

```python
from luckylab.tasks import register_task
from luckylab.envs import ManagerBasedRlEnvCfg
from luckylab.rl import RlRunnerCfg

env_cfg = ManagerBasedRlEnvCfg(
    decimation=4,
    robot="unitreego2",
    scene="velocity",
    observations={...},
    actions={...},
    rewards={...},
    terminations={...},
)

register_task(
    "my_task",
    env_cfg,
    rl_cfgs={"ppo": RlRunnerCfg(algorithm="ppo", max_iterations=3000)},
)
```

---

## Architecture

LuckyLab uses a manager-based environment where each MDP component is handled by a dedicated manager, configured with direct function references:

```
ManagerBasedRlEnv
├── ObservationManager   Observation groups with noise, delay, and history
├── ActionManager        Action scaling, offset, and joint commands
├── RewardManager        Weighted sum of reward terms
├── TerminationManager   Episode termination conditions
└── CurriculumManager    Progressive difficulty adjustment
```

```python
from luckylab.managers import RewardTermCfg, TerminationTermCfg
from luckylab.tasks.velocity import mdp

rewards = {
    "track_velocity": RewardTermCfg(func=mdp.track_linear_velocity, weight=2.0, params={"std": 0.5}),
    "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
}

terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(func=mdp.bad_orientation, params={"limit_angle": 1.2}),
}
```

---

## Visualization & Logging

**Policy Viewer** — a web-based MuJoCo viewer powered by [Viser](https://github.com/nerfstudio-project/viser) for inspecting trained RL policies. Renders the robot in a browser with velocity command sliders, pause/play, speed control, and keyboard input — no LuckyEngine connection required.

```bash
# Open http://localhost:8080 after starting
python -m luckylab.viewer.run_policy runs/go2_velocity_sac/checkpoints/best_agent.pt
```

**Rerun** — live step-by-step inspection of observations, actions, rewards, and camera feeds. No LuckyEngine connection required.

```bash
# Dataset viewer
python -m luckylab.scripts.visualize_dataset --repo-id piper/pickandplace --web

# Attach to evaluation
python -m luckylab.scripts.play go2_velocity_flat --algorithm sac --checkpoint best_agent.pt --rerun
```

**Weights & Biases** — enabled by default for both RL and IL. Disable with `--agent.wandb false` or `--il.wandb false`.

---

## Project Structure

```
src/luckylab/
├── configs/          Simulation contract and shared configs
├── entity/           Robot entity and observation data
├── envs/             ManagerBasedRlEnv and MDP functions
│   └── mdp/          Observations, actions, rewards, terminations, curriculum
├── il/               Imitation learning
│   └── lerobot/      LeRobot integration (trainer, wrapper)
├── managers/         Observation, action, reward, termination, curriculum managers
├── rl/               Reinforcement learning
│   ├── skrl/         skrl backend
│   ├── sb3/          Stable Baselines3 backend
│   ├── config.py     RlRunnerCfg and algorithm configs
│   └── common.py     Shared utilities
├── scene/            Scene management
├── scripts/          CLI entry points (train, play, list_envs, visualize_dataset)
├── tasks/            Task definitions and registry
│   ├── velocity/     Locomotion velocity tracking
│   └── pickandplace/ Manipulation (IL)
├── utils/            NaN guard, noise models, rerun logger, keyboard, buffers
└── viewer/           Debug visualization with Viser
```

---

## Development

```bash
uv sync --all-groups
uv run pre-commit install

# Tests
uv run pytest tests -v

# Lint
uv run ruff check src tests
uv run ruff format src tests
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## Acknowledgments

LuckyLab is inspired by:
- [MJLab](https://github.com/google-deepmind/mujoco_playground) — manager-based, config-driven environment architecture
- [LeRobot](https://github.com/huggingface/lerobot) — imitation learning policies and dataset format

Built on top of [skrl](https://github.com/Toni-SM/skrl) and [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) for RL training.

## License

MIT License — see [LICENSE](LICENSE) for details.
