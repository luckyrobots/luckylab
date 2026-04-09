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

LuckyLab handles RL training, IL training, and policy inference for robots simulated in [LuckyEngine](https://luckyrobots.com). It communicates with the engine over gRPC (port 50051) via [luckyrobots](https://github.com/luckyrobots/luckyrobots).

| Robot | Task | Learning |
|-------|------|----------|
| Unitree Go2 | Velocity tracking | RL (PPO, SAC) |
| SO-100 | Pick-and-place | IL (ACT via LeRobot) |

---

## Setup

### 1. Install LuckyLab

```bash
git clone https://github.com/luckyrobots/luckylab.git
cd luckylab
```

LuckyLab uses [uv](https://docs.astral.sh/uv/) for dependency management. Install it if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install the dependency group for your use case:

```bash
# RL only
uv sync --group rl

# IL only (LeRobot)
uv sync --group il

# Everything (RL + IL + Rerun + dev tools)
uv sync --all-groups
```

### 2. Start LuckyEngine

LuckyLab does not launch the engine, you need to start it yourself first.

1. Open LuckyEngine
2. Load a scene (e.g. the Go2 velocity scene or SO-100 pick-and-place scene)
3. Enable the **gRPC panel**, this starts the gRPC server on port 50051
4. LuckyLab will connect to `localhost:50051` by default

If the engine is not running or gRPC is not enabled, LuckyLab will fail to connect.

---

## Training

### RL — Go2 velocity tracking

```bash
uv run python -m luckylab.scripts.train go2_velocity_flat \
    --agent.algorithm sac --agent.backend skrl --device cuda
```

Checkpoints are saved to `runs/go2_velocity_sac/checkpoints/` every 5,000 steps, named by step count:

```
runs/go2_velocity_sac/checkpoints/
├── agent_5000.pt
├── agent_10000.pt
├── agent_15000.pt
└── ...
```

### IL — SO-100 pick-and-place

```bash
uv run python -m luckylab.scripts.train so100_pickandplace \
    --il.policy act \
    --il.dataset-repo-id luckyrobots/so100_pickandplace_sim \
    --device cuda
```

Datasets are loaded from the [HuggingFace Hub](https://huggingface.co/datasets) or from a local directory at `~/.luckyrobots/data/`.

---

## Inference

### RL

```bash
# Run a trained SAC policy
uv run python -m luckylab.scripts.play go2_velocity_flat \
    --algorithm sac --backend skrl \
    --checkpoint runs/go2_velocity_sac/checkpoints/agent_25000.pt

# With keyboard velocity command control
uv run python -m luckylab.scripts.play go2_velocity_flat \
    --algorithm sac --backend skrl \
    --checkpoint runs/go2_velocity_sac/checkpoints/agent_25000.pt \
    --keyboard
```

Keyboard controls: **W/S** forward/back, **A/D** strafe, **Q/E** turn, **Space** zero, **Esc** quit.

### IL

```bash
uv run python -m luckylab.scripts.play so100_pickandplace \
    --policy act --checkpoint runs/so100_pickandplace_act/final
```

---

## Available Tasks

```bash
# List all registered tasks
uv run python -m luckylab.scripts.list_envs
```

| Task ID | Robot | Type | Algorithms |
|---------|-------|------|------------|
| `go2_velocity_flat` | Unitree Go2 | RL | PPO, SAC |
| `so100_pickandplace` | SO-100 | IL | ACT |

Any algorithm supported by [skrl](https://github.com/Toni-SM/skrl) or [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) can be used for RL, and any policy supported by [LeRobot](https://github.com/huggingface/lerobot) can be used for IL — you just need to define the configs for them.

---

## Visualization

**Rerun** — live inspection of observations, actions, rewards, and camera feeds:

```bash
# Browse a dataset
uv run python -m luckylab.scripts.visualize_dataset \
    --repo-id luckyrobots/so100_pickandplace_sim --episode-index 0 --web

# Attach to an evaluation run
uv run python -m luckylab.scripts.play go2_velocity_flat \
    --algorithm sac --backend skrl \
    --checkpoint runs/go2_velocity_sac/checkpoints/agent_25000.pt --rerun
```

**Weights & Biases** — enabled by default for RL training. Disable with `--agent.wandb false`.

---

## Development

```bash
uv sync --all-groups
uv run pre-commit install

uv run pytest tests -v
uv run ruff check src tests
uv run ruff format src tests
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
