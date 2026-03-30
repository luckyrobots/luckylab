<p align="center">
  <h1 align="center">LuckyLab</h1>
  <p align="center">
    <strong>RL and IL training framework for <a href="https://github.com/luckyrobots/LuckyEngine">LuckyEngine</a></strong>
  </p>
  <p align="center">
    <a href="https://luckyrobots.com"><img src="https://img.shields.io/badge/Lucky_Robots-ff6600?style=flat&logoColor=white" alt="Lucky Robots"></a>
    <a href="https://github.com/luckyrobots/LuckyEngine"><img src="https://img.shields.io/badge/LuckyEngine-0984e3?style=flat&logoColor=white" alt="LuckyEngine"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
  </p>
  <p align="center">
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white" alt="Python 3.10+"></a>
    <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-%3E%3D2.0-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch >= 2.0"></a>
    <a href="https://github.com/luckyrobots/luckyrobots"><img src="https://img.shields.io/badge/luckyrobots-%3E%3D0.1.84-00b894" alt="luckyrobots >= 0.1.84"></a>
    <a href="https://docs.astral.sh/ruff/"><img src="https://img.shields.io/badge/code%20style-ruff-000000.svg" alt="Ruff"></a>
  </p>
</p>

LuckyLab is the training and inference layer for robots simulated in [LuckyEngine](https://github.com/luckyrobots/LuckyEngine). It connects to LuckyEngine over gRPC (via the [luckyrobots](https://github.com/luckyrobots/luckyrobots) client), sends joint-level actions, and receives observations each step — all physics and rendering runs in LuckyEngine.

---
## Quick Start
### 1. Installation

```bash
git clone -b mick/release-2026-1 --single-branch https://github.com/luckyrobots/luckylab.git
cd luckylab

# Run the setup script for your OS 
./setup.bat # Windows 
./setup.sh # Linux
```
### 2. Prepare LuckyEngine

1. Launch LuckyEngine
2. Download the Piper Block Stacking project
3. Open the Piper Block Stacking scene
4. Open the gRPC Panel
<table><tr><td>

5. Follow the prompts to ensure:
   - Action Gate is **Enabled**
   - Server is **Running**
   - Scene is **Playing**

</td><td>

<img width="300" alt="gRPC Panel" src="https://github.com/user-attachments/assets/352bd83e-29d7-4c6f-af79-b27ba412c4e4" />

</td></tr></table>

### 3. Run Debug Viewer

```bash
# Run the gRPC viewer script for your OS 
./run_debug_viewer.bat # Windows 
./run_debug_viewer.sh # Linux
```

If everything has been configured correctly, this script will log the inputs/outputs between LuckyLab and LuckyEngine, and display the camera feed being exported from LuckyEngine to LuckyLab.
### 4. Download & Run Piper Block Stacking Demo Model

```bash
# Run the model download script for your OS
# Windows 
./download_demo.bat 
./run_demo.bat

# Linux
./download_demo.sh
./run_demo.sh
```

Manually downloaded models need to be placed within their own subfolder within the /runs/ directory of LuckyLab, where-as the download scripts already extract to the appropriate nested location.

---

## How It Works

```mermaid
graph TD
    LE[LuckyEngine]

    LE <--> LR[luckyrobots client]

    LR --> ENV

    subgraph LuckyLab ["&nbsp;&nbsp;&nbsp;&nbsp;LuckyLab&nbsp;&nbsp;&nbsp;&nbsp;"]
        ENV[ManagerBasedEnv]
        ENV --- OBS[Observations]
        ENV --- ACT[Actions]
        ENV --- REW[Rewards]
        ENV --- TERM[Terminations]
        ENV --- CURR[Curriculum]
    end

    subgraph Backends ["Training Backends"]
        SKRL[skrl — RL]
        SB3[SB3 — RL]
        LEROBOT[LeRobot — IL]
    end

    ENV --> Backends

    style LE fill:#1a1a2e,stroke:#0984e3,stroke-width:2px,color:#74b9ff
    style LR fill:#1a1a2e,stroke:#00b894,stroke-width:2px,color:#55efc4

    style LuckyLab fill:#16213e,stroke:#6c5ce7,stroke-width:2px,color:#a29bfe
    style ENV fill:#1a1a2e,stroke:#6c5ce7,stroke-width:2px,color:#a29bfe
    style OBS fill:#1a1a2e,stroke:#636e72,stroke-width:1px,color:#dfe6e9
    style ACT fill:#1a1a2e,stroke:#636e72,stroke-width:1px,color:#dfe6e9
    style REW fill:#1a1a2e,stroke:#636e72,stroke-width:1px,color:#dfe6e9
    style TERM fill:#1a1a2e,stroke:#636e72,stroke-width:1px,color:#dfe6e9
    style CURR fill:#1a1a2e,stroke:#636e72,stroke-width:1px,color:#dfe6e9

    style Backends fill:#16213e,stroke:#e17055,stroke-width:2px,color:#fab1a0
    style SKRL fill:#1a1a2e,stroke:#e17055,stroke-width:2px,color:#fab1a0
    style SB3 fill:#1a1a2e,stroke:#e17055,stroke-width:2px,color:#fab1a0
    style LEROBOT fill:#1a1a2e,stroke:#fdcb6e,stroke-width:2px,color:#ffeaa7
```

LuckyEngine handles all physics simulation (built on MuJoCo). LuckyLab is purely a training orchestrator — it does not run physics locally. The [luckyrobots](https://github.com/luckyrobots/luckyrobots) package manages the gRPC connection, engine lifecycle, and domain randomization protocol.

---

## Status

LuckyLab is in **early development (alpha)**. The Piper block-stacking demo above is the current focus. The codebase also includes scaffolding for reinforcement learning (Go2 velocity tracking via [skrl](https://github.com/Toni-SM/skrl) / [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)) and additional imitation learning policies via [LeRobot](https://github.com/huggingface/lerobot).

---

## Development

```bash
# Manual install with uv (instead of setup scripts)
uv sync --all-groups
uv run pre-commit install

# Tests
uv run pytest tests -v

# Lint
uv run ruff check src tests
uv run ruff format src tests
```
