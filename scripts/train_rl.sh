#!/usr/bin/env bash
set -euo pipefail

# Train RL — Go2 velocity tracking with SAC/skrl on CUDA.
# Any extra arguments are forwarded to the training script.
#
# Usage:
#   ./train_rl.sh
#   ./train_rl.sh --device cpu
#   ./train_rl.sh --agent.algorithm ppo

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "Starting RL training (Go2 velocity — SAC / skrl / cuda) ..."
uv run python -m luckylab.scripts.train go2_velocity_flat \
    --agent.algorithm sac \
    --agent.backend skrl \
    --device cuda \
    "$@"
