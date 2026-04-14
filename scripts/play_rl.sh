#!/usr/bin/env bash
set -euo pipefail

# Run a trained RL policy (Go2 velocity — SAC / skrl).
# First argument is the checkpoint path. Extra arguments are forwarded.
#
# Usage:
#   ./play_rl.sh runs/go2_velocity_sac/checkpoints/agent_25000.pt
#   ./play_rl.sh runs/go2_velocity_sac/checkpoints/agent_25000.pt --keyboard
#   ./play_rl.sh runs/go2_velocity_sac/checkpoints/agent_25000.pt --rerun

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [ $# -lt 1 ]; then
    echo "Usage: ./play_rl.sh <checkpoint> [extra args...]"
    echo ""
    echo "Example:"
    echo "  ./play_rl.sh runs/go2_velocity_sac/checkpoints/agent_25000.pt"
    echo "  ./play_rl.sh runs/go2_velocity_sac/checkpoints/agent_25000.pt --keyboard"
    exit 1
fi

CHECKPOINT="$1"
shift

echo "Running RL inference (Go2 velocity — SAC / skrl) ..."
echo "  Checkpoint: ${CHECKPOINT}"
uv run python -m luckylab.scripts.play go2_velocity_flat \
    --algorithm sac \
    --backend skrl \
    --checkpoint "${CHECKPOINT}" \
    "$@"
