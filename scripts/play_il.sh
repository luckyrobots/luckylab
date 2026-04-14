#!/usr/bin/env bash
set -euo pipefail

# Run a trained IL policy (SO-100 pick-and-place — ACT).
# First argument is the checkpoint path. Extra arguments are forwarded.
#
# Usage:
#   ./play_il.sh runs/so100_pickandplace_act/final
#   ./play_il.sh runs/so100_pickandplace_act/final --episodes 20
#   ./play_il.sh runs/so100_pickandplace_act/final --rerun

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [ $# -lt 1 ]; then
    echo "Usage: ./play_il.sh <checkpoint> [extra args...]"
    echo ""
    echo "Example:"
    echo "  ./play_il.sh runs/so100_pickandplace_act/final"
    echo "  ./play_il.sh runs/so100_pickandplace_act/final --episodes 20"
    exit 1
fi

CHECKPOINT="$1"
shift

echo "Running IL inference (SO-100 pick-and-place — ACT) ..."
echo "  Checkpoint: ${CHECKPOINT}"
uv run python -m luckylab.scripts.play so100_pickandplace \
    --policy act \
    --checkpoint "${CHECKPOINT}" \
    "$@"
