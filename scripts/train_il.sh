#!/usr/bin/env bash
set -euo pipefail

# Train IL — SO-100 pick-and-place with ACT policy on CUDA.
# Any extra arguments are forwarded to the training script.
#
# Usage:
#   ./train_il.sh
#   ./train_il.sh --device cpu
#   ./train_il.sh --il.dataset-repo-id my_org/my_dataset

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "Starting IL training (SO-100 pick-and-place — ACT / cuda) ..."
uv run python -m luckylab.scripts.train so100_pickandplace \
    --il.policy act \
    --il.dataset-repo-id luckyrobots/so100_pickandplace_sim \
    --device cuda \
    "$@"
