#!/usr/bin/env bash
set -euo pipefail

# Visualize a LeRobot dataset in a web-based Rerun viewer.
# Defaults to the SO-100 pick-and-place dataset, episode 0.
# Any extra arguments are forwarded to the visualization script.
#
# Usage:
#   ./visualize_dataset.sh
#   ./visualize_dataset.sh --episode-index 3
#   ./visualize_dataset.sh --repo-id my_org/my_dataset --episode-index 0

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "Starting dataset visualization (web viewer) ..."
uv run python -m luckylab.scripts.visualize_dataset \
    --repo-id luckyrobots/so100_pickandplace_sim \
    --episode-index 0 \
    --web \
    "$@"
