#!/usr/bin/env bash
set -euo pipefail

# List all registered LuckyLab tasks (RL and IL).
#
# Usage:
#   ./list_tasks.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

uv run python -m luckylab.scripts.list_envs
