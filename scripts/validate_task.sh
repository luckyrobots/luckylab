#!/usr/bin/env bash
set -euo pipefail

# Validate a task contract against engine capabilities.
#
# Usage:
#   ./validate_task.sh go2_velocity_flat
#   ./validate_task.sh go2_velocity_flat --host localhost --port 50051

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

uv run python -m luckylab.scripts.validate_task "$@"
