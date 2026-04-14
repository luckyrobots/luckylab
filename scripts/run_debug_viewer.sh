#!/usr/bin/env bash
set -euo pipefail

# Resolve the directory this script lives in (the luckylab root)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "Starting gRPC debug viewer (wiggle mode) ..."
uv run --no-sync --group il python grpc_debug_viewer.py \
    --cameras Camera \
    --width 256 \
    --height 256 \
    --wiggle
