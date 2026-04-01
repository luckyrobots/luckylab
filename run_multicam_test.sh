#!/usr/bin/env bash
set -euo pipefail

# Multi-camera stress test for gRPC camera streaming.
# Auto-discovers all cameras in the scene and streams them simultaneously.
# Usage: ./run_multicam_test.sh [host]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

HOST="${1:-127.0.0.1}"

echo "Starting multi-camera stress test (host=${HOST}) ..."
uv run --no-sync --group il python grpc_multicam_test.py \
    --width 256 \
    --height 256 \
    --wiggle \
    --host "${HOST}"
