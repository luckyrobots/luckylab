#!/usr/bin/env bash
set -euo pipefail

# List available MDP capabilities from a running LuckyEngine instance.
#
# Usage:
#   ./list_capabilities.sh
#   ./list_capabilities.sh --robot unitreego2
#   ./list_capabilities.sh --host 192.168.1.10 --port 50051

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

uv run python -m luckylab.scripts.list_capabilities "$@"
