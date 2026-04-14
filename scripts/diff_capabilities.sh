#!/usr/bin/env bash
set -euo pipefail

# Compare capability manifests between engine versions.
#
# Usage:
#   ./diff_capabilities.sh --save manifest-v1.json
#   ./diff_capabilities.sh --old manifest-v1.json --new manifest-v1.1.json
#   ./diff_capabilities.sh --old manifest-v1.json --live

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

uv run python -m luckylab.scripts.diff_capabilities "$@"
