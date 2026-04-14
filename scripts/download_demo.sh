#!/usr/bin/env bash
set -euo pipefail

REPO="luckyrobots/luckylab"
TAG="demo-v0.1.0"
DEMO_NAME="piper_blockstacking_act"
ZIP_NAME="${DEMO_NAME}.zip"
DOWNLOAD_URL="https://github.com/${REPO}/releases/download/${TAG}/${ZIP_NAME}"

# Resolve the directory this script lives in (the luckylab root)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Downloading demo from ${DOWNLOAD_URL} ..."
curl -L "${DOWNLOAD_URL}" -o "${ROOT_DIR}/${ZIP_NAME}"

echo "Extracting demo ..."
unzip -o "${ROOT_DIR}/${ZIP_NAME}" -d "${ROOT_DIR}"

chmod -R u+rwX "${ROOT_DIR}/runs"
chmod +x "${ROOT_DIR}/run_demo.sh"

rm "${ROOT_DIR}/${ZIP_NAME}"

echo ""
echo "Demo installed successfully."
echo "  Model:  runs/${DEMO_NAME}/final/"
echo "  Script: run_demo.sh"
echo ""
echo "Run './run_demo.sh' to start the demo."
