#!/usr/bin/env bash
set -e

echo "Checking for uv..."
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "uv installed successfully."
else
    echo "uv is already installed."
fi

echo
echo "Running uv sync --all-groups..."
uv sync --all-groups

echo
echo "Installing pre-commit hooks..."
uv run pre-commit install

echo
echo "Setup complete!"
