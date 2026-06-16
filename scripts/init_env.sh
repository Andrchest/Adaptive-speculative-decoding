#!/usr/bin/env bash
# =============================================================================
# Initialize development environment
# =============================================================================
# Usage: bash scripts/init_env.sh
# =============================================================================
set -euo pipefail

echo "🔧 Initializing development environment..."

# Check Python version
PYTHON_VER=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
if [[ "$PYTHON_VER" != "3.12" ]]; then
    echo "⚠️  Python 3.12 recommended, found $PYTHON_VER"
    read -p "Continue anyway? [y/N] " -n 1 -r
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check uv
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Sync dependencies
echo "📦 Installing dependencies..."
uv sync --all-extras

# Install pre-commit hooks
echo "🪝 Installing pre-commit hooks..."
pre-commit install

echo "✅ Done! Run 'uv run python src/main.py --help' to get started."
