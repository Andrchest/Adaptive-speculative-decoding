#!/usr/bin/env bash
# =============================================================================
# Clean up artifacts and temporary files
# =============================================================================
# Usage: bash scripts/cleanup.sh [--all]
# =============================================================================
set -euo pipefail

CLEAN_RESULTS=false
if [[ "${1:-}" == "--all" ]]; then
    CLEAN_RESULTS=true
fi

echo "🧹 Cleaning up..."

# Python artifacts
echo "  Removing __pycache__/..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Test artifacts
echo "  Removing .pytest_cache/..."
rm -rf .pytest_cache/
rm -rf .mypy_cache/
rm -rf .ruff_cache/
rm -rf htmlcov/

# Build artifacts
echo "  Removing build artifacts..."
rm -rf dist/
rm -rf *.egg-info/

if [[ "$CLEAN_RESULTS" == "true" ]]; then
    echo "  Removing research results (permanent)..."
    find research -name "*.csv" -delete 2>/dev/null || true
    find research -name "*.json" -delete 2>/dev/null || true
    find research -name "mlflow.db" -delete 2>/dev/null || true
    find research -name "mlruns" -type d -exec rm -rf {} + 2>/dev/null || true
    echo "  ⚠️  Research data cleaned!"
fi

echo "✅ Cleanup complete."
