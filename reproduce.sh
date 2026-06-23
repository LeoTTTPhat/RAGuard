#!/usr/bin/env bash
# reproduce.sh — run all RAGentGuard experiments and tests reproducibly.
#
# Usage:
#   chmod +x reproduce.sh
#   ./reproduce.sh
#
# Requirements: Python 3.10+ (tested with 3.11).
# No external API keys or GPU required for the synthetic scaffold evaluation.
# See SOURCE_VALIDATION_REPORT.md for what additional infrastructure is needed
# to reproduce paper-grade results.

set -euo pipefail

PYTHON=${PYTHON:-python3.11}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== RAGentGuard Reproducibility Script ==="
echo "Python: $($PYTHON --version)"
echo "Working directory: $SCRIPT_DIR"
echo ""

# ------------------------------------------------------------------ #
# 1. Install package (editable, no external deps)                     #
# ------------------------------------------------------------------ #
echo "[1/4] Installing ragentguard (editable)..."
$PYTHON -m pip install -e . -q

# Create output directories up-front so tee never fails on a clean clone
mkdir -p output/experiments

# ------------------------------------------------------------------ #
# 2. Run unit tests                                                    #
# ------------------------------------------------------------------ #
echo ""
echo "[2/4] Running unit tests (23 pytest cases)..."
$PYTHON -m pytest tests/ -v --tb=short 2>&1 | tee output/test_results.txt
echo "Unit tests complete."

# ------------------------------------------------------------------ #
# 3. Run E1-E5 experiments                                            #
# ------------------------------------------------------------------ #
echo ""
echo "[3/4] Running E1-E5 experiments (synthetic scaffold)..."
$PYTHON examples/run_experiments.py 2>&1 | tee output/experiment_results.txt

# ------------------------------------------------------------------ #
# 4. Run basic audit example                                           #
# ------------------------------------------------------------------ #
echo ""
echo "[4/4] Running basic audit example..."
$PYTHON examples/basic_audit.py 2>&1 | tee output/basic_audit_results.txt

echo ""
echo "=== Reproduction complete ==="
echo "Results:"
echo "  Test results:      output/test_results.txt"
echo "  Experiment tables: output/experiment_results.txt"
echo "  E1-E5 JSON:        output/experiments/"
echo "  Audit HTML report: output/audit_report.html"
echo "  Audit JSON:        output/audit_report.json"
echo ""
echo "NOTE: These are synthetic scaffold results (no real RAG pipeline)."
echo "See SOURCE_VALIDATION_REPORT.md for the gap to paper-grade evaluation."
