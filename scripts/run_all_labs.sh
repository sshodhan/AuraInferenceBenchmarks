#!/usr/bin/env bash
# Run all inference benchmark labs sequentially.
#
# Prerequisites:
#   - vLLM server running (start with scripts/start_server.sh)
#   - Python dependencies installed (pip install -r requirements.txt)
#
# Usage:
#   ./scripts/run_all_labs.sh          # Run labs 1-3 + 6 (single-server labs)
#   ./scripts/run_all_labs.sh --all    # Run ALL labs (4 & 5 manage their own servers)

set -euo pipefail

cd "$(dirname "$0")/.."

echo "============================================"
echo "  AuraInferenceBenchmarks — Running Labs"
echo "============================================"
echo ""

# Labs 1-3, 6 assume a pre-started server
echo "[Lab 1] Deploy & first measurements"
python -m benchmarks.lab1_deploy
echo ""

echo "[Lab 2] KV cache impact"
python -m benchmarks.lab2_kvcache
echo ""

echo "[Lab 3] Batching & throughput"
python -m benchmarks.lab3_batching
echo ""

echo "[Lab 6] Prefix caching simulation"
python -m benchmarks.lab6_prefix_caching
echo ""

if [ "${1:-}" = "--all" ]; then
    echo "[Lab 4] Model size comparison (manages its own servers)"
    python -m benchmarks.lab4_model_comparison
    echo ""

    echo "[Lab 5] Quantization impact (manages its own servers)"
    python -m benchmarks.lab5_quantization
    echo ""
fi

echo "============================================"
echo "  All labs complete! Results in results/"
echo "============================================"
ls -la results/
