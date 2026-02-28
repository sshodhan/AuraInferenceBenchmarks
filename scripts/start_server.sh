#!/usr/bin/env bash
# Start vLLM OpenAI-compatible API server
#
# Usage:
#   ./scripts/start_server.sh                              # default: Qwen2.5-1.5B
#   ./scripts/start_server.sh Qwen/Qwen2.5-0.5B-Instruct  # custom model
#   ENABLE_PREFIX_CACHE=1 ./scripts/start_server.sh        # with prefix caching
#   PORT=8001 ./scripts/start_server.sh                    # custom port

set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
PORT="${PORT:-8000}"
ENABLE_PREFIX_CACHE="${ENABLE_PREFIX_CACHE:-0}"

ARGS=(
    --model "$MODEL"
    --port "$PORT"
    --trust-remote-code
)

if [ "$ENABLE_PREFIX_CACHE" = "1" ]; then
    ARGS+=(--enable-prefix-caching)
    echo "Prefix caching: ENABLED"
fi

echo "Starting vLLM server..."
echo "  Model: $MODEL"
echo "  Port:  $PORT"
echo "  URL:   http://localhost:$PORT/v1"
echo ""

python -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
