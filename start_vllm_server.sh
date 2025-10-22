#!/bin/bash

# Start vLLM server with gpt-oss-20b model
# This script starts an OpenAI-compatible API server

# Activate virtual environment
source .venv/bin/activate

# Default configuration
MODEL_NAME="${MODEL_NAME:-openai/gpt-oss-20b}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.9}"

echo "Starting vLLM server with configuration:"
echo "  Model: $MODEL_NAME"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Tensor Parallel Size: $TENSOR_PARALLEL_SIZE GPUs"
echo "  GPU Memory Utilization: $GPU_MEMORY_UTIL"
echo ""

# Start the server
vllm serve "$MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTIL"

