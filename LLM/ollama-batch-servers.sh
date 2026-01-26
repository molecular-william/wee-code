#!/bin/bash

# Constants
HOST="0.0.0.0"
BASE_PORT=11432
OLLAMA_BINARY="/mnt/data/wee/ollama/bin/ollama"
LOG_DIR="ollama-server-logs"
SLEEP_INTERVAL=1

# Check if the number of GPUs is provided as an argument
if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <num_gpus>"
    exit 1
fi

# Command-line argument
NUM_GPUS=$1

# Validate that NUM_GPUS is a positive integer
if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]] || [[ "$NUM_GPUS" -le 0 ]]; then
    echo "Error: <num_gpus> must be a positive integer."
    exit 1
fi

# Check if the Ollama binary exists
if [[ ! -x "$OLLAMA_BINARY" ]]; then
    echo "Error: Ollama binary not found or not executable at $OLLAMA_BINARY"
    exit 1
fi

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Kill existing processes using the ports
for ((i=0; i<NUM_GPUS; i++)); do
    PORT=$((BASE_PORT + i))
    PID=$(lsof -ti :$PORT)
    if [[ ! -z "$PID" ]]; then
        echo "Killing existing process on port ${PORT} (PID: $PID)"
        kill -9 $PID
    fi
done

# Start server instances
for ((i=0; i<NUM_GPUS; i++)); do
    PORT=$((BASE_PORT + i))
    LOG_FILE="${LOG_DIR}/${PORT}.log"

    # Environment variables
    export OLLAMA_LOAD_TIMEOUT="1m"
    export OLLAMA_KEEP_ALIVE="10s"
    export OLLAMA_NUM_PARALLEL="4" # our gpus probably too weak to handle anything larger than 1
    export OLLAMA_HOST="${HOST}:${PORT}"
    export CUDA_VISIBLE_DEVICES="$i"
    export OLLAMA_CONTEXT_LENGTH="4500"  # cant go above 18k for qwen3:8b models, cant go above 5k for qwen3:14b
    export OLLAMA_FLASH_ATTENTION="1"  # flash attention
    export OLLAMA_KV_CACHE_TYPE="q8_0"  # default is fp16
    export OLLAMA_NEW_ENGINE="1"
    export OLLAMA_NEW_ESTIMATES="1"

    # Start server with nohup and log output
    nohup "$OLLAMA_BINARY" serve > "$LOG_FILE" 2>&1 &

    if [[ $? -eq 0 ]]; then
        echo "Started server instance $i on port ${PORT}, logging to ${LOG_FILE}"
    else
        echo "Error: Failed to start server instance $i on port ${PORT}"
    fi

    # Sleep interval between starting instances
    sleep "$SLEEP_INTERVAL"
done

echo "All server instances started successfully."

