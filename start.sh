#!/bin/bash
set -e

echo "[start.sh] Starting inventory FastAPI server..."
uvicorn server.inventory_env:app --host 0.0.0.0 --port 8000 &
SERVER_PID=$!

echo "[start.sh] Waiting for server to be ready..."
until curl -sf http://localhost:8000/docs > /dev/null 2>&1; do
  sleep 1
done
echo "[start.sh] Server ready (PID $SERVER_PID)"

echo "[start.sh] Starting GRPO training..."
python agent/train_grpo.py \
  --base-url http://localhost:8000 \
  --n-iterations 5 \
  --episodes-per-iter 4 \
  --max-new-tokens 128 \
  --device cuda

echo "[start.sh] Training complete. Keeping server alive..."
wait $SERVER_PID
