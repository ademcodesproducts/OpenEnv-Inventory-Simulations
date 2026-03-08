#!/bin/bash
set -e

LOG_DIR="/app/grpo_inventory"
mkdir -p "$LOG_DIR"

echo "[start.sh] Starting OpenEnv inventory server..."
uvicorn server.app:app --host 0.0.0.0 --port 8000 >> "$LOG_DIR/server.log" 2>&1 &
SERVER_PID=$!

echo "[start.sh] Waiting for server to be ready..."
until curl -sf http://localhost:8000/health > /dev/null 2>&1; do
  sleep 1
done
echo "[start.sh] Server ready (PID $SERVER_PID)"

echo "[start.sh] Starting GRPO training... (log: $LOG_DIR/training.log)"
python agent/train_grpo.py \
  --base-url http://localhost:8000 \
  --output-dir "$LOG_DIR" \
  --n-iterations 5 \
  --episodes-per-iter 4 \
  --max-new-tokens 128 \
  --device cuda \
  2>&1 | tee "$LOG_DIR/training.log"

echo "[start.sh] Training complete. Keeping server alive..."
wait $SERVER_PID
