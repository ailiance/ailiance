#!/usr/bin/env bash
# scripts/start.sh — Launch all eu-kiki workers + gateway
# Ports 930x to avoid conflict with micro-kiki (920x)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="${ROOT}/.venv/bin/python"
LOG_DIR="/tmp/eu-kiki"
mkdir -p "$LOG_DIR"

echo "[$(date '+%H:%M:%S')] Starting eu-kiki workers..."

"$PYTHON" -m uvicorn src.worker.server:make_apertus_app --factory \
    --host 127.0.0.1 --port 9301 > "$LOG_DIR/apertus.log" 2>&1 &
echo "[$(date '+%H:%M:%S')] Apertus worker started (PID $!, port 9301)"

"$PYTHON" -m uvicorn src.worker.server:make_devstral_app --factory \
    --host 127.0.0.1 --port 9302 > "$LOG_DIR/devstral.log" 2>&1 &
echo "[$(date '+%H:%M:%S')] Devstral worker started (PID $!, port 9302)"

"$PYTHON" -m uvicorn src.worker.server:make_eurollm_app --factory \
    --host 127.0.0.1 --port 9303 > "$LOG_DIR/eurollm.log" 2>&1 &
echo "[$(date '+%H:%M:%S')] EuroLLM worker started (PID $!, port 9303)"

sleep 15

"$PYTHON" -m uvicorn src.gateway.server:make_gateway_app --factory \
    --host 127.0.0.1 --port 9300 > "$LOG_DIR/gateway.log" 2>&1 &
echo "[$(date '+%H:%M:%S')] Gateway started (PID $!, port 9300)"

echo "[$(date '+%H:%M:%S')] eu-kiki running. Logs in $LOG_DIR/"
echo "  Gateway:  http://localhost:9300"
echo "  Apertus:  http://localhost:9301"
echo "  Devstral: http://localhost:9302"
echo "  EuroLLM:  http://localhost:9303"
