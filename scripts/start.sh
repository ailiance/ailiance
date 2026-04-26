#!/usr/bin/env bash
# scripts/start.sh — Launch all eu-kiki workers + gateway
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="${ROOT}/.venv/bin/python"
LOG_DIR="/tmp/eu-kiki"
mkdir -p "$LOG_DIR"

echo "[$(date '+%H:%M:%S')] Starting eu-kiki workers..."

"$PYTHON" -m uvicorn src.worker.server:make_apertus_app --factory \
    --host 127.0.0.1 --port 9201 > "$LOG_DIR/apertus.log" 2>&1 &
echo "[$(date '+%H:%M:%S')] Apertus worker started (PID $!, port 9201)"

"$PYTHON" -m uvicorn src.worker.server:make_devstral_app --factory \
    --host 127.0.0.1 --port 9202 > "$LOG_DIR/devstral.log" 2>&1 &
echo "[$(date '+%H:%M:%S')] Devstral worker started (PID $!, port 9202)"

"$PYTHON" -m uvicorn src.worker.server:make_eurollm_app --factory \
    --host 127.0.0.1 --port 9203 > "$LOG_DIR/eurollm.log" 2>&1 &
echo "[$(date '+%H:%M:%S')] EuroLLM worker started (PID $!, port 9203)"

sleep 5

"$PYTHON" -m uvicorn src.gateway.server:make_gateway_app --factory \
    --host 127.0.0.1 --port 9200 > "$LOG_DIR/gateway.log" 2>&1 &
echo "[$(date '+%H:%M:%S')] Gateway started (PID $!, port 9200)"

echo "[$(date '+%H:%M:%S')] eu-kiki running. Logs in $LOG_DIR/"
echo "  Gateway:  http://localhost:9200"
echo "  Apertus:  http://localhost:9201"
echo "  Devstral: http://localhost:9202"
echo "  EuroLLM:  http://localhost:9203"
