#!/usr/bin/env bash
# Phase 3 launcher — full v1 + v2 eval after Phase 1 validation.
#
# Pre-flight:
#   1. Phase 1 quick eval succeeded (perplexity_v1-only_*.json present).
#   2. EuroLLM worker (:9303) stopped to free GPU memory.
#   3. iogpu.wired_limit_mb=458752 (already set on Studio).
#
# Output:
#   $AILIANCE/output/eval/raw/{perplexity,efficiency,generation,speed}_*.json
#   $AILIANCE/output/eval/eval_report_v1_vs_v2.md
#
# Wall-clock estimate: 3-4 h sustained MLX on M3 Ultra 512 GB.

set -euo pipefail

AILIANCE="${AILIANCE_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
RAW="$AILIANCE/output/eval/raw"

# 1. Pre-flight: confirm Phase 1 ran
if ! ls "$RAW"/perplexity_v1-only_*.json >/dev/null 2>&1; then
    echo "ABORT: Phase 1 raw results not found in $RAW"
    echo "Run first: bash $AILIANCE/scripts/run_eval.sh --quick --v1-only"
    exit 1
fi
LATEST_QUICK=$(ls -t "$RAW"/perplexity_v1-only_*.json | head -1)
echo "Phase 1 anchor: $LATEST_QUICK"

# 2. Stop EuroLLM worker if running
WORKER_PID=$(lsof -tiTCP:9303 -sTCP:LISTEN 2>/dev/null || true)
if [[ -n "$WORKER_PID" ]]; then
    echo "Stopping EuroLLM worker (PID $WORKER_PID) to free GPU memory..."
    # Try graceful first
    kill -TERM "$WORKER_PID" 2>/dev/null || true
    sleep 5
    if kill -0 "$WORKER_PID" 2>/dev/null; then
        echo "  Worker still alive, sending SIGKILL..."
        kill -KILL "$WORKER_PID" 2>/dev/null || true
        sleep 2
    fi
    echo "  Worker stopped. Production EuroLLM endpoint :9303 is OFFLINE."
    WORKER_WAS_UP=1
else
    echo "EuroLLM worker not running (or not on :9303). Continuing."
    WORKER_WAS_UP=0
fi

# 3. Launch full eval (v1+v2 compare mode)
LOG="$AILIANCE/output/eval/full-$(date +%Y%m%d-%H%M).log"
echo "Launching full eval. Log: $LOG"
echo "Wall-clock estimate: 3-4 h."
START=$(date +%s)
bash "$AILIANCE/scripts/run_eval.sh" --compare 2>&1 | tee "$LOG"
END=$(date +%s)
ELAPSED=$(( (END - START) / 60 ))
echo "Full eval done in ${ELAPSED} minutes."

# 4. Restart EuroLLM worker if we stopped it
if [[ "$WORKER_WAS_UP" == 1 ]]; then
    echo "Restarting EuroLLM worker..."
    if launchctl list | grep -q ai.eurollm.worker 2>/dev/null; then
        launchctl start ai.eurollm.worker
    else
        echo "  WARN: launchctl service ai.eurollm.worker not found."
        echo "        Restart manually with the saved command."
    fi
fi

# 5. Summary
echo "=== Artefacts ==="
ls -lah "$RAW" | head -20
echo
echo "Report: $AILIANCE/output/eval/eval_report_v1_vs_v2.md"
