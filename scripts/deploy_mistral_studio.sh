#!/usr/bin/env bash
# Deploy ailiance-mistral worker to MacStudio.
#
# Replaces the legacy Apertus-70B :9301 worker with Mistral-Medium-3.5-128B
# Q8 (MLX). Idempotent: rerun safely. Requires passwordless SSH alias 'studio'.
#
# IMPORTANT: pins mlx-core to <0.31 due to a regression introduced in mlx 0.31.x
# where streams are bound to the creating thread. mlx_lm.server uses a worker
# thread for token generation, which crashes with:
#   RuntimeError: There is no Stream(gpu, 0) in current thread
# See https://github.com/ml-explore/mlx for upstream tracking. Reassess on
# each mlx upgrade.
#
# Usage: bash scripts/deploy_mistral_studio.sh [--reload]
#
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-studio}"
REMOTE_USER="${REMOTE_USER:-clems}"
PLIST_LABEL="cc.ailiance.mistral"
PLIST_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/launchd/${PLIST_LABEL}.plist"
PLIST_DST="/Users/${REMOTE_USER}/Library/LaunchAgents/${PLIST_LABEL}.plist"
LOG_DIR="/Users/${REMOTE_USER}/KIKI-Mac_tunner/logs"
MODEL_DIR="/Users/${REMOTE_USER}/KIKI-Mac_tunner/models/Mistral-Medium-3.5-128B-MLX-Q8"
VENV_DIR="/Users/${REMOTE_USER}/.venv-mistral"
PORT=9301

reload_flag=0
[[ "${1:-}" == "--reload" ]] && reload_flag=1

echo "[deploy] checking remote prerequisites on ${REMOTE_HOST}..."
ssh "${REMOTE_HOST}" "test -d '${MODEL_DIR}'" \
    || { echo "[deploy] FAIL: model missing on ${REMOTE_HOST}"; exit 1; }

if ! ssh "${REMOTE_HOST}" "test -x '${VENV_DIR}/bin/python'"; then
    echo "[deploy] creating dedicated venv ${VENV_DIR}..."
    ssh "${REMOTE_HOST}" "/opt/homebrew/bin/uv venv ${VENV_DIR} --python 3.12"
fi

echo "[deploy] pinning mlx_lm with mlx<0.31 (thread-stream regression workaround)..."
ssh "${REMOTE_HOST}" "/opt/homebrew/bin/uv pip install --python ${VENV_DIR}/bin/python --force-reinstall 'mlx<0.31' 'mlx_lm<0.31'" >/dev/null

echo "[deploy] copying plist..."
scp -q "${PLIST_SRC}" "${REMOTE_HOST}:${PLIST_DST}"
ssh "${REMOTE_HOST}" "mkdir -p '${LOG_DIR}' && plutil -lint '${PLIST_DST}'"

if [[ $reload_flag -eq 1 ]]; then
    echo "[deploy] killing existing worker..."
    ssh "${REMOTE_HOST}" "pkill -f 'mlx_lm.server.*${PORT}' 2>/dev/null || true"
    sleep 3
fi

if ! ssh "${REMOTE_HOST}" "lsof -nP -iTCP:${PORT} -sTCP:LISTEN >/dev/null 2>&1"; then
    echo "[deploy] starting worker via nohup (launchd bootstrap requires GUI session)..."
    ssh "${REMOTE_HOST}" "cd /Users/${REMOTE_USER}/KIKI-Mac_tunner && nohup ${VENV_DIR}/bin/python -m mlx_lm.server --model '${MODEL_DIR}' --host 0.0.0.0 --port ${PORT} --log-level INFO > logs/mistral.out.log 2> logs/mistral.err.log < /dev/null & echo started_pid=\$!"
fi

echo "[deploy] waiting for :${PORT} to listen (up to 6 min, MLX 128B Q8 cold-start)..."
for i in $(seq 1 36); do
    if ssh "${REMOTE_HOST}" "lsof -nP -iTCP:${PORT} -sTCP:LISTEN >/dev/null 2>&1"; then
        echo "[deploy] listening on :${PORT} after $((i*10))s"
        break
    fi
    sleep 10
done

echo "[deploy] smoke test..."
ssh "${REMOTE_HOST}" "curl -s --max-time 60 -X POST http://localhost:${PORT}/v1/chat/completions -H 'Content-Type: application/json' -d '{\"messages\":[{\"role\":\"user\",\"content\":\"OK?\"}],\"max_tokens\":5,\"temperature\":0}' | head -c 400"

echo
echo "[deploy] done."
