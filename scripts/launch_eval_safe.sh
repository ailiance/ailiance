#!/usr/bin/env bash
# Wraps run_eval.sh and emits output/eval/last_run_status.json so downstream
# automation can react without parsing free-form logs.
set -euo pipefail
AILIANCE="${AILIANCE_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
STAMP=$(date +%Y%m%d-%H%M%S)
LOG="$AILIANCE/output/eval/launch-${STAMP}.log"
STATUS_FILE="$AILIANCE/output/eval/last_run_status.json"
mkdir -p "$AILIANCE/output/eval"

START=$(date -u +%s)
set +e
bash "$AILIANCE/scripts/run_eval.sh" "$@" 2>&1 | tee "$LOG"
EC=${PIPESTATUS[0]}
set -e
END=$(date -u +%s)
WALL=$((END - START))

# 137 = SIGKILL (128 + 9), 139 = SIGSEGV (128 + 11)
SIGNAL=""
if [[ $EC -gt 128 && $EC -lt 160 ]]; then
    SIGNAL=$((EC - 128))
fi

# JSON encoded via Python so quotes/backslashes/newlines/control chars in
# args or paths can never break the file.
EC=$EC SIGNAL_JSON="${SIGNAL:-null}" WALL=$WALL \
python3 - "$LOG" "$@" <<'PY' > "$STATUS_FILE"
import json, os, sys, time
log_path = sys.argv[1]
args_list = sys.argv[2:]
sig_raw = os.environ.get("SIGNAL_JSON", "null")
status = {
    "stamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "exit_code": int(os.environ["EC"]),
    "signal": None if sig_raw == "null" else int(sig_raw),
    "wall_seconds": int(os.environ["WALL"]),
    "log_path": log_path,
    "args": args_list,
}
print(json.dumps(status, indent=2))
PY

echo "Status: $STATUS_FILE (exit=$EC, signal=${SIGNAL:-none}, wall=${WALL}s)"
exit $EC
