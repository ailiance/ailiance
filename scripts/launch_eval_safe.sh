#!/usr/bin/env bash
# Wraps run_eval.sh and emits output/eval/last_run_status.json so downstream
# automation can react without parsing free-form logs.
set -euo pipefail
EU_KIKI="$HOME/eu-kiki"
STAMP=$(date +%Y%m%d-%H%M%S)
LOG="$EU_KIKI/output/eval/launch-${STAMP}.log"
STATUS_FILE="$EU_KIKI/output/eval/last_run_status.json"
mkdir -p "$EU_KIKI/output/eval"

START=$(date -u +%s)
set +e
bash "$EU_KIKI/scripts/run_eval.sh" "$@" 2>&1 | tee "$LOG"
EC=${PIPESTATUS[0]}
set -e
END=$(date -u +%s)
WALL=$((END - START))

# 137 = SIGKILL (128 + 9), 139 = SIGSEGV (128 + 11)
SIGNAL=""
if [[ $EC -gt 128 && $EC -lt 160 ]]; then
    SIGNAL=$((EC - 128))
fi

cat > "$STATUS_FILE" <<JSON
{
  "stamp_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "exit_code": $EC,
  "signal": ${SIGNAL:-null},
  "wall_seconds": $WALL,
  "log_path": "$LOG",
  "args": "$*"
}
JSON
echo "Status: $STATUS_FILE (exit=$EC, signal=${SIGNAL:-none}, wall=${WALL}s)"
exit $EC
