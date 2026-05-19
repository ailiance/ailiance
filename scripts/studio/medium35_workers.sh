#!/usr/bin/env bash
# Unload / reload Studio inference workers around a medium35 training campaign.
# Discovers each worker's launchd label at unload time and records it so the
# matching reload can restart it. Workers with no launchd label were launched
# manually and cannot be restarted automatically (reported RELOAD_FAILED).
set -uo pipefail

GUI="gui/$(id -u)"
STATE_DIR="$HOME/.ailiance-training"
STATE_FILE="$STATE_DIR/unloaded.tsv"

label_for_pid() {
  launchctl list | awk -v pid="$1" '$1 == pid {print $3; exit}'
}

healthy() {
  curl -sf "http://127.0.0.1:$1/health" >/dev/null 2>&1 \
    || curl -sf "http://127.0.0.1:$1/v1/models" >/dev/null 2>&1
}

do_unload() {
  mkdir -p "$STATE_DIR"
  : >"$STATE_FILE"
  local port pid label
  for port in "$@"; do
    pid="$(lsof -ti tcp:"$port" 2>/dev/null | head -1 || true)"
    if [[ -z "$pid" ]]; then
      echo "ALREADY_DOWN $port"
      continue
    fi
    label="$(label_for_pid "$pid")"
    if [[ -n "$label" ]]; then
      launchctl bootout "$GUI/$label" 2>/dev/null || true
      printf '%s\t%s\n' "$port" "$label" >>"$STATE_FILE"
      echo "UNLOADED $port label=$label"
    else
      kill "$pid" 2>/dev/null || true
      printf '%s\t-\n' "$port" >>"$STATE_FILE"
      echo "UNLOADED $port (manual, no label)"
    fi
  done
}

do_reload() {
  [[ -f "$STATE_FILE" ]] || { echo "no unload state, nothing to reload"; return 0; }
  local port label ok
  while IFS=$'\t' read -r port label; do
    [[ -z "$port" ]] && continue
    if [[ "$label" != "-" ]]; then
      launchctl bootstrap "$GUI" \
        "$HOME/Library/LaunchAgents/$label.plist" 2>/dev/null || true
    fi
    ok=""
    for _ in $(seq 1 30); do
      if healthy "$port"; then ok=1; break; fi
      sleep 2
    done
    if [[ -n "$ok" ]]; then
      echo "RELOADED $port"
    else
      echo "RELOAD_FAILED $port"
    fi
  done <"$STATE_FILE"
}

case "${1:-}" in
  unload) shift; do_unload "$@" ;;
  reload) do_reload ;;
  *) echo "usage: $0 {unload|reload}" >&2; exit 2 ;;
esac
