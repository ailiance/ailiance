#!/usr/bin/env bash
# Unload / reload Studio inference workers around a medium35 training campaign.
# At unload, captures each worker's full command line (base64) so reload can
# restart it via nohup — independent of launchd, which cannot be driven over
# SSH. Worker ports are passed as arguments by the gateway (studio_ops.py is
# the single source of the port list).
set -uo pipefail

GUI="gui/$(id -u)"
STATE_DIR="$HOME/.ailiance-training"
STATE_FILE="$STATE_DIR/unloaded.tsv"

healthy() {
  curl -sf "http://127.0.0.1:$1/health" >/dev/null 2>&1 \
    || curl -sf "http://127.0.0.1:$1/v1/models" >/dev/null 2>&1
}

label_for_pid() {
  launchctl list 2>/dev/null | awk -v pid="$1" '$1 == pid {print $3; exit}'
}

do_unload() {
  mkdir -p "$STATE_DIR"
  : >"$STATE_FILE"
  local port pid label cmd
  for port in "$@"; do
    pid="$(lsof -ti tcp:"$port" 2>/dev/null | head -1 || true)"
    if [[ -z "$pid" ]]; then
      echo "ALREADY_DOWN $port"
      continue
    fi
    # Capture the worker's exact argv so reload can re-run it. NOTE: if the
    # worker was started without an explicit non-loopback --host, the reloaded
    # process rebinds to 127.0.0.1 and the local healthcheck still passes
    # while the gateway cannot reach it. Studio workers must be launched with
    # --host 0.0.0.0.
    cmd="$(ps -o command= -p "$pid" 2>/dev/null)"
    label="$(label_for_pid "$pid")"
    # record: port <TAB> label-or-"-" <TAB> base64(command line)
    printf '%s\t%s\t%s\n' "$port" "${label:--}" \
      "$(printf '%s' "$cmd" | base64 | tr -d '\n')" >>"$STATE_FILE"
    if [[ -n "$label" ]]; then
      launchctl bootout "$GUI/$label" 2>/dev/null || true
    fi
    kill "$pid" 2>/dev/null || true
    echo "UNLOADED $port"
  done
}

do_reload() {
  [[ -f "$STATE_FILE" ]] || { echo "no unload state, nothing to reload"; return 0; }
  local port label cmd_b64 cmd ok
  while IFS=$'\t' read -r port label cmd_b64; do
    [[ -z "$port" ]] && continue
    if healthy "$port"; then echo "RELOADED $port"; continue; fi
    cmd="$(printf '%s' "$cmd_b64" | base64 --decode 2>/dev/null || true)"
    if [[ -n "$cmd" ]]; then
      nohup bash -c "$cmd" >>"$STATE_DIR/reload-$port.log" 2>&1 </dev/null &
    elif [[ "$label" != "-" ]]; then
      launchctl bootstrap "$GUI" \
        "$HOME/Library/LaunchAgents/$label.plist" 2>/dev/null || true
    fi
    ok=""
    for _ in $(seq 1 60); do
      if healthy "$port"; then ok=1; break; fi
      sleep 5
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
  *) echo "usage: $0 {unload <ports...>|reload}" >&2; exit 2 ;;
esac
