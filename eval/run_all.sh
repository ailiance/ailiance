#!/usr/bin/env bash
# run_all.sh — orchestrate full publishable benchmark suite for one model+adapter
#
# Spawns mlx_lm server with the given adapter, runs HumanEval+/MBPP+/GSM8K/
# MMLU-Pro/IFEval/BBH (and BBEH if --extended), captures env snapshot, and
# writes a single report.md.
#
# Usage:
#   bash eval/run_all.sh \\
#       --model models/Devstral-Small-2-24B-MLX-4bit \\
#       --adapter output/adapters/devstral/python \\
#       --label devstral-v1-python \\
#       [--quick]                       # smoke run with fewer samples
#       [--extended]                    # add BBEH, MultiPL-E
#       [--port 8000] [--max-samples N]

set -euo pipefail

# ---- Defaults --------------------------------------------------------------
MODEL=""
ADAPTER=""
LABEL=""
QUICK=0
EXTENDED=0
PORT=8000
MAX_SAMPLES=""
SUITE_ROOT="$(cd "$(dirname "$0")" && pwd)"
EU_KIKI_ROOT="$(cd "$SUITE_ROOT/.." && pwd)"
DATE="$(date +%Y-%m-%d)"

# ---- Parse -----------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)        MODEL="$2"; shift 2 ;;
        --adapter)      ADAPTER="$2"; shift 2 ;;
        --label)        LABEL="$2"; shift 2 ;;
        --quick)        QUICK=1; shift ;;
        --extended)     EXTENDED=1; shift ;;
        --port)         PORT="$2"; shift 2 ;;
        --max-samples)  MAX_SAMPLES="$2"; shift 2 ;;
        -h|--help)      sed -n '1,20p' "$0"; exit 0 ;;
        *)              echo "Unknown: $1" >&2; exit 2 ;;
    esac
done

[[ -n "$MODEL" ]] || { echo "ERROR: --model required" >&2; exit 2; }
[[ -n "$LABEL" ]] || LABEL="$(basename "$MODEL")$([ -n "$ADAPTER" ] && echo "-$(basename "$ADAPTER")")"

OUT="$EU_KIKI_ROOT/eval/results/$DATE/$LABEL"
mkdir -p "$OUT"

PY="$EU_KIKI_ROOT/.venv/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python3)"

echo "============================================================"
echo " eu-kiki publishable bench — $LABEL"
echo " Model:    $MODEL"
echo " Adapter:  ${ADAPTER:-<none>}"
echo " Output:   $OUT"
echo " Quick:    $QUICK   Extended: $EXTENDED"
echo "============================================================"

# ---- Snapshot environment FIRST (before loading model) ---------------------
"$PY" -m runners.mlx_server_runner \
    --model "$MODEL" \
    ${ADAPTER:+--adapter "$ADAPTER"} \
    --port "$PORT" \
    --env-out "$OUT/env.json" \
    || { echo "Env snapshot failed" >&2; exit 1; }

# ---- Spawn server in background --------------------------------------------
SERVER_LOG="$OUT/mlx_server.log"
"$PY" -m runners.mlx_server_runner \
    --model "$MODEL" \
    ${ADAPTER:+--adapter "$ADAPTER"} \
    --port "$PORT" \
    --log-file "$SERVER_LOG" \
    --block &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT INT TERM

# Wait for ready (the runner above blocks until it's ready, but in --block
# mode it stays up; we re-poll just in case).
for _ in $(seq 1 60); do
    if curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
        break
    fi
    sleep 1
done
curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 \
    || { echo "ERROR: server not responding" >&2; exit 1; }

BASE_URL="http://127.0.0.1:$PORT/v1"
MODEL_NAME="$LABEL"

# ---- Build task list -------------------------------------------------------
LIGHTEVAL_TASKS=""
add_task() { LIGHTEVAL_TASKS="${LIGHTEVAL_TASKS:+$LIGHTEVAL_TASKS,}$1"; }

if (( QUICK )); then
    add_task "lighteval|humaneval|0|0"
    add_task "lighteval|gsm8k|5|0"
else
    add_task "lighteval|humaneval|0|0"
    add_task "lighteval|mbpp|3|0"
    add_task "lighteval|gsm8k|5|0"
    add_task "lighteval|mmlu_pro|5|0"
    add_task "lighteval|ifeval|0|0"
    add_task "lighteval|hellaswag|10|0"
fi
if (( EXTENDED )); then
    add_task "extended|bbeh|0|0"
    add_task "lighteval|big_bench_hard|3|1"
fi

# ---- Run lighteval ---------------------------------------------------------
echo ">>> Lighteval tasks: $LIGHTEVAL_TASKS"
"$PY" -m runners.lighteval_runner \
    --base-url "$BASE_URL" \
    --model "$MODEL_NAME" \
    --tasks "$LIGHTEVAL_TASKS" \
    --output-dir "$OUT/lighteval" \
    ${MAX_SAMPLES:+--max-samples "$MAX_SAMPLES"} \
    || echo "WARN: lighteval had errors, continuing"

# ---- Run EvalPlus HumanEval+ -----------------------------------------------
echo ">>> EvalPlus humanevalplus"
"$PY" -m runners.evalplus_runner \
    --base-url "$BASE_URL" \
    --model "$MODEL_NAME" \
    --task humanevalplus \
    --output-dir "$OUT/evalplus_humanevalplus" \
    --temperature 0.0 \
    --n-samples 1 \
    || echo "WARN: evalplus humanevalplus failed, continuing"

if (( !QUICK )); then
    echo ">>> EvalPlus mbppplus"
    "$PY" -m runners.evalplus_runner \
        --base-url "$BASE_URL" \
        --model "$MODEL_NAME" \
        --task mbppplus \
        --output-dir "$OUT/evalplus_mbppplus" \
        --temperature 0.0 \
        --n-samples 1 \
        || echo "WARN: evalplus mbppplus failed, continuing"
fi

# ---- Generate report -------------------------------------------------------
"$PY" -c "
import json, pathlib, time
out = pathlib.Path('$OUT')
metrics = {}
for f in out.rglob('results.json'):
    rel = f.relative_to(out).parent
    try:
        d = json.loads(f.read_text())
        if 'metrics' in d:
            for k, v in d['metrics'].items():
                metrics[f'{rel}/{k}'] = v
        if 'pass_at_k' in d:
            for k, v in d['pass_at_k'].items():
                metrics[f'{rel}/{k}'] = v
    except Exception as e:
        metrics[f'{rel}/__error__'] = str(e)

env = json.loads((out / 'env.json').read_text()) if (out / 'env.json').exists() else {}
report = ['# eu-kiki bench — $LABEL', '']
report += ['Generated: ' + time.strftime('%Y-%m-%dT%H:%M:%S%z'), '']
report += ['## Environment', '', '\`\`\`json', json.dumps(env, indent=2), '\`\`\`', '']
report += ['## Metrics', '', '| Task | Value |', '|---|---|']
for k, v in sorted(metrics.items()):
    val = f'{v:.4f}' if isinstance(v, (int, float)) else str(v)
    report.append(f'| \`{k}\` | {val} |')
report.append('')
(out / 'report.md').write_text('\n'.join(report))
print(f'Report → {out}/report.md')
"

echo ""
echo "============================================================"
echo " Done. Report at $OUT/report.md"
echo "============================================================"
