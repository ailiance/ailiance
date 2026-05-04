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
# mlx_lm.server registers the model under its absolute path. The OpenAI client
# `model` field must match this exactly, otherwise HF tries to look it up online.
MODEL_NAME="$(cd "$(dirname "$MODEL")" && pwd)/$(basename "$MODEL")"

# ---- Build task list -------------------------------------------------------
# Lighteval 0.13 task name format: <task>|<fewshot>  (no suite prefix).
# HumanEval/MBPP are NOT in Lighteval 0.13 — use EvalPlus runner for code.
# BBH/BBEH renamed; subtasks live under bigbench:<task>.
LIGHTEVAL_TASKS=""
add_task() { LIGHTEVAL_TASKS="${LIGHTEVAL_TASKS:+$LIGHTEVAL_TASKS,}$1"; }

if (( QUICK )); then
    add_task "gsm8k|5"
else
    add_task "gsm8k|5"
    add_task "mmlu_pro|5"
    add_task "ifeval|0"
    add_task "hellaswag|10"
    add_task "truthfulqa:mc|0"
fi
if (( EXTENDED )); then
    # bigbench has 200+ subtasks; pick a representative reasoning sample
    add_task "bigbench:logical_deduction_seven_objects|3"
    add_task "bigbench:tracking_shuffled_objects_seven_objects|3"
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

# ---- Stop server before report -----------------------------------------------
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

# ---- Generate methodology, rerun, and report -------------------------------
EVALPLUS_TASKS_LIST="humanevalplus"
if (( !QUICK )); then
    EVALPLUS_TASKS_LIST="$EVALPLUS_TASKS_LIST,mbppplus"
fi

"$PY" -m runners.result_writer \
    --output-dir "$OUT" \
    --label "$LABEL" \
    --lighteval-tasks "$LIGHTEVAL_TASKS" \
    --evalplus-tasks "$EVALPLUS_TASKS_LIST" \
    --port "$PORT" \
    --temperature 0.0 \
    --max-tokens 1024 \
    --n-samples 1 \
    --seed 42 \
    $( (( QUICK )) && echo "--quick" ) \
    $( (( EXTENDED )) && echo "--extended" )

echo ""
echo "============================================================"
echo " Done."
echo "  Report:        $OUT/report.md"
echo "  Methodology:   $OUT/methodology.md"
echo "  Rerun script:  $OUT/rerun.sh"
echo "  Env snapshot:  $OUT/env.json"
echo "============================================================"
