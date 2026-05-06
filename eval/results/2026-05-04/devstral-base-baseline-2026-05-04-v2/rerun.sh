#!/usr/bin/env bash
# Auto-generated rerun script for eval result devstral-base-baseline-2026-05-04-v2
# Generated at 2026-05-04T18:32:39+0200
#
# This re-runs the SAME benchmarks against the SAME model+adapter.
# To reproduce identically, check out git commit fd120b337f624bee2ebb932c9e22c19637bd1ba2 first.

set -euo pipefail

LABEL="devstral-base-baseline-2026-05-04-v2"
MODEL="/Users/electron/Projets/KIKI-Mac_tunner/models/Devstral-Small-2-24B-MLX-4bit"
ADAPTER=""
PORT=8801

EVAL_DIR="$(cd "$(dirname "$0")/../../.." && pwd)/eval"
cd "$EVAL_DIR/.."   # ailiance root

bash eval/run_all.sh \
    --model "$MODEL" \
     \
    --label "$LABEL-rerun-$(date +%Y%m%d-%H%M)" \
    --port "$PORT" \
    --quick
