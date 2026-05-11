#!/usr/bin/env bash
# Auto-generated rerun script for eval result devstral-python-adapter-2026-05-04
# Generated at 2026-05-04T19:51:40+0200
#
# This re-runs the SAME benchmarks against the SAME model+adapter.
# To reproduce identically, check out git commit b1ef93e2a142dfc706117139fd3b95fc6afa1dac first.

set -euo pipefail

LABEL="devstral-python-adapter-2026-05-04"
MODEL="/Users/electron/Projets/ailiance-mac-tuner/models/Devstral-Small-2-24B-MLX-4bit"
ADAPTER="/Users/electron/Projets/ailiance/output/adapters/devstral/python"
PORT=8802

EVAL_DIR="$(cd "$(dirname "$0")/../../.." && pwd)/eval"
cd "$EVAL_DIR/.."   # ailiance root

bash eval/run_all.sh \
    --model "$MODEL" \
    --adapter "$ADAPTER" \
    --label "$LABEL-rerun-$(date +%Y%m%d-%H%M)" \
    --port "$PORT" \
    --quick
