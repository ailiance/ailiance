#!/usr/bin/env bash
# fuse_adapter.sh — bake a LoRA adapter into a base model and produce a
# self-contained 4-bit MLX model.
#
# Workaround for the mlx_lm.server `--adapter-path` issue on 4-bit models
# (LoRA silently NOT applied to QuantizedLinear modules). Fusing the adapter
# into the base produces a new checkpoint that can be served without
# `--adapter-path`, guaranteed to include the LoRA contribution.
#
# Workflow:
#   1. Fuse adapter into BF16 base -> produces new BF16 checkpoint
#   2. Quantize to 4-bit MLX -> produces servable model
#
# Usage:
#   bash eval/runners/fuse_adapter.sh \\
#       --base   /Users/clems/ailiance-mac-tuner/models/Devstral-Small-2-24B-Instruct-2512 \\
#       --adapter /Users/clems/ailiance/output/adapters/devstral/python \\
#       --out-name Devstral-2-24B-MLX-4bit-python \\
#       --models-dir /Users/clems/ailiance-mac-tuner/models
#
# Output:
#   <models-dir>/<out-name>/  (self-contained MLX 4-bit model)

set -euo pipefail

BASE=""
ADAPTER=""
OUT_NAME=""
MODELS_DIR=""
SKIP_QUANTIZE=0
KEEP_FUSED=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base)              BASE="$2"; shift 2 ;;
        --adapter)           ADAPTER="$2"; shift 2 ;;
        --out-name)          OUT_NAME="$2"; shift 2 ;;
        --models-dir)        MODELS_DIR="$2"; shift 2 ;;
        --skip-quantize)     SKIP_QUANTIZE=1; shift ;;
        --keep-fused)        KEEP_FUSED=1; shift ;;
        -h|--help)           sed -n '1,25p' "$0"; exit 0 ;;
        *)                   echo "Unknown: $1" >&2; exit 2 ;;
    esac
done

[[ -n "$BASE" && -n "$ADAPTER" && -n "$OUT_NAME" && -n "$MODELS_DIR" ]] \
  || { echo "ERROR: --base, --adapter, --out-name, --models-dir required" >&2; exit 2; }

[[ -d "$BASE" ]]    || { echo "ERROR: base not found: $BASE" >&2; exit 2; }
[[ -d "$ADAPTER" ]] || { echo "ERROR: adapter not found: $ADAPTER" >&2; exit 2; }
[[ -f "$ADAPTER/adapters.safetensors" ]] || { echo "ERROR: missing adapters.safetensors in $ADAPTER" >&2; exit 2; }
[[ -f "$ADAPTER/adapter_config.json" ]]  || { echo "ERROR: missing adapter_config.json in $ADAPTER" >&2; exit 2; }

FUSED_DIR="$MODELS_DIR/${OUT_NAME}-fused-bf16"
QUANT_DIR="$MODELS_DIR/${OUT_NAME}"
PY="${PY:-$(command -v python3)}"

echo "============================================================"
echo " Fuse adapter"
echo " Base:    $BASE"
echo " Adapter: $ADAPTER"
echo " Fused:   $FUSED_DIR"
echo " Final:   $QUANT_DIR"
echo "============================================================"

# Step 1: fuse to BF16 (de-quantize and merge LoRA)
"$PY" -m mlx_lm fuse \
    --model "$BASE" \
    --adapter-path "$ADAPTER" \
    --save-path "$FUSED_DIR" \
    --dequantize 2>&1 | tail -5

# Step 2: quantize to 4-bit (skipped on --skip-quantize)
if (( SKIP_QUANTIZE )); then
    echo "Skipped 4-bit quantization (--skip-quantize). Fused model at: $FUSED_DIR"
    exit 0
fi

"$PY" -m mlx_lm convert \
    --hf-path "$FUSED_DIR" \
    -q --q-bits 4 --q-group-size 64 \
    --mlx-path "$QUANT_DIR" 2>&1 | tail -5

# Step 3: validate via /v1/models smoke
echo ">>> Smoke: load model + 1 generation"
"$PY" -c "
from mlx_lm.utils import load
m, t = load('$QUANT_DIR')
from mlx_lm.generate import generate
out = generate(m, t, t.apply_chat_template([{'role':'user','content':'Test, dis bonjour.'}], tokenize=False, add_generation_prompt=True), max_tokens=30, verbose=False)
print('OUT:', out[:200])
"

if (( !KEEP_FUSED )); then
    echo "Removing intermediate BF16 fused dir: $FUSED_DIR"
    rm -rf "$FUSED_DIR"
fi

echo ""
echo "Done. Servable model: $QUANT_DIR"
echo "Use as drop-in: mlx_lm server --model $QUANT_DIR --port 8810"
