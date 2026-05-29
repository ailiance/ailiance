#!/usr/bin/env bash
# ==============================================================================
# ailiance VLM PoC — Vision LoRA training on Devstral Small 2
#
# Trains a vision-enabled LoRA adapter using schematic/diagram images
# from ST Application Notes and Espressif datasheets.
#
# Uses mlx-vlm (not mlx-lm) with --train-vision flag to unfreeze
# the vision tower and multimodal projector.
#
# Model: Devstral-Small-2-24B-MLX-4bit (QLoRA for memory efficiency)
# Dataset: 924 VLM pairs from PDF extraction pipeline
#
# EU AI Act compliance: full provenance tracked in vlm-compliance-report.md
# Legal basis: DSM Directive Art.4 TDM (no opt-out from sources)
#
# Prerequisites: sudo sysctl -w iogpu.wired_limit_mb=458752
# ==============================================================================

set -euo pipefail

AILIANCE="${AILIANCE_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
KIKI_TUNNER="${KIKI_TUNNER_HOME:-$(dirname "$AILIANCE")/ailiance-mac-tuner}"
PYTHON="$KIKI_TUNNER/.venv/bin/python3"
MODEL="$KIKI_TUNNER/models/Devstral-Small-2-24B-BF16"
DATA="$AILIANCE/data/vlm-dataset"
OUTPUT="$KIKI_TUNNER/output/ailiance-hf/devstral-vlm-schematic"
ADAPTER_DEST="$AILIANCE/output/adapters/devstral/vlm-schematic"
LOG_DIR="$AILIANCE/output/training-logs"
LOG_FILE="$LOG_DIR/devstral-vlm-schematic.log"

mkdir -p "$LOG_DIR" "$OUTPUT"

echo "============================================================"
echo " ailiance VLM PoC — Vision LoRA Training"
echo " Date: $(date '+%Y-%m-%d %H:%M')"
echo " Model: $(basename $MODEL)"
echo " Dataset: $DATA ($(wc -l < $DATA/train.jsonl | tr -d ' ') train)"
echo " Output: $OUTPUT"
echo "============================================================"

STARTED=$(date +%s)

cd "$KIKI_TUNNER"

# Run mlx-vlm LoRA training with vision unfreezing
# mlx-vlm uses --dataset (not --data), --model-path (not --model),
# --output-path (not --adapter-path), --steps-per-save (not --save-every)
# No --train flag, no --lora-layers, no --seed
$PYTHON -m mlx_vlm.lora \
    --model-path "$MODEL" \
    --dataset "$DATA" \
    --output-path "$OUTPUT" \
    --train-vision \
    --batch-size 1 \
    --lora-rank 16 \
    --lora-alpha 32 \
    --lora-dropout 0.05 \
    --iters 500 \
    --learning-rate 1e-5 \
    --steps-per-report 10 \
    --steps-per-eval 100 \
    --val-batches 5 \
    --steps-per-save 200 \
    --grad-checkpoint \
    --gradient-accumulation-steps 4 \
    2>&1 | tee "$LOG_FILE"

ENDED=$(date +%s)
ELAPSED=$(( (ENDED - STARTED) / 60 ))

# Copy adapter to ailiance output
if [[ -f "$OUTPUT/adapters.safetensors" ]]; then
    mkdir -p "$ADAPTER_DEST"
    cp "$OUTPUT/adapters.safetensors" "$ADAPTER_DEST/"
    SIZE_MB=$(du -m "$ADAPTER_DEST/adapters.safetensors" | cut -f1)
    echo ""
    echo "✓ VLM adapter saved: $ADAPTER_DEST (${SIZE_MB} MB)"
else
    echo ""
    echo "WARNING: No adapter produced"
fi

echo ""
echo "============================================================"
echo " VLM PoC COMPLETE — ${ELAPSED} min"
echo " Adapter: $ADAPTER_DEST"
echo " Log: $LOG_FILE"
echo "============================================================"
