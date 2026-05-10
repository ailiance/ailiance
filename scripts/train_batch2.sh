#!/usr/bin/env bash
# ==============================================================================
# ailiance LoRA Batch 2 — Enriched/Rebuilt Domains
#
# Trains domains that were enriched or rebuilt after batch 1:
#   Devstral: cpp (embedded MCU), shell, html-css, ml-training
#   Apertus: math-reasoning (Orca-Math MIT)
#
# Prerequisites:
#   sudo sysctl -w iogpu.wired_limit_mb=458752
#
# Usage:
#   bash ~/eu-kiki/scripts/train_batch2.sh
#   bash ~/eu-kiki/scripts/train_batch2.sh --dry-run
#
# Generated: 2026-04-29
# ==============================================================================

set -euo pipefail

KIKI_TUNNER="$HOME/KIKI-Mac_tunner"
EU_KIKI="$HOME/eu-kiki"
HF_DATA="$EU_KIKI/data/hf-traced"
ADAPTERS="$EU_KIKI/output/adapters"
OUTPUT_ROOT="$KIKI_TUNNER/output/ailiance-hf"
LOG_DIR="$EU_KIKI/output/training-logs"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

mkdir -p "$LOG_DIR"

# -----------------------------------------------------------------------------
# Training function (same as batch 1)
# -----------------------------------------------------------------------------
train_one() {
    local model_name="$1"
    local domain="$2"
    local model_path="$3"
    local grad_accum="$4"
    local max_seq="$5"
    local n_train="$6"
    local iters="$7"

    local adapter_path="$ADAPTERS/$model_name/$domain"
    local output_dir="$OUTPUT_ROOT/${model_name}-${domain}"
    local data_dir="$HF_DATA/$domain"
    local log_file="$LOG_DIR/${model_name}-${domain}.log"

    # Skip if adapter already exists
    if [[ -f "$adapter_path/adapters.safetensors" ]]; then
        echo "  SKIP $model_name/$domain — adapter already exists"
        return 0
    fi

    echo ""
    echo "  ▶ Training $model_name/$domain ($n_train examples, $iters iters)"
    echo "    Data:   $data_dir"
    echo "    Output: $output_dir"
    echo "    Log:    $log_file"

    if $DRY_RUN; then
        return 0
    fi

    cd "$KIKI_TUNNER"

    "$KIKI_TUNNER/.venv/bin/python" - "$model_path" "$data_dir" "$output_dir" \
        "$adapter_path" "$grad_accum" "$max_seq" "$iters" <<'PYTHON_SCRIPT' 2>&1 | tee "$log_file"
import mlx.core as mx
mx.set_memory_limit(480 * 1024**3)
mx.set_cache_limit(32 * 1024**3)

import os
import sys
import time
import yaml
import shutil
from pathlib import Path

sys.path.insert(0, "/Users/clems/KIKI-Mac_tunner/lib")

model_path = sys.argv[1]
data_dir = sys.argv[2]
output_dir = sys.argv[3]
adapter_dest = sys.argv[4]
grad_accum = int(sys.argv[5])
max_seq = int(sys.argv[6])
iters = int(sys.argv[7])

os.makedirs(output_dir, exist_ok=True)

train_file = Path(data_dir) / "train.jsonl"
n_train = sum(1 for _ in open(train_file))
actual_iters = min(iters, n_train)

config = {
    "model": model_path,
    "fine_tune_type": "lora",
    "lora_parameters": {"rank": 16, "alpha": 32, "dropout": 0.05, "scale": 2.0},
    "num_layers": -1,
    "learning_rate": 1e-5,
    "batch_size": 1,
    "grad_accumulation_steps": grad_accum,
    "iters": actual_iters,
    "max_seq_length": max_seq,
    "grad_checkpoint": True,
    "save_every": 200,
    "steps_per_report": 10,
    "steps_per_eval": 200,
    "val_batches": 5,
    "train": True,
    "seed": 42,
}

config_path = Path(output_dir) / "train_config.yaml"
with open(config_path, "w") as f:
    yaml.dump(config, f, default_flow_style=False)

print(f"Config written: {config_path}")
print(f"Model: {Path(model_path).name}")
print(f"Data: {data_dir} ({n_train} examples)")
print(f"Iters: {actual_iters}")

t0 = time.time()

from mlx_lm_fork.lora import main as lora_main
sys.argv = ["lora", "-c", str(config_path), "--data", data_dir, "--adapter-path", output_dir]
lora_main()

elapsed = time.time() - t0
print(f"\nDone in {elapsed/60:.1f} min")

# Copy adapter
adapter_src = Path(output_dir) / "adapters.safetensors"
if adapter_src.exists():
    dest = Path(adapter_dest)
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(adapter_src), str(dest / "adapters.safetensors"))
    size_mb = adapter_src.stat().st_size / 1048576
    print(f"Copied adapter ({size_mb:.0f} MB) -> {dest}")
else:
    print("WARNING: no adapters.safetensors produced")
    exit(1)
PYTHON_SCRIPT

    echo "  ✓ $model_name/$domain complete"
}

# -----------------------------------------------------------------------------
# Model paths
# -----------------------------------------------------------------------------
APERTUS="$KIKI_TUNNER/models/Apertus-70B-Instruct-2509"
DEVSTRAL="$KIKI_TUNNER/models/Devstral-Small-2-24B-Instruct-2512"

# -----------------------------------------------------------------------------
# Execution plan — Batch 2
# -----------------------------------------------------------------------------
echo "============================================================"
echo " ailiance LoRA Batch 2 — Enriched Domains"
echo " Date: $(date '+%Y-%m-%d %H:%M')"
echo " Data source: $HF_DATA"
if $DRY_RUN; then
    echo " Mode: DRY RUN (no training will be launched)"
fi
echo "============================================================"

STARTED=$(date +%s)

# --- GROUP 1: Devstral-24B (enriched coding domains) ---
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " GROUP 1: Devstral-Small-2-24B (enriched coding)"
echo " Params: grad_accum=4, max_seq=2048, iters=500"
echo " Est. ~25 min/domain × 4 domains = ~100 min"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

#                model      domain         model_path  ga  seq  n_train  iters
train_one "devstral" "cpp"          "$DEVSTRAL"  4  2048  2850     500
train_one "devstral" "shell"        "$DEVSTRAL"  4  2048  2850     500
train_one "devstral" "html-css"     "$DEVSTRAL"  4  2048  2850     500
train_one "devstral" "ml-training"  "$DEVSTRAL"  4  2048  2850     500

# --- GROUP 2: Apertus-70B (rebuilt math-reasoning) ---
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " GROUP 2: Apertus-70B (math-reasoning rebuilt)"
echo " Params: grad_accum=8, max_seq=1024, iters=500"
echo " Est. ~45 min × 1 domain"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

train_one "apertus" "math-reasoning" "$APERTUS"  8  1024  2850  500

# --- Summary ---
ENDED=$(date +%s)
ELAPSED=$(( (ENDED - STARTED) / 60 ))

echo ""
echo "============================================================"
echo " BATCH 2 COMPLETE"
echo " Total wall time: ${ELAPSED} min"
echo " Logs: $LOG_DIR"
echo " Adapters: $ADAPTERS"
echo "============================================================"
