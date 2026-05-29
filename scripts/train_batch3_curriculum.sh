#!/usr/bin/env bash
# ==============================================================================
# ailiance LoRA Batch 3 — Curriculum Retrain with Intelligent Splitting
#
# Retrains 3 domains with:
#   - Smart sequence splitting for very long records (>2x max_seq)
#   - Curriculum learning (short -> long ordering)
#   - Increased max_seq and grad_accum for 512GB M3 Ultra
#
# Training config:
#   devstral/cpp:              max_seq=8192, grad_accum=16 (~150GB peak)
#   apertus/emc-dsp-power:     max_seq=4096, grad_accum=16
#   apertus/security-fenrir:   max_seq=4096, grad_accum=16
#
# Prerequisites:
#   sudo sysctl -w iogpu.wired_limit_mb=458752
#
# Usage:
#   bash $AILIANCE/scripts/train_batch3_curriculum.sh           # run all
#   bash $AILIANCE/scripts/train_batch3_curriculum.sh --dry-run  # show plan only
#
# Generated: 2026-04-28
# ==============================================================================

set -euo pipefail

AILIANCE="${AILIANCE_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
KIKI_TUNNER="${KIKI_TUNNER_HOME:-$(dirname "$AILIANCE")/ailiance-mac-tuner}"
HF_DATA="$AILIANCE/data/hf-traced"
ADAPTERS="$AILIANCE/output/adapters"
OUTPUT_ROOT="$KIKI_TUNNER/output/ailiance-hf"
LOG_DIR="$AILIANCE/output/training-logs"
BACKUP_DIR="$AILIANCE/output/adapters-backup-pre-curriculum"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

mkdir -p "$LOG_DIR"

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
echo "============================================================"
echo " ailiance LoRA Batch 3 — Curriculum Retrain (Split + Sort)"
echo " Date: $(date '+%Y-%m-%d %H:%M')"
echo " Data source: $HF_DATA"
if $DRY_RUN; then
    echo " Mode: DRY RUN (no training will be launched)"
fi
echo "============================================================"

# -----------------------------------------------------------------------------
# Step 1: Prepare curriculum-sorted data with intelligent splitting
# -----------------------------------------------------------------------------
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " STEP 1: Prepare curriculum data (split + sort short->long)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

PREPARE_SCRIPT="$AILIANCE/scripts/prepare_curriculum.py"

if $DRY_RUN; then
    echo ""
    echo "  [Devstral domains — max_seq=8192]"
    echo "  Would run: prepare_curriculum.py --domains cpp --max-seq 8192 --stats-only"
    cd "$AILIANCE" && uv run python "$PREPARE_SCRIPT" \
        --domains "cpp" --max-seq 8192 --stats-only

    echo ""
    echo "  [Apertus domains — max_seq=4096]"
    echo "  Would run: prepare_curriculum.py --domains emc-dsp-power,security-fenrir --max-seq 4096 --stats-only"
    cd "$AILIANCE" && uv run python "$PREPARE_SCRIPT" \
        --domains "emc-dsp-power,security-fenrir" --max-seq 4096 --stats-only
else
    echo ""
    echo "  [Devstral domains — max_seq=8192]"
    cd "$AILIANCE" && uv run python "$PREPARE_SCRIPT" \
        --domains "cpp" --max-seq 8192

    echo ""
    echo "  [Apertus domains — max_seq=4096]"
    cd "$AILIANCE" && uv run python "$PREPARE_SCRIPT" \
        --domains "emc-dsp-power,security-fenrir" --max-seq 4096
fi

# -----------------------------------------------------------------------------
# Step 2: Backup existing adapters
# -----------------------------------------------------------------------------
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " STEP 2: Backup existing adapters"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

backup_adapter() {
    local model_name="$1"
    local domain="$2"
    local src="$ADAPTERS/$model_name/$domain"
    local dst="$BACKUP_DIR/$model_name/$domain"

    if [[ -f "$src/adapters.safetensors" ]]; then
        echo "  Backup: $model_name/$domain"
        if ! $DRY_RUN; then
            mkdir -p "$dst"
            mv "$src/adapters.safetensors" "$dst/adapters.safetensors"
            echo "    Moved to $dst/"
        else
            echo "    Would move $src/adapters.safetensors -> $dst/"
        fi
    else
        echo "  No adapter to backup for $model_name/$domain"
    fi
}

backup_adapter "devstral" "cpp"
backup_adapter "apertus" "emc-dsp-power"
backup_adapter "apertus" "security-fenrir"

# -----------------------------------------------------------------------------
# Training function (uses curriculum data with swap)
# -----------------------------------------------------------------------------
train_one_curriculum() {
    local model_name="$1"
    local domain="$2"
    local model_path="$3"
    local grad_accum="$4"
    local max_seq="$5"
    local iters="$6"

    local adapter_path="$ADAPTERS/$model_name/$domain"
    local output_dir="$OUTPUT_ROOT/${model_name}-${domain}-curriculum"
    local data_dir="$HF_DATA/$domain"
    local log_file="$LOG_DIR/${model_name}-${domain}-curriculum.log"

    # Skip if adapter already exists (safety check after backup)
    if [[ -f "$adapter_path/adapters.safetensors" ]]; then
        echo "  SKIP $model_name/$domain — adapter already exists"
        return 0
    fi

    local curriculum_train="$data_dir/train_curriculum.jsonl"
    local n_train
    n_train=$(wc -l < "$curriculum_train" 2>/dev/null || echo "0")
    n_train=$(echo "$n_train" | tr -d ' ')
    local actual_iters=$((iters < n_train ? iters : n_train))

    echo ""
    echo "  Training: $model_name/$domain"
    echo "    Data:        $data_dir (curriculum: $n_train records)"
    echo "    max_seq:     $max_seq"
    echo "    grad_accum:  $grad_accum"
    echo "    iters:       $actual_iters (requested=$iters, available=$n_train)"
    echo "    Output:      $output_dir"
    echo "    Log:         $log_file"

    if $DRY_RUN; then
        return 0
    fi

    # Swap curriculum file in: copy train_curriculum.jsonl -> train.jsonl
    # (mlx_lm_fork expects train.jsonl)
    local original_train="$data_dir/train.jsonl"
    local backup_train="$data_dir/train_original.jsonl"

    if [[ ! -f "$curriculum_train" ]]; then
        echo "  ERROR: $curriculum_train not found. Run prepare_curriculum.py first."
        return 1
    fi

    # Backup original, swap in curriculum
    cp "$original_train" "$backup_train"
    cp "$curriculum_train" "$original_train"
    echo "    Swapped train.jsonl with curriculum-sorted version"

    cd "$KIKI_TUNNER"

    "$KIKI_TUNNER/.venv/bin/python" - "$model_path" "$data_dir" "$output_dir" \
        "$adapter_path" "$grad_accum" "$max_seq" "$actual_iters" <<'PYTHON_SCRIPT' 2>&1 | tee "$log_file"
import mlx.core as mx
mx.set_memory_limit(480 * 1024**3)
mx.set_cache_limit(32 * 1024**3)

import os
import sys
import time
import yaml
import shutil
from pathlib import Path

sys.path.insert(0, "/Users/clems/ailiance-mac-tuner/lib")

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
print(f"Max seq length: {max_seq}")
print(f"Grad accumulation: {grad_accum}")
print(f"Curriculum learning: ENABLED (split + short->long)")

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

    # Restore original train.jsonl
    mv "$backup_train" "$original_train"
    echo "    Restored original train.jsonl"

    echo "  Done: $model_name/$domain (curriculum)"
}

# -----------------------------------------------------------------------------
# Model paths
# -----------------------------------------------------------------------------
APERTUS="$KIKI_TUNNER/models/Apertus-70B-Instruct-2509"
DEVSTRAL="$KIKI_TUNNER/models/Devstral-Small-2-24B-Instruct-2512"

# -----------------------------------------------------------------------------
# Step 3: Train Devstral/cpp — max_seq=8192, grad_accum=16
# -----------------------------------------------------------------------------
STARTED=$(date +%s)

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " STEP 3: Devstral/cpp — max_seq=8192, grad_accum=16"
echo " (Devstral-24B on 512GB — ~150GB peak estimated)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

#                         model      domain    model_path  ga  seq   iters
train_one_curriculum "devstral" "cpp"          "$DEVSTRAL"  16  8192  500

# -----------------------------------------------------------------------------
# Step 4: Train Apertus/emc-dsp-power — max_seq=4096, grad_accum=16
# -----------------------------------------------------------------------------
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " STEP 4: Apertus/emc-dsp-power — max_seq=4096, grad_accum=16"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

train_one_curriculum "apertus" "emc-dsp-power"    "$APERTUS"  16  4096  500

# -----------------------------------------------------------------------------
# Step 5: Train Apertus/security-fenrir — max_seq=4096, grad_accum=16
# -----------------------------------------------------------------------------
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " STEP 5: Apertus/security-fenrir — max_seq=4096, grad_accum=16"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

train_one_curriculum "apertus" "security-fenrir"  "$APERTUS"  16  4096  500

# -----------------------------------------------------------------------------
# Step 6: Train Devstral/rust-embedded — max_seq=4096, grad_accum=8
# (New domain, no backup needed — uses standard train_one from batch 1 pattern)
# -----------------------------------------------------------------------------
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " STEP 6: Devstral/rust-embedded — max_seq=4096, grad_accum=8"
echo " (New domain, 1501 examples, curriculum pre-sorted)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# rust-embedded data is already sorted curriculum-style from the scrape script
# Use train_one_curriculum with max_seq=4096 (embedded Rust files can be long)
train_one_curriculum "devstral" "rust-embedded" "$DEVSTRAL" 8 4096 500

# --- Summary ---
ENDED=$(date +%s)
ELAPSED=$(( (ENDED - STARTED) / 60 ))

echo ""
echo "============================================================"
echo " BATCH 3 (CURRICULUM RETRAIN) COMPLETE"
echo " Total wall time: ${ELAPSED} min"
echo " Logs: $LOG_DIR"
echo " Adapters: $ADAPTERS"
echo " Old adapters backed up: $BACKUP_DIR"
echo "============================================================"
