#!/usr/bin/env bash
# ==============================================================================
# ailiance LoRA Batch 4 — Devstral BF16 Retrain (all 12 domains)
#
# Re-trains ALL Devstral domains on the properly loaded BF16 model.
# Previous Devstral adapters were trained on broken FP8 weights (loss ~12).
#
# Model: akoumpa/Devstral-Small-2-24B-Instruct-2512-BF16 (Apache 2.0)
# Provenance: mistralai official FP8 → community BF16 dequantization
#
# Usage:
#   bash ~/eu-kiki/scripts/train_batch4_bf16_retrain.sh
#   bash ~/eu-kiki/scripts/train_batch4_bf16_retrain.sh --dry-run
# ==============================================================================

set -euo pipefail

KIKI_TUNNER="$HOME/KIKI-Mac_tunner"
EU_KIKI="$HOME/eu-kiki"
HF_DATA="$EU_KIKI/data/hf-traced"
ADAPTERS="$EU_KIKI/output/adapters"
OUTPUT_ROOT="$KIKI_TUNNER/output/ailiance-hf"
LOG_DIR="$EU_KIKI/output/training-logs"
BACKUP_DIR="$EU_KIKI/output/adapters-backup-pre-bf16"

# NEW: BF16 model path
DEVSTRAL_BF16="$KIKI_TUNNER/models/Devstral-Small-2-24B-BF16"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

mkdir -p "$LOG_DIR"

echo "============================================================"
echo " ailiance LoRA Batch 4 — Devstral BF16 Retrain"
echo " Date: $(date '+%Y-%m-%d %H:%M')"
echo " Model: $DEVSTRAL_BF16"
echo " Data:  $HF_DATA"
if $DRY_RUN; then
    echo " Mode: DRY RUN"
fi
echo "============================================================"

# -----------------------------------------------------------------------------
# Step 1: Backup all existing Devstral adapters
# -----------------------------------------------------------------------------
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " STEP 1: Backup existing Devstral adapters (broken FP8)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for domain in python rust typescript sql docker-devops llm-ops cpp shell html-css ml-training rust-embedded; do
    src="$ADAPTERS/devstral/$domain"
    if [[ -f "$src/adapters.safetensors" ]]; then
        dst="$BACKUP_DIR/devstral/$domain"
        if ! $DRY_RUN; then
            mkdir -p "$dst"
            mv "$src/adapters.safetensors" "$dst/adapters.safetensors"
            echo "  Backed up devstral/$domain"
        else
            echo "  Would backup devstral/$domain"
        fi
    fi
done

# -----------------------------------------------------------------------------
# Training function
# -----------------------------------------------------------------------------
train_one() {
    local domain="$1"
    local grad_accum="$2"
    local max_seq="$3"
    local iters="$4"

    local adapter_path="$ADAPTERS/devstral/$domain"
    local output_dir="$OUTPUT_ROOT/devstral-${domain}-bf16"
    local data_dir="$HF_DATA/$domain"
    local log_file="$LOG_DIR/devstral-${domain}-bf16.log"

    if [[ -f "$adapter_path/adapters.safetensors" ]]; then
        echo "  SKIP devstral/$domain — adapter already exists"
        return 0
    fi

    # Use curriculum file if available, otherwise regular train.jsonl
    local train_file="$data_dir/train.jsonl"
    if [[ -f "$data_dir/train_curriculum.jsonl" ]]; then
        # Swap in curriculum
        cp "$train_file" "$data_dir/train_backup.jsonl"
        cp "$data_dir/train_curriculum.jsonl" "$train_file"
        echo "  Using curriculum-sorted data for $domain"
    fi

    local n_train
    n_train=$(wc -l < "$train_file" | tr -d ' ')

    echo ""
    echo "  ▶ Training devstral/$domain ($n_train examples, $iters iters, seq=$max_seq)"

    if $DRY_RUN; then
        # Restore if we swapped
        if [[ -f "$data_dir/train_backup.jsonl" ]]; then
            mv "$data_dir/train_backup.jsonl" "$train_file"
        fi
        return 0
    fi

    cd "$KIKI_TUNNER"

    "$KIKI_TUNNER/.venv/bin/python" - "$DEVSTRAL_BF16" "$data_dir" "$output_dir" \
        "$adapter_path" "$grad_accum" "$max_seq" "$iters" <<'PYTHON_SCRIPT' 2>&1 | tee "$log_file"
import mlx.core as mx
mx.set_memory_limit(480 * 1024**3)
mx.set_cache_limit(32 * 1024**3)

import os, sys, time, yaml, shutil
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

print(f"Model: {Path(model_path).name}")
print(f"Data: {data_dir} ({n_train} examples)")
print(f"Iters: {actual_iters}, max_seq: {max_seq}, grad_accum: {grad_accum}")

t0 = time.time()

from mlx_lm_fork.lora import main as lora_main
sys.argv = ["lora", "-c", str(config_path), "--data", data_dir, "--adapter-path", output_dir]
lora_main()

elapsed = time.time() - t0
print(f"\nDone in {elapsed/60:.1f} min")

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

    # Restore original train.jsonl if we swapped
    if [[ -f "$data_dir/train_backup.jsonl" ]]; then
        mv "$data_dir/train_backup.jsonl" "$data_dir/train.jsonl"
    fi

    echo "  ✓ devstral/$domain complete (BF16)"
}

# -----------------------------------------------------------------------------
# Execution plan — 12 Devstral domains
# -----------------------------------------------------------------------------
STARTED=$(date +%s)

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " GROUP 1: Standard coding domains (grad_accum=4, max_seq=2048)"
echo " Est. ~25 min/domain × 7 domains = ~175 min"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

train_one "python"        4  2048  500
train_one "rust"          4  2048  500
train_one "typescript"    4  2048  500
train_one "sql"           4  2048  500
train_one "shell"         4  2048  500
train_one "html-css"      4  2048  500
train_one "docker-devops" 4  2048  500

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " GROUP 2: Niche domains (grad_accum=4, max_seq=2048)"
echo " Est. ~25 min/domain × 2 domains = ~50 min"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

train_one "llm-ops"       4  2048  500
train_one "ml-training"   4  2048  500

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " GROUP 3: Embedded domains (grad_accum=8, max_seq=4096)"
echo " Longer sequences for firmware files"
echo " Est. ~35 min/domain × 2 domains = ~70 min"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

train_one "cpp"             8  4096  500
train_one "rust-embedded"   8  4096  500

# --- Summary ---
ENDED=$(date +%s)
ELAPSED=$(( (ENDED - STARTED) / 60 ))

echo ""
echo "============================================================"
echo " BATCH 4 (BF16 RETRAIN) COMPLETE"
echo " Total wall time: ${ELAPSED} min"
echo " Model: Devstral-Small-2-24B-BF16 (properly dequantized)"
echo " Logs: $LOG_DIR"
echo " Adapters: $ADAPTERS"
echo " Old FP8 adapters: $BACKUP_DIR"
echo "============================================================"
