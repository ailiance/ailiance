#!/usr/bin/env bash
# ==============================================================================
# ailiance LoRA Batch 5 — New domains on Devstral BF16
#
# Trains 9 newly created domains. All on Devstral BF16 except spice-sim (Apertus).
# ==============================================================================

set -euo pipefail

KIKI_TUNNER="$HOME/ailiance-mac-tuner"
AILIANCE="$HOME/ailiance"
HF_DATA="$AILIANCE/data/hf-traced"
ADAPTERS="$AILIANCE/output/adapters"
OUTPUT_ROOT="$KIKI_TUNNER/output/ailiance-hf"
LOG_DIR="$AILIANCE/output/training-logs"

DEVSTRAL_BF16="$KIKI_TUNNER/models/Devstral-Small-2-24B-BF16"
APERTUS="$KIKI_TUNNER/models/Apertus-70B-Instruct-2509"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

mkdir -p "$LOG_DIR"

echo "============================================================"
echo " ailiance LoRA Batch 5 — New Domains"
echo " Date: $(date '+%Y-%m-%d %H:%M')"
if $DRY_RUN; then echo " Mode: DRY RUN"; fi
echo "============================================================"

STARTED=$(date +%s)

# Training function
train_one() {
    local model_name="$1"
    local domain="$2"
    local model_path="$3"
    local grad_accum="$4"
    local max_seq="$5"
    local iters="$6"

    local adapter_path="$ADAPTERS/$model_name/$domain"
    local output_dir="$OUTPUT_ROOT/${model_name}-${domain}"
    local data_dir="$HF_DATA/$domain"
    local log_file="$LOG_DIR/${model_name}-${domain}.log"

    if [[ -f "$adapter_path/adapters.safetensors" ]]; then
        echo "  SKIP $model_name/$domain — adapter already exists"
        return 0
    fi

    if [[ ! -f "$data_dir/train.jsonl" ]]; then
        echo "  SKIP $model_name/$domain — no training data"
        return 0
    fi

    local n_train
    n_train=$(wc -l < "$data_dir/train.jsonl" | tr -d ' ')

    echo ""
    echo "  ▶ Training $model_name/$domain ($n_train examples, $iters iters, seq=$max_seq)"

    if $DRY_RUN; then return 0; fi

    cd "$KIKI_TUNNER"

    "$KIKI_TUNNER/.venv/bin/python" - "$model_path" "$data_dir" "$output_dir" \
        "$adapter_path" "$grad_accum" "$max_seq" "$iters" <<'PYTHON_SCRIPT' 2>&1 | tee "$log_file"
import mlx.core as mx
mx.set_memory_limit(480 * 1024**3)
mx.set_cache_limit(32 * 1024**3)
import os, sys, time, yaml, shutil
from pathlib import Path
sys.path.insert(0, "/Users/clems/ailiance-mac-tuner/lib")
model_path, data_dir, output_dir = sys.argv[1], sys.argv[2], sys.argv[3]
adapter_dest, grad_accum, max_seq, iters = sys.argv[4], int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
os.makedirs(output_dir, exist_ok=True)
train_file = Path(data_dir) / "train.jsonl"
n_train = sum(1 for _ in open(train_file))
actual_iters = min(iters, n_train)
config = {
    "model": model_path, "fine_tune_type": "lora",
    "lora_parameters": {"rank": 16, "alpha": 32, "dropout": 0.05, "scale": 2.0},
    "num_layers": -1, "learning_rate": 1e-5, "batch_size": 1,
    "grad_accumulation_steps": grad_accum, "iters": actual_iters,
    "max_seq_length": max_seq, "grad_checkpoint": True,
    "save_every": 200, "steps_per_report": 10, "steps_per_eval": 200,
    "val_batches": 5, "train": True, "seed": 42,
}
config_path = Path(output_dir) / "train_config.yaml"
with open(config_path, "w") as f: yaml.dump(config, f)
print(f"Model: {Path(model_path).name}\nData: {data_dir} ({n_train} examples)\nIters: {actual_iters}, max_seq: {max_seq}")
t0 = time.time()
from mlx_lm_fork.lora import main as lora_main
sys.argv = ["lora", "-c", str(config_path), "--data", data_dir, "--adapter-path", output_dir]
lora_main()
print(f"\nDone in {(time.time()-t0)/60:.1f} min")
adapter_src = Path(output_dir) / "adapters.safetensors"
if adapter_src.exists():
    dest = Path(adapter_dest); dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(adapter_src), str(dest / "adapters.safetensors"))
    print(f"Copied adapter ({adapter_src.stat().st_size/1048576:.0f} MB) -> {dest}")
else: print("WARNING: no adapter produced"); exit(1)
PYTHON_SCRIPT

    echo "  ✓ $model_name/$domain complete"
}

# --- Devstral domains ---
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Devstral BF16 — 8 new coding/niche domains"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

train_one "devstral" "web-backend"   "$DEVSTRAL_BF16"  4  2048  500
train_one "devstral" "web-frontend"  "$DEVSTRAL_BF16"  4  2048  500
train_one "devstral" "yaml-json"     "$DEVSTRAL_BF16"  4  2048  500
train_one "devstral" "llm-orch"      "$DEVSTRAL_BF16"  4  2048  500
train_one "devstral" "iot"           "$DEVSTRAL_BF16"  4  2048  500
train_one "devstral" "lua-upy"       "$DEVSTRAL_BF16"  4  2048  500
train_one "devstral" "platformio"    "$DEVSTRAL_BF16"  4  2048  500
train_one "devstral" "music-audio"   "$DEVSTRAL_BF16"  4  2048  500

# --- Apertus domain ---
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Apertus — spice-sim (circuit simulation)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

train_one "apertus" "spice-sim"  "$APERTUS"  8  2048  500

# --- Summary ---
ENDED=$(date +%s)
ELAPSED=$(( (ENDED - STARTED) / 60 ))

echo ""
echo "============================================================"
echo " BATCH 5 COMPLETE"
echo " Total wall time: ${ELAPSED} min"
echo " Adapters: $ADAPTERS"
echo "============================================================"
