#!/usr/bin/env bash
# ==============================================================================
# ailiance Batch 7 — KiCad curriculum retrain + FreeCAD
#
# Retrains kicad-dsl and kicad-pcb with curriculum (short→long) and higher max_seq
# to avoid truncation. FreeCAD kept from batch 6 (only 62 records, no truncation).
#
# Prerequisites: sudo sysctl -w iogpu.wired_limit_mb=458752
# ==============================================================================

set -euo pipefail

KIKI_TUNNER="$HOME/ailiance-mac-tuner"
AILIANCE="$HOME/ailiance"
HF_DATA="$AILIANCE/data/hf-traced"
ADAPTERS="$AILIANCE/output/adapters"
OUTPUT_ROOT="$KIKI_TUNNER/output/ailiance-hf"
LOG_DIR="$AILIANCE/output/training-logs"
BACKUP_DIR="$AILIANCE/output/adapters-backup-pre-kicad-curriculum"
DEVSTRAL_BF16="$KIKI_TUNNER/models/Devstral-Small-2-24B-BF16"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=true; fi

mkdir -p "$LOG_DIR"

echo "============================================================"
echo " ailiance Batch 7 — KiCad Curriculum Retrain"
echo " Date: $(date '+%Y-%m-%d %H:%M')"
if $DRY_RUN; then echo " Mode: DRY RUN"; fi
echo "============================================================"

STARTED=$(date +%s)

# Backup existing kicad adapters from batch 6
for domain in kicad-dsl kicad-pcb; do
    src="$ADAPTERS/devstral/$domain"
    if [[ -f "$src/adapters.safetensors" ]]; then
        dst="$BACKUP_DIR/devstral/$domain"
        if ! $DRY_RUN; then
            mkdir -p "$dst"
            mv "$src/adapters.safetensors" "$dst/"
            echo "  Backed up devstral/$domain"
        fi
    fi
done

train_curriculum() {
    local domain="$1" max_seq="$2" grad_accum="$3" iters="$4"
    local adapter_path="$ADAPTERS/devstral/$domain"
    local output_dir="$OUTPUT_ROOT/devstral-${domain}-curriculum"
    local data_dir="$HF_DATA/$domain"
    local log_file="$LOG_DIR/devstral-${domain}-curriculum.log"
    local curriculum="$data_dir/train_curriculum.jsonl"
    local original="$data_dir/train.jsonl"
    local backup="$data_dir/train_original.jsonl"

    if [[ -f "$adapter_path/adapters.safetensors" ]]; then
        echo "  SKIP $domain — adapter exists"; return 0
    fi
    if [[ ! -f "$curriculum" ]]; then
        echo "  SKIP $domain — no curriculum file"; return 0
    fi

    local n_train=$(wc -l < "$curriculum" | tr -d ' ')
    local actual_iters=$((iters < n_train ? iters : n_train))

    echo ""
    echo "  ▶ Training devstral/$domain (curriculum, $n_train records, $actual_iters iters, seq=$max_seq)"

    if $DRY_RUN; then return 0; fi

    # Swap curriculum into train.jsonl
    cp "$original" "$backup"
    cp "$curriculum" "$original"

    cd "$KIKI_TUNNER"
    "$KIKI_TUNNER/.venv/bin/python" - "$DEVSTRAL_BF16" "$data_dir" "$output_dir" \
        "$adapter_path" "$grad_accum" "$max_seq" "$actual_iters" <<'PYTHON_SCRIPT' 2>&1 | tee "$log_file"
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
config = {"model": model_path, "fine_tune_type": "lora",
    "lora_parameters": {"rank": 16, "alpha": 32, "dropout": 0.05, "scale": 2.0},
    "num_layers": -1, "learning_rate": 1e-5, "batch_size": 1,
    "grad_accumulation_steps": grad_accum, "iters": actual_iters,
    "max_seq_length": max_seq, "grad_checkpoint": True,
    "save_every": 200, "steps_per_report": 10, "steps_per_eval": 200,
    "val_batches": 5, "train": True, "seed": 42}
config_path = Path(output_dir) / "train_config.yaml"
with open(config_path, "w") as f: yaml.dump(config, f)
print(f"Curriculum training: devstral/{Path(data_dir).name}")
print(f"Model: {Path(model_path).name}\nData: {data_dir} ({n_train} records)")
print(f"Iters: {actual_iters}, max_seq: {max_seq}, grad_accum: {grad_accum}")
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

    # Restore original
    mv "$backup" "$original"
    echo "  ✓ devstral/$domain complete (curriculum)"
}

# kicad-dsl: 7694 records, max token 6620 — max_seq=8192 covers all
train_curriculum "kicad-dsl" 8192 8 1000

# kicad-pcb: 11681 records (after split), max token 8192 — max_seq=8192
train_curriculum "kicad-pcb" 8192 8 1000

ENDED=$(date +%s)
ELAPSED=$(( (ENDED - STARTED) / 60 ))
echo ""
echo "============================================================"
echo " BATCH 7 COMPLETE — ${ELAPSED} min"
echo "============================================================"
