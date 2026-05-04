#!/usr/bin/env bash
# ==============================================================================
# eu-kiki Batch 8 — KiCad PCB full-sequence retrain (max_seq=16384)
#
# Retrains kicad-pcb with max_seq=16384 so NO footprint is truncated.
# Uses train_fullseq.jsonl (S-expression-aware splitting, curriculum sorted).
#
# Prerequisites: sudo sysctl -w iogpu.wired_limit_mb=458752
# ==============================================================================

set -euo pipefail

KIKI_TUNNER="$HOME/KIKI-Mac_tunner"
EU_KIKI="$HOME/eu-kiki"
HF_DATA="$EU_KIKI/data/hf-traced"
ADAPTERS="$EU_KIKI/output/adapters"
OUTPUT_ROOT="$KIKI_TUNNER/output/eu-kiki-hf"
LOG_DIR="$EU_KIKI/output/training-logs"
BACKUP_DIR="$EU_KIKI/output/adapters-backup-pre-kicad-fullseq"
DEVSTRAL_BF16="$KIKI_TUNNER/models/Devstral-Small-2-24B-BF16"

DOMAIN="kicad-pcb"
MAX_SEQ=16384
GRAD_ACCUM=16
ITERS=1000

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=true; fi

mkdir -p "$LOG_DIR"

echo "============================================================"
echo " eu-kiki Batch 8 — KiCad PCB Full-Sequence Retrain"
echo " max_seq=$MAX_SEQ, grad_accum=$GRAD_ACCUM"
echo " Date: $(date '+%Y-%m-%d %H:%M')"
if $DRY_RUN; then echo " Mode: DRY RUN"; fi
echo "============================================================"

STARTED=$(date +%s)

DATA_DIR="$HF_DATA/$DOMAIN"
ADAPTER_PATH="$ADAPTERS/devstral/$DOMAIN"
OUTPUT_DIR="$OUTPUT_ROOT/devstral-${DOMAIN}-fullseq"
LOG_FILE="$LOG_DIR/devstral-${DOMAIN}-fullseq.log"
FULLSEQ="$DATA_DIR/train_fullseq.jsonl"
ORIGINAL="$DATA_DIR/train.jsonl"
BACKUP_TRAIN="$DATA_DIR/train_pre_fullseq.jsonl"

# Verify fullseq file exists
if [[ ! -f "$FULLSEQ" ]]; then
    echo "ERROR: $FULLSEQ not found. Run prepare_kicad_pcb_fullseq.py first."
    exit 1
fi

N_TRAIN=$(wc -l < "$FULLSEQ" | tr -d ' ')
ACTUAL_ITERS=$((ITERS < N_TRAIN ? ITERS : N_TRAIN))

echo ""
echo "  Dataset: $FULLSEQ ($N_TRAIN records)"
echo "  Effective iters: $ACTUAL_ITERS"
echo "  Model: $(basename $DEVSTRAL_BF16)"

# Backup existing adapter
SRC_ADAPTER="$ADAPTER_PATH/adapters.safetensors"
if [[ -f "$SRC_ADAPTER" ]]; then
    DST_BACKUP="$BACKUP_DIR/devstral/$DOMAIN"
    if ! $DRY_RUN; then
        mkdir -p "$DST_BACKUP"
        cp "$SRC_ADAPTER" "$DST_BACKUP/"
        echo "  Backed up existing adapter -> $DST_BACKUP"
    else
        echo "  Would backup existing adapter -> $DST_BACKUP"
    fi
fi

if $DRY_RUN; then
    echo ""
    echo "  DRY RUN — would train devstral/$DOMAIN"
    echo "    max_seq=$MAX_SEQ, grad_accum=$GRAD_ACCUM, iters=$ACTUAL_ITERS"
    ENDED=$(date +%s)
    echo ""
    echo "============================================================"
    echo " DRY RUN COMPLETE"
    echo "============================================================"
    exit 0
fi

# Swap fullseq into train.jsonl
cp "$ORIGINAL" "$BACKUP_TRAIN"
cp "$FULLSEQ" "$ORIGINAL"

echo ""
echo "  ▶ Training devstral/$DOMAIN (fullseq, $N_TRAIN records, $ACTUAL_ITERS iters, seq=$MAX_SEQ)"

cd "$KIKI_TUNNER"
"$KIKI_TUNNER/.venv/bin/python" - "$DEVSTRAL_BF16" "$DATA_DIR" "$OUTPUT_DIR" \
    "$ADAPTER_PATH" "$GRAD_ACCUM" "$MAX_SEQ" "$ACTUAL_ITERS" <<'PYTHON_SCRIPT' 2>&1 | tee "$LOG_FILE"
import mlx.core as mx
mx.set_memory_limit(480 * 1024**3)
mx.set_cache_limit(64 * 1024**3)
import os, sys, time, yaml, shutil
from pathlib import Path
sys.path.insert(0, "/Users/clems/KIKI-Mac_tunner/lib")
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
print(f"Full-sequence training: devstral/{Path(data_dir).name}")
print(f"Model: {Path(model_path).name}\nData: {data_dir} ({n_train} records)")
print(f"Iters: {actual_iters}, max_seq: {max_seq}, grad_accum: {grad_accum}")
print(f"Memory limit: 480GB, Cache limit: 64GB")
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

# Restore original train.jsonl
mv "$BACKUP_TRAIN" "$ORIGINAL"
echo "  ✓ Restored original train.jsonl"

ENDED=$(date +%s)
ELAPSED=$(( (ENDED - STARTED) / 60 ))
echo ""
echo "============================================================"
echo " BATCH 8 COMPLETE — ${ELAPSED} min"
echo " Adapter: $ADAPTER_PATH"
echo " Log: $LOG_FILE"
echo "============================================================"
