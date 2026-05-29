#!/usr/bin/env bash
# ==============================================================================
# ailiance LoRA Batch 10 — Mistral Medium 3.5 128B: 8 reasoning + 3 multilingual
#
# Model: mistralai/Mistral-Medium-3.5-128B-Instruct (Modified MIT license)
# Base model provenance: mistralai official FP8 → community BF16 dequantization
# Verified base loss: 2.56
#
# EU AI Act compliance:
#   - License: Modified MIT (Mistral Community License, Art. 53 compliant)
#   - Training data: all domains from hf-traced/ with _provenance metadata
#   - Legal basis: legitimate interest (Art. 6(1)(f) GDPR), publicly available
#     data and documentation with provenance tracking per EU AI Act Art. 53(1)(d)
#   - Reasoning domains from Apertus (EU-origin model)
#   - Multilingual domains from EuroLLM (EU-funded project)
#
# Peak memory: ~280-300 GB (128B BF16 + gradients)
# Estimated wall time: ~8h (11 domains × ~45 min each)
#
# Reminder: run before launching if wired limit not yet set:
#   sudo sysctl -w iogpu.wired_limit_mb=458752
#
# Usage:
#   bash $AILIANCE/scripts/train_batch10_medium35_reasoning.sh
#   bash $AILIANCE/scripts/train_batch10_medium35_reasoning.sh --dry-run
# ==============================================================================

set -euo pipefail

AILIANCE="${AILIANCE_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
KIKI_TUNNER="${KIKI_TUNNER_HOME:-$(dirname "$AILIANCE")/ailiance-mac-tuner}"
HF_DATA="$AILIANCE/data/hf-traced"
ADAPTERS="$AILIANCE/output/adapters-v2/medium35"
OUTPUT_ROOT="$KIKI_TUNNER/output/ailiance-v2"
LOG_DIR="$AILIANCE/output/training-logs"

MODEL="$KIKI_TUNNER/models/Mistral-Medium-3.5-128B-BF16"
MODEL_LICENSE="Modified-MIT (Mistral Community License)"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

mkdir -p "$LOG_DIR" "$ADAPTERS"

echo "============================================================"
echo " ailiance LoRA Batch 10 — Mistral Medium 3.5 128B"
echo "   8 reasoning + 3 multilingual domains"
echo " Date: $(date '+%Y-%m-%d %H:%M')"
echo " Model: $MODEL"
echo " License: $MODEL_LICENSE"
echo " Data:  $HF_DATA"
echo " Adapters (v2): $ADAPTERS"
if $DRY_RUN; then
    echo " Mode: DRY RUN"
fi
echo "============================================================"

# -----------------------------------------------------------------------------
# Training function
# -----------------------------------------------------------------------------
train_one() {
    local domain="$1"
    local grad_accum="$2"
    local max_seq="$3"
    local iters="$4"

    local adapter_path="$ADAPTERS/$domain"
    local output_dir="$OUTPUT_ROOT/medium35-${domain}"
    local data_dir="$HF_DATA/$domain"
    local log_file="$LOG_DIR/batch10-medium35-${domain}.log"

    if [[ ! -d "$data_dir" ]]; then
        echo "  SKIP $domain — data directory not found: $data_dir"
        return 0
    fi

    if [[ -f "$adapter_path/adapters.safetensors" ]]; then
        echo "  SKIP $domain — adapter already exists"
        return 0
    fi

    # Use curriculum or fullseq file if available
    local train_file="$data_dir/train.jsonl"
    local using_alt=""
    if [[ -f "$data_dir/train_fullseq.jsonl" ]]; then
        cp "$train_file" "$data_dir/train_backup.jsonl"
        cp "$data_dir/train_fullseq.jsonl" "$train_file"
        using_alt="fullseq"
    elif [[ -f "$data_dir/train_curriculum.jsonl" ]]; then
        cp "$train_file" "$data_dir/train_backup.jsonl"
        cp "$data_dir/train_curriculum.jsonl" "$train_file"
        using_alt="curriculum"
    fi

    if [[ ! -f "$train_file" ]]; then echo "  SKIP $domain — no train.jsonl"; return 0; fi
    local n_train
    n_train=$(wc -l < "$train_file" | tr -d ' ')

    echo ""
    echo "  ▶ Training medium35/$domain ($n_train examples, $iters iters, seq=$max_seq, grad_accum=$grad_accum)"
    if [[ -n "$using_alt" ]]; then
        echo "    Using $using_alt-sorted data"
    fi

    if $DRY_RUN; then
        if [[ -f "$data_dir/train_backup.jsonl" ]]; then
            mv "$data_dir/train_backup.jsonl" "$train_file"
        fi
        return 0
    fi

    cd "$KIKI_TUNNER"

    "$KIKI_TUNNER/.venv/bin/python" - "$MODEL" "$data_dir" "$output_dir" \
        "$adapter_path" "$grad_accum" "$max_seq" "$iters" "$MODEL_LICENSE" "$domain" <<'PYTHON_SCRIPT' 2>&1 | tee "$log_file"
import mlx.core as mx
mx.set_memory_limit(480 * 1024**3)
mx.set_cache_limit(64 * 1024**3)

import os, sys, time, yaml, shutil, json
from pathlib import Path

sys.path.insert(0, "/Users/clems/ailiance-mac-tuner/lib")

model_path = sys.argv[1]
data_dir = sys.argv[2]
output_dir = sys.argv[3]
adapter_dest = sys.argv[4]
grad_accum = int(sys.argv[5])
max_seq = int(sys.argv[6])
iters = int(sys.argv[7])
model_license = sys.argv[8]
domain_name = sys.argv[9]

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

# EU AI Act provenance tracking
provenance = {
    "pipeline": "ailiance-v2",
    "batch": "batch10",
    "base_model": Path(model_path).name,
    "base_model_license": model_license,
    "domain": domain_name,
    "data_source": str(data_dir),
    "n_train": n_train,
    "actual_iters": actual_iters,
    "max_seq_length": max_seq,
    "grad_accumulation_steps": grad_accum,
    "eu_ai_act_compliance": {
        "license": model_license,
        "data_provenance": True,
        "legal_basis": "Art. 6(1)(f) GDPR — legitimate interest, publicly available data",
        "transparency": "Art. 53(1)(d) — training data summary with provenance"
    },
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
}
provenance_path = Path(output_dir) / "provenance.json"
with open(provenance_path, "w") as f:
    json.dump(provenance, f, indent=2)

print(f"Model: {Path(model_path).name} (license: {model_license})")
print(f"Domain: {domain_name}")
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
    shutil.copy2(str(provenance_path), str(dest / "provenance.json"))
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

    echo "  ✓ medium35/$domain complete"
}

# -----------------------------------------------------------------------------
# Execution plan — 8 reasoning + 3 multilingual domains
# -----------------------------------------------------------------------------
STARTED=$(date +%s)

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " GROUP 1: Reasoning domains from Apertus (grad_accum=16, max_seq=2048)"
echo " 8 domains, est. ~45 min each"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# electronics and stm32 are PDF-supplement only (no train.jsonl), skipped
# train_one "electronics"       16  2048  500
train_one "math-gsm8k"        16  2048  500
train_one "math-reasoning"    16  2048  500
train_one "embedded"          16  2048  500
train_one "emc-dsp-power"     16  2048  500
train_one "security-fenrir"   16  2048  500
train_one "spice-sim"         16  2048  500

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " GROUP 2: Multilingual domains from EuroLLM (grad_accum=16, max_seq=2048)"
echo " 3 domains, est. ~45 min each"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

train_one "chat-fr"           16  2048  500
train_one "multilingual-eu"   16  2048  500
train_one "traduction-tech"   16  2048  500

# --- Summary ---
ENDED=$(date +%s)
ELAPSED=$(( (ENDED - STARTED) / 60 ))

echo ""
echo "============================================================"
echo " BATCH 10 (Mistral Medium 3.5 128B REASONING) COMPLETE"
echo " Total wall time: ${ELAPSED} min"
echo " Model: Mistral-Medium-3.5-128B-BF16 ($MODEL_LICENSE)"
echo " Domains trained: 7 reasoning + 3 multilingual"
echo " Adapters (v2): $ADAPTERS"
echo " Logs: $LOG_DIR"
echo "============================================================"
