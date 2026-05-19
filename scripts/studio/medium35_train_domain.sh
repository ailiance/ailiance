#!/usr/bin/env bash
# medium35 single-domain 3-phase LoRA trainer (Mistral-Medium-128B).
set -uo pipefail

REPO="/Users/clems/KIKI-Mac_tunner"
VENV="$REPO/.venv/bin/activate"
MODEL="Mistral-Medium-3.5-128B-BF16"
DATA_DIR="$REPO/eu-kiki/data/hf-traced"
LOG_DIR="$REPO/logs"

# Verified hyperparameters (3 healthy medium35 runs, 2026-05-19).
RANK=16; SCALE=32.0; DROPOUT=0.01; LAYERS=-1
# phase -> "seq lr iters batch grad_accum"
declare -A PHASE=(
  [1]="512 8e-6 500 1 16"
  [2]="1280 5e-6 800 2 8"
  [3]="2048 3e-6 500 2 8"
)

run_domain() {
  local domain="$1"
  local out_dir="$REPO/output/eu-kiki-v2-curriculum/medium35-$domain"
  local data="$DATA_DIR/$domain"
  mkdir -p "$out_dir"
  if [[ ! -f "$data/train.jsonl" ]]; then
    echo "### DOMAIN NO_DATA $domain"; return 0
  fi
  # shellcheck disable=SC1090
  source "$VENV"
  for phase in 1 2 3; do
    [[ -f "$out_dir/phase${phase}_done" ]] && { echo "### PHASE $phase/3 SKIP"; continue; }
    read -r seq lr iters batch ga <<<"${PHASE[$phase]}"
    echo "### PHASE $phase/3 domain=$domain seq=$seq"
    local cfg="$out_dir/config-phase${phase}.yaml"
    cat >"$cfg" <<YAML
model: $MODEL
train: true
fine_tune_type: lora
optimizer: adam
num_layers: $LAYERS
seed: 42
grad_checkpoint: true
data: $data
adapter_path: $out_dir
iters: $iters
learning_rate: $lr
max_seq_length: $seq
batch_size: $batch
grad_accumulation_steps: $ga
save_every: 200
steps_per_report: 50
lora_parameters:
  rank: $RANK
  scale: $SCALE
  dropout: $DROPOUT
YAML
    if python -m mlx_lm.lora -c "$cfg"; then
      touch "$out_dir/phase${phase}_done"
    else
      echo "### DOMAIN FAILED_OOM $domain phase=$phase"; return 1
    fi
  done
  local vl
  vl="$(grep -oE 'Val loss [0-9.]+' "$LOG_DIR/medium35-$domain.log" | tail -1 \
        | awk '{print $3}')"
  echo "### DOMAIN COMPLETE $domain final_val_loss=${vl:-0.0}"
}

case "${1:-}" in
  spawn)
    domain="${2:?domain required}"
    log="$LOG_DIR/medium35-$domain.log"
    nohup bash "$0" run "$domain" >"$log" 2>&1 </dev/null &
    echo "$!"
    ;;
  run)  run_domain "${2:?domain required}" ;;
  *) echo "usage: $0 {spawn <domain>|run <domain>}" >&2; exit 2 ;;
esac
