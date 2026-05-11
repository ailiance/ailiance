#!/usr/bin/env bash
# Run all 6 M2 LoRA training jobs sequentially on Studio.
# DEFERRED execution: requires --actually-run to disarm the safety guard.
# Owner kicks off manually after C10 smoke passes and F1 frees the GPU.
set -euo pipefail

ACTUALLY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --actually-run) ACTUALLY_RUN=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

cd "${HOME}/ailiance"
mkdir -p "${HOME}/ailiance-mac-tuner/logs"

for model in qwen36 gemma4; do
  for split in D1 D2 D3; do
    cfg="${HOME}/ailiance-mac-tuner/configs/ailiance-v3-${model}-kicad-sch-${split}.yaml"
    log="${HOME}/ailiance-mac-tuner/logs/ailiance-v3-${model}-kicad-sch-${split}-$(date +%Y%m%d-%H%M).log"
    echo "[$(date -Iseconds)] plan ${model} ${split} cfg=${cfg}"
    if [[ "${ACTUALLY_RUN}" -eq 1 ]]; then
      echo "[$(date -Iseconds)] start ${model} ${split}"
      uv run python -m scripts.kicad_sch.train_lora \
          --config "${cfg}" --actually-run 2>&1 | tee "${log}"
    else
      echo "[$(date -Iseconds)] dry-run (pass --actually-run to launch)"
      uv run python -m scripts.kicad_sch.train_lora --config "${cfg}"
    fi
  done
done
