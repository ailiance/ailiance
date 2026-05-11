#!/usr/bin/env bash
# Smoke train (C10): 50 D2 samples -> strip -> 100-iter qwen36 LoRA.
# DEFERRED execution: this script is committed but not auto-run.
# Owner triggers manually after F1 iact validators free the Studio GPU.
set -euo pipefail
cd "${HOME}/ailiance"

uv run python -m scripts.kicad_sch.synth_d2 \
    --n-samples 50 --compilers skidl

uv run python -m scripts.kicad_sch.strip_lib_symbols \
    --input  "${HOME}/ailiance-data/kicad-sch-synth" \
    --output "${HOME}/ailiance-data/kicad-sch-synth-stripped"

cp "${HOME}/ailiance-mac-tuner/configs/ailiance-v3-qwen36-kicad-sch-D2.yaml" \
   "${HOME}/ailiance-mac-tuner/configs/ailiance-v3-qwen36-kicad-sch-D2-smoke.yaml"
sed -i.bak \
    -e 's/^iters: .*/iters: 100/' \
    -e 's/^save_every: .*/save_every: 100/' \
    "${HOME}/ailiance-mac-tuner/configs/ailiance-v3-qwen36-kicad-sch-D2-smoke.yaml"

echo "[smoke] config patched; pass --actually-run to disarm dry-run."
uv run python -m scripts.kicad_sch.train_lora \
    --config "${HOME}/ailiance-mac-tuner/configs/ailiance-v3-qwen36-kicad-sch-D2-smoke.yaml"
