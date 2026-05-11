#!/usr/bin/env bash
# F1 3rd-axis bench dispatcher.
#
# Path A (stub mode) is the active strategy on 2026-05-11:
#   iact-bench submodule not vendored; IactBenchValidator falls back to
#   StubValidator and cells are tagged stub_mode=true. Real Docker
#   validators will activate when ailiance issue #23 lands the submodule
#   at vendored/iact-bench. The Python orchestrator transparently picks
#   the operational validator when available — no script change needed.
#
# Compute footprint per heavy run (when chain is free):
#   versions=2 × runnable_cells ≈ 65 base + 13 tuned = 78 cells
#   × N_SAMPLES=10 prompts × ~5s/prompt (MLX 4-bit) ≈ 65 min wall
#   + model-load overhead (one load per cell, ~30s) → ~104 min wall.
#   Stub-only run (no model load): ~30s total.
#
# Outputs:
#   output/iact_validator_runs/{base,tuned}_2026-05-11.json
#   output/iact_validator_runs/{base,tuned}_2026-05-11_skipped.json
#   output/iact_validator_runs/run_<ts>.log
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p output/iact_validator_runs
ts="$(date +%Y%m%d_%H%M%S)"
log="output/iact_validator_runs/run_${ts}.log"
echo "[F1] start $(date -Iseconds)" | tee "$log"
.venv/bin/python scripts/run_iact_validators.py "$@" 2>&1 | tee -a "$log"
echo "[F1] done $(date -Iseconds)" | tee -a "$log"
