#!/usr/bin/env bash
# Track-D hybrid LLM->DSL->compiler pipelines launcher.
# Usage:
#   scripts/run_track_d.sh smoke   # 1 cell (qwen36 + skidl + 1 prompt)
#   scripts/run_track_d.sh full    # 5 models * 4 compilers * 20 prompts * 5 seeds
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_track_d.sh <mode>
  smoke   1 cell: qwen36 base x skidl compiler x 1 prompt x 1 seed.
  full    Full grid: 5 base models x 4 compilers x 20 prompts x 5 seeds.

Output goes to $AILIANCE/output/track-d/<timestamp>/.
USAGE
}

if [[ $# -ne 1 ]] || [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  [[ $# -eq 0 ]] && exit 1 || exit 0
fi

MODE="$1"
case "$MODE" in
  smoke|full) ;;
  *) echo "error: unknown mode '$MODE'" >&2; usage >&2; exit 2 ;;
esac

TS=$(date +%Y-%m-%dT%H-%M-%S)
ROOT="${AILIANCE_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
OUT="${ROOT}/output/track-d/${TS}"
mkdir -p "${OUT}/artefacts"

cd "${ROOT}"
uv run python -m scripts.kicad_sch.hybrid_pipeline \
  --mode "${MODE}" \
  --out-dir "${OUT}/artefacts" \
  --audit-path "${OUT}/audit.ndjson" \
  --summary-path "${OUT}/summary.json"

echo "Track-D ${MODE} run complete: ${OUT}"
