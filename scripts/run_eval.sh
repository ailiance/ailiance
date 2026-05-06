#!/usr/bin/env bash
# ==============================================================================
# AILIANCE Evaluation Runner — v1 vs v2 adapter comparison
#
# Usage:
#   bash run_eval.sh --v1-only       # eval v1 adapters only
#   bash run_eval.sh --v2-only       # eval v2 adapters only
#   bash run_eval.sh --compare       # eval both and compare (default)
#   bash run_eval.sh --quick         # quick mode (5 records/domain, no generation)
#   bash run_eval.sh --domains python rust typescript  # eval specific domains
#
# Combines flags:
#   bash run_eval.sh --compare --quick
#   bash run_eval.sh --v2-only --domains python cpp
#   bash run_eval.sh --compare --skip-generation
#
# Output:
#   ~/ailiance/output/eval/eval_report_v1_vs_v2.md    — comparison report
#   ~/ailiance/output/eval/raw/                        — raw JSON results
#
# EU AI Act Art. 53(1)(d): eval methodology documented for transparency
# ==============================================================================

set -euo pipefail

AILIANCE="$HOME/ailiance"
KIKI_TUNNER="$HOME/KIKI-Mac_tunner"
EVAL_SCRIPT="$AILIANCE/scripts/eval_framework.py"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

echo "============================================================"
echo " AILIANCE Evaluation Framework"
echo " Date: $(date '+%Y-%m-%d %H:%M')"
echo "============================================================"

# Check eval script exists
if [[ ! -f "$EVAL_SCRIPT" ]]; then
    echo "ERROR: eval_framework.py not found at $EVAL_SCRIPT"
    exit 1
fi

# Check mlx_lm_fork is available
if [[ ! -d "$KIKI_TUNNER/lib/mlx_lm_fork" ]]; then
    echo "ERROR: mlx_lm_fork not found at $KIKI_TUNNER/lib/mlx_lm_fork"
    exit 1
fi

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

MODE="compare"
QUICK=false
SKIP_GEN=false
SKIP_SPEED=false
DOMAINS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --v1-only)
            MODE="v1-only"
            shift
            ;;
        --v2-only)
            MODE="v2-only"
            shift
            ;;
        --compare)
            MODE="compare"
            shift
            ;;
        --quick)
            QUICK=true
            shift
            ;;
        --skip-generation)
            SKIP_GEN=true
            shift
            ;;
        --skip-speed)
            SKIP_SPEED=true
            shift
            ;;
        --domains)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                DOMAINS+=("$1")
                shift
            done
            ;;
        -h|--help)
            echo "Usage: bash run_eval.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --v1-only          Evaluate v1 adapters only"
            echo "  --v2-only          Evaluate v2 adapters only"
            echo "  --compare          Evaluate both and compare (default)"
            echo "  --quick            Quick mode (5 records/domain, no generation/speed)"
            echo "  --skip-generation  Skip generation quality evaluation"
            echo "  --skip-speed       Skip inference speed benchmark"
            echo "  --domains D1 D2    Evaluate only specified domains"
            echo "  -h, --help         Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage."
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Report available adapters
# ---------------------------------------------------------------------------

echo ""
echo "Mode: $MODE"
echo ""

# v1 adapters
V1_DIR="$AILIANCE/output/adapters"
echo "v1 adapters ($V1_DIR):"
for model_dir in "$V1_DIR"/*/; do
    model_name=$(basename "$model_dir")
    n_adapters=$(find "$model_dir" -name "adapters.safetensors" 2>/dev/null | wc -l | tr -d ' ')
    echo "  $model_name: $n_adapters domains"
done

# v2 adapters
V2_DIR="$AILIANCE/output/adapters-v2"
echo ""
echo "v2 adapters ($V2_DIR):"
for model_dir in "$V2_DIR"/*/; do
    model_name=$(basename "$model_dir")
    n_adapters=$(find "$model_dir" -name "adapters.safetensors" 2>/dev/null | wc -l | tr -d ' ')
    echo "  $model_name: $n_adapters domains"
done

# Check if there's anything to eval
if [[ "$MODE" == "v1-only" || "$MODE" == "compare" ]]; then
    v1_count=$(find "$V1_DIR" -name "adapters.safetensors" 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$v1_count" -eq 0 ]]; then
        echo ""
        echo "WARNING: No v1 adapters found. v1 evaluation will be empty."
    fi
fi

if [[ "$MODE" == "v2-only" || "$MODE" == "compare" ]]; then
    v2_count=$(find "$V2_DIR" -name "adapters.safetensors" 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$v2_count" -eq 0 ]]; then
        echo ""
        echo "WARNING: No v2 adapters found. v2 evaluation will be empty."
        if [[ "$MODE" == "v2-only" ]]; then
            echo "Training may still be running. Check with:"
            echo "  ps aux | grep train_batch"
            echo "  tail -f $AILIANCE/output/training-logs/batch9-*.log"
            echo ""
            read -rp "Continue anyway? [y/N] " confirm
            if [[ ! "$confirm" =~ ^[yY] ]]; then
                echo "Aborted."
                exit 0
            fi
        fi
    fi
fi

# ---------------------------------------------------------------------------
# Set memory limits for MLX
# ---------------------------------------------------------------------------

echo ""
echo "Setting memory limits..."

# Check wired limit
current_wired=$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo "unknown")
echo "  Current iogpu.wired_limit_mb: $current_wired"
if [[ "$current_wired" != "unknown" && "$current_wired" -lt 400000 ]]; then
    echo "  WARNING: wired_limit_mb is low ($current_wired). Consider:"
    echo "    sudo sysctl -w iogpu.wired_limit_mb=458752"
fi

# ---------------------------------------------------------------------------
# Build python command
# ---------------------------------------------------------------------------

PYTHON="$KIKI_TUNNER/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    # Fallback: try ailiance's own venv or system python
    if [[ -x "$AILIANCE/.venv/bin/python" ]]; then
        PYTHON="$AILIANCE/.venv/bin/python"
    else
        PYTHON=$(which python3)
    fi
fi

CMD=("$PYTHON" "$EVAL_SCRIPT" "--mode" "$MODE")

if $QUICK; then
    CMD+=("--quick")
fi

if $SKIP_GEN; then
    CMD+=("--skip-generation")
fi

if $SKIP_SPEED; then
    CMD+=("--skip-speed")
fi

if [[ ${#DOMAINS[@]} -gt 0 ]]; then
    CMD+=("--domains" "${DOMAINS[@]}")
fi

# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------

echo ""
echo "Running: ${CMD[*]}"
echo ""

mkdir -p "$AILIANCE/output/eval/raw"

START_TIME=$(date +%s)

"${CMD[@]}" 2>&1 | tee "$AILIANCE/output/eval/eval_run_$(date +%Y%m%d_%H%M).log"

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
ELAPSED_MIN=$(( ELAPSED / 60 ))

echo ""
echo "============================================================"
echo " Evaluation complete in ${ELAPSED_MIN} minutes"
echo " Report: $AILIANCE/output/eval/eval_report_v1_vs_v2.md"
echo " Raw data: $AILIANCE/output/eval/raw/"
echo " Log: $AILIANCE/output/eval/eval_run_$(date +%Y%m%d_%H%M).log"
echo "============================================================"
