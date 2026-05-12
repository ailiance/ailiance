#!/bin/bash
# Run after corpus gen completes: curate -> train -> eval
set -euo pipefail
cd /home/electron/ailiance
echo "=== Curate ==="
./.venv/bin/python tools/router_v7/curate_split.py

echo "=== Train (MiniLM-L6-v2 hidden=256, matches v6 prod arch) ==="
./.venv/bin/python scripts/train_router.py \
    --data-dir data/router-v7-multimodel \
    --output-dir output/router-v7-multimodel \
    --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
    --hidden-dim 256 \
    --epochs 40 \
    --batch-size 128 \
    --lr 1e-3

echo "=== Eval ==="
./.venv/bin/python tools/router_v7/eval_v7.py

echo "=== DONE ==="
ls -la output/router-v7-multimodel/
