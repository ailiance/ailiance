# kicad_sch package — N3 5-axis eval

## Eval N3 — 5-axis evaluator

Run on a directory of generated `.kicad_sch` files:

    uv run python scripts/run_eval_n3.py \
        --sch-dir output/kicad_sch_gen/qwen36-D3/ \
        --ref-dir ~/eu-kiki-data/kicad-sch-refs/ \
        --model-key kicad-sch-qwen36-D3 \
        --domain kicad-sch \
        --out output/eval/raw/eval_n3_qwen36-D3.json \
        --out-aggregate output/eval/raw/eval_n3_qwen36-D3.agg.json \
        --audit-dir output/audit/kicad-sch-2026-05-11/

Feed aggregates into bench_comparison:

    uv run python scripts/bench_comparison.py \
        --validator-tuned output/eval/raw/eval_n3_qwen36-D3.agg.json \
        --metric-axes parse_ok,erc_clean,sch_render,drc_clean,sem_equiv

Composite weights: 0.30·parse_ok + 0.30·erc_clean + 0.15·sch_render
+ 0.10·drc_clean + 0.15·sem_equiv (locked, sums to 1.0).
