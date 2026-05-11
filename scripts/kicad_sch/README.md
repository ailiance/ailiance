# kicad_sch package — N3 5-axis eval

## Eval N3 — 5-axis evaluator

Run on a directory of generated `.kicad_sch` files:

    uv run python scripts/run_eval_n3.py \
        --sch-dir output/kicad_sch_gen/qwen36-D3/ \
        --ref-dir ~/ailiance-data/kicad-sch-refs/ \
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


## Track C — LoRA training infrastructure (kicad-sch v10)

Five entry points cover the D1/D2/D3 datasets plus M2 LoRA training.
All scripts emit Annex-IV manifest rows and NDJSON audit events.

### 1. `strip_lib_symbols` — context-size pre-processor

    uv run python -m scripts.kicad_sch.strip_lib_symbols \
        --input  ~/ailiance-data/kicad-sch-scraped \
        --output ~/ailiance-data/kicad-sch-scraped-stripped

Removes the inline `(lib_symbols ...)` block; `lib_id` placement
refs remain and kicad-cli reloads symbols at parse time.

### 2. `scrape_d1` — github scraper (D1)

    uv run python -m scripts.kicad_sch.scrape_d1 \
        --max-files 10000 \
        --license-allowlist MIT,Apache-2.0,CC0-1.0,GPL-3.0

Searches `gh search code extension:kicad_sch`, filters by SPDX, runs
`kicad-cli sch update`, dedupes via UUID-stripped sha256.

### 3. `synth_d2` — random circuit synth (D2)

    uv run python -m scripts.kicad_sch.synth_d2 \
        --n-samples 10000 --compilers skidl,atopile,circuit-synth

Renders 10 analog templates through 3 compilers, ERC-gated by
`kicad-cli sch erc`.

### 4. `mix_d3` — 50/50 stratified mixer (D3)

    uv run python -m scripts.kicad_sch.mix_d3 \
        --d1 ~/ailiance-data/kicad-sch-scraped \
        --d2 ~/ailiance-data/kicad-sch-synth \
        --d3 ~/ailiance-data/kicad-sch-mixed \
        --n-total 10000

Symlinks half D1 + half D2 (stratified across compilers) into D3.

### 5. `train_lora` — MLX LoRA orchestrator

    uv run python -m scripts.kicad_sch.train_lora \
        --config ~/ailiance-mac-tuner/configs/ailiance-v3-qwen36-kicad-sch-D2.yaml

Dry-run by default. Pass `--actually-run` to disarm and invoke
`mlx_lm.lora`. Audit events: `train_start`, `train_dry_run` or
`train_done`.

### Full M2 sweep

    bash scripts/kicad_sch/run_m2_all.sh                  # dry-run all 6
    bash scripts/kicad_sch/run_m2_all.sh --actually-run   # launch

### Configs

Six M2 configs (qwen36 + gemma4 × D1/D2/D3) live in
`configs/v3-track-c/` and are mirrored to `~/ailiance-mac-tuner/configs/`.
Twelve M3/M4 stubs (devstral, apertus, eurollm, medium35) carry a
`STATUS: M3/M4 stub` header and are skipped by default launchers.

### Dependencies

- Foundation track: `scripts.kicad_sch.manifest.DatasetManifest`,
  `scripts.kicad_sch.audit_log.AuditLogger`.
- Eval N3 track: `scripts.kicad_sch.eval_n3` for post-train scoring.
- External: `kicad-cli 10.0.2`, `gh` CLI ≥ 2.40, `mlx_lm`, Python 3.14.
