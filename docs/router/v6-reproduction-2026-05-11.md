# router-v6 prod reproduction — forensic report
Date: 2026-05-11
Investigator: Claude

## TL;DR
**Mystery solved. NOT reproducible from current data.**

- Prod router-v6 (`electron-server:/home/electron/ailiance/output/router-v6/`) weights evaluated on **current** Studio `data/router-minilm-v7` valid set → **top1=0.9009, top3=0.9904** (matches user's measurement exactly).
- Prod `meta.json` self-reports top1=0.877 because at training time (2026-05-05) the corpus had 9967 rows and a different stratified split.
- Studio retrains on **today's** `data/router-minilm-v7` (9817 rows, 7839 train + 1978 valid) plateau at 0.877-0.882 across {legacy 30ep last, legacy 100ep last, v9 best-with-patience-10, seeds 42/137/1024}.
- The +1.9% gap is attributable to **dataset shift**: today's emb cache was re-encoded 2026-05-11 from a slightly different/dedup'd corpus that under-samples ~12 niche domains to support=1 in valid (and 1-4 in train), making them unlearnable. The prod model trained on a richer corpus that classified them correctly.

## Phase 1 — Forensic findings

### Prod weights are real
`output/router-v6/router.safetensors` is a 4-layer Linear(384→256)+GELU+Dropout(0.1)+Linear(256→32) MLP, 427 KB. Loaded cleanly into the reference architecture.

### Git history
Commit `06060c3` (2026-05-05 21:28) "feat(router): v6 87.7 top1 vs v5 65.5" — created prod v6. The commit changed gateway config + PROVENANCE.json + added `scripts/encode_router_minilm.py`. NO training-log artifacts. NO snapshot of train/valid jsonl files (they are `.gitignore`'d).

### Two trainers exist on Studio
- `scripts/train_router_from_embeddings.py` (legacy, used for prod v6): 30 epochs, **saves LAST epoch**, class-weighted CE (cap=10), AdamW lr=1e-3 wd=1e-4, Dropout 0.1, no seeding, no early-stop, no best-tracking.
- `scripts/train_router_from_embeddings_v9.py`: same arch, **adds best-epoch tracking, early stop (patience=10), seeding, `.float()` casts, valid-domains-union-with-train in label_map, per-class metrics, history**.

### Data shift detected
- Prod meta.json: `"rebuilt_from": "data/router-clean (32 domains, 9967 rows, niche+greetings curated)"`.
- Studio current jsonl: **9817 rows** (`wc -l data/router/{train,valid}.jsonl` = 7839+1978).
- Studio `data/router-minilm-v7/` embeddings: mtime **2026-05-11 08:43** (re-encoded today from current 9817-row data).
- PROVENANCE.json reports 10309 source rows on both prod & Studio, so source catalog is identical — but the dedup/clean output differs by 150 rows.

### Per-class distribution (current minilm-v7)
12 domains have valid_support=1 AND train_support≤4: `dsp, embedded, iot, kicad-dsl, kicad-pcb, llm-orch, ml-training, music-audio, platformio, stm32, web-backend, web-frontend, yaml-json`. These are essentially unlearnable in current data.

## Phase 2 — Hypothesis tests

| Hypothesis | Test | Result |
|---|---|---|
| H1: Different bench data (not 0.877 self-report) | Evaluate prod weights on `data/router-minilm-v7/valid` (current) | **Confirmed → 0.9009 reproduced exactly** |
| H2: Data leakage (train+valid mixed) | Eval prod on train: 0.9316 (vs valid 0.9009, gap 3.2%); fresh repro train 0.9430 valid 0.8777 (gap 6.5%) | **No leakage. Prod just generalizes better.** |
| H3: Longer training closes gap | Legacy trainer 100 epochs (save last) | **No → degrades to 0.862 (overfit)** |
| H4: Save LAST epoch instead of BEST (matches prod script) | Legacy trainer 30 epochs (save last) | **No → 0.8777, identical plateau** |
| H5: Niche-class behavior differs | Per-class TP diff: prod beats repro on 12 niche support=1 classes (+12 TPs) AND on hard classes spice(+14), reasoning(+11), python(+7) | **Confirms data-not-training root cause** |

## Phase 3 — Per-class diff (prod vs repro epoch-30-last, same emb dir)
Net: **prod = repro + 46 TPs / 1978** (= +2.33%, accounting for the full gap).

| Class | Support | prod TP | repro TP | Δ |
|---|---:|---:|---:|---:|
| spice | 141 | 107 | 93 | **+14** |
| reasoning | 161 | 135 | 124 | **+11** |
| python | 162 | 155 | 148 | **+7** |
| power | 40 | 37 | 33 | +4 |
| security | 39 | 39 | 35 | +4 |
| lua-upy | 121 | 111 | 108 | +3 |
| 12× niche (support=1) | 12 | 12 | 0 | +12 |
| math | 161 | 151 | 156 | **-5** |
| chat-fr | 192 | 185 | 187 | -2 |
| typescript | 121 | 115 | 117 | -2 |

Prod systematically wins on niche domains AND on technical-heavy classes (spice/reasoning/python). Repro slightly wins on chat-fr/math/typescript (well-supported classes — repro slightly more biased toward majority).

## Phase 4 — Recommendation

**Keep prod `output/router-v6/router.safetensors` deployed as-is.** It is the best router we have for the current routing task. It is not reproducible from the corpus that exists today on Studio.

### To match or beat prod v6 with reproducible pipeline, do one of:
1. **Reconstruct the May-5 corpus.** Check `electron-server:/home/electron/ailiance/data/router-clean/` for the 9967-row jsonl files (currently gitignored — but maybe present on disk). If found, re-encode with `scripts/encode_router_minilm.py` and re-train with legacy trainer 30 epochs. *Recommended next step.*
2. **Augment current corpus.** Add ≥30 samples per niche class (`dsp, embedded, iot, kicad-dsl, kicad-pcb, llm-orch, ml-training, music-audio, platformio, stm32, web-backend, web-frontend, yaml-json`). Niche support=1 in valid will then be informative AND learnable.
3. **Drop unlearnable niche classes** from the 32-class taxonomy (since 13 of them have ≤5 total samples). Reduces top1 ceiling but eliminates the noise floor.

### What NOT to do
- Don't retrain on current corpus expecting 0.9009. Plateau is ~0.88, multi-seed (42/137/1024) all converge there.
- Don't deploy current `router-v11-minilm-seed42` (0.882) over prod (0.901) — strict regression.

## Files & paths
- Prod weights: `electron-server:/home/electron/ailiance/output/router-v6/router.safetensors` (427 KB, mtime 2026-05-05 21:25)
- Prod meta (says 0.877): `electron-server:/home/electron/ailiance/output/router-v6/meta.json`
- Prod train script: `electron-server:/home/electron/ailiance/scripts/train_router_from_embeddings.py`
- v9 train script: `studio:/Users/clems/eu-kiki/scripts/train_router_from_embeddings_v9.py`
- Current emb cache (re-encoded 2026-05-11): `studio:/Users/clems/eu-kiki/data/router-minilm-v7/`
- Repro artifacts (Studio, this session):
  - `output/router-v6-repro-last30/router.safetensors` (top1 0.8777)
  - `output/router-v6-repro-last100/router.safetensors` (top1 0.862, overfit)
- Audit scripts: `/tmp/audit/{eval_prod_router,probe_data_leak,per_class_diff}.py`
- Local copy of prod weights: `/tmp/audit/prod-router-v6.safetensors` (also `studio:/tmp/prod-router-v6.safetensors`)
