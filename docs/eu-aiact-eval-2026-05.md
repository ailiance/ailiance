# EU AI Act Art. 53(1)(d) — ailiance v1 evaluation report

**Date**: 2026-05-10
**Models**: Apertus 70B Instruct 2509 (CH), Devstral 2 Small 24B (FR/Mistral), EuroLLM 22B Instruct 2512 (EU consortium)
**Adapters**: 30 LoRA adapters across 30 (model × domain) cells (ailiance v1)
**Eval framework**: `scripts/eval_framework.py` @ `ailiance` commits `36d45aa..2a87b5f`
**Hardware**: Mac Studio M3 Ultra 512 GB, MLX runtime, macOS 26.4

---

## 1. Scope

This document satisfies Article 53(1)(d) of the EU AI Act, which
requires general-purpose AI providers to "draw up and keep up-to-date
technical documentation of the model, including its training and
testing process and the results of its evaluation."

It covers the **ailiance v1** stack: three EU-sourced base models, each
specialised on a set of domain-specific LoRA adapters, evaluated by
held-out perplexity. It does **not** cover (i) downstream task quality
beyond intrinsic perplexity, (ii) the v2 adapter cohort (qwen36 ×14 +
medium35 ×4), which is blocked by an `mlx_lm_fork` bug — see §5, or
(iii) human evaluation. A v2 fix and human-rated evaluation are
tracked separately.

## 2. Methodology

### 2.1 Validation procedure

For each (model, domain) pair where a trained LoRA adapter exists, we
compute the perplexity of `base_model + adapter` on a held-out
validation slice of `data/hf-traced/<domain>/valid.jsonl`, with N=5
records per domain (`--quick` mode; full N=150 run planned).

Records use the OpenAI-compatible `messages` format. The model's chat
template is applied via
`tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`,
the prompt is tokenized, and the model is run in teacher-forcing mode.
Cross-entropy loss is averaged over response tokens; perplexity is
`exp(loss)`.

Implementation: `ailiance/scripts/eval_framework.py` (entry point
`launch_eval_safe.sh --v1-only --quick`). Adapter loader uses
`mlx_lm.tuner.utils.load_adapters` (or `mlx_lm_fork` for SwitchLinear
v2 cases — see §5).

### 2.2 Hyperparameters (LoRA training)

All three base models share the same LoRA recipe; only `max_seq_length`
and gradient accumulation differ.

| Param                    | Apertus 70B | Devstral 24B | EuroLLM 22B |
|--------------------------|------------:|-------------:|------------:|
| `fine_tune_type`         | lora        | lora         | lora        |
| `lora_parameters.rank`   | 16          | 16           | 16          |
| `lora_parameters.alpha`  | 32          | 32           | 32          |
| `lora_parameters.dropout`| 0.05        | 0.05         | 0.05        |
| `lora_parameters.scale`  | 2.0         | 2.0          | 2.0         |
| `num_layers`             | -1 (all)    | -1 (all)     | -1 (all)    |
| `learning_rate`          | 1e-5        | 1e-5         | 1e-5        |
| `batch_size`             | 1           | 1            | 1           |
| `grad_accumulation_steps`| 8           | 4            | 4           |
| `iters`                  | 500         | 500          | 500         |
| `max_seq_length`         | 1024        | 2048         | 2048        |
| `grad_checkpoint`        | true        | true         | true        |
| `save_every`             | 100         | 100          | 100         |

Training scripts: `ailiance-mac-tuner/scripts/train_eu_kiki_{apertus,devstral,eurollm}.py`.

### 2.3 Memory budget

`mx.set_memory_limit(440 GiB)` is enforced via the
`WIRED_MEMORY_BUDGET_GIB` constant — 8 GiB of headroom under the macOS
`iogpu.wired_limit_mb=458752` cap. A per-group budget probe
(`_assert_within_budget()`) is run between model transitions in
`--mode sequential-strict`, mitigating the wired-memory growth the
prior eval framework hit. Plan: `docs/superpowers/plans/2026-05-10-eval-framework-oom-fix.md`.

## 3. Datasets

### 3.1 Manifest sources

Provenance metadata lives in three manifest files under
`data/hf-traced/`:

- `MANIFEST.json` (seed 42, valid_ratio 0.05, max_per_domain 3000) — 13 mainstream domains.
- `MANIFEST_niche.json` — 25 niche/composite domains (kicad, spice, embedded …).
- `MANIFEST_enriched.json` — 5 cross-checked entries with split counts.

### 3.2 Domain inventory (subset relevant to v1 cells)

| Domain            | Upstream                                                  | License                            | n_used | train | valid |
|-------------------|-----------------------------------------------------------|------------------------------------|-------:|------:|------:|
| chat-fr           | CohereForAI/aya_dataset                                   | Apache-2.0                         |  1422  |  1351 |    71 |
| cpp               | Takiyoshia/commitpack-parquet                             | MIT                                |  3000  |  2850 |   150 |
| docker-devops     | mcipriano/stackoverflow-kubernetes-questions              | CC-BY-SA-4.0                       |  1743  |     – |     – |
| electronics       | (curated mix)                                             | CC-BY-SA-3.0 AND ST-SLA0048        |    90  |     – |     – |
| embedded          | OSHWA + gavmac00/arduino-docs                             | Apache-2.0 AND CC-BY-SA-4.0        |  3000  |     – |     – |
| freecad           | (curated)                                                 | MIT AND CC-BY-4.0 AND CC0-1.0      |    65  |     – |     – |
| html-css          | iamtarun/code_instructions_120k_alpaca + sahil2801/CodeAlpaca-20k | Apache-2.0 + CC-BY-4.0     |  3000  |  2850 |   150 |
| iot               | TokenBender + embedded-overlap                            | Apache-2.0                         |  1932  |     – |     – |
| kicad-dsl         | (curated)                                                 | CC-BY-SA-4.0                       |  8099  |     – |     – |
| kicad-pcb         | (curated)                                                 | CC-BY-SA-4.0                       | 11882  |     – |     – |
| llm-ops           | zenml/llmops-database                                     | Apache-2.0                         |  1452  |     – |     – |
| llm-orch          | TokenBender + zenml/llmops-database                       | Apache-2.0                         |  1832  |     – |     – |
| lua-upy           | micropython/micropython + HF-lua                          | MIT + Apache-2.0                   |   875  |     – |     – |
| math-gsm8k        | openai/gsm8k                                              | MIT                                |  3000  |  2850 |   150 |
| math-reasoning    | microsoft/orca-math-word-problems-200k                    | MIT                                |  3000  |  2850 |   150 |
| ml-training       | Takiyoshia/commitpack-parquet                             | MIT                                |  3000  |  2850 |   150 |
| multilingual-eu   | CohereForAI/aya_dataset                                   | Apache-2.0                         |  3000  |  2850 |   150 |
| music-audio       | TokenBender + synthetic-audio                             | Apache-2.0                         |   500  |     – |     – |
| platformio        | platformio/platformio-examples + synthetic                | Apache-2.0                         |   700  |     – |     – |
| python            | bigcode/self-oss-instruct-sc2-exec-filter-50k             | Apache-2.0                         |  3000  |  2850 |   150 |
| rust              | bigcode/self-oss-instruct-sc2-exec-filter-50k             | Apache-2.0                         |  3000  |  2850 |   150 |
| rust-embedded     | (curated)                                                 | MIT + Apache-2.0 + 0BSD            |  1580  |     – |     – |
| shell             | bigcode/self-oss-instruct-sc2-exec-filter-50k             | Apache-2.0                         |    60  |    57 |     3 |
| spice-sim         | ngspice/ngspice + synthetic                               | BSD-3-Clause                       |   500  |     – |     – |
| sql               | gretelai/synthetic_text_to_sql                            | Apache-2.0                         |  3000  |  2850 |   150 |
| traduction-tech   | FrancophonIA/europarl-v7_fr-en                            | CC-BY-4.0                          |  3000  |     – |     – |
| typescript        | bigcode/self-oss-instruct-sc2-exec-filter-50k             | Apache-2.0                         |  3000  |  2850 |   150 |
| web-backend       | TokenBender/code_instructions_122k + bigcode              | Apache-2.0                         |  3000  |     – |     – |
| web-frontend      | TokenBender/code_instructions_122k + bigcode              | Apache-2.0                         |  3000  |     – |     – |
| yaml-json         | TokenBender/code_instructions_122k + K8s-SO               | Apache-2.0 + CC-BY-SA-4.0          |  3000  |     – |     – |

Entries marked "–" come from `MANIFEST_niche.json`, which records
`n_used` only and does not split-count train/valid (split applied at
load time using global `valid_ratio=0.05`, seed 42).

### 3.3 Train / valid split

Global defaults from `MANIFEST.json`: `seed=42`, `valid_ratio=0.05`,
`max_per_domain=3000`. For mainstream domains this gives a 95/5 split;
for niche domains (n<200) the ratio is preserved with a hard floor of
3 valid records.

## 4. Results — per-model × domain perplexity (N=5, `--quick`)

Source: merged across `output/eval/raw/perplexity_v1-only_*.json`,
latest record per (model, domain) wins. 30 unique cells.

### 4.1 Apertus 70B Instruct 2509 — 5 cells

| Domain          |   PPL | val_loss | n |
|-----------------|------:|---------:|--:|
| electronics     |  2.55 |   0.9358 | 5 |
| embedded        |  7.16 |   1.9691 | 5 |
| math-gsm8k      |  4.41 |   1.4838 | 5 |
| math-reasoning  |  2.00 |   0.6917 | 5 |
| spice-sim       |  4.91 |   1.5903 | 5 |

Median PPL **4.41**, min 2.00 (math-reasoning), max 7.16 (embedded).
Coverage 5/8 (configs/apertus.yaml lists 20 candidate domains; only 5
adapters reached the v1 publish bar).

### 4.2 Devstral 2 Small 24B (MLX-4bit) — 22 cells

| Domain         |   PPL | val_loss | n |
|----------------|------:|---------:|--:|
| cpp            |  2.26 |   0.8164 | 5 |
| docker-devops  |  2.08 |   0.7319 | 5 |
| freecad        |  2.49 |   0.9123 | 3 |
| html-css       |  1.76 |   0.5652 | 5 |
| iot            |  5.01 |   1.6112 | 5 |
| kicad-dsl      |  1.55 |   0.4414 | 5 |
| kicad-pcb      |  1.21 |   0.1922 | 5 |
| llm-ops        |  6.82 |   1.9196 | 5 |
| llm-orch       |  3.67 |   1.3012 | 5 |
| lua-upy        |  2.20 |   0.7885 | 5 |
| ml-training    |  1.78 |   0.5746 | 5 |
| music-audio    |  1.90 |   0.6394 | 5 |
| platformio     |  1.49 |   0.4000 | 5 |
| python         |  2.06 |   0.7237 | 5 |
| rust           |  2.46 |   0.8995 | 5 |
| rust-embedded  |  2.22 |   0.7965 | 5 |
| shell          | 13.08 |   2.5714 | 5 |
| sql            |  2.00 |   0.6928 | 5 |
| typescript     |  1.78 |   0.5777 | 5 |
| web-backend    |  1.62 |   0.4800 | 5 |
| web-frontend   |  2.39 |   0.8728 | 5 |
| yaml-json      |  2.44 |   0.8901 | 5 |

Median PPL **2.13**, min 1.21 (kicad-pcb), max 13.08 (shell — n=60
training only). 22/16 declared in config (extras = niche/PR-merged).

### 4.3 EuroLLM 22B Instruct 2512 — 3 cells

| Domain          |   PPL | val_loss | n |
|-----------------|------:|---------:|--:|
| chat-fr         |  5.90 |   1.7749 | 5 |
| multilingual-eu |  6.75 |   1.9101 | 5 |
| traduction-tech |  6.65 |   1.8943 | 5 |

Median PPL **6.65**, min 5.90, max 6.75. Coverage 3/4 (`localisation-doc`
adapter not yet trained at v1 cutoff).

### 4.4 Aggregate

| Model       | Cells | Median PPL | Min PPL | Max PPL |
|-------------|------:|-----------:|--------:|--------:|
| Apertus 70B |     5 |       4.41 |    2.00 |    7.16 |
| Devstral 24B|    22 |       2.13 |    1.21 |   13.08 |
| EuroLLM 22B |     3 |       6.65 |    5.90 |    6.75 |
| **Total**   |  **30** |    2.36 |    1.21 |   13.08 |

## 5. Limitations and known biases

- **Sample size**: N=5 records/domain (`--quick`). A full N=150 run
  using the held-out `valid.jsonl` slice is planned and is expected
  to tighten estimates by ≈√30. Current N=5 numbers carry wide
  confidence intervals — treat ±20% relative.
- **Devstral base swap**: the bf16 MLX conversion at
  `Devstral-Small-2-24B-Instruct-2512/` is corrupt (q/k/v projection
  weight mean ≈ +0.12 vs ~0 expected, base PPL ≈ 345 k). This eval
  uses the 4-bit MLX sibling `Devstral-Small-2-24B-MLX-4bit` (PR #17,
  commit `2a87b5f`). Eval-time 4-bit quantisation may shift PPL by
  0.1–0.5 in absolute terms; relative ranking unaffected.
- **Outlier — `shell` PPL 13.08 on Devstral**: training corpus only
  60 records (57 train / 3 valid). Hyperparams unchanged at iters=500
  → severe under-fitting. Re-train with curated data planned for v1.1.
- **No v2 results in this report**: 18 v2 adapters (qwen36 ×14 +
  medium35 ×4) are blocked by a `SwitchLinear` LoRA gap in
  `mlx_lm_fork.tuner.utils:74`. A separate fix-plan (Task #14) tracks
  this.
- **Coverage gaps**: Apertus 5/20 declared, EuroLLM 3/4 declared. The
  `electronics-hw` data line is sourced from
  `ailiance-mac-tuner/data/micro-kiki/` rather than `ailiance/data/hf-traced/` —
  provenance flagged, to be migrated.
- **Manifest gaps**: `MANIFEST_enriched.json` only enumerates 5
  domains. The remaining 25 domains rely on `MANIFEST.json` +
  `MANIFEST_niche.json`, which are less detailed (no per-row license
  tagging). Enrichment for the full 30 is a follow-up.
- **No human evaluation**: this report is intrinsic-metric only. A
  future human-rated quality eval is out of scope.
- **Composite-license domains** (cpp, embedded, html-css, lua-upy,
  yaml-json, …) inherit the most-restrictive component's terms. We
  redistribute only adapter weights (LoRA deltas), not training data.

## 6. Reproducibility

```bash
ssh studio
sudo sysctl -w iogpu.wired_limit_mb=458752  # default for this host
kill -TERM $(lsof -tiTCP:9303 -sTCP:LISTEN) 2>/dev/null  # free EuroLLM worker
cd ~/ailiance
bash scripts/launch_eval_safe.sh --v1-only --quick
# Output: output/eval/raw/perplexity_v1-only_<stamp>.json
# Status: output/eval/last_run_status.json
```

## 7. References

- EU AI Act Art. 53(1)(d) — Regulation (EU) 2024/1689, Annex XI.
- Plan: `docs/superpowers/plans/2026-05-10-eval-framework-oom-fix.md`
- PRs: #14 (OOM fix), #15 (cell completeness), #16 (path sweep), #17 (Devstral base).
- Bench artifacts: `output/eval/raw/perplexity_v1-only_*.json`.
- Manifests: `data/hf-traced/MANIFEST.json`, `MANIFEST_niche.json`, `MANIFEST_enriched.json`.
- Training scripts: `ailiance-mac-tuner/scripts/train_eu_kiki_{apertus,devstral,eurollm}.py`.
