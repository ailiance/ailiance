# eu-kiki

100% EU-sovereign multi-model LLM serving pipeline running locally on Mac Studio M3 Ultra (512 GB unified memory). EU AI Act Article 52/53 transparency-compliant.

## What This Does

Routes requests to **3 European foundation models** via a MiniLM domain classifier, each fine-tuned with **LoRA adapters trained on HF-traceable domains** (~10 K curated examples after v6 rebuild).

| Model | Origin | Domains | Port |
|-------|--------|---------|------|
| **Apertus-70B-Instruct-2509** | EPFL + ETH Zürich + CSCS (CH) | 20 — electronics, EMC, DSP, SPICE, KiCad, STM32, IoT, embedded, MISRA-C, AUTOSAR, IEC norms… | `:9301` |
| **Devstral-Small-2-24B-MLX-4bit** | Mistral AI (FR) | 16 — Python, Rust, TypeScript, C++, shell, SQL, web, Docker, devops, llm-ops, ml-training… | `:9302` |
| **EuroLLM-22B-Instruct-2512** | utter-project (EU consortium) | 4 — chat-fr, traduction-tech, redaction-multilingue, localisation-doc | `:9303` |

Router: **all-MiniLM-L6-v2** (384d, 22 M params) + MLP head (256 hidden) — sigmoid multi-label routing on **32 domains**, `top-k=4`, `threshold=0.50`. Active checkpoint: `output/router-v6` — **87.7 % top-1 / 98.7 % top-3** on validation (vs v5 65.5 % / 85.3 %, +22 / +13 pts).

Encoder caches: L1 LRU 1024 (exact-hit ~0.01 ms) + L2 cosine ≥ 0.95 (paraphrase ~0.2 ms) + auto-prewarm at boot. Auto-resolves device (MPS / CUDA / CPU). Cold compute ~9 ms on Studio MPS, ~17 ms on electron-server CPU.

Gateway: FastAPI on `:9300` (electron-server, systemd unit `eu-kiki-gateway.service`, env `EU_KIKI_WORKERS_JSON` for Tailscale worker URLs). Prometheus metrics at `/metrics`. Live at `https://ml.saillant.cc/api/public/chat` (Cloudflare Tunnel → kiki-cockpit on electron-server → gateway → workers on Studio over Tailscale).

⚠️ **Quarantined adapters** (verified 2026-05-05, source: training-data chat-template leak): EuroLLM `chat-fr` and `traduction-tech` produce `"user user user…"` loops; the worker silently falls back to the base EuroLLM model for those domains. See `MLXWorkerRuntime.QUARANTINED_DOMAINS` in `src/worker/runtime.py`. Re-train pending.

## Why EU-sovereign?

Every component is auditable, EU/Swiss-origin, and Apache-2.0 licensed. Datasets are HF-traceable with `hf_dataset_id`, license, download date, and used-row count documented per domain. Local-only deployment — no cloud, no telemetry. Full provenance chain in [`docs/eu-ai-act-transparency.md`](docs/eu-ai-act-transparency.md).

## Quick Start

```bash
# Setup
uv venv && uv pip install -e ".[dev,router,data]"

# Build datasets (HF-traceable)
uv run python scripts/build_hf_datasets.py
uv run python scripts/scrape_oshwa.py            # 3265 OSHWA-certified projects
uv run python scripts/scrape_arxiv_eess.py
uv run python scripts/scrape_wikipedia_electronics.py

# Train LoRA adapters (3 models, sequential)
bash scripts/train_eu_kiki_batch.sh              # or run individually:
uv run python scripts/train_apertus.py
uv run python scripts/train_devstral.py
uv run python scripts/train_eurollm.py

# Train router (full pipeline, ~25 min on macM1 MPS)
uv run python scripts/rebuild_router_dataset.py        # HF + niche + greetings → data/router-clean/
uv run python scripts/build_router_data.py             # split train/valid 80/20
uv run python scripts/encode_router_minilm.py          # MiniLM embeddings → data/router-minilm-vN/
uv run python scripts/train_router_from_embeddings.py --emb-dir data/router-minilm-vN --hidden-dim 256 --output-dir output/router-vN

# Launch all services
bash scripts/start.sh

# Test
uv run python -m pytest
uv run python -m pytest tests/test_xielu.py -v   # single file
```

## Data pipeline

### Sources

| Source | Script | Items |
|--------|--------|-------|
| OSHWA-certified projects | `scrape_oshwa.py` | 3265 |
| arXiv EESS papers | `scrape_arxiv_eess.py` | — |
| Wikipedia electronics | `scrape_wikipedia_electronics.py` | — |
| Hackaday writeups | `scrape_hackaday.py` | — |
| Arduino examples | `scrape_arduino_examples.py` | — |
| ESP-IDF examples | `scrape_espidf_examples.py` | — |
| STM32 examples | `scrape_stm32_examples.py` | — |
| Rust embedded | `scrape_rust_embedded.py` | — |
| KiCad schematics | `scrape_kicad_schematics.py` | — |
| HuggingFace datasets | `build_hf_datasets.py` | 48K (20 domains) |
| StackExchange (manuals) | — | (PDF pipeline) |

### HF-traced datasets

`data/hf-traced/` (404 MB) — 35 domain folders, format `train.jsonl` / `valid.jsonl` (split 95/5, max 3000/domain, seed 42). Sources include `bigcode/self-oss-instruct-sc2-exec-filter-50k`, `CohereForAI/aya_dataset`, etc. Each `MANIFEST.json` documents `hf_dataset_id`, `license`, `download_date`, `n_source_rows`, `n_used`.

### PDF pipeline

`scripts/pdf_pipeline/` + `scan_pii.py` + `fix_provenance.py` — robots.txt-respectful scraping under EU DSM Art. 4 TDM exception (ST/Espressif/TI/NXP/KiCad), SHA-256 manifests, 360 training pairs. Audit: [`docs/pdf-compliance-report.md`](docs/pdf-compliance-report.md).

### VLM POC

`scripts/vlm_poc_pipeline.py` — visual-language model POC, same legal frame. Audit: [`docs/vlm-compliance-report.md`](docs/vlm-compliance-report.md).

## Router

| Component | Path |
|-----------|------|
| Active checkpoint | `output/router-v6/` (87.7 % top-1 / 98.7 % top-3 on validation) |
| Encoder | `sentence-transformers/all-MiniLM-L6-v2` (384d, 22 M) |
| MLP head | 384 → 256 → 32 (sigmoid, threshold 0.50) |
| Train data | `data/router-clean/` (32 JSONL, 9967 rows, niche+greetings curated) |
| Embeddings | `data/router-minilm-v6/{train,valid}_embs.npy` (MPS-encoded) |
| Classifier | `src/router/classifier.py` (auto-device, L1 LRU + L2 cosine cache, auto-prewarm) |
| Train pipeline | `scripts/{rebuild_router_dataset,build_router_data,encode_router_minilm,train_router_from_embeddings}.py` |
| Confusion top-10 | [`docs/transparency/confusion-top10.md`](docs/transparency/confusion-top10.md) |
| Provenance | [`docs/transparency/router-training-data.md`](docs/transparency/router-training-data.md) |

## Configuration

| File | Role |
|------|------|
| `configs/apertus.yaml` | Apertus 70B worker (port 9301, BF16, 20 domains) |
| `configs/devstral.yaml` | Devstral 24B worker (port 9302, BF16, 16 domains) |
| `configs/eurollm.yaml` | EuroLLM 22B worker (port 9303, BF16, 4 domains) |
| `configs/gateway.yaml` | FastAPI gateway + router config |

## Source layout

```
src/
├── gateway/          # FastAPI :9200, request dispatch, Prometheus metrics
├── router/           # Jina v3 + MLP classifier (40 domains)
├── worker/           # 1 model / process, BF16, shared memory pool
└── mlx_models/       # Apertus MLX impl + custom xielu activation
```

## Tests

`tests/` — `test_apertus_model.py` (MLX model), `test_xielu.py` (custom activation), `test_worker.py`, `test_integration.py`, `test_runtime.py`, `test_router.py`, `test_gateway.py`.

## Compliance docs (`docs/`)

| Doc | Contenu |
|-----|---------|
| [`MODEL_CARD.md`](MODEL_CARD.md) | **Carte de modèle — performance mesurée, limitations connues, intended/out-of-scope use, Art. 53(1)(d)**. |
| [`eu-ai-act-transparency.md`](docs/eu-ai-act-transparency.md) | Doc principale — Art. 52/53 EU AI Act, "limited risk" classification, full provenance chain (v0.4.0 + evaluation summary §8.bis). |
| [`pdf-compliance-report.md`](docs/pdf-compliance-report.md) | Audit pipeline PDF (robots.txt, SHA-256, DSM Art.4 TDM). |
| [`vlm-compliance-report.md`](docs/vlm-compliance-report.md) | Audit pipeline VLM POC. |
| [`eval/WORKFLOW.md`](eval/WORKFLOW.md) | Bench pipeline trace (3-host topology, bug history, fuse workaround, full results). |
| [`eval/results/SUMMARY.md`](eval/results/SUMMARY.md) | Aggregated benchmark table — KIKI-DSL v3, HumanEval+, MT-Bench, GSM8K. |
| `docs/specs/2026-04-26-eu-kiki-design.md` | Design initial du système. |
| `docs/specs/2026-04-26-eu-kiki-plan.md` | Plan d'implémentation. |

## Headline benchmark results

| Bench | Subject | Result |
|---|---|---|
| HumanEval+ | Devstral 24B 4-bit base (Linux EvalPlus) | 87.20 / 82.90 |
| HumanEval+ | + python / cpp / rust adapters | −1.80 / −1.22 / −0.61 |
| MT-Bench (full 80q, judge Mistral-Medium 128B) | Devstral base | **8.892 / 10** |
| GSM8K 5-shot, n=200 | Qwen 35B-A3B-4bit base | **94.5 %** |
| GSM8K | + reasoning / + math adapters | 0 / −4.5 |
| KIKI-DSL v3 (15 prompts, balanced) | Qwen base | 73.3 % pass / 0.704 avg |
| KIKI-DSL v3 | best adapter (`reasoning`) | **+13.4 pass** |
| KIKI-DSL v3 | worst adapter (`kicad-dsl`, narrow) | −27 pass |

Cognitive adapter wins on KIKI-DSL v3 do **not** transfer to GSM8K
(saturated). See [`MODEL_CARD.md`](MODEL_CARD.md) §4.5 for cross-bench
analysis and §7 for known limitations.

## Logs

```bash
tail -f /tmp/eu-kiki/gateway.log
tail -f /tmp/eu-kiki/apertus.log
tail -f /tmp/eu-kiki/devstral.log
tail -f /tmp/eu-kiki/eurollm.log
```

## Key Design Decisions

- **BF16 for all models** — 512 GB unified memory makes quantization unnecessary
- **Multi-process workers** — 1 model per process, shared memory pool
- **Sigmoid routing** — domains overlap, not mutually exclusive
- **LoRA on `q/k/v/o_proj` only** — minimal footprint, full provenance
- **`xielu` activation custom-implemented** for Apertus MLX support
- **HF-traceable datasets** — each domain has a `MANIFEST.json` with provenance

## Sister project

[`KIKI-Mac_tunner`](https://github.com/L-electron-Rare/KIKI-Mac_tunner) — non-EU foundation distillation track (Mistral Large, Qwen3.5-122B, Devstral 2 123B dense). eu-kiki training scripts (`train_eu_kiki_*.py`) and configs (`eu-kiki-*.yaml`) are mirrored there.

## License

Apache-2.0
