# eu-kiki

100% EU-sovereign multi-model LLM serving pipeline running locally on Mac Studio M3 Ultra (512 GB unified memory). EU AI Act Article 52/53 transparency-compliant.

## What This Does

Routes requests to **3 European foundation models** via a Jina v3 domain classifier (Berlin), each fine-tuned with **LoRA adapters trained on 20 HF-traceable domains** (48K curated examples).

| Model | Origin | Domains | Port |
|-------|--------|---------|------|
| **Apertus-70B-Instruct-2509** | EPFL + ETH Zürich + CSCS (CH) | 20 — electronics, EMC, DSP, SPICE, KiCad, STM32, IoT, embedded, MISRA-C, AUTOSAR, IEC norms… | `:9301` |
| **Devstral-Small-2-24B-MLX-4bit** | Mistral AI (FR) | 16 — Python, Rust, TypeScript, C++, shell, SQL, web, Docker, devops, llm-ops, ml-training… | `:9302` |
| **EuroLLM-22B-Instruct-2512** | utter-project (EU consortium) | 4 — chat-fr, traduction-tech, redaction-multilingue, localisation-doc | `:9303` |

Router: **Jina Embeddings v3** (Berlin) + MLP classifier — sigmoid multi-label routing on 40 domains, `top-k=4`, `threshold=0.12`.

Gateway: FastAPI on `:9200` with Prometheus metrics.

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

# Train router
uv run python scripts/build_router_data.py
uv run python scripts/encode_router_jina.py
uv run python scripts/train_router_from_embeddings.py

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
| Embeddings | `data/router-jina-v3/` (Jina v3 pre-encoded, 39 MB) |
| Train data | `data/router/train.jsonl` (46100 lines) + `valid.jsonl` (11532) |
| Classifier | `src/router/classifier.py` (Jina v3, 1024d → MLP hidden=512 → 40 domains) |
| Domain map | `src/router/domain_map.py` (static domain → worker port) |
| Train script | `scripts/train_router.py` (175 lines) |

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
| [`eu-ai-act-transparency.md`](docs/eu-ai-act-transparency.md) | Doc principale — Art. 52/53 EU AI Act, "limited risk" classification, full provenance chain. |
| [`pdf-compliance-report.md`](docs/pdf-compliance-report.md) | Audit pipeline PDF (robots.txt, SHA-256, DSM Art.4 TDM). |
| [`vlm-compliance-report.md`](docs/vlm-compliance-report.md) | Audit pipeline VLM POC. |
| `specs/2026-04-26-eu-kiki-design.md` | Design initial du système. |
| `specs/2026-04-26-eu-kiki-plan.md` | Plan d'implémentation. |

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
