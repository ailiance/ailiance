# EU AI Act — Transparency & Traceability Documentation

**Document ID:** EU-KIKI-TRANS-001
**Date:** 2026-05-06
**Version:** 0.4.1
**System:** eu-kiki — EU-sovereign multi-model LLM serving pipeline
**Risk Classification:** Limited risk (general-purpose AI system, Article 52)

---

## 1. System Overview

EU-KIKI is a multi-model routing system that dispatches user queries to one of three EU-origin language models, each enhanced with domain-specific LoRA adapters. The system runs locally on a single machine (no cloud dependencies).

**Purpose:** Provide domain-specialized AI assistance using exclusively European AI models and infrastructure.

**Deployment:** Local-only, single-machine (Mac Studio M3 Ultra 512GB).

**Training domains:** 35 total (30 HF-traced + 2 PDF-supplement + 3 in progress: misra-c, normes-iec, doc-technique-ce).

---

## 2. Foundation Models

### 2.1 Apertus-70B-Instruct-2509

| Field | Value |
|-------|-------|
| **Provider** | Swiss AI Initiative (EPFL + ETH Zürich + CSCS) |
| **Country** | Switzerland 🇨🇭 |
| **Parameters** | 70.6B (dense transformer) |
| **Training data** | 15T tokens, 1811 languages, ~40% non-English |
| **Training infrastructure** | CSCS Alps (4096 GH200 GPUs) |
| **Data compliance** | Goldfish objective (suppresses memorization), robots.txt respected, PII filtered |
| **License** | Apache 2.0 |
| **HuggingFace** | `swiss-ai/Apertus-70B-Instruct-2509` |
| **Paper** | arXiv:2509.14233 |
| **Data reconstruction** | github.com/swiss-ai/pretrain-data (fully reproducible) |
| **EU AI Act** | Art. 53 transparency provided by Swiss AI |

### 2.2 Devstral-Small-2-24B-Instruct-2512

| Field | Value |
|-------|-------|
| **Provider** | Mistral AI |
| **Country** | France 🇫🇷 |
| **Parameters** | 24B (dense transformer) |
| **Training data** | Proprietary (Mistral AI internal dataset) |
| **Specialization** | Agentic coding, software engineering |
| **License** | Apache 2.0 |
| **HuggingFace** | `mistralai/Devstral-Small-2-24B-Instruct-2512` |
| **Benchmark** | 68.0% SWE-bench Verified |
| **Weight source** | Official FP8 checkpoint, dequantized to BF16 via akoumpa community conversion |

### 2.3 EuroLLM-22B-Instruct-2512

| Field | Value |
|-------|-------|
| **Provider** | EU Consortium (utter-project) |
| **Country** | European Union 🇪🇺 (multi-national) |
| **Funding** | Horizon Europe, European Research Council, EuroHPC |
| **Parameters** | 22.6B (dense transformer) |
| **Training data** | ~4T tokens, 35 languages (all 24 EU official + 11 additional) |
| **Training infrastructure** | MareNostrum 5 supercomputer (BSC, Barcelona) — 400 H100 GPUs |
| **License** | Apache 2.0 |
| **HuggingFace** | `utter-project/EuroLLM-22B-Instruct-2512` |
| **Paper** | arXiv:2602.05879 |
| **Data sources** | Web data, parallel corpora (OPUS, Europarl), Wikipedia, ArXiv, code |

### 2.4 Mistral Small 4 119B (pending evaluation)

| Field | Value |
|-------|-------|
| **Provider** | Mistral AI |
| **Country** | France |
| **Parameters** | 119B (MoE, 24B active) |
| **License** | Apache 2.0 |
| **Status** | Downloaded, pending evaluation. Not yet integrated into routing. |

### 2.5 Mistral Medium 3.5 128B (teacher/eval only)

| Field | Value |
|-------|-------|
| **Provider** | Mistral AI |
| **Country** | France |
| **Parameters** | 128B (dense) |
| **License** | Modified MIT (Mistral Research License) |
| **Purpose** | Teacher model for evaluation and contrastive pair generation ONLY |
| **Deployment restriction** | NOT deployed for production inference. Modified MIT license does not qualify for open-source exemption under EU AI Act Art. 53(2). Used exclusively offline for training data quality evaluation. |

---

## 3. Router / Embedding Model

### 3.1 Current: all-MiniLM-L6-v2 (bootstrap)

| Field | Value |
|-------|-------|
| **Provider** | Microsoft Research |
| **Country** | USA 🇺🇸 (temporary — to be replaced) |
| **Parameters** | 33M |
| **Embedding dim** | 384 |
| **License** | Apache 2.0 |
| **Purpose** | Bootstrap router — to be replaced by Jina v3 |

### 3.2 Target: Jina Embeddings v3

| Field | Value |
|-------|-------|
| **Provider** | Jina AI GmbH |
| **Country** | Germany 🇩🇪 (Berlin) |
| **Parameters** | 570M |
| **Embedding dim** | 1024 |
| **License** | Apache 2.0 |
| **HuggingFace** | `jinaai/jina-embeddings-v3` |
| **Status** | Validated, pending router retraining |

---

## 4. Training Data Sources (Updated)

*Access date for all sources: 2026-04-27 to 2026-05-02*

### 4.1 HuggingFace Datasets

| Dataset | HF ID | SPDX License | Records Used | Domain(s) |
|---------|-------|-------------|-------------|-----------|
| StarCoder2 Self-Instruct | `bigcode/self-oss-instruct-sc2-exec-filter-50k` | Apache-2.0 | 3000 py + 3000 rust + 3000 ts + 201 cpp + 60 shell + 168 sql + 99 html | python, rust, typescript, cpp, shell, sql, html-css |
| CommitPackFT | `Takiyoshia/commitpack-parquet` | MIT | 3000 shell + 3000 cpp + 3000 ml | shell, cpp, ml-training |
| CommitPackFT (Rust) | `bigcode/commitpackft` | MIT | 282 | rust-embedded |
| Code Instructions 120k | `iamtarun/code_instructions_120k_alpaca` | Apache-2.0 | 3000 | html-css |
| CodeAlpaca 20k | `sahil2801/CodeAlpaca-20k` | CC-BY-4.0 | merged w/ above | html-css |
| Code Instructions 122k | `TokenBender/code_instructions_122k` | Apache-2.0 | 3000 web-be + 3000 web-fe + 3000 yaml + 1832 llm-orch + 1932 iot + 500 music | web-backend, web-frontend, yaml-json, llm-orch, iot, music-audio |
| Synthetic Text-to-SQL | `gretelai/synthetic_text_to_sql` | Apache-2.0 | 3000 | sql |
| Linux Commands RU-EN | `NickIBrody/linux-commands-ru-en` | CC-BY-4.0 | 3000 | shell |
| Aya Dataset | `CohereForAI/aya_dataset` | Apache-2.0 | 1422 fra + 3000 EU langs | chat-fr, multilingual-eu |
| GSM8K | `openai/gsm8k` | MIT | 3000 | math-gsm8k |
| Orca-Math | `microsoft/orca-math-word-problems-200k` | MIT | 3000 | math-reasoning |
| Cybersecurity Fenrir v2 | `AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.0` | Apache-2.0 | 3000 | security-fenrir |
| CertiCoder | `wuog/CertiCoder` | Research | 37.4K (planned) | misra-c (planned) |
| Masala-CHAI | `Masala-CHAI/masala-chai` (GitHub) | Open | 7.5K netlists (ref) | spice-sim |
| Open Schematics | `bshada/open-schematics` | Open | ~4K (ref) | electronics, kicad-pcb |
| Common Corpus | `PleIAs/common_corpus` | Permissive | ref only | multilingual-eu |
| OPUS-100 | `Helsinki-NLP/opus-100` | CC-BY-4.0 | 55M pairs (ref) | multilingual-eu |
| StackOverflow K8s | `mcipriano/stackoverflow-kubernetes-questions` | CC-BY-SA-4.0 | 1743 | docker-devops |
| ZenML LLMOps | `zenml/llmops-database` | Apache-2.0 | 1452 | llm-ops |
| Europarl FR-EN | `FrancophonIA/europarl-v7_fr-en` | Open | 3000 | traduction-tech |
| Arduino Docs | `gavmac00/arduino-docs` | CC-BY-SA-4.0 | merged | embedded |

### 4.2 Scraped GitHub Repositories

| Source | GitHub URL | SPDX License | Records | Domain(s) |
|--------|-----------|-------------|---------|-----------|
| ESP-IDF examples | `espressif/esp-idf` | Apache-2.0 | 687 | cpp (embedded) |
| STM32CubeF4 | `STMicroelectronics/STM32CubeF4` | BSD-3-Clause | 1812 | cpp (embedded) |
| Arduino examples | `arduino/arduino-examples` | CC0-1.0 / MIT / Apache-2.0 | 99 | cpp (embedded) |
| Embassy (Rust embedded) | `embassy-rs/embassy` | MIT OR Apache-2.0 | 939 | rust-embedded |
| RTIC | `rtic-rs/rtic` | MIT OR Apache-2.0 | 82 | rust-embedded |
| STM32F4xx HAL (Rust) | `stm32-rs/stm32f4xx-hal` | 0BSD | 57 | rust-embedded |
| Rust Embedded Discovery | `rust-embedded/discovery` | MIT OR Apache-2.0 | 52 | rust-embedded |
| ESP-HAL (Rust) | `esp-rs/esp-hal` | MIT OR Apache-2.0 | 46 | rust-embedded |
| Cortex-M | `rust-embedded/cortex-m` | MIT OR Apache-2.0 | 38 | rust-embedded |
| nRF HAL | `nrf-rs/nrf-hal` | MIT OR Apache-2.0 | 30 | rust-embedded |
| defmt | `knurling-rs/defmt` | MIT OR Apache-2.0 | 26 | rust-embedded |
| embedded-hal | `rust-embedded/embedded-hal` | MIT OR Apache-2.0 | 24 | rust-embedded |
| KiCad symbols | `KiCad/kicad-symbols` | CC-BY-SA-4.0 | 8098 | kicad-dsl |
| KiCad footprints | `KiCad/kicad-footprints` | CC-BY-SA-4.0 | 11882 | kicad-pcb |
| KiCad demos | `KiCad/kicad-source-mirror` | CC-BY-SA-4.0 | 1 | kicad-dsl |
| FreeCAD macros | `FreeCAD/FreeCAD-macros` | Per-file (MIT / CC-BY / CC0) | 65 | freecad |
| MicroPython examples | `micropython/micropython` | MIT | merged | lua-upy |
| PlatformIO examples | `platformio/platformio-examples` | Apache-2.0 | 47 real + synthetic | platformio |
| ngspice examples | `ngspice/ngspice` | BSD-3-Clause | merged | spice-sim |

### 4.3 Other Scraped / API Sources

| Source | URL | SPDX License | Size | Provenance | Domain(s) |
|--------|-----|-------------|------|------------|-----------|
| OSHWA API | https://certificationapi.oshwa.org | Per-project (open hardware) | API | Certified open hardware project metadata | embedded |
| Hackaday.io API | https://api.hackaday.io | CC-BY-SA-4.0 | REST API | Maker/hardware project descriptions via official API | electronics, iot |
| EUR-Lex CELLAR | https://eur-lex.europa.eu/sparql | CC-BY-4.0 | SPARQL | EU legal texts via official SPARQL endpoint | normes-iec, doc-technique-ce |
| Wikipedia dumps | https://dumps.wikimedia.org | CC-BY-SA-3.0 | Bulk dumps | Official download, electronics/science articles | emc-dsp-power, reasoning |
| arXiv eess.* | https://arxiv.org/help/bulk_data_s3 | arXiv license (TDM exception Art. 3-4 DSM Directive) | S3 bulk | Electrical engineering & systems science papers | emc-dsp-power |
| KiCad Documentation | https://docs.kicad.org | CC-BY-SA-4.0 | Official docs | KiCad EDA official documentation | kicad-dsl, kicad-pcb |
| CircuitSnips | https://circuitsnips.io | CERN-OHL-S-2.0 (verified) | 4300 schematics | Open hardware schematics | electronics, kicad-pcb |

### 4.4 Per-Domain Training Data Inventory

| Domain | Records (train) | Source(s) | SPDX License(s) | Model Target |
|--------|----------------|-----------|-----------------|-------------|
| chat-fr | 1,351 | Aya Dataset (fra) | Apache-2.0 | EuroLLM |
| cpp | 2,850 | CommitPackFT + ESP-IDF + STM32Cube + Arduino | MIT + Apache-2.0 + BSD-3-Clause + CC0-1.0 | Devstral |
| docker-devops | 1,656 | StackOverflow K8s | CC-BY-SA-4.0 | Devstral |
| electronics | 90 (PDF) | Datasheets + Wikipedia extracts | CC-BY-SA-3.0 AND ST-SLA0048 | Apertus (supplement) |
| embedded | 2,850 | OSHWA + Arduino docs + synthetic | Apache-2.0 AND CC-BY-SA-4.0 | Apertus |
| emc-dsp-power | 2,850 | arXiv eess + Wikipedia | arXiv-TDM-DSM-Art4 AND CC-BY-SA-3.0 | Apertus |
| freecad | 62 | FreeCAD macros | MIT AND CC-BY-4.0 AND CC0-1.0 | Apertus |
| html-css | 2,850 | Code Instructions + CodeAlpaca | Apache-2.0 + CC-BY-4.0 | Devstral |
| iot | 1,835 | TokenBender + embedded overlap | Apache-2.0 | Apertus |
| kicad-dsl | 7,694 | KiCad symbols + demos | CC-BY-SA-4.0 | Apertus |
| kicad-pcb | 11,288 | KiCad footprints | CC-BY-SA-4.0 | Apertus |
| llm-ops | 1,379 | ZenML LLMOps | Apache-2.0 | Devstral |
| llm-orch | 1,740 | TokenBender + ZenML | Apache-2.0 | Devstral |
| lua-upy | 831 | MicroPython + Lua HF | MIT + Apache-2.0 | Devstral |
| math-gsm8k | 2,850 | GSM8K | MIT | Apertus |
| math-reasoning | 2,850 | Orca-Math | MIT | Apertus |
| ml-training | 2,850 | CommitPackFT (Python+ML) | MIT | Devstral |
| multilingual-eu | 2,850 | Aya Dataset (EU langs) | Apache-2.0 | EuroLLM |
| music-audio | 475 | TokenBender + synthetic | Apache-2.0 | Apertus |
| platformio | 665 | PlatformIO examples + synthetic | Apache-2.0 | Apertus |
| python | 2,850 | StarCoder2 Self-Instruct | Apache-2.0 | Devstral |
| rust | 2,850 | StarCoder2 Self-Instruct | Apache-2.0 | Devstral |
| rust-embedded | 1,501 | Embassy + RTIC + cortex-m + esp-hal + defmt + embedded-hal | MIT + Apache-2.0 + 0BSD | Devstral |
| security-fenrir | 2,850 | Cybersecurity Fenrir v2 | Apache-2.0 | Apertus |
| shell | 2,850 | CommitPackFT + Linux Commands | MIT + CC-BY-4.0 | Devstral |
| spice-sim | 475 | ngspice + synthetic | BSD-3-Clause | Apertus |
| sql | 2,850 | Synthetic Text-to-SQL + StarCoder2 | Apache-2.0 | Devstral |
| stm32 | 90 (PDF) | STM32 datasheets | ST-SLA0048 (TDM) | Apertus (supplement) |
| traduction-tech | 2,850 | Europarl FR-EN | CC-BY-4.0 | EuroLLM |
| typescript | 2,850 | StarCoder2 Self-Instruct | Apache-2.0 | Devstral |
| web-backend | 2,850 | TokenBender + bigcode | Apache-2.0 | Devstral |
| web-frontend | 2,850 | TokenBender + bigcode | Apache-2.0 | Devstral |
| yaml-json | 2,850 | TokenBender + K8s-SO | Apache-2.0 + CC-BY-SA-4.0 | Devstral |
| **TOTAL** | **82,432** | | | |

**PDF-supplement domains (2):** stm32, electronics — extracted from datasheets, no dedicated LoRA adapter.

**Domains in progress (3):** misra-c, normes-iec, doc-technique-ce — synthetic generation planned.

### 4.5 PII Scan Results

| Scan Date | Scan Tool | Files Scanned | Findings | Action |
|-----------|-----------|--------------|----------|--------|
| 2026-04-28 | Presidio (Microsoft) + en_core_web_lg | 21 JSONL files | 1 email address in `traduction-tech` | Redacted and replaced |
| 2026-04-28 | Presidio (Microsoft) + en_core_web_lg | ALL 35+ domains (hf-traced + scraped) | Re-scan after provenance fix | See pii-scan-report.json |

Coverage: All directories under `data/hf-traced/` (35 domain dirs) and `data/scraped/` (8 dirs) are now scanned.
No high-signal PII (email, phone, credit card, SSN, IBAN) detected outside the previously-redacted traduction-tech finding.
Low-signal detections (PERSON, LOCATION, DATE_TIME) are common false positives in technical text and do not constitute PII risk.

---

## EU AI Act PST Compliance Checklist

- [x] Public Summary Template (PST) documented
- [x] Provenance per sub-corpus documented
- [x] **Per-record _provenance fields** — DONE v0.3.0 (49,956 records across 21 domains, 9 domains already had provenance)
- [x] Synthetic data marked as synthetic
- [x] **License verification per dataset (SPDX IDs verified)** — DONE v0.3.0 (non-SPDX values corrected: embedded, emc-dsp-power, traduction-tech, cpp, freecad, rust-embedded)
- [x] JWT secret management — DONE (moved from hardcoded to env var `EU_KIKI_JWT_SECRET`)
- [x] **PII removal verification** — DONE (Presidio scan, ALL 35+ domains scanned, 1 email redacted in traduction-tech)
- [x] MathInstruct license issue — DONE (replaced with `microsoft/orca-math-word-problems-200k`, MIT)
- [x] web-backend/yaml-json provenance — DONE (quarantined, then rebuilt with traced data from TokenBender/code_instructions_122k Apache-2.0 + K8s-SO CC-BY-SA-4.0)
- [x] **Scraping manifests** — DONE v0.3.0 (manifest.json created for all 8 scraped directories with source_url, license, legal_basis, robots_status)
- [x] **Model config licenses** — DONE v0.3.0 (license field added to all 7 model config.json files)
- [x] **rust-strand phantom entry** — DONE v0.3.0 (annotated as deprecated in MANIFEST.json)
- [x] **stm32/electronics PDF-supplement** — DONE v0.3.0 (added to MANIFEST_niche.json and inventory)
- [ ] Opt-out/robots.txt verification (pending for scraped sources)
  - TODO: Verify robots.txt compliance for each non-API scraped source at collection time
  - TODO: Implement opt-out mechanism for content owners (contact email + removal script)

---

## 4-LEGACY. Training Data (Original — Deprecated)

> **Deprecated.** The information below describes the initial bootstrap dataset from KIKI-Mac_tunner.
> It has been superseded by section 4 above. Retained for audit continuity only.

### 4-LEGACY.1 Source: KIKI-Mac_tunner classified dataset

### 4-LEGACY.1 Source: KIKI-Mac_tunner classified dataset

| Field | Value |
|-------|-------|
| **Location** | `~/KIKI-Mac_tunner/data/micro-kiki/classified/` |
| **Total examples** | 57,632 |
| **Format** | JSONL, `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}` |
| **Domains** | 32 classified domains |
| **Origin** | Mixed synthetic + distilled from Claude Opus 4.6, Qwen 3.5, Mistral Large |
| **Curation** | Domain classification via MiniLM embeddings + manual review |
| **Languages** | ~60% English, ~30% French, ~10% other EU languages |
| **PII** | No PII included (synthetic generation) |
| **Copyright** | Synthetic data — no copyrighted material |

### 4-LEGACY.2 Domain breakdown

| Domain | Examples | Model Target | Notes |
|--------|----------|-------------|-------|
| chat-fr | 3,000 | EuroLLM | French conversational |
| cpp | 3,000 | Devstral | C++ coding |
| devops | 3,000 | Devstral | Infrastructure/CI |
| docker | 3,000 | Devstral | Container orchestration |
| dsp | 169 | Apertus | Digital signal processing |
| electronics | 1,439 | Apertus | Hardware engineering |
| embedded | 3,000 | Apertus | Embedded systems |
| emc | 210 | Apertus | Electromagnetic compatibility |
| freecad | 2 | Apertus | CAD modeling (sparse) |
| html-css | 3,000 | Devstral | Web frontend |
| iot | 268 | Apertus | IoT devices |
| kicad-dsl | 2,003 | Apertus | KiCad design |
| kicad-pcb | 284 | Apertus | PCB design |
| llm-orch | 1,190 | Devstral | LLM orchestration |
| lua-upy | 3,000 | Devstral | Lua/MicroPython |
| math | 3,000 | Apertus | Mathematics |
| music-audio | 225 | Apertus | Audio/music tech |
| platformio | 32 | Apertus | PlatformIO embedded (sparse) |
| power | 948 | Apertus | Power electronics |
| python | 3,000 | Devstral | Python coding |
| reasoning | 3,000 | Apertus | General reasoning |
| rust | 3,000 | Devstral | Rust coding |
| security | 3,000 | Apertus | Cybersecurity |
| shell | 2,136 | Devstral | Shell/bash scripting |
| spice | 3,000 | Apertus | SPICE simulation |
| spice-sim | 593 | Apertus | SPICE simulation (alt) |
| sql | 3,000 | Devstral | Database queries |
| stm32 | 29 | Apertus | STM32 firmware (sparse) |
| typescript | 3,000 | Devstral | TypeScript coding |
| web-backend | 1,896 | Devstral | **QUARANTINED then RESOLVED** — rebuilt with traced data (TokenBender Apache-2.0) |
| web-frontend | 748 | Devstral | Frontend web dev |
| yaml-json | 460 | Devstral | **QUARANTINED then RESOLVED** — rebuilt with traced data (TokenBender Apache-2.0 + K8s-SO CC-BY-SA-4.0) |

### 4-LEGACY.3 New EU-specific domains (planned)

| Domain | Status | Data Source |
|--------|--------|-------------|
| misra-c | Planned | Synthetic from MISRA-C:2012 guidelines |
| autosar-cert | Planned | Synthetic from AUTOSAR/CERT-C standards |
| doc-technique-ce | Planned | Synthetic technical documentation templates |
| calcul-normatif | Planned | Synthetic engineering calculations per EN standards |
| normes-iec | Planned | Synthetic from IEC 61508/62443 guidelines |

---

## 5. LoRA Training Configuration

| Parameter | Apertus-70B | Devstral-24B | EuroLLM-22B |
|-----------|-------------|-------------|-------------|
| LoRA rank | 16 | 16 | 16 |
| LoRA alpha | 32 | 32 | 32 |
| Dropout | 0.05 | 0.05 | 0.05 |
| Learning rate | 1e-5 | 1e-5 | 1e-5 |
| Batch size | 1 | 1 | 1 |
| Grad accumulation | 8 | 4 | 4 |
| Max seq length | 1024 | 2048 | 2048 |
| Precision | BF16 | BF16 | BF16 |
| Framework | MLX (mlx_lm_fork) | MLX (mlx_lm_fork) | MLX (mlx_lm_fork) |
| Target modules | All linear layers | All linear layers | All linear layers |

---

## 6. Router Training

| Field | Value |
|-------|-------|
| **Architecture** | MLP: Linear(dim, 256) → GELU → Dropout(0.1) → Linear(256, 32) → Sigmoid |
| **Training data** | 46,100 train / 11,532 valid (from classified dataset) |
| **Loss** | CrossEntropyLoss with inverse-frequency class weights |
| **Optimizer** | AdamW (lr=1e-3, weight_decay=1e-4) |
| **Epochs** | 30 |
| **Accuracy** | Top-1: 64.8%, Top-3: 84.9% (MiniLM bootstrap) |

---

## 7. Infrastructure

| Component | Provider | Country | License |
|-----------|----------|---------|---------|
| ML framework | Apple MLX | USA 🇺🇸 | Apache 2.0 |
| Hardware | Apple Silicon M3 Ultra | USA 🇺🇸 | Proprietary HW |
| Serving | FastAPI + Uvicorn | Open source | MIT / BSD |
| Metrics | Prometheus client | Open source | Apache 2.0 |

**Note:** MLX framework and Apple Silicon hardware are US-origin. This is unavoidable for local Apple Silicon deployment. All model weights, training data, and embedding models are EU-sourced.

---

## 8. Risk Assessment

| Risk | Level | Mitigation |
|------|-------|-----------|
| Hallucination | Medium | Not deployed in safety-critical contexts |
| Bias in training data | Medium | Synthetic data, no user data |
| PII in outputs | Low | No PII in training data, local deployment |
| Copyright infringement | Low | All synthetic training data |
| Model memorization | Low | Apertus uses Goldfish objective |

---

## 8.bis Evaluation summary (Art. 53(1)(d))

Full results, methodology, and reproduction scripts:
[`eval/results/SUMMARY.md`](../eval/results/SUMMARY.md),
[`eval/WORKFLOW.md`](../eval/WORKFLOW.md), and the consolidated model
card [`MODEL_CARD.md`](../MODEL_CARD.md).

Headline metrics (all measured on this hardware, all reproducible):

| Bench | Subject | Result |
|---|---|---|
| HumanEval+ (Linux EvalPlus) | Devstral 24B 4-bit base | 87.20 / 82.90 |
| HumanEval+ | Devstral + python adapter | −1.80 HE+ |
| HumanEval+ | Devstral + cpp adapter | −1.22 HE base (custom scorer) |
| HumanEval+ | Devstral + rust adapter | −0.61 HE base (custom scorer) |
| MT-Bench (full 80q) | Devstral 24B 4-bit base | 8.892/10 (37/160 turns parseable) |
| GSM8K 5-shot, n=200 | Qwen 35B-A3B-4bit base | 94.5 % |
| GSM8K | + reasoning fused | 0 |
| GSM8K | + math fused | −4.5 |
| KIKI-DSL v3 (custom, 15 prompts) | Qwen 35B-A3B-4bit base | 73.3 % pass / 0.704 avg |
| KIKI-DSL v3 | + reasoning fused | +13.4 pass |
| KIKI-DSL v3 | + math fused | +6.7 pass |
| KIKI-DSL v3 | + kicad-dsl narrow fused | −27 pass (negative transfer) |

**Honest limitations disclosed (Art. 53(1)(d) requires faithful summary):**

1. The KIKI-DSL v1 test set was biased (named-IC heavy); v3 corrects it.
   On v3, `chat-fr` loses its v1 +10 win (artifact), and domain-narrow
   adapters show real but reduced regressions (−20 to −27 vs −30 on v1).
2. The cognitive cluster (`reasoning`, `math`) wins on KIKI-DSL v3 but
   does **not** transfer to public GSM8K (0 and −4.5 deltas). Saturated
   public-bench performance is the base model's, not the adapter's.
3. eu-kiki Devstral adapters slightly **degrade** HumanEval+ (style
   mismatch). They are intended for chat-style production, not raw
   algorithmic completion benchmarks.

## 9. Contact

- **System operator:** L'Electron Rare (electron-rare)
- **Privacy requests:** See individual model providers
- **Apertus PII/copyright:** llm-privacy-requests@swiss-ai.org / llm-copyright-requests@swiss-ai.org

---

## 10. Changelog

| Date | Version | Change |
|------|---------|--------|
| 2026-04-27 | 0.1.0-dev | Initial documentation |
| 2026-04-28 | 0.1.1-dev | Quarantine web-backend/yaml-json (missing provenance); JWT secret resolved; PII/opt-out TODOs; verified zenml/llmops-database (Apache-2.0) and AYI-NEDJIMI/mlops-infrastructure-en (MIT) |
| 2026-04-28 | 0.1.2-dev | Replace TIGER-Lab/MathInstruct (unclear sub-source licenses) with microsoft/orca-math-word-problems-200k (MIT, 200K examples) for math-reasoning domain |
| 2026-04-28 | 0.2.0 | Comprehensive update: all 33 domains inventoried with SPDX licenses and record counts; added all scraped repo sources (ESP-IDF, STM32Cube, Arduino, Embassy, KiCad symbols/footprints, FreeCAD macros, OSHWA API); added Mistral Small 4 119B (pending eval) and Mistral Medium 3.5 128B (teacher/eval only, Modified MIT); PII scan completed (Presidio, 1 finding redacted); web-backend/yaml-json provenance resolved; Devstral BF16 dequantization source documented |
| 2026-05-05 | 0.4.1 | **Router v6 + ops hardening:** (1) Re-trained router head on rebuilt clean corpus (`data/router-clean/`, 9 967 rows × 32 domains, niche-augmented + greetings-grounded) — top-1 65.5 % → 87.7 %, top-3 85.3 % → 98.7 %; (2) `MLXWorkerRuntime.QUARANTINED_DOMAINS` introduced, EuroLLM `chat-fr` and `traduction-tech` adapters fall back to the base model after a chat-template-leakage bug surfaced as `"user user…"` loops in production (worker behaviour now openly reflects the disabled adapters); (3) router classifier patched: auto-device resolution (MPS / CUDA / CPU), `max_seq_length=128`, L1 LRU + L2 cosine ≥ 0.95 semantic cache, auto-prewarm at boot to kill p95 spikes; (4) production gateway runs as systemd unit `eu-kiki-gateway.service` on electron-server with worker URLs supplied via `EU_KIKI_WORKERS_JSON` env (Tailscale endpoints to Studio MLX workers). |
| 2026-05-05 | 0.4.0 | **Art. 53(1)(d) evaluation summary added** (§8.bis): published full benchmark suite (HumanEval+ Devstral base/python/cpp/rust, MT-Bench Devstral 8.89/10, GSM8K Qwen base/reasoning/math, KIKI-DSL v3 8 adapters). New model card at [`MODEL_CARD.md`](../MODEL_CARD.md). Honest limitations disclosed: v1→v3 test-set bias correction, no cognitive transfer to public GSM8K, slight HumanEval+ degradation by Devstral adapters. All result directories self-contained (`env.json` + `methodology.md` + `rerun.sh`); cross-platform pipeline (macM1 + studio + kx6tm-23) traced in [`eval/WORKFLOW.md`](../eval/WORKFLOW.md). |
| 2026-04-28 | 0.3.0 | **EU AI Act compliance remediation:** (1) Per-record `_provenance` added to 49,956 records across 21 HF-traced domains (source, SPDX license, record_idx, access_date); (2) Non-SPDX license values corrected in MANIFEST_niche.json: embedded (Apache-2.0 AND CC-BY-SA-4.0), emc-dsp-power (arXiv-TDM-DSM-Art4 AND CC-BY-SA-3.0), traduction-tech (CC-BY-4.0), cpp sub-sources (removed "Open"/"various"), freecad (MIT AND CC-BY-4.0 AND CC0-1.0), rust-embedded "various" sub-source (MIT AND Apache-2.0); (3) PII scan extended to ALL 35+ domains; (4) Scraping manifests (manifest.json) created for all 8 scraped directories; (5) License field added to all 7 model config.json files; (6) stm32 and electronics added as PDF-supplement domains; (7) rust-strand annotated as deprecated phantom entry |
