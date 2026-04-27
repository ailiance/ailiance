# EU AI Act — Transparency & Traceability Documentation

**Document ID:** EU-KIKI-TRANS-001
**Date:** 2026-04-27
**Version:** 0.1.0-dev
**System:** eu-kiki — EU-sovereign multi-model LLM serving pipeline
**Risk Classification:** Limited risk (general-purpose AI system, Article 52)

---

## 1. System Overview

EU-KIKI is a multi-model routing system that dispatches user queries to one of three EU-origin language models, each enhanced with domain-specific LoRA adapters. The system runs locally on a single machine (no cloud dependencies).

**Purpose:** Provide domain-specialized AI assistance using exclusively European AI models and infrastructure.

**Deployment:** Local-only, single-machine (Mac Studio M3 Ultra 512GB).

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

## 4. Training Data for LoRA Adapters

### 4.1 Source: KIKI-Mac_tunner classified dataset

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

### 4.2 Domain breakdown

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
| web-backend | 1,896 | Devstral | Backend web dev |
| web-frontend | 748 | Devstral | Frontend web dev |
| yaml-json | 460 | Devstral | Config file formats |

### 4.3 New EU-specific domains (planned)

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

## 9. Contact

- **System operator:** L'Electron Rare (electron-rare)
- **Privacy requests:** See individual model providers
- **Apertus PII/copyright:** llm-privacy-requests@swiss-ai.org / llm-copyright-requests@swiss-ai.org

---

## 10. Changelog

| Date | Version | Change |
|------|---------|--------|
| 2026-04-27 | 0.1.0-dev | Initial documentation |
