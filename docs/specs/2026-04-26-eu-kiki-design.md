# EU-KIKI Design Specification

**Date:** 2026-04-26
**Status:** Draft
**Author:** Clems + Claude

## 1. Vision

EU-KIKI is a 100% EU-sovereign LLM serving pipeline running on a single Mac Studio M3 Ultra 512GB. It routes user requests to one of three specialized European open-source models, each enhanced with domain-specific LoRA adapters. The system follows the same proven pattern as micro-kiki (single-model LoRA routing) but extends it to multi-model dispatch with full European data sovereignty.

## 2. Goals

- **EU sovereignty**: All models, embeddings, and training data from EU/CH sources. Zero non-EU model dependencies.
- **Multi-model routing**: Intelligent dispatch to the best model for the task (reasoning, code, or multilingual).
- **Domain specialization**: 39 LoRA adapters across 3 models, covering technical engineering + EU-specific domains.
- **Local-first**: Entire stack runs on a single machine, no cloud dependencies.
- **OpenAI-compatible API**: Drop-in replacement via `/v1/chat/completions`.

## 3. Models

### 3.1 Apertus-70B (Reasoning & Hardware EU)

- **Origin:** Swiss AI Initiative (EPFL + ETH Zurich + CSCS)
- **Architecture:** 70B dense transformer, custom (xielu activation, QK-norm, GQA)
- **Context:** 65,536 tokens
- **License:** Apache 2.0
- **Languages:** 1,811 natively supported
- **RAM BF16:** ~141 GB
- **Role:** Technical reasoning, hardware engineering, EU normative domains
- **MLX status:** Inference supported (4bit/8bit quants available). xielu activation + `models/apertus.py` must be implemented in mlx-lm fork.

### 3.2 Devstral Small 2 24B (Code)

- **Origin:** Mistral AI (Paris, France)
- **Architecture:** 24B dense transformer (Mistral arch)
- **Context:** 256,000 tokens
- **License:** Apache 2.0
- **Benchmark:** 68.0% SWE-bench Verified
- **RAM BF16:** ~48 GB
- **Role:** Code generation, agentic coding, SWE tasks
- **MLX status:** Fully supported (Mistral architecture native in mlx-lm).

### 3.3 EuroLLM-22B (EU Languages)

- **Origin:** EU Consortium (Horizon Europe, ERC, EuroHPC), trained on MareNostrum 5
- **Architecture:** 22B dense transformer, custom (SwiGLU, GQA, RoPE)
- **Context:** 32,768 tokens
- **License:** Apache 2.0
- **Languages:** 35 (all 24 EU official + 11 additional)
- **RAM BF16:** ~45 GB
- **Role:** Technical translation, multilingual documentation, EU language tasks
- **MLX status:** Custom architecture — `models/eurollm.py` to verify/implement in mlx-lm fork.

### 3.4 Jina Embeddings v3 (Router)

- **Origin:** Jina AI (Berlin, Germany)
- **Architecture:** 570M params, 1024-dim embeddings
- **Context:** 8,192 tokens
- **License:** Apache 2.0
- **Role:** Encode user queries for domain classification router
- **Replaces:** all-mpnet-base-v2 (Microsoft) from micro-kiki

## 4. Architecture

### 4.1 Overview

```
Client ──▶ Gateway (:9200) ──▶ Router (Jina v3 + MLP)
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
           Apertus Worker    Devstral Worker   EuroLLM Worker
              (:9201)           (:9202)           (:9203)
           + 20 LoRA           + 15 LoRA        + 4 LoRA
```

### 4.2 Gateway (port 9200)

- FastAPI/Uvicorn process
- Exposes OpenAI-compatible `/v1/chat/completions`
- Encodes incoming request with Jina v3 (1024d embedding)
- Runs MLP router head: 1024d → 512 hidden → 39 sigmoid outputs
- Maps predicted domain to worker via static `DOMAIN_TO_WORKER` table
- Forwards request to worker with `X-Lora-Domain` header
- Streams response back to client
- Exposes `/health`, `/metrics` (Prometheus), `/v1/models`

### 4.3 Workers (ports 9201-9203)

Each worker is an independent uvicorn process:
- Loads 1 MLX model at startup (BF16 or quantized)
- Pre-loads its LoRA adapter pool into memory
- Exposes `/v1/chat/completions` (OpenAI-compat)
- Reads `X-Lora-Domain` header to select active LoRA
- Hot-swaps LoRA via PEFT `set_adapter()` / MLX equivalent
- Exposes per-worker `/health` and `/metrics`
- Independent crash/restart without affecting other workers

### 4.4 Memory Layout (BF16, all 3 models)

| Component           | RAM       |
|---------------------|-----------|
| Apertus-70B BF16    | 141 GB    |
| Devstral Small 2    | 48 GB     |
| EuroLLM-22B BF16    | 45 GB     |
| Jina v3 router      | 1 GB      |
| 39 LoRA adapters    | ~20 GB    |
| KV caches (3 models)| ~30 GB    |
| OS + processes       | ~15 GB    |
| **Total serving**   | **~300 GB** |
| **Margin (training)**| **~212 GB** |

Apple Silicon unified memory: all processes share the same physical memory pool. No duplication across workers.

Metal GPU compute is serialized (one active inference at a time), but tokenization, LoRA switching, and request queuing happen concurrently across workers.

## 5. Domains (39 total)

### 5.1 Apertus-70B Domains (~20)

**Existing (from micro-kiki):**
electronics-hw, emc, dsp, spice, kicad, stm32, platformio, iot, embedded, math, reasoning, security, music-audio, freecad, power

**New EU-specific:**
- `misra-c` — MISRA-C/C++ automotive coding standard generation
- `autosar-cert` — AUTOSAR, CERT-C, IEC 62443 cybersec industrial code
- `doc-technique-ce` — Technical documentation for CE marking (FR/EN/DE)
- `calcul-normatif` — Engineering calculations per EN/IEC standards
- `normes-iec` — IEC 61508 functional safety, IEC 62443 cybersecurity

### 5.2 Devstral Small 2 Domains (~15)

python, rust, typescript, cpp, shell, html-css, sql, web-backend, web-frontend, docker, devops, yaml-json, llm-ops, llm-orch, ml-training, lua-upy

### 5.3 EuroLLM-22B Domains (~4)

- `chat-fr` — French conversational
- `traduction-tech` — Technical translation between EU languages
- `redaction-multilingue` — Multilingual technical writing (reports, specs)
- `localisation-doc` — Documentation localization (i18n, cultural adaptation)

## 6. Router Design

### 6.1 Embedding

- Model: `jinaai/jina-embeddings-v3` (570M, 1024d)
- Input: last user message content
- Output: 1024-dimensional dense vector

### 6.2 Classification Head

- Architecture: Linear(1024, 512) → ReLU → Linear(512, 39) → Sigmoid
- Training: CrossEntropyLoss with inverse-frequency class weights
- Dataset: Extended from micro-kiki router data + new EU domain examples
- Target: >70% top-1 accuracy, >90% top-3

### 6.3 Domain-to-Worker Mapping

Static Python dict, updated at deploy time:
```python
DOMAIN_TO_WORKER = {
    # Apertus (:9201)
    "electronics-hw": 9201, "emc": 9201, "dsp": 9201,
    "spice": 9201, "kicad": 9201, "stm32": 9201, ...
    # Devstral (:9202)
    "python": 9202, "rust": 9202, "typescript": 9202, ...
    # EuroLLM (:9203)
    "chat-fr": 9203, "traduction-tech": 9203, ...
}
```

## 7. MLX Implementation Requirements

### 7.1 xielu Activation (Apertus)

Custom activation with 2 learnable parameters per layer:
```
x > 0:  f(x) = softplus(log_alpha_p) * x^2 + beta * x
x <= 0: f(x) = (softplus(log_alpha_n) + beta) * (exp(min(x, eps)) - 1 - x) + beta * x
```

All required MLX ops exist (`mx.where`, `mx.softplus`, `mx.exp`, `mx.minimum`). ~30 lines of code.

### 7.2 models/apertus.py

Based on Llama architecture with modifications:
- xielu activation instead of SiLU in MLP (SwiGLU → xIELU-GLU)
- QK-norm (RMSNorm on Q and K before attention)
- vocab_size=131072
- Dense model (no MoE complexity)

Estimated effort: copy `models/llama.py`, modify MLP and attention. ~200-300 lines.

### 7.3 models/eurollm.py

EuroLLM uses custom architecture (SwiGLU, GQA, RoPE). Likely very similar to Llama. Needs verification — may work with existing `llama.py` if `model_type` mapping is added.

### 7.4 Devstral Small 2

Mistral architecture, already supported in mlx-lm. No work needed.

## 8. Training Strategy

### 8.1 LoRA Configuration

Same approach as micro-kiki — LoRA on attention projections (q_proj, k_proj, v_proj, o_proj):

| Model | LoRA rank | LoRA alpha | LR | Precision |
|-------|-----------|------------|-----|-----------|
| Apertus-70B | 48 | 96 | 1.5e-5 | BF16 |
| Devstral Small 2 | 16 | 16 | 1e-5 | BF16 |
| EuroLLM-22B | 16 | 16 | 1e-5 | BF16 |

### 8.2 Training Data Sources

- **Existing:** micro-kiki classified domain data (46K+ train examples)
- **New EU domains:** Synthetic generation from EU normative texts + expert-written examples
- **Translation:** EuroLLM trained on parallel corpora from EUR-Lex, OPUS, Europarl
- **Router:** Augmented dataset from micro-kiki + new EU domain examples, re-embedded with Jina v3

### 8.3 Training Schedule

Training can happen while serving (212 GB margin). Kill one worker at a time for its domain training, restart after.

## 9. API Specification

### 9.1 Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI-compatible chat (auto-routed) |
| `/v1/chat/completions` | POST | With `model: "eu-kiki-apertus"` to force model |
| `/v1/models` | GET | List available models and domains |
| `/health` | GET | Gateway + all workers health |
| `/metrics` | GET | Prometheus metrics (per-worker, per-domain) |

### 9.2 Model Aliases

- `eu-kiki` — auto-routed (default)
- `eu-kiki-apertus` — force Apertus-70B
- `eu-kiki-devstral` — force Devstral Small 2
- `eu-kiki-eurollm` — force EuroLLM-22B

## 10. Project Structure

```
eu-kiki/
├── src/
│   ├── gateway/
│   │   ├── server.py          # FastAPI gateway, routing logic
│   │   ├── router.py          # Jina v3 + MLP domain classifier
│   ��   └── config.py          # Domain-to-worker mapping
│   ├── worker/
│   │   ├── server.py          # Worker FastAPI (1 model, N LoRA)
│   │   ├── mlx_runtime.py     # MLX model loading + LoRA hot-swap
│   │   └── config.py          # Per-worker config
│   └── mlx_models/
│       ├── apertus.py         # Apertus MLX implementation (xielu + QK-norm)
│       ├── xielu.py           # xIELU activation for MLX
│       └── eurollm.py         # EuroLLM MLX implementation (if needed)
├── configs/
│   ├── gateway.yaml           # Router weights, worker endpoints
│   ├── apertus-worker.yaml    # Model path, LoRA dir, domains
│   ├── devstral-worker.yaml
│   └── eurollm-worker.yaml
├── scripts/
│   ├── start.sh               # Launch all workers + gateway
│   ├── train_router.py        # Router training (Jina v3 + MLP)
│   ├── train_lora.py          # LoRA adapter training
│   └── build_router_data.py   # Dataset preparation
├── data/
│   └── router/                # Train/valid JSONL for router
├── output/
│   ├── router/                # Router weights (safetensors + meta)
│   └── adapters/              # Per-model LoRA adapters
│       ���── apertus/
│       ├── devstral/
│       └── eurollm/
├── tests/
├── docs/
│   └── specs/
│       └── 2026-04-26-eu-kiki-design.md  # This file
├── pyproject.toml
├── CLAUDE.md
└── README.md
```

## 11. Sovereignty Audit

| Component | Provider | Country | License |
|-----------|----------|---------|---------|
| Reasoning model | Swiss AI (EPFL/ETH) | Switzerland | Apache 2.0 |
| Code model | Mistral AI | France | Apache 2.0 |
| Language model | EU Consortium | EU | Apache 2.0 |
| Router embeddings | Jina AI | Germany | Apache 2.0 |
| ML framework | Apple MLX | USA * | Apache 2.0 |
| Hardware | Apple Silicon | USA * | — |

*\* MLX and hardware are US — unavoidable for local Apple Silicon deployment. All model weights, training data, and embeddings are EU-sourced.*

## 12. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| xielu not in mlx-lm | Blocks Apertus | Implement (~1h), all ops exist in MLX |
| EuroLLM arch incompatible with mlx-lm | Blocks EuroLLM | Likely works as Llama variant; verify first |
| Router accuracy drop with Jina vs MiniLM | Worse routing | Retrain with more data; Jina is higher quality |
| Metal serializes GPU across processes | No true parallel inference | Pre-process in parallel; acceptable for 3 models |
| Apertus-70B slow inference (dense 70B) | High latency | BF16 quality tradeoff; no quantization needed |
| New EU domains lack training data | Weak LoRA adapters | Synthetic generation + distillation from experts |
