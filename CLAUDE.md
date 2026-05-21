# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Context

**ailiance** is the EU-sovereign multi-model LLM serving pipeline of L'Electron Rare. The system routes OpenAI-compatible requests across **11 named backends + 13 mascarade hardware-specialist LoRA experts** via a 47-domain MiniLM classifier (router v7), with a FastAPI gateway deployed as the `ailiance-gateway.service` systemd unit on `electron-server`, and public exposure on `https://gateway.ailiance.fr` (Cloudflare Tunnel).

The brand was carved out of `L-electron-Rare` on **2026-05-11** into the dedicated GitHub org [`ailiance`](https://github.com/ailiance); domain `ailiance.fr` was verified the same day. HuggingFace models live under [`Ailiance-fr`](https://huggingface.co/Ailiance-fr) (10 models + 13 datasets; 20/20 models Apache-2.0).

### Repo path note (legacy)

The local clone directory is still `~/Documents/Projets/eu-kiki/` for historical reasons. The upstream remote points at `ailiance/ailiance` (org carve-out). The local path is **slated for rename** to `~/Documents/Projets/ailiance/`; the rename is documented but **out of scope for this branch** (filesystem ops, requires updating IDE workspaces + symlinks + a couple of helper scripts).

### Service rename (cutover 2026-05-11)

- Active unit: **`ailiance-gateway.service`** (FastAPI, port 9300, EnvFile carries `AILIANCE_WORKERS_JSON`).
- Legacy unit: `eu-kiki-gateway.service` is `disabled` + `inactive` and kept on disk as a rollback path.
- Production filesystem on electron-server moved `/home/electron/eu-kiki/` → `/home/electron/ailiance/`.

## Architecture

```
client → gateway.ailiance.fr (Cloudflare Tunnel 2c6b04a3…)
       → ailiance-gateway.service (electron-server :9300, FastAPI)
       → router v7 (MiniLM-L6-v2 384d + MLP 256→47, sigmoid multi-label)
       → backend selection:
           ├─ 11 named workers (Studio MLX / macM1 MLX / Tower llama.cpp / kxkm-ai llama.cpp)
           └─ 13 mascarade LoRA experts (MacStudio MLX bf16 :9340, PR #100)
       → FIFO per-worker_url asyncio.Lock (PR #68, prevents head-of-line blocking)
       → SSE token stream back to client
```

### Router v7 (live since 2026-05-12, PR #77 `133a9b5`)

| Property | Value |
|---|---|
| Encoder | `sentence-transformers/all-MiniLM-L6-v2` (384d, 22M params) |
| Head | 384 → 256 → 47 MLP, sigmoid multi-label, threshold 0.50 |
| Labels | **47 domains** (`output/router-prod/meta.json`) |
| Training corpus | **5 696 examples from 14 LLMs** (multi-model adversarial sampling) |
| Top-1 / Top-3 | **0.8895 / ~0.98** (was 0.4862 / ~0.85 on v6 base corpus) |
| Δ vs v6 | +0.4033 top-1 |
| Caches | L1 LRU 1024 (~0.01 ms hit) + L2 cosine ≥0.95 (~0.2 ms hit) + auto-prewarm |
| Smart truncation | short→full · medium→128-tok left-trunc · long (>1000 chars)→head 256 + tail 256 |
| Disk footprint | ~88 MB (MiniLM 88 MB + MLP head 436 KB) |
| Active checkpoint | `output/router-prod/` stable symlink → `router-v7-multimodel` |

Note: Jina v3 was evaluated as router-v6 candidate and **rejected on bench** (top-1 0.874 vs 0.876, encode 9.7 vs 1.6 ms/prompt, Δ separation 0.15 vs 0.34). Cache `models--jinaai--*` may still be present on electron-server but is **not loaded**.

### 11 named backends (gateway aliases)

| Alias | Model | Host:Port | Quant | Notes |
|---|---|---|---|---|
| `ailiance` | Auto-router (default) | — | — | Picks best backend per router v7 + override map |
| `ailiance-mistral-medium` | Mistral-Medium-3.5-128B-Instruct | studio :9301 | MLX Q8 (~130 GB) | **Main heavyweight** (replaces decommissioned Apertus 70B) |
| `ailiance-mistral` | Mistral-Small-3.1-24B-Instruct | studio :9326 | MLX 4-bit (~13 GB) | launchd auto-restart |
| `ailiance-gemma` | Gemma 3 4B IT (GGUF Q4_K_M) | tower :9304 | llama.cpp | n_ctx_train=131072 |
| `ailiance-gemma2` | Gemma 3 4B IT (MLX 4-bit) | macm1 :8502 | MLX 4-bit | Hot-swap on `:8502` server |
| `ailiance-gemma4` | Gemma-4-E4B-it-MLX-4bit + LoRA `gemma4-e4b-eukiki` | macm1 :8502 | MLX 4-bit | **Default on macm1 :8502** |
| `ailiance-eurollm` | EuroLLM-22B-Instruct-2512 | studio :9303 | MLX BF16 (~45 GB) | UP since 2026-05-12 |
| `ailiance-qwen` | Qwen3.5-9B-MLX-4bit | macm1 :8502 | MLX 4-bit | Hot-swap |
| `ailiance-granite` | granite-4.1-30b-4bit | macm1 :8502 | MLX 4-bit | Hot-swap |
| `ailiance-ministral` | Ministral-3-14B-Instruct-2512-4bit | macm1 :8502 | MLX 4-bit | Hot-swap |
| `ailiance-ministral-reasoning` | Ministral-3-14B-Reasoning-2512-4bit | macm1 :8502 | MLX 4-bit | Hot-swap, needs `max_tokens≥2048` |

Plus complementary auxiliaries used internally / via direct alias on the router map: `ailiance-reasoning-r1` (DeepSeek-R1-Distill-Qwen-32B :9323 MLX 4bit), `ailiance-llama` (Llama-3.3-70B :9324 MLX 4-bit), Pixtral-12B vision worker :9325, Qwen3-Coder-30B :9327, plus `kxkm-ai` heavyweights: Qwen3-Next 80B MoE :18888 (via autossh tunnel `electron-server:8002`) and Granite-4.1-30B :18889 (via `:8003`).

### 13 mascarade LoRA experts (hardware specialists)

Served from MacStudio MLX bf16 (port `:9340`) since the **2026-05-18 cutover (PR #100)** — Qwen3-4B base + 10 mascarade LoRAs merged + converted to MLX bf16, with no quantization loss. Routed by the 9-domain `MASCARADE_DOMAINS` override (see "Domain routing overrides" below).

Aliases: `ailiance-{kicad, spice, stm32, emc, embedded, platformio, freecad, dsp, iot, power, components-review, coder, embed}`.

LoRAs trained from `ailiance-models-tuning` (Qwen3-4B-Instruct-2507, r=16 / α=32, 126–522 real steps with checkpoints, **all 10 hardware LoRAs really trained** — earlier "5/10 dirs empty" note was incorrect; audit 2026-05-18 confirmed checkpoints for all 10). Published on HF under [`Ailiance-fr`](https://huggingface.co/Ailiance-fr).

Legacy: the same LoRAs are still available as Tower-Ollama Q4_K_M `mascarade-*:latest` (tunnel `tower-ollama-tunnel.service` `electron-server:8004 → tower:11434`). Kept as rollback only — **production routing no longer hits Tower :8004** since PR #100.

### Domain routing overrides (live state)

| Domain(s) | Routes to | Source |
|---|---|---|
| `kicad`, `stm32`, `emc`, `embedded`, `platformio`, `freecad`, `dsp`, `iot`, `power` | MacStudio MLX bf16 `:9340` (mascarade) | PR #100 cutover 2026-05-18; replaces Tower Q4 `:8004` |
| `spice` | `ailiance-apertus` → fallback Apertus :9301 (now Mistral-Medium) | Removed from MASCARADE_DOMAINS PR #55 (2026-05-11) — bench shows −25 on spice-sim |
| `kicad-dsl`, `kicad-pcb` | macm1 `:8502` (eu-kiki Gemma-4 + LoRA) | PR #54 — Gemma-4 champion P1 bench (+55 DSL, +42 PCB) |

A request with a `tools[]` field is **force-routed to `qwen-32b-awq`** regardless of the classifier output, because Mistral-Medium 128B has no native function-calling and hallucinates XML otherwise (gateway hardcoded since 2026-05-12; see `reference_ailiance_gateway_tools_force_route`).

### Inference defaults registry

`src/gateway/inference_defaults.py` exposes a per-alias registry of caller-wins defaults (`temperature`, `max_tokens`, `top_p`, `repetition_penalty`, `stop`, `chat_template_kwargs`). When adding a new backend alias, also add an entry here. Notable entries:

- Reasoning models (`ailiance-reasoning-r1`, `ailiance-ministral-reasoning`, Gemma-3 thinking, Apertus math reasoning): `max_tokens ≥ 2048` (default 1024 truncates chain-of-thought).
- Qwen3.5 family (`ailiance-qwen`): inject `chat_template_kwargs.enable_thinking=false` for short outputs.
- Pixtral: low temperature, custom stop tokens to prevent `USER:` fabrication.
- Coder aliases: `temperature=0.2`.

## Repo layout

```
ailiance/                              # upstream org/repo (local path still ./eu-kiki/)
├── configs/                           # apertus.yaml (legacy), devstral.yaml, eurollm.yaml,
│   │                                  # gateway.yaml, gemma4.yaml, chain_policies.yaml,
│   │                                  # models-display.yaml, reflector_prompts.yaml
├── src/
│   ├── gateway/                       # FastAPI :9300, dispatch, FIFO lock per worker_url,
│   │                                  # tenant_isolation, alias_inventory, inference_defaults,
│   │                                  # observability (Prometheus + Langfuse)
│   ├── router/                        # MiniLM + MLP classifier v7, domain_map, L1+L2 caches
│   ├── worker/                        # 1 model / process, QUARANTINED_DOMAINS guard
│   └── mlx_models/                    # Apertus MLX impl + xielu activation (legacy refs)
├── scripts/                           # ~50 scripts (scrape, build, train, eval, router pipeline)
│   ├── scrape_*.py                    # OSHWA, arXiv, Wikipedia, Hackaday, Arduino, ESP-IDF, STM32, Rust, KiCad
│   ├── build_hf_datasets.py
│   ├── train_ailiance_batch.py
│   ├── rebuild_router_dataset.py + augment_niche_domains.py + augment_short_greetings.py
│   ├── build_router_data.py / encode_router_minilm.py / train_router_from_embeddings.py
│   ├── pdf_pipeline/ + scan_pii.py + fix_provenance.py
│   └── eval_framework.py / run_eval.sh / launch_eval_safe.sh
├── tools/router_v7/                   # multi-LLM corpus generators (PR #77)
├── vendored/iact-bench/               # submodule, real validators (v0.2.0)
├── data/
│   ├── hf-traced/                     # domain folders, train/valid JSONL + MANIFEST.json
│   ├── router/                        # train/valid (split from router-clean)
│   ├── router-clean/                  # per-domain JSONL (niche+greetings curated)
│   └── router-minilm-v7-multimodel/   # pre-encoded MiniLM embeddings (npy)
├── docs/
│   ├── eu-ai-act-transparency.md      # Master Art. 52/53 dossier
│   ├── transparency/
│   │   ├── router-training-data.md
│   │   ├── confusion-top10.md
│   │   ├── 2026-05-05-encoder-bench.log     # Jina v3 rejection record
│   │   └── 2026-05-19-phase1-ram-demotion.md
│   ├── router-mascarade-override-2026-05-11.md
│   ├── router-v0.3-deliberate.md
│   ├── pdf-compliance-report.md
│   └── provenance/                    # per-alias JSON (Annex IV §1(c))
├── tests/                             # pytest — gateway, router, worker, integration, runtime
├── pyproject.toml                     # Python ≥3.13, Apache-2.0
└── uv.lock
```

## Commands

```bash
# Setup
uv venv && uv pip install -e ".[dev,router,data]"

# Tests
uv run python -m pytest
uv run python -m pytest tests/test_router.py -v
uv run python -m pytest -k "test_name"

# Build datasets (HF-traceable, EU AI Act-compliant)
uv run python scripts/build_hf_datasets.py
uv run python scripts/scrape_oshwa.py

# Train LoRA adapters (sequential)
uv run python scripts/train_ailiance_batch.py

# Train router v7 (multi-model corpus → MLP head)
bash tools/router_v7/finalize.sh                       # full pipeline
# or step-by-step:
uv run python tools/router_v7/gen_corpus_multi.py
uv run python tools/router_v7/gen_augment.py
uv run python scripts/build_router_data.py
uv run python scripts/encode_router_minilm.py
uv run python scripts/train_router_from_embeddings.py \
    --emb-dir data/router-minilm-v7-multimodel \
    --hidden-dim 256 \
    --output-dir output/router-v7-multimodel

# Router QA
uv run python scripts/build_confusion_matrix.py
uv run python scripts/calibrate_threshold.py

# Local dev (launch gateway + workers in one process tree)
bash scripts/start.sh

# Production
# - Gateway: `ailiance-gateway.service` on electron-server (EnvFile carries AILIANCE_WORKERS_JSON)
# - Workers: launchd plists on studio (auto-restart, see ~/CLAUDE.md MacStudio block),
#   nohup mlx_lm.server on macm1, llama.cpp on tower + kxkm-ai
# - Router weights: `git pull` then repoint output/router-prod symlink, restart service

# Logs
ssh electron-server "sudo journalctl -u ailiance-gateway -f"
ssh studio "tail -f /tmp/ailiance-eurollm.log /tmp/ailiance-mistral-medium.log"
```

## Public endpoint

`https://gateway.ailiance.fr/v1/*` — OpenAI-compatible, **no auth** (live since 2026-05-12 11:47, via Cloudflare Tunnel `2c6b04a3…` → `localhost:9300`). 46 aliases enumerated at `/v1/models`. LiteLLM proxy on `electron-server:8800` sits in front for cost tracking (SpendLogs in local postgres).

```bash
curl -sN https://gateway.ailiance.fr/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"ailiance","messages":[{"role":"user","content":"Hello"}]}'
```

## EU AI Act compliance

- **Art. 52/53** — `docs/eu-ai-act-transparency.md` covers system, models, provenance, datasets, "limited risk" classification.
- **DSM Art. 4 TDM** — `docs/pdf-compliance-report.md` audits robots.txt + SHA-256 manifests for ST/Espressif/TI/NXP/KiCad PDFs.
- **Datasets HF-traceable** — `data/hf-traced/{domain}/MANIFEST.json` documents `hf_dataset_id`, license, download_date, `n_source_rows`, `n_used`.
- **All 20 HF models Apache-2.0**, full provenance. 13 datasets remain CC-BY-SA-4.0 / GPL-3.0 (Stack Exchange / KiCad upstream constraint).

## Key Design Decisions

- BF16 (Studio 512 GB unified memory) / 4-bit MLX (macm1 32 GB) / Q4_K_M GGUF (tower, kxkm-ai).
- One model per process (isolation, predictable memory).
- Sigmoid multi-label routing (domains overlap: `docker` ∩ `devops`, `embedded` ∩ `stm32`).
- LoRA on `q/k/v/o_proj` only.
- FIFO `asyncio.Lock` per `worker_url` (PR #68 `9440122`, prevents head-of-line blocking under load).
- Inference defaults registry per-alias (`src/gateway/inference_defaults.py`).
- Local-only deployment, no cloud, no telemetry.

## Notes & sister projects

- **`Ailiance-fr` on HuggingFace** is the product-distribution namespace. **`electron-rare`** on HF remains the IP source-of-truth.
- **`ailiance-mac-tuner`** (`~/Documents/Projets/ailiance-mac-tuner/`) — non-EU foundation distillation track. Some training scripts and configs mirror across.
- **`ailiance-bench`** ([github.com/ailiance/ailiance-bench](https://github.com/ailiance/ailiance-bench)) — Phase 6 scoreboard (7 tasks, eu-kiki champion 4/7, mascarade champion P3 +48, kicad9plus catastrophic forgetting −31 P3).
- **`iact-bench`** — vendored at `vendored/iact-bench` (v0.2.0). Real `IactBenchValidator` is the default; clients can opt out via `AILIANCE_VALIDATOR=stub` for local dev without docker.
- **kxkm-ai** is a *different machine* from `kx6tm-23` (Proxmox PVE host, no GPU). The two are conflated in some legacy notes — see corrected mapping in `docs/eu-ai-act-transparency.md` §2.7.

## Decommissioned / archived

- **Apertus 70B (source)** — deleted 2026-05-12 from Studio (~1.3 TB freed). `Apertus-70B-Instruct-2509-4bit-MLX` (37 GB) still on disk but **not served**. Alias `ailiance-apertus` is kept as back-compat redirect to `ailiance-mistral-medium`.
- **Devstral :9302** — decommissioned on macm1 (pre-2026-05-10). Devstral multi-LoRA :9330 on Studio is currently DOWN post-reboot 2026-05-12.
- **Tower-Ollama mascarade Q4 (:8004)** — legacy fallback since PR #100. Production no longer routes there.
- **`eu-kiki-gateway.service`** — disabled+inactive, kept for rollback.

## See also

- `~/CLAUDE.md` — cluster-wide infrastructure (electron-server / Studio / macm1 / Tower / kxkm-ai block).
- `~/.claude/projects/-Users-electron/memory/reference_ailiance_gateway_2026_05_11.md` — cutover record.
- `~/.claude/projects/-Users-electron/memory/reference_router_v7_2026_05_12.md` — v7 training details.
- `~/.claude/projects/-Users-electron/memory/reference_inference_defaults_registry.md` — per-alias defaults.
- `~/.claude/projects/-Users-electron/memory/reference_gateway_fifo_2026_05_12.md` — FIFO lock details.
