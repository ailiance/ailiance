# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Context

EU-KIKI is a 100% EU-sovereign multi-model LLM serving pipeline. It routes requests to **3 European/Swiss foundation models** via a small MiniLM domain classifier, each fine-tuned with LoRA adapters on HF-traceable datasets. Distributed deployment: workers on Mac Studio M3 Ultra (512 GB) MLX, gateway on `electron-server` (systemd, FastAPI), public exposure via Cloudflare Tunnel → `kiki-cockpit`.

## Architecture

```
client → ml.saillant.cc (Cloudflare Tunnel)
       → kiki-cockpit (electron-server :443, Traefik)
       → eu-kiki gateway (electron-server :9300, systemd `eu-kiki-gateway.service`)
       → router classifier (MiniLM v6, L1+L2 cache, smart truncation)
       → worker via Tailscale (URLs from EU_KIKI_WORKERS_JSON env):
            - Apertus :9301 (Studio, MLX BF16)
            - Devstral :9302 (Studio, MLX BF16, currently offline)
            - EuroLLM :9303 (Studio, MLX BF16)
            - Gemma 3 :9304 (Tower, llama-server)
```

Workers (3 EU/CH base models):
- **Apertus-70B-Instruct-2509** (`:9301`) — EPFL+ETH+CSCS — reasoning, hardware, EU normative (20 LoRA domains)
- **Devstral-Small-2-24B-MLX-4bit** (`:9302`) — Mistral AI — code generation (16 LoRA domains)
- **EuroLLM-22B-Instruct-2512** (`:9303`) — utter-project — multilingual EU (4 LoRA domains)

Router: **all-MiniLM-L6-v2** (384d, 22M) + MLP head (256 hidden) → 32 domains, sigmoid multi-label, top-k=4, threshold=0.50. Active checkpoint `output/router-v6/` — **87.7% top-1 / 98.7% top-3** on validation. Auto-device (MPS/CUDA/CPU). L1 LRU cache (exact match) + L2 cosine ≥ 0.95 cache (paraphrase) + auto-prewarm on boot. Length-aware smart truncation: short → full encode, medium → 128-tok left-truncate, long (> 1000 chars) → head 256 + tail 256.

⚠️ **Quarantined adapters**: EuroLLM `chat-fr` and `traduction-tech` produce `"user user user…"` loops (training chat-template leakage). The worker silently falls back to base EuroLLM for those domains. See `MLXWorkerRuntime.QUARANTINED_DOMAINS` in `src/worker/runtime.py`. Re-train pending.

## Repo layout

```
eu-kiki/
├── configs/                     # apertus.yaml, devstral.yaml, eurollm.yaml, gateway.yaml
├── src/
│   ├── gateway/                 # FastAPI :9300, dispatch, Prometheus, env-driven worker URLs
│   ├── router/                  # MiniLM + MLP classifier (32 domains, smart truncation, caches)
│   ├── worker/                  # 1 model / process, BF16, shared memory pool, QUARANTINED_DOMAINS guard
│   └── mlx_models/              # Apertus MLX impl + xielu activation
├── scripts/                     # ~50 scripts (scrape, build, train, eval, router pipeline)
│   ├── scrape_*.py              # OSHWA, arXiv, Wikipedia, Hackaday, Arduino, ESP-IDF, STM32, Rust, KiCad
│   ├── build_hf_datasets.py     # 729 lines — HF→JSONL orchestrator
│   ├── train_apertus.py / train_devstral.py / train_eurollm.py
│   ├── train_eu_kiki_batch.py / train_eu_kiki_hf_batch.py    # sequential
│   ├── rebuild_router_dataset.py + augment_niche_domains.py + augment_short_greetings.py
│   ├── build_router_data.py / encode_router_minilm.py / train_router_from_embeddings.py
│   ├── build_confusion_matrix.py / calibrate_threshold.py    # router QA
│   ├── pdf_pipeline/ + scan_pii.py + fix_provenance.py       # PDF compliance
│   ├── vlm_poc_pipeline.py                                   # VLM POC
│   └── eval_framework.py / run_eval.sh
├── data/
│   ├── hf-traced/               # 35 domain folders, train.jsonl + valid.jsonl + MANIFEST.json
│   ├── router/                  # train.jsonl + valid.jsonl (split from router-clean)
│   ├── router-clean/            # 32 per-domain JSONL (9 967 rows, niche+greetings curated)
│   └── router-minilm-v6/        # pre-encoded MiniLM embeddings (npy)
├── docs/
│   ├── eu-ai-act-transparency.md   # Doc principale Art. 52/53
│   ├── pdf-compliance-report.md    # Audit PDF pipeline (DSM Art.4 TDM)
│   ├── vlm-compliance-report.md    # Audit VLM POC
│   └── specs/                      # 2026-04-26 design + plan
├── tests/                       # pytest — apertus, xielu, worker, integration, runtime, router, gateway
├── pyproject.toml               # Python ≥3.13, Apache-2.0
└── uv.lock
```

## Commands

```bash
# Setup
uv venv && uv pip install -e ".[dev,router,data]"

# Tests
uv run python -m pytest
uv run python -m pytest tests/test_xielu.py -v     # single file
uv run python -m pytest -k "test_name"             # single test

# Build datasets (HF-traceable, EU AI Act-compliant)
uv run python scripts/build_hf_datasets.py
uv run python scripts/scrape_oshwa.py              # 3265 OSHWA-certified projects

# Train LoRA adapters (3 modèles, séquentiel)
uv run python scripts/train_eu_kiki_batch.py

# Train router (full v6 pipeline, ~25 min on macM1 MPS)
uv run python scripts/rebuild_router_dataset.py        # HF + niche + greetings → data/router-clean/
uv run python scripts/build_router_data.py             # split 80/20 → data/router/
uv run python scripts/encode_router_minilm.py          # MiniLM 384d → data/router-minilm-vN/
uv run python scripts/train_router_from_embeddings.py --emb-dir data/router-minilm-vN --hidden-dim 256 --output-dir output/router-vN

# Router QA
uv run python scripts/build_confusion_matrix.py        # → docs/transparency/confusion-top10.md
uv run python scripts/calibrate_threshold.py

# Local dev: launch all workers + gateway in one process tree
bash scripts/start.sh

# Production deploy
# - Gateway runs as systemd unit `eu-kiki-gateway.service` on electron-server
#   with EU_KIKI_WORKERS_JSON env (Tailscale URLs to Studio MLX workers)
# - Workers run as nohup uvicorn on Studio (PIDs in /tmp/eu-kiki-*.log)
# - Router weights deployed via rsync (output/router-vN/ is git-ignored)

# Logs
tail -f /tmp/eu-kiki-eurollm.log    # studio
tail -f /tmp/eu-kiki-apertus.log    # studio
ssh electron-server "sudo journalctl -u eu-kiki-gateway -f"
```

## Domains par modèle

### Apertus 70B (20 domaines)
electronics-hw, emc, dsp, spice, kicad, stm32, platformio, iot, embedded, math, reasoning, security, music-audio, freecad, power, misra-c, autosar-cert, doc-technique-ce, calcul-normatif, normes-iec

### Devstral 24B (16 domaines)
python, rust, typescript, cpp, shell, html-css, sql, web-backend, web-frontend, docker, devops, yaml-json, llm-ops, llm-orch, ml-training, lua-upy

### EuroLLM 22B (4 domaines)
chat-fr, traduction-tech, redaction-multilingue, localisation-doc

## Conformité EU AI Act

- **Art. 52/53** : `docs/eu-ai-act-transparency.md` couvre système, modèles, provenance, datasets, classification "limited risk"
- **DSM Art. 4 TDM** : audit PDF pipeline (`docs/pdf-compliance-report.md`) — robots.txt, SHA-256, ST/Espressif/TI/NXP/KiCad
- **Datasets HF-traceable** : `data/hf-traced/{domain}/MANIFEST.json` documente `hf_dataset_id`, license, download_date, `n_source_rows`, `n_used`
- **Tous modèles Apache-2.0** avec provenance documentée

## Key Design Decisions

- BF16 for all models (512 GB unified memory allows it)
- Multi-process workers (1 model per process, shared memory pool)
- Sigmoid routing (domains overlap, not mutually exclusive)
- LoRA on attention projections only (`q/k/v/o_proj`)
- `xielu` activation custom-implemented for Apertus MLX support
- Local-only deployment, no cloud, no telemetry

## Production deployment (live 2026-05-06)

| Component | Host | Port | Notes |
|---|---|---|---|
| Cloudflare Tunnel | — | 443 | `ml.saillant.cc` → cockpit |
| kiki-cockpit | electron-server | 443 | React SPA + Python API (`/api/public/chat`) |
| eu-kiki gateway | electron-server | 9300 | systemd unit, FastAPI, MiniLM router |
| EuroLLM worker | studio (Tailscale 100.116.92.12) | 9303 | MLX BF16, ~22 GB |
| Apertus worker | studio | 9301 | MLX BF16, ~140 GB |
| Devstral worker | (TBD) | 9302 | currently offline |
| Gemma 3 worker | tower (100.78.6.122) | 9304 | llama-server |

Worker URLs are supplied to the gateway via `EU_KIKI_WORKERS_JSON` env var (set in the systemd unit). Override at boot to redirect traffic without code changes.

## Roadmap (audit 2026-05-04)

### État réel d'entraînement

| Modèle | Adapters entraînés | Cible | Statut |
|--------|--------------------|-------|--------|
| Apertus 70B | 6 (electronics-hw, embedded, math, math-gsm8k, math-reasoning, spice-sim) | 8 | 🟠 PARTIAL — manque `emc-dsp-power`, `security-fenrir` |
| Devstral 2 24B | 22/22 | 22 | 🟢 DONE |
| EuroLLM 22B | 3 (chat-fr, multilingual-eu, traduction-tech) | 4 | 🟠 PARTIAL — 1 manquant |
| Router 32-domain | trained | — | 🟢 DONE (`output/router/router.safetensors`) |
| Eval framework | code prêt (52 ko) | — | 🔴 **JAMAIS LANCÉ** — `output/eval/raw/` vide |
| VLM PoC | 6 runs, loss diverge >5 | — | 🔴 CRASHED — `vlm_poc_run6.log` "No adapter produced" |
| Pipeline PDF (360 pairs) | intégré dans Apertus spice-sim/freecad/embedded/electronics-hw | — | 🟢 DONE |
| Batch v2 medium-35 (Mistral-Medium-3.5-128B) | math-gsm8k done, math-reasoning iter 400 val 0.511 | 4+ | 🟠 EN COURS |

### 🔴 Bloquants

1. **Re-train EuroLLM `chat-fr` + `traduction-tech`** — adapters quarantined depuis 2026-05-05 (training chat-template leak → "user user…" loop). Audit `scripts/train_eurollm.py` pour vérifier le strip du template. ~6 h chacun.
2. **Démarrer un Devstral worker** — port 9302 vide actuellement. Devrait tourner sur Studio (Apertus + EuroLLM y sont déjà). MLX 4-bit ~13 GB. ~30 min.
3. **Stabiliser ou abandonner VLM PoC** — loss diverge sur 6 runs successifs. Revoir prepro images, lr, ou archiver. 1-2 j.

### ✅ Récents (résolus 2026-05-05/06)

- ✅ **Router divergence résolue** : `classifier.py` aligné sur MiniLM 384d + 32 domaines (matches `gateway.yaml`).
- ✅ **Router v6 trained** — top-1 65.5% → 87.7% (+22 pts), top-3 85.3% → 98.7%. 9 967 rows curated, niche-augmented, greetings-grounded.
- ✅ **Smart truncation length-aware** — short (full), medium (left-trunc 128), long (head 256 + tail 256).
- ✅ **L1 LRU + L2 cosine cache + auto-prewarm** — p95 spike éliminé, L1 hit ~0.01 ms.
- ✅ **systemd unit gateway** — `eu-kiki-gateway.service` enabled+active sur electron-server, env `EU_KIKI_WORKERS_JSON` persisté.
- ✅ **Quarantine adapters mécanisme** — `MLXWorkerRuntime.QUARANTINED_DOMAINS` fallback base.
- ✅ **Bench suite publishable** — HumanEval+, MT-Bench full, GSM8K, KIKI-DSL v3, model card v0.4.1.

### 🟠 Important

- Compléter **Apertus** : `emc-dsp-power` et `security-fenrir` adapters finals. Logs présents → reprendre. ~6 h chacun.
- Compléter **EuroLLM** : domaine manquant (`localisation-doc` ?). ~6 h.
- Finir **batch v2 medium-35** (Mistral-Medium-3.5-128B) : math-reasoning en cours, manque chat-fr/multilingual-eu/traduction-tech. ~6-8 h.
- Réintégrer `data/quarantine/` (5 dirs PII flaggés) après filtrage propre. 0.5 j.

### 🟡 Cleanup

- Publier sur HF les adapters EU validés (Apache-2.0, full provenance EU AI Act) — aucun adapter EU-KIKI sur `clemsail/` ni `electron-rare/` à ce jour (audit 2026-05-04). Le script existe en sister project (`KIKI-Mac_tunner/scripts/release_hf.py`). 3-4 h.
- Standardiser le format de sortie eval (`output/eval/<YYYY-MM-DD>-<scope>.{json,md}`).
- Documenter le différentiel "20 domains HF-traced (48K examples)" du commit `f2c9cee` vs ~81K lignes mesurées sur 24 domaines (probablement sous-ensemble curé/dédupliqué).

### 🟢 Future

- Réconcilier les `data/scraped/` non utilisés (arduino, hackaday, oshwa, kicad — finalisé seulement pour kicad/freecad).
- VLM full pipeline si convergence trouvée.
- Évaluations adverses (red team) sur les modèles EU pour conformité Art. 55.

## Notes

- **48K vs 81K examples** : commit `f2c9cee` annonce "20 domains HF-traced (48K examples)" — la mesure brute donne ~81K lignes train sur 24 domaines. Les 48K désignent vraisemblablement le sous-ensemble curé/dédupliqué. À confirmer.
- **Repo GitHub** : `L-electron-Rare/eu-kiki` (privé) — poussé le 2026-05-04. Mirror local sur studio + electron-server + macM1.
- **Router weights `output/router-vN/` git-ignored** — déploiement par rsync (`rsync -avz output/router-v6/ electron-server:/home/electron/eu-kiki/output/router-v6/`) puis `sudo systemctl restart eu-kiki-gateway`.
- **kxkm-ai** : RTX 4090 24 GB, joignable via electron-server bastion (`ssh kxkm@10.2.0.237`). Actuellement sert llama-server Qwen3-Next-80B + ComfyUI + SearXNG + OpenWebUI (stack chat avec web-search tool). Pas utilisé par eu-kiki à ce jour.

## Sister project

`~/Documents/Projets/KIKI-Mac_tunner/` — non-EU foundation distillation. Les scripts `train_eu_kiki_*.py` et configs `eu-kiki-*.yaml` y sont mirrorés.
