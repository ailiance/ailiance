# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Context

EU-KIKI is a 100% EU-sovereign multi-model LLM serving pipeline. It routes requests to **3 European/Swiss foundation models** via a Jina v3 domain classifier, each fine-tuned with LoRA adapters on HF-traceable datasets. Local-only deployment on Mac Studio M3 Ultra (512 GB).

## Architecture

Gateway (`:9200`) dispatches to 3 workers:
- **Apertus-70B-Instruct-2509** (`:9301`) — EPFL+ETH+CSCS — reasoning, hardware, EU normative (20 LoRA domains)
- **Devstral-Small-2-24B-MLX-4bit** (`:9302`) — Mistral AI — code generation (16 LoRA domains)
- **EuroLLM-22B-Instruct-2512** (`:9303`) — utter-project — multilingual EU (4 LoRA domains)

Router: **Jina Embeddings v3** (Berlin) + MLP classifier (40 domains, sigmoid multi-label, top-k=4, threshold=0.12).

## Repo layout

```
eu-kiki/
├── configs/                     # apertus.yaml, devstral.yaml, eurollm.yaml, gateway.yaml
├── src/
│   ├── gateway/                 # FastAPI :9200, dispatch, Prometheus
│   ├── router/                  # Jina v3 + MLP classifier (40 domains)
│   ├── worker/                  # 1 model / process, BF16, shared memory pool
│   └── mlx_models/              # Apertus MLX impl + xielu activation
├── scripts/                     # ~40 scripts (scrape, build, train, eval)
│   ├── scrape_*.py              # OSHWA, arXiv, Wikipedia, Hackaday, Arduino, ESP-IDF, STM32, Rust, KiCad
│   ├── build_hf_datasets.py     # 729 lines — HF→JSONL orchestrator
│   ├── train_apertus.py / train_devstral.py / train_eurollm.py
│   ├── train_eu_kiki_batch.py / train_eu_kiki_hf_batch.py    # sequential
│   ├── train_router.py / encode_router_jina.py
│   ├── pdf_pipeline/ + scan_pii.py + fix_provenance.py       # PDF compliance
│   ├── vlm_poc_pipeline.py                                   # VLM POC
│   └── eval_framework.py / run_eval.sh
├── data/
│   ├── hf-traced/               # 35 domain folders, train.jsonl + valid.jsonl + MANIFEST.json
│   ├── router/                  # train.jsonl (46100) + valid.jsonl (11532)
│   └── router-jina-v3/          # pre-encoded embeddings (39 MB)
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

# Train router
uv run python scripts/build_router_data.py
uv run python scripts/encode_router_jina.py
uv run python scripts/train_router_from_embeddings.py

# Launch all services
bash scripts/start.sh

# Logs
tail -f /tmp/eu-kiki/gateway.log
tail -f /tmp/eu-kiki/apertus.log
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

1. **Lancer `bash scripts/run_eval.sh --mode compare`** — 31 adapters EU produits, zéro métrique de qualité. Eval framework prêt avec 4 dimensions Art. 53(1)(d) (perplexité, generation quality, adapter efficiency, inference speed). 3-5 h compute.
2. **Stabiliser ou abandonner VLM PoC** — loss diverge sur 6 runs successifs. Revoir prepro images, lr, ou archiver. 1-2 j.
3. **Aligner divergence routeur** — `gateway.yaml` annonce 32 domaines + MiniLM-384d, `classifier.py` annonce 40 + Jina-v3-1024d. Choisir l'un, supprimer l'autre. 1 h.

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

- **Divergence routeur à clarifier** : `gateway.yaml` indique 32 domaines + MiniLM-384d, alors que `classifier.py` et ce CLAUDE.md annoncent 40 domaines + Jina-v3-1024d. Possible config legacy — gateway.yaml à aligner.
- **48K vs 81K examples** : commit `f2c9cee` annonce "20 domains HF-traced (48K examples)" — la mesure brute donne ~81K lignes train sur 24 domaines. Les 48K désignent vraisemblablement le sous-ensemble curé/dédupliqué. À confirmer.
- **Repo GitHub** : `L-electron-Rare/eu-kiki` (privé) — poussé le 2026-05-04. Mirror local sur studio + GrosMac (rsync).

## Sister project

`~/Documents/Projets/KIKI-Mac_tunner/` — non-EU foundation distillation. Les scripts `train_eu_kiki_*.py` et configs `eu-kiki-*.yaml` y sont mirrorés.
