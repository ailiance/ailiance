# Router Quality Implementation Plan (Phases 1 + 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Push the ailiance router from 87.6 % top-1 (router-v5) toward >92 % by (a) deciding empirically if Jina v3 beats MiniLM on the clean corpus, (b) calibrating the confidence threshold so ambiguous prompts fall back to Gemma instead of being force-routed, and (c) covering 13 niche domains currently <10 rows with curated prompts.

**Architecture:** Pure data-and-training pass on the existing router stack at `~/Documents/Projets/ailiance/`. No gateway code changes — only the head checkpoint at `output/router/` is replaced and `configs/gateway.yaml` is updated if the encoder changes. The collaborative niche-domain script accepts `--domain X --interactive` to let the human curator iterate without touching code.

**Tech Stack:** Python 3.12 (uv venv at `~/KIKI-Mac_tunner/.venv`), `sentence-transformers`, `safetensors`, `torch`, `huggingface_hub`. Training runs on Mac Studio M3 Ultra. Deployment target: gateway tmux session on `electron-server`.

**Out of scope (separate plans):**
- Phase 3 — MLX MCP router on macm1 (new repo `mlx-mcp-router`)
- Phase 4 — Observability + UX on `ailiance.fr` (kiki-cockpit repo)

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `scripts/router_benchmark.py` | exists | Already runs MiniLM + Jina bench. Updated to read `data/router-clean/` instead of hard-coded smoke set. |
| `scripts/calibrate_threshold.py` | **new** | Grid-search the router threshold on the validation split, pick the value that maximises top-1 while routing ≥95 % of prompts (rest fall back to Gemma). |
| `scripts/build_confusion_matrix.py` | **new** | Compute the 32×32 confusion matrix on the validation split, dump CSV + Markdown summary of the top 10 confused pairs. |
| `scripts/augment_niche_domains.py` | **new** | Per-domain curated prompt list for the 13 niche domains. Designed to be edited by a human collaborator: each domain has a Python list literal with empty slots and inline guidance. |
| `scripts/rebuild_router_dataset.py` | exists | No changes. It already pulls `augment_niche_domains.py` symbols if exposed. |
| `configs/gateway.yaml` | modify | Bump `embedding_model` and `embedding_dim` if Jina wins. |
| `output/router-v6` | **new dir** | Re-trained head if Jina chosen. |
| `output/router-v7` | **new dir** | Final head with niche-augmented corpus + chosen encoder. |
| `output/router/` | replace | Symlink-copy of `router-v7` once validated. |
| `data/router-clean/PROVENANCE.json` | auto-updated | Picks up new domain rows automatically when `rebuild_router_dataset.py` runs. |
| `docs/transparency/router-training-data.md` | modify | Bump v3 → v7 in the active checkpoint section, update accuracy numbers. |

All work happens in the `~/Documents/Projets/ailiance/` checkout on the local Mac, then is rsynced to studio for training and to electron-server for deployment.

---

## Phase 1 — Encoder bench + threshold calibration

### Task 1.1: Patch `router_benchmark.py` to use the clean corpus

**Files:**
- Modify: `scripts/router_benchmark.py`

The current script has a 10-prompt smoke set hard-coded. We want it to also report top-1/top-3 over the full validation split for a fair MiniLM-vs-Jina comparison.

- [ ] **Step 1: Read the current `router_benchmark.py`**

Run:
```bash
cd ~/Documents/Projets/ailiance
sed -n '1,60p' scripts/router_benchmark.py
```
Expected: see the existing `TESTS = [...]` constant and the `bench_encoder()` function.

- [ ] **Step 2: Add a `--full` flag that loads the validation split**

Edit `scripts/router_benchmark.py`. After the `TESTS` definition add:

```python
def load_full_valid() -> list[tuple[str, str]]:
    """Read data/router/valid.jsonl, return (prompt, domain) pairs."""
    import json
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent
    valid_path = repo_root / "data/router/valid.jsonl"
    pairs: list[tuple[str, str]] = []
    with valid_path.open() as f:
        for line in f:
            obj = json.loads(line)
            pairs.append((obj["prompt"], obj["domain"]))
    return pairs
```

In `main()`, add at the top:

```python
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--full", action="store_true",
                help="Run on data/router/valid.jsonl instead of the 10-prompt smoke set")
args = ap.parse_args()

global TESTS
if args.full:
    TESTS = load_full_valid()
    print(f"FULL VALIDATION MODE: {len(TESTS)} prompts")
```

- [ ] **Step 3: Sync to studio and run smoke-only first**

Run:
```bash
ssh studio "cd ~/ailiance && git stash && git pull --ff-only"
cd ~/Documents/Projets/ailiance
git add scripts/router_benchmark.py
git commit -m "bench: --full flag reads valid split"
git push
ssh studio "cd ~/ailiance && git pull --ff-only && ~/KIKI-Mac_tunner/.venv/bin/python scripts/router_benchmark.py 2>&1 | tail -15"
```
Expected: same 10-prompt smoke output as before.

- [ ] **Step 4: Run on full valid set with both encoders**

Run:
```bash
ssh studio "cd ~/ailiance && ~/KIKI-Mac_tunner/.venv/bin/python scripts/router_benchmark.py --full 2>&1 | tee /tmp/bench-full.log | tail -40"
```
Expected: load time + per-prompt latency for MiniLM and Jina v3 (classification + separation tasks). Cosine separation Δ for each. ~1-2 min total.

- [ ] **Step 5: Commit the bench artefact**

Run:
```bash
ssh studio "cat /tmp/bench-full.log" > ~/Documents/Projets/ailiance/docs/transparency/2026-05-05-encoder-bench.log
cd ~/Documents/Projets/ailiance
git add docs/transparency/2026-05-05-encoder-bench.log
git commit -m "docs: encoder bench MiniLM vs Jina v3"
git push
```

---

### Task 1.2: Decide encoder migration

**Files:**
- Read-only: `docs/transparency/2026-05-05-encoder-bench.log`

Decision rule: migrate to Jina v3 only if **(top-1 with Jina-trained head ≥ MiniLM-trained head + 3 pp) AND (per-prompt latency < 30 ms)**. Otherwise keep MiniLM (faster, cheaper).

- [ ] **Step 1: Read the bench log**

Run:
```bash
cat docs/transparency/2026-05-05-encoder-bench.log
```

- [ ] **Step 2: Train a Jina-head if encoder cosine separation is ≥ MiniLM**

Only do this step if the bench shows Jina has a higher Δ separation. Otherwise SKIP to Task 1.3 with `--embedding-model sentence-transformers/all-MiniLM-L6-v2`.

Run:
```bash
ssh studio "cd ~/ailiance && ~/KIKI-Mac_tunner/.venv/bin/python scripts/train_router.py --embedding-model jinaai/jina-embeddings-v3 --hidden-dim 512 --epochs 30 --output-dir output/router-v6 2>&1 | tail -10"
```
Expected: 30 epochs, final `top1=` and `top3=`. Takes ~10-15 min on Mac Studio.

- [ ] **Step 3: Compare against MiniLM v5 baseline (87.6/98.7)**

If Jina v6 top-1 ≥ 90.6 % AND per-prompt encode ≤ 30 ms (from Task 1.1 step 4): migrate. Otherwise abandon Jina and keep v5 as the MiniLM base for Phase 2.

Document the decision:

```bash
cat <<EOF >> docs/transparency/router-training-data.md

## Encoder migration decision (2026-05-05)

After running scripts/router_benchmark.py --full on the v5 clean corpus:

- MiniLM L6 v2 (active): top-1 = 87.6 %, encode = 4 ms/prompt
- Jina v3 (candidate v6): top-1 = X.X %, encode = X ms/prompt
- Δ top-1: +/-X.X pp
- Δ latency: +/-X ms

Decision: KEEP MiniLM / MIGRATE TO JINA (cross out one).

Rationale: <fill in>.
EOF
```

- [ ] **Step 4: Commit the decision**

Run:
```bash
git add docs/transparency/router-training-data.md
git commit -m "docs: encoder migration decision"
git push
```

---

### Task 1.3: Build threshold calibration script

**Files:**
- Create: `scripts/calibrate_threshold.py`

The current router config in `configs/gateway.yaml` has `threshold: 0.12` — meaning anything above 12 % sigmoid score wins. This is too low: ambiguous prompts get force-routed instead of falling back to Gemma. We grid-search over [0.1, 0.3, 0.5, 0.7] and pick the best.

- [ ] **Step 1: Write the calibration script**

Create `scripts/calibrate_threshold.py`:

```python
"""Calibrate the router confidence threshold against the validation split.

For each candidate threshold, compute:
  - top-1 accuracy on prompts where the head is confident enough
  - coverage: fraction of prompts where any score >= threshold
  - fallback rate: 1 - coverage (these would route to Gemma)

Pick the threshold that maximises top-1 accuracy while keeping
coverage >= 95 % (i.e. we don't bail to Gemma more than 5 % of the
time on the labelled validation set).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parent.parent
WEIGHTS = REPO / "output/router"  # whichever checkpoint is currently active
VALID = REPO / "data/router/valid.jsonl"


def load_head() -> tuple[torch.nn.Module, list[str], str, int]:
    """Return (mlp, domains, encoder_name, embedding_dim)."""
    meta = json.loads((WEIGHTS / "meta.json").read_text())
    domains = meta["domains"]
    encoder_name = meta["embedding_model"]
    dim = meta["embedding_dim"]
    hidden = meta["hidden_dim"]
    n = len(domains)

    import torch.nn as tnn
    class RouterMLP(tnn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = tnn.Sequential(
                tnn.Linear(dim, hidden), tnn.GELU(),
                tnn.Dropout(0.1), tnn.Linear(hidden, n),
            )
        def forward(self, x): return torch.sigmoid(self.net(x))

    mlp = RouterMLP()
    state = load_file(str(WEIGHTS / "router.safetensors"))
    if not any(k.startswith("net.") for k in state):
        state = {f"net.{k}": v for k, v in state.items()}
    mlp.load_state_dict(state)
    mlp.eval()
    return mlp, domains, encoder_name, dim


def main() -> None:
    import os
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from sentence_transformers import SentenceTransformer

    mlp, domains, enc_name, dim = load_head()
    print(f"Loaded head: {len(domains)} domains, encoder={enc_name}")

    pairs: list[tuple[str, str]] = []
    with VALID.open() as f:
        for line in f:
            obj = json.loads(line)
            pairs.append((obj["prompt"], obj["domain"]))
    print(f"Validation set: {len(pairs)} prompts")

    enc = SentenceTransformer(enc_name)
    embs = enc.encode([p for p, _ in pairs], normalize_embeddings=True,
                      convert_to_tensor=True, show_progress_bar=False)
    with torch.no_grad():
        scores = mlp(embs.cpu())  # (N, num_domains)

    domain_idx = {d: i for i, d in enumerate(domains)}

    print("\n  threshold | coverage | top-1 (covered) | top-1 (overall)")
    print("  ----------+----------+-----------------+----------------")
    for thr in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        n_total = len(pairs)
        n_covered = 0
        n_correct_covered = 0
        n_correct_overall = 0
        for i, (_, target) in enumerate(pairs):
            target_idx = domain_idx.get(target)
            if target_idx is None:
                continue
            row = scores[i]
            top_idx = int(row.argmax().item())
            top_score = float(row[top_idx].item())
            covered = top_score >= thr
            if covered:
                n_covered += 1
                if top_idx == target_idx:
                    n_correct_covered += 1
                    n_correct_overall += 1
            # if not covered the gateway falls back to Gemma → counted wrong
        cov = n_covered / n_total
        top1_cov = n_correct_covered / n_covered if n_covered else 0.0
        top1_all = n_correct_overall / n_total
        print(f"   {thr:.2f}     |  {cov:.3f}   |    {top1_cov:.3f}        |   {top1_all:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Sync + run the script**

Run:
```bash
git add scripts/calibrate_threshold.py
git commit -m "scripts: calibrate router threshold"
git push
ssh studio "cd ~/ailiance && git pull --ff-only && ~/KIKI-Mac_tunner/.venv/bin/python scripts/calibrate_threshold.py 2>&1 | tail -20"
```
Expected: 8-row table from threshold 0.10 to 0.80 with coverage and top-1 columns.

- [ ] **Step 3: Pick the optimal threshold**

Apply the rule: pick the **largest** threshold where `coverage >= 0.95`. If none reaches 0.95, pick the largest threshold where `top-1 (overall) >= 0.85`.

- [ ] **Step 4: Update `configs/gateway.yaml`**

Edit `configs/gateway.yaml`:

```yaml
router:
  weights_dir: output/router
  embedding_model: sentence-transformers/all-MiniLM-L6-v2  # or jina if migrated
  embedding_dim: 384  # or 1024 if jina
  hidden_dim: 256
  num_domains: 32
  threshold: <chosen_value>
  max_active: 4
```

- [ ] **Step 5: Commit + deploy**

Run:
```bash
git add configs/gateway.yaml
git commit -m "config: bump router threshold (calibrated)"
git push
ssh electron-server "cd ~/ailiance && git pull --ff-only"
```

The gateway will pick up the new threshold on next restart (Task 2.5 step 5).

---

## Phase 2 — Confusion matrix + niche domain curation

### Task 2.1: Confusion matrix script

**Files:**
- Create: `scripts/build_confusion_matrix.py`

- [ ] **Step 1: Write the confusion matrix script**

Create `scripts/build_confusion_matrix.py`:

```python
"""Compute the router confusion matrix on the validation split.

Outputs:
  output/confusion-<sha>.csv      — full matrix (domains × domains)
  output/confusion-<sha>-top10.md — markdown summary of worst pairs

The script reuses the same head loading code as calibrate_threshold.py.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import torch
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parent.parent
WEIGHTS = REPO / "output/router"
VALID = REPO / "data/router/valid.jsonl"
OUT_DIR = REPO / "output"


def load_head() -> tuple[torch.nn.Module, list[str], str]:
    meta = json.loads((WEIGHTS / "meta.json").read_text())
    domains = meta["domains"]
    encoder_name = meta["embedding_model"]
    dim = meta["embedding_dim"]
    hidden = meta["hidden_dim"]
    n = len(domains)
    import torch.nn as tnn
    class RouterMLP(tnn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = tnn.Sequential(
                tnn.Linear(dim, hidden), tnn.GELU(),
                tnn.Dropout(0.1), tnn.Linear(hidden, n),
            )
        def forward(self, x): return torch.sigmoid(self.net(x))
    mlp = RouterMLP()
    state = load_file(str(WEIGHTS / "router.safetensors"))
    if not any(k.startswith("net.") for k in state):
        state = {f"net.{k}": v for k, v in state.items()}
    mlp.load_state_dict(state)
    mlp.eval()
    return mlp, domains, encoder_name


def main() -> None:
    import os
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from sentence_transformers import SentenceTransformer

    mlp, domains, enc_name = load_head()
    domain_idx = {d: i for i, d in enumerate(domains)}
    n = len(domains)

    pairs: list[tuple[str, str]] = []
    with VALID.open() as f:
        for line in f:
            obj = json.loads(line)
            pairs.append((obj["prompt"], obj["domain"]))

    enc = SentenceTransformer(enc_name)
    embs = enc.encode([p for p, _ in pairs], normalize_embeddings=True,
                      convert_to_tensor=True, show_progress_bar=False)
    with torch.no_grad():
        scores = mlp(embs.cpu())

    matrix = [[0] * n for _ in range(n)]
    confusions: Counter = Counter()
    for i, (_, target) in enumerate(pairs):
        ti = domain_idx.get(target)
        if ti is None:
            continue
        pi = int(scores[i].argmax().item())
        matrix[ti][pi] += 1
        if pi != ti:
            confusions[(target, domains[pi])] += 1

    # CSV
    csv_path = OUT_DIR / "confusion.csv"
    with csv_path.open("w") as f:
        w = csv.writer(f)
        w.writerow(["target↓ / predicted→"] + domains)
        for ti, target in enumerate(domains):
            w.writerow([target] + matrix[ti])
    print(f"  wrote {csv_path}")

    # Markdown top-10
    md_path = OUT_DIR / "confusion-top10.md"
    with md_path.open("w") as f:
        f.write("# Top 10 confusion pairs\n\n")
        f.write("| Target | Predicted | Count |\n|---|---|---|\n")
        for (t, p), c in confusions.most_common(10):
            f.write(f"| {t} | {p} | {c} |\n")
    print(f"  wrote {md_path}")
    print()
    print("Top 10 confusions:")
    for (t, p), c in confusions.most_common(10):
        print(f"  {t:18s} -> {p:18s}  ({c})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Sync + run**

Run:
```bash
git add scripts/build_confusion_matrix.py
git commit -m "scripts: confusion matrix"
git push
ssh studio "cd ~/ailiance && git pull --ff-only && ~/KIKI-Mac_tunner/.venv/bin/python scripts/build_confusion_matrix.py 2>&1 | tail -25"
```
Expected: top-10 confusion pairs printed + 2 files in `output/`.

- [ ] **Step 3: Pull artefacts back to local for inspection**

Run:
```bash
ssh studio "cat ~/ailiance/output/confusion-top10.md" > /tmp/confusion-top10.md
cat /tmp/confusion-top10.md
mkdir -p ~/Documents/Projets/ailiance/docs/transparency
cp /tmp/confusion-top10.md docs/transparency/confusion-top10.md
git add docs/transparency/confusion-top10.md
git commit -m "docs: confusion top-10 snapshot"
git push
```

---

### Task 2.2: Niche-domain augmentation scaffold

**Files:**
- Create: `scripts/augment_niche_domains.py`

This script provides ~100 manually-curated prompts per niche domain. The first batch is filled in by you with conservative, high-confidence prompts; the second batch comes from the human collaborator iterating. Each domain has its own list literal so editing one doesn't risk breaking another.

Niche domains under 10 rows in v5: `embedded`, `kicad-pcb`, `kicad-dsl`, `stm32`, `dsp`, `iot`, `music-audio`, `platformio`, `lua-upy` (already 603), `ml-training`, `web-backend`, `web-frontend`, `yaml-json`, `electronics-hw`, `llm-orch`. We focus on the 13 that are still <10 rows after v5.

- [ ] **Step 1: Create the scaffold with starter prompts**

Create `scripts/augment_niche_domains.py`:

```python
"""Curated prompts for niche technical domains.

Each domain has a list of short user-style prompts in the same format
the legacy corpus used. rebuild_router_dataset.py imports these lists
and appends them to the clean corpus.

ONLY add prompts you, the human curator, can verify match the domain.
The router is more useful with 100 high-quality prompts than 1000 noisy
ones — see docs/transparency/router-training-data.md §4.
"""
from __future__ import annotations

# === KiCad PCB design ===
KICAD_PCB = [
    "Comment vérifier les règles DRC sur un PCB KiCad 9 ?",
    "Calculer la largeur de piste pour 3A en cuivre 35µm",
    "Comment importer une netlist Altium dans KiCad ?",
    "Définir une stack-up 4 couches impédance contrôlée",
    "Configurer le DRC pour un produit certifié IPC-A-610 classe 2",
    "Comment router un signal différentiel 100Ω en KiCad ?",
    "Plan de masse séparé analogique/digital — bonne pratique KiCad",
    "Générer les fichiers Gerber pour JLCPCB depuis KiCad",
    "Comment placer correctement les vias de cousure sur un plan de masse ?",
    "Router un bus SPI haute vitesse sur PCB 4 couches",
    # ... add ~80 more here, focused on real KiCad PCB workflow questions.
]

# === KiCad DSL / Schematic editor ===
KICAD_DSL = [
    "Créer un symbole hiérarchique en KiCad Eeschema",
    "Comment lier une feuille hiérarchique à un schéma parent ?",
    "Définir des annotations multi-feuilles pour un design modulaire",
    "Comment générer une BOM filtrée par sous-feuille ?",
    "Création d'un symbole custom pour un composant non standard",
    # ...
]

# === STM32 / ARM Cortex-M firmware ===
STM32 = [
    "Configurer un timer en input capture sur STM32F4 avec HAL",
    "Comment activer le DMA sur un ADC STM32G4 ?",
    "Démarrer un projet STM32CubeIDE pour un STM32H7",
    "Bare-metal blink LED sur STM32F0 sans HAL",
    "Configurer FreeRTOS avec deux tâches sur STM32F7",
    "Activer le mode low-power Stop2 sur STM32L4",
    "Initialiser SPI1 en master sur STM32F411",
    "Linker script custom pour placer une variable en CCM RAM",
    "Migrer un projet de F1 vers F4 — checklist",
    "Calibration ADC interne avec VREFINT sur STM32",
    # ...
]

# === Embedded systems (generic, non STM32-specific) ===
EMBEDDED = [
    "Comment écrire un driver I2C bare-metal en C ?",
    "Implémenter une FIFO circulaire en C pour un microcontrôleur",
    "Calculer la fréquence d'horloge requise pour un sampling 100 kHz",
    "Comment utiliser malloc dans un système embarqué temps-réel ?",
    "Bonne pratique pour gérer les interruptions imbriquées sur ARM",
    # ...
]

# === DSP — digital signal processing ===
DSP = [
    "Concevoir un filtre FIR passe-bas avec scipy.signal",
    "Calculer la réponse fréquentielle d'un filtre IIR Butterworth ordre 4",
    "Implémenter une FFT radix-2 en C pour un microcontrôleur",
    "Différence entre fenêtre Hamming et Hanning pour analyse spectrale",
    "Comment éviter le repliement spectral lors du sous-échantillonnage ?",
    # ...
]

# === IoT — internet of things ===
IOT = [
    "Connecter un ESP32 à un broker MQTT via TLS",
    "Comment implémenter MQTT QoS 1 en MicroPython ?",
    "Architecture LoRaWAN end-to-end pour capteurs distants",
    "Sécuriser une session BLE avec passkey entry",
    "Optimiser la consommation Wi-Fi en mode deep sleep ESP32",
    # ...
]

# === Music / audio synthesis ===
MUSIC_AUDIO = [
    "Implémenter un oscillateur wavetable en C++ pour VST3",
    "Générer un fichier MIDI programmatiquement en Python avec mido",
    "Calculer les coefficients d'un filtre biquad pour un EQ",
    "Architecture d'un synthétiseur soustractif numérique",
    "Comment mesurer la latence de traitement audio en JUCE ?",
    # ...
]

# === PlatformIO build system ===
PLATFORMIO = [
    "Créer un nouveau projet PlatformIO pour ESP32-S3",
    "Configurer plusieurs environnements dans platformio.ini",
    "Comment ajouter une bibliothèque privée à un projet PlatformIO ?",
    "Différence entre framework=arduino et framework=espidf",
    "Configurer le debugger sur PlatformIO avec une sonde J-Link",
    # ...
]

# === ML training (PyTorch / Hugging Face / training-ops) ===
ML_TRAINING = [
    "Fine-tuner un modèle Hugging Face avec PEFT/LoRA",
    "Configurer un DataLoader PyTorch avec num_workers",
    "Quantifier un modèle PyTorch en INT8 pour edge",
    "Différence entre training et evaluation mode en PyTorch",
    "Implémenter un learning rate scheduler cosine annealing",
    # ...
]

# === LLM orchestration ===
LLM_ORCH = [
    "Configurer LiteLLM comme proxy multi-provider",
    "Implémenter un agent ReAct avec LangChain",
    "Comment streamer une réponse LLM en SSE depuis FastAPI ?",
    "Mettre en place un retrieval-augmented generation avec ChromaDB",
    "Comparaison vLLM vs llama.cpp pour le serving",
    # ...
]

# === Web backend ===
WEB_BACKEND = [
    "Créer un endpoint FastAPI avec validation Pydantic v2",
    "Implémenter rate-limiting en Express.js avec Redis",
    "Comment ajouter un middleware d'auth JWT en NestJS ?",
    "Stream un fichier large depuis FastAPI sans le charger en mémoire",
    "Connection pooling Postgres en SQLAlchemy 2.0 async",
    # ...
]

# === Web frontend ===
WEB_FRONTEND = [
    "Configurer Vite avec React 19 et Tailwind v4",
    "Implémenter le routing avec TanStack Router",
    "Optimiser le re-render d'un composant React avec memo",
    "Créer un store Zustand avec persistence localStorage",
    "Différence entre Suspense et lazy loading en React",
    # ...
]

# === YAML/JSON config ===
YAML_JSON = [
    "Valider un manifest Kubernetes avec un schéma JSON",
    "Convertir un YAML en JSON avec yq en ligne de commande",
    "Structure d'un GitHub Actions workflow YAML multi-job",
    "Schéma OpenAPI 3.1 — différences avec OpenAPI 3.0",
    "Comment merger deux fichiers YAML avec anchors et aliases ?",
    # ...
]

# Aggregate every list into one mapping so rebuild_router_dataset.py can iterate.
NICHE_DOMAIN_PROMPTS: dict[str, list[str]] = {
    "kicad-pcb": KICAD_PCB,
    "kicad-dsl": KICAD_DSL,
    "stm32": STM32,
    "embedded": EMBEDDED,
    "dsp": DSP,
    "iot": IOT,
    "music-audio": MUSIC_AUDIO,
    "platformio": PLATFORMIO,
    "ml-training": ML_TRAINING,
    "llm-orch": LLM_ORCH,
    "web-backend": WEB_BACKEND,
    "web-frontend": WEB_FRONTEND,
    "yaml-json": YAML_JSON,
}
```

- [ ] **Step 2: Commit the scaffold**

Run:
```bash
git add scripts/augment_niche_domains.py
git commit -m "scripts: niche domain prompts scaffold"
git push
```

---

### Task 2.3: Wire `rebuild_router_dataset.py` to import niche prompts

**Files:**
- Modify: `scripts/rebuild_router_dataset.py`

- [ ] **Step 1: Import the niche prompts**

Edit `scripts/rebuild_router_dataset.py`. After the existing FreeCAD-special-handling block but before the `# === HF-sourced domains ===` loop, append:

```python
    # === Niche-domain manually-curated prompts ===
    try:
        from augment_niche_domains import NICHE_DOMAIN_PROMPTS  # type: ignore
        for domain, prompts in NICHE_DOMAIN_PROMPTS.items():
            kept = 0
            for p in prompts:
                if _ok(p):
                    all_rows.append({
                        "prompt": p,
                        "domain": domain,
                        "source": "L'Électron Rare internal (niche curation)",
                        "license": "apache-2.0",
                    })
                    kept += 1
            if kept:
                provenance["sources"].append({
                    "domain": domain,
                    "huggingface_repo": None,
                    "config": None,
                    "split": None,
                    "license_spdx": "apache-2.0",
                    "rows_used": kept,
                    "note": "Curated by L'Électron Rare. See scripts/augment_niche_domains.py.",
                })
            print(f"  niche {domain:14s} +{kept}")
    except Exception as e:
        print(f"  niche import failed: {e}")
```

- [ ] **Step 2: Commit**

Run:
```bash
git add scripts/rebuild_router_dataset.py
git commit -m "feat(router): pull niche curated prompts"
git push
```

---

### Task 2.4: Iterative human curation pass

**Files:**
- Modify (heavily): `scripts/augment_niche_domains.py`

This is the manual step. The collaborator (you) goes through each domain list and adds prompts in batches. The agent runs the rebuild + retrain loop after every batch to give feedback.

- [ ] **Step 1: Pick a target domain (start with the worst one from the confusion matrix)**

Run:
```bash
head -20 docs/transparency/confusion-top10.md
```
Inspect — pick whichever niche domain appears most often in the "Target" column.

- [ ] **Step 2: Add ~50 prompts to that domain in `augment_niche_domains.py`**

Edit the corresponding list (e.g. `KICAD_PCB`) and append ~50 new short prompts. Each prompt should:
- Be 8-300 characters
- Mention domain-specific tools, components, or workflows
- Be in FR or EN (mix is fine — the encoder is multilingual)
- NOT overlap obviously with another domain (a prompt "How do I run pytest?" belongs in `python`, not `embedded`, even if the project is embedded)

- [ ] **Step 3: Rebuild + retrain + evaluate**

Run:
```bash
git add scripts/augment_niche_domains.py
git commit -m "data(router): +50 curated prompts for <domain>"
git push
ssh studio "cd ~/ailiance && git pull --ff-only && rm -rf data/router data/router-clean && \
  ~/KIKI-Mac_tunner/.venv/bin/python scripts/rebuild_router_dataset.py 2>&1 | tail -3 && \
  ~/KIKI-Mac_tunner/.venv/bin/python scripts/build_router_data.py 2>&1 | tail -3 && \
  ~/KIKI-Mac_tunner/.venv/bin/python scripts/train_router.py \
    --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
    --hidden-dim 256 --epochs 30 \
    --output-dir output/router-vN 2>&1 | tail -5"
```
(Replace `vN` with the next version number.)

Expected: final epoch reports `top1=` and `top3=` numbers above v5's 87.6/98.7. If the new domain's confusion improves but overall top-1 drops, rollback the prompts that look weakest.

- [ ] **Step 4: Repeat for the next worst domain**

Loop steps 1-3 for the top 5 niche domains (kicad-pcb, stm32, dsp, iot, web-backend are the typical priorities).

---

### Task 2.5: Final retrain + deployment

**Files:**
- Read: `output/router-vN` (the latest checkpoint after curation)
- Replace: `output/router/` on electron-server
- Update: `docs/transparency/router-training-data.md`

- [ ] **Step 1: Train the final v7 with all niche prompts**

Run:
```bash
ssh studio "cd ~/ailiance && rm -rf data/router data/router-clean && \
  ~/KIKI-Mac_tunner/.venv/bin/python scripts/rebuild_router_dataset.py 2>&1 | tail -3 && \
  ~/KIKI-Mac_tunner/.venv/bin/python scripts/build_router_data.py 2>&1 | tail -3 && \
  ~/KIKI-Mac_tunner/.venv/bin/python scripts/train_router.py \
    --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
    --hidden-dim 256 --epochs 30 \
    --output-dir output/router-v7 2>&1 | tail -5"
```
Expected: top-1 ≥ 90 % (target: 92 %).

- [ ] **Step 2: Verify routing on the production smoke set**

Run on electron-server first to make sure the gateway can load v7 cleanly:

```bash
ssh studio "cd ~/ailiance && tar cz output/router-v7" | \
  ssh electron-server "cd ailiance && tar xz"
ssh electron-server "cd ~/ailiance && BACKUP=output/router-v6-backup-\$(date +%s) && \
  cp -r output/router \$BACKUP && rm -rf output/router && \
  cp -r output/router-v7 output/router && head -10 output/router/meta.json"
```
Expected: meta.json shows `num_domains: 32` (or 34 if Phase 1 chose a different head shape — check Task 1.3 step 4).

- [ ] **Step 3: Restart the gateway**

Run:
```bash
ssh electron-server "tmux kill-session -t ailiance 2>/dev/null; sleep 2; \
  tmux new-session -d -s ailiance \"cd ~/ailiance && \
  KXKM_QWEN_KEY='vllm-er-2026' \
  AILIANCE_WORKERS_JSON='{\\\"9301\\\":\\\"http://100.116.92.12:9301\\\",\\\"9302\\\":\\\"http://100.112.121.126:9302\\\",\\\"9303\\\":\\\"http://100.116.92.12:9303\\\",\\\"9304\\\":\\\"http://100.78.6.122:9304\\\",\\\"8002\\\":\\\"http://localhost:8002\\\"}' \
  .venv/bin/uvicorn src.gateway.server:make_gateway_app --factory --host 0.0.0.0 --port 9300 2>&1 | tee /tmp/ailiance-gateway.log\""
ssh electron-server "until curl -fsS -m 3 http://localhost:9300/health 2>/dev/null | grep -q router_loaded; do sleep 3; done; echo READY"
```
Expected: `READY` within 30 s.

- [ ] **Step 4: Run a 12-prompt smoke test**

Run:
```bash
for q in 'Bonjour' 'Hello' 'Comment dimensionner un buck DC-DC ?' \
         'Write Rust async code' 'Comment dockeriser une app FastAPI ?' \
         'Simule un filtre RC dans NGSPICE' 'Quelle norme IEC 61010' \
         'Configurer un timer STM32F4 en input capture' \
         'Implémenter MQTT QoS 1 sur ESP32' 'Concevoir un filtre FIR' \
         'Créer un endpoint FastAPI Pydantic' \
         'Solve x^2-5x+6=0 step by step'; do
  echo -n "  '$q' -> "
  ssh electron-server "curl -fsS -X POST http://localhost:9300/v1/route \
    -H 'Content-Type: application/json' -d '{\"prompt\":\"$q\"}'" | \
    python3 -c "import json,sys; d=json.load(sys.stdin); \
      print('port=' + str(d.get('chosen_port')) + ' domain=' + str(d.get('chosen_domain')))"
done
```
Expected: at least 11/12 correct routings. If <11, return to Task 2.4 and add more prompts to the failing domain.

- [ ] **Step 5: Update transparency doc**

Edit `docs/transparency/router-training-data.md`. Update §1 (active checkpoint, accuracy numbers), §2.2 (new niche curation row), and §4 (mark the §4 limitation as resolved if top-1 ≥ 90 %).

- [ ] **Step 6: Commit + push the docs**

Run:
```bash
git add docs/transparency/router-training-data.md
git commit -m "docs: router-v7 promoted to active"
git push
```

- [ ] **Step 7: Push v7 to HF for traceability**

Run:
```bash
# Reuse the existing /tmp/hf_push_router_v5.py with REPO + LOCAL changed
sed -i.bak 's|router-v5|router-v7|g; s|ailiance-router-v5-minilm|ailiance-router-v7-minilm|g' /tmp/hf_push_router_v5.py
mv /tmp/hf_push_router_v5.py /tmp/hf_push_router_v7.py
scp /tmp/hf_push_router_v7.py electron-server:/tmp/
ssh electron-server "scp /tmp/hf_push_router_v7.py kxkm@10.2.0.237:/tmp/"
ssh studio "cd ~/ailiance && tar cz output/router-v7" | \
  ssh electron-server "tar xz -C /tmp/ && scp -r /tmp/output/router-v7 kxkm@10.2.0.237:ailiance/output/"
ssh electron-server "ssh kxkm@10.2.0.237 'python3 /tmp/hf_push_router_v7.py 2>&1 | tail -10'"
```
Expected: `clemsail/ailiance-router-v7-minilm` repo created with README + 3 weight files.

---

## Self-Review

Re-read the spec the user gave:

> 1. MAINTENANT : bench Jina v3 vs MiniLM sur le corpus propre + threshold calibration. Décide si on migre l'encoder. → 30 min.
> 2. Ensuite : confusion matrix + script augment_niche_domains.py que tu remplis collaborativement. → 1-2 h.

**Coverage:**
- (1) Bench → Task 1.1 + 1.2
- (1) Threshold calibration → Task 1.3
- (1) Encoder migration decision → Task 1.2 step 3-4
- (2) Confusion matrix → Task 2.1
- (2) `augment_niche_domains.py` collaborative → Task 2.2 + 2.4
- Final deployment → Task 2.5

**Placeholder scan:** All code blocks contain real code. Niche domain lists are starter prompts (~5-10 each) — Task 2.4 explicitly directs the curator to expand them. The "v6/v7" version numbers are concrete.

**Type consistency:**
- `RouterMLP` definition is duplicated in `calibrate_threshold.py` and `build_confusion_matrix.py` to keep each script standalone (the existing `classifier.py` builds the same shape). Both use `tnn.Sequential(Linear → GELU → Dropout(0.1) → Linear)` matching `_build_mlp` in `src/router/classifier.py`.
- `meta.json` schema (`embedding_model`, `embedding_dim`, `hidden_dim`, `num_domains`, `domains`) is the same shape produced by `train_router.py`.
- `data/router/valid.jsonl` row format `{prompt, domain}` is what `build_router_data.py` writes.

No issues found.
