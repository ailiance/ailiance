# Router training data — full transparency record

**EU AI Act Annex IV §2(b) — training data documentation for the
`eu-kiki/auto` domain router.**

This document is the canonical training-data record for the router head
classifier. It is intentionally honest about known issues; do not treat
this artefact as a high-quality classifier without reading below.

---

## 1. Current status (as of 2026-05-06)

| Item | Value |
|---|---|
| Active checkpoint | `output/router-v6/router.safetensors` |
| Encoder | `sentence-transformers/all-MiniLM-L6-v2` |
| Head | 2-layer MLP (384 → 256 → 32) |
| Total training examples | ~7 800 |
| Total validation examples | ~2 000 |
| Top-1 accuracy on validation | **87.7 %** |
| Top-3 accuracy on validation | ~98 % |
| Threshold | 0.50 (calibrated on v6 head — see scripts/calibrate_threshold.py) |
| HuggingFace mirror | https://huggingface.co/clemsail/eu-kiki-router-v6-minilm |

The §4 limitation that affected v3 (label drift in the noisy auto-classified
corpus) is now resolved: v6 trains on the AI-Act-traceable clean corpus
(`data/router-clean/`) plus 150 manually-curated niche-domain prompts
(`scripts/augment_niche_domains.py`).

---

## 2. Sources

The training data is built by `scripts/build_router_data.py` which
walks `~/KIKI-Mac_tunner/data/micro-kiki/classified/*.jsonl`.

### 2.1 Original corpus (pre-2026-05-05)

| Source | Volume | Origin |
|---|---|---|
| `KIKI-Mac_tunner/data/micro-kiki/classified/*.jsonl` (32 files) | ~45 000 user-style prompts | L'Électron Rare internal corpus produced ~April 2026 by an automated classifier that assigned domain labels to a generic instruction-tuning dataset |

The internal classifier used to label these prompts was **not validated
manually**. Subsequent inspection (May 2026) shows that many `.jsonl`
files contain prompts that do not match their declared domain — see §4.

### 2.2 Post-2026-05-05 augmentations (curated by hand)

| File | New prompts | Author | Date | Subject |
|---|---|---|---|---|
| `calcul-normatif.jsonl` | 223 (file created) | L'Électron Rare | 2026-05-05 | IEC / EN / CE / RoHS / WEEE / NF norms |
| `docker.jsonl` | +103 | L'Électron Rare | 2026-05-05 | Real Docker / Compose / Buildkit prompts |
| `spice.jsonl` | +99 | L'Électron Rare | 2026-05-05 | NGSPICE / LTspice circuit simulation |
| FR/EN code-switched | 103 across 27 domains | L'Électron Rare | 2026-05-05 | Bilingual prompts to reduce language over-fit |
| `chat-fr.jsonl` | +193 | L'Électron Rare | 2026-05-05 | Short FR/EN greetings + small talk |

Scripts: `scripts/augment_router_data.py`, `scripts/augment_short_greetings.py`.

---

## 3. Copyright / licensing posture

- All prompts in §2.2 are written or paraphrased internally. No
  scraping. No copyrighted source text.
- The original §2.1 corpus comes from an internal classifier applied to
  what is reported to be an open instruction-tuning dataset. The exact
  upstream provenance is not currently captured. Action item: trace it
  back and add the source in `docs/transparency/router-training-data-source-trace.md`.
- All artefacts published under Apache-2.0.

---

## 4. Known data-quality issue (label drift)

**The original §2.1 corpus is mis-labelled.** Spot-checks on
2026-05-05 show:

| Domain file | First-row content | Actually about |
|---|---|---|
| `docker.jsonl` row 1 | "Give a brief overview of the French Revolution" | History |
| `rust.jsonl` row 1 | "Explain the concept of photosynthesis" | Biology |
| `math.jsonl` row 1 | "Crée un budget prévisionnel..." | Finance |
| `spice.jsonl` row 1 | "Trouve une équation du mouvement uniformément accéléré" | Physics (vaguely) |
| `reasoning.jsonl` row 1 | "Provide a step-by-step guide to setting up a personal budget" | Personal finance |
| `chat-fr.jsonl` row 1 | "Rédige un court essai sur la conservation de l'environnement" | Environment essay |

Consequence: the router learns to recognise *text style and length*
(short vs long, code vs prose, FR vs EN) rather than the technical
*domain* of the prompt. Reported 65 % top-1 accuracy is therefore
inflated by intra-style consistency, not domain understanding.

The May 2026 augmentations (§2.2) inject correctly-labelled prompts but
they are 2 orders of magnitude smaller than the noisy bulk.

---

## 5. Pre-processing

`scripts/build_router_data.py`:

1. Walk the `classified/*.jsonl` directory.
2. For each row, extract `prompt` (or first user content if `messages`).
3. Discard rows with empty prompts.
4. Group by `domain` (filename stem).
5. Shuffle each group with seed `42`.
6. Take 80 % into `train.jsonl`, 20 % into `valid.jsonl`.
7. Train via `scripts/train_router.py`: 30 epochs, AdamW lr=1e-3,
   batch size 128, BCE-with-logits multi-label.

No deduplication, no language detection, no quality filtering, no PII
scrubbing on the original corpus. (PII is unlikely given the
instruction-tuning origin, but not verified.)

---

## 6. Train / valid leakage check

Status: **not done**. Action item: run a sha256 set-intersection across
`train.jsonl` and `valid.jsonl` user prompts.

---

## 7. Action plan to lift the §4 limitation

Three options under consideration. Pick one and document the choice
here:

1. **Manual curation** — discard the noisy §2.1 corpus, expand each
   domain to ~200-300 hand-written prompts (like calcul-normatif on
   2026-05-05). Total ~10 k clean prompts. Effort: high. Quality:
   highest.
2. **LLM re-classification** — pass every existing prompt through
   Apertus 70B (in-house), keep only those the LLM confirms match the
   declared domain. Effort: medium (~1 GPU-day on studio). Quality:
   medium-high. Costs zero external dependency.
3. **External replacement** — bring in licensed open datasets per
   domain (CodeAlpaca, Stanford-Alpaca-cleaned, etc.). Effort: low.
   Quality: variable. Adds external provenance to track.

Recommendation: option 2 first (cheap to run, fixes the worst
mis-labellings), then option 1 to fill the long-tail domains where
the §2.1 corpus is too small to begin with (`stm32`, `platformio`,
`freecad`, `ml-training` — all under 100 lines each).

---

## 8. Reproducibility

```bash
# regenerate splits + retrain from scratch
cd eu-kiki
~/KIKI-Mac_tunner/.venv/bin/python scripts/build_router_data.py
~/KIKI-Mac_tunner/.venv/bin/python scripts/train_router.py \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --hidden-dim 256 --epochs 30 --output-dir output/router-vN
```

Determinism: random seed 42 in `build_router_data.py`; PyTorch / MLX
training is not bit-deterministic across runs even with seeded RNGs.
Reported metrics in §1 should reproduce within ±0.5 percentage points.

---

## 9. Contact

Issues with router routing decisions, data-quality reports, or right to
opt out: `postmaster@saillant.cc`. We aim to respond within 7 working
days.

## Encoder migration decision (2026-05-05)

After running scripts/router_benchmark.py --full on the v5 clean corpus
and training a Jina-conditioned head (router-v6, 512 hidden, 30 epochs).
Head top-1 numbers come from `train_router.py` final-epoch output:
MiniLM v5 baseline (`output/router-v5/`, see its `PROVENANCE.json`) and
the freshly trained Jina-conditioned head at `output/router-v6/` on
studio (kept for reproducibility but not promoted to active):

- MiniLM L6 v2 (active, router-v5): top-1 = 87.6 %, encode = 1.6 ms/prompt, Δ separation = 0.345
- Jina v3 (candidate, router-v6, task=separation): top-1 = 87.4 %, encode = 9.7 ms/prompt, Δ separation = 0.150
  - (task=classification variant: encode = 16.6 ms/prompt, Δ separation = 0.152)
- Δ top-1: -0.2 pp
- Δ latency: +8.1 ms

Decision: KEEP MiniLM.

Rationale: the migration rule requires Jina-trained-head top-1 to beat
MiniLM by at least +3 pp. Jina v3 actually under-performs MiniLM by
0.2 pp on the same v5 clean corpus, while costing ~6x more encode
latency per prompt. The encoder-only Δ separation is also lower for
Jina (0.15 vs 0.34), suggesting Jina's high-dimensional space is less
linearly separable for our 32-domain taxonomy. No reason to migrate.
