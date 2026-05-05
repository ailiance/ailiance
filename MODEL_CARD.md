# eu-kiki — Model Card

**System:** eu-kiki — EU-sovereign multi-model LLM serving pipeline
**Version:** 0.4.0
**Date:** 2026-05-05
**License:** Apache-2.0
**Risk classification (EU AI Act):** Limited risk — Article 52
**Operator:** L'Electron Rare (`electron-rare` / `L-electron-Rare`)
**Repo:** https://github.com/L-electron-Rare/eu-kiki

---

## 1. System overview

eu-kiki dispatches user queries via a Jina v3 + MLP router (40 domains,
sigmoid multi-label) to one of three EU/CH foundation models, each
augmented with LoRA adapters trained on HF-traceable datasets. Local-only
deployment, no cloud, no telemetry.

```
client → gateway:9200 → router → worker:9301/9302/9303 → response
                                       │
                                       └── base model + LoRA(domain)
```

## 2. Models served

| Model | Origin | Params | Adapters | Port |
|---|---|---:|---:|---|
| Apertus-70B-Instruct-2509 | Swiss AI (EPFL/ETH/CSCS) 🇨🇭 | 70.6 B | 20 | `:9301` |
| Devstral-Small-2-24B-Instruct-2512 | Mistral AI 🇫🇷 | 24 B | 16 | `:9302` |
| EuroLLM-22B-Instruct-2512 | utter-project 🇪🇺 | 22.6 B | 4 | `:9303` |

All three Apache-2.0. Full provenance per model in
[`docs/eu-ai-act-transparency.md`](docs/eu-ai-act-transparency.md) §2.

## 3. Adapter training

| Parameter | Value |
|---|---|
| Method | LoRA (rank 16, alpha 32, dropout 0.05) |
| Targets | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Precision | BF16 |
| LR | 1e-5 (AdamW) |
| Batch | 1 × grad-accum 4–8 |
| Framework | MLX (`mlx_lm` fork) |
| Iterations | 500–4000 per domain |
| Datasets | HF-traceable, ≤ 3000 rows/domain, deduplicated |

Per-record provenance (`_provenance.source`, `_provenance.license`,
`_provenance.access_date`) attached to 49,956 training records across 21
HF-traced domains (compliance remediation 2026-04-28, transparency.md
v0.3.0).

## 4. Measured performance

All benchmarks reproducible from this repo
([`eval/results/SUMMARY.md`](eval/results/SUMMARY.md),
[`eval/WORKFLOW.md`](eval/WORKFLOW.md)). Each result directory contains
`env.json` (hardware/git/pip), `methodology.md`, `rerun.sh`, and the
official scorer artifacts.

### 4.1 HumanEval+ — Devstral 24B 4-bit + eu-kiki v1 adapters

164 problems, 1 sample, temperature 0.0, greedy. Linux scoring on
`kx6tm-23` (Proxmox PVE 6.17, EvalPlus official sandbox) for base/python.
macOS custom-subprocess scorer (extra-tests skipped) for cpp/rust.

| Model | HE base | HE+ | Δ HE+ vs base | Scorer |
|---|---:|---:|---:|---|
| Devstral base | 87.20 % | 82.90 % | ref | EvalPlus Linux |
| + python | 86.00 % | 81.10 % | **−1.80** | EvalPlus Linux |
| + cpp | 85.98 % | 85.98 % | **−1.22** | macOS custom |
| + rust | 86.59 % | 86.59 % | **−0.61** | macOS custom |

### 4.2 MT-Bench — Devstral 24B 4-bit (no adapter)

80 questions × 2 turns, judge = Mistral-Medium-3.5-128B-MLX-4bit (local).

| Metric | Value |
|---|---|
| Overall | **8.892 / 10** |
| Turn 1 / Turn 2 | 9.42 / 8.33 |
| writing | 9.33 |
| math, coding, stem | 10.0 |
| roleplay | 7.40 |
| reasoning | 7.83 |
| Caveat | 37/160 turns parseable; bias toward verbose-judge categories |

### 4.3 GSM8K 5-shot, n=200 — Qwen3.6-35B-A3B-4bit

External validation. Custom runner (Lighteval/LiteLLM bypassed — both
hit HuggingFace tokenizer 404 even offline). Direct OpenAI-compat call,
last-numeric-token extraction.

| Model | Accuracy | Δ vs base |
|---|---:|---:|
| base | **94.5 %** (189/200) | ref |
| + reasoning fused | 94.5 % | 0 |
| + math fused | 90.0 % | **−4.5** |

### 4.4 KIKI-DSL v3 — Qwen3.6-35B-A3B-4bit

Custom KIKI-native bench, 15 prompts (10 named-IC + 5 SPICE-pure
abstract net labels). v3 corrects the v1 named-IC bias.

| Adapter | Pass | Avg | Δ pass v3 |
|---|---:|---:|---:|
| reasoning | 86.7 % | 0.705 | **+13.4** |
| math | 80.0 % | 0.726 | +6.7 |
| security | 80.0 % | 0.651 | +6.7 |
| **base** | **73.3 %** | **0.704** | ref |
| chat-fr | 73.3 % | 0.615 | 0 |
| kicad-pcb | 66.7 % | 0.598 | −6.7 |
| spice-sim | 53.3 % | 0.614 | −20 |
| components | 53.3 % | — | −20 |
| kicad-dsl | 46.7 % | 0.477 | **−27** |

### 4.5 Cross-bench reading

| Adapter | KIKI-DSL v3 (Δ pass) | GSM8K (Δ acc) |
|---|---:|---:|
| reasoning | **+13.4** | 0 |
| math | +6.7 | **−4.5** |

The KIKI-DSL v3 wins of `reasoning` and `math` **do not transfer to
saturated public math benchmarks**. Honest read: cognitive scaffolding
helps when the base is at ~73 % on a format-complex task, not when the
base is near-ceiling.

## 5. Intended use

- Domain-specialized assistance on the 40 routed domains, French-first.
- Local development, R&D, prototype tooling.
- KiCad / electronics / embedded / EU-norm-adjacent technical writing.

## 6. Out-of-scope use

- **Safety-critical** decisions (medical, legal, structural, life-safety).
- **High-stakes individual decisions** (hiring, credit, biometric, law
  enforcement) — would re-classify under EU AI Act Art. 6 high-risk.
- Production systems requiring guaranteed factuality on saturated public
  benchmarks (reasoning/math adapters do not transfer; see §4.3).
- Multi-tenant cloud deployment (system is local-only by design).

## 7. Known limitations

1. **Hallucination** present (medium-frequency on long-tail factual
   queries). Not mitigated by current adapters.
2. **Domain-narrow adapters** (`kicad-dsl`, `components`, `spice-sim`)
   show **negative transfer** of −20 to −27 pts on balanced KIKI-DSL v3.
   Their training favored named-IC SPICE-compact style; they degrade on
   abstract-label prompts.
3. **`chat-fr` adapter** shows a +10 win on KIKI-DSL v1 that **does not
   replicate on v3** — the win was test-set-specific (refusal-preamble
   suppression on biased prompts).
4. **Public-bench saturation:** Qwen 35B-A3B-4bit is at 94.5 % GSM8K
   without adapter; reasoning/math adapters cannot improve from there.
5. **HumanEval+ adapters slightly degrade** the base (-0.6 to -1.8 pts).
   Style mismatch (verbose chat-instructional vs terse algorithmic
   completion). Safe in chat production, not on raw HumanEval.
6. **MT-Bench score (8.89) is partial** — only 23 % of turns produced a
   parseable `[[rating]]` (judge-runner regex bug). Score biased toward
   writing-heavy categories.
7. **Custom Studio scorer** for cpp/rust HumanEval+ does not run
   EvalPlus extra-tests (Linux-only sandbox); HE base = HE+ in those
   rows. For rigorous Δ HE+, samples must be re-scored on Linux.

## 8. Reproducibility

```bash
git clone https://github.com/L-electron-Rare/eu-kiki
cd eu-kiki
uv venv && uv pip install -e ".[dev,router,data]"

# Re-run any benchmark from its result directory
bash eval/results/2026-05-04/<run-id>/rerun.sh
```

Each `rerun.sh` is self-contained and prints the captured `env.json`
(model SHA, adapter SHA, hardware, git commit, pip freeze). Linux scoring
requires SSH access to a Linux host with EvalPlus installed (we used
`kx6tm-23`).

Three machines were involved in the eu-kiki bench runs:
- **GrosMac** — code/git transit (M5, 16 GB)
- **macM1** — codegen + 4-bit fuse + bench (M1 Max, 32 GB)
- **studio** — heavy training + BF16 fuse + bench (M3 Ultra, 512 GB)
- **kx6tm-23** — EvalPlus official scoring (Linux x86_64, sandbox)

Topology + bug history + workarounds: [`eval/WORKFLOW.md`](eval/WORKFLOW.md).

## 9. Compliance summary (EU AI Act)

| Article | Coverage |
|---|---|
| Art. 52 (transparency to users) | Disclosed: AI-generated, model identity, EU origin |
| Art. 53(1)(a) (technical doc) | This file + transparency.md |
| Art. 53(1)(b) (training data summary) | transparency.md §4 + per-domain `MANIFEST.json` |
| Art. 53(1)(c) (copyright policy) | DSM Art. 4 TDM + robots.txt + opt-out logged in `data/quarantine/` history |
| Art. 53(1)(d) (evaluation summary) | This file §4 + `eval/results/SUMMARY.md` |
| Art. 53(2) (open-source exemption) | All 3 served models Apache-2.0; teacher Mistral-Medium 128B is research-only, never deployed |
| Art. 55 (systemic risk) | N/A — no foundation model > 10²⁵ FLOPs trained here; we only fine-tune via LoRA |

## 10. Contact

- **System operator:** L'Electron Rare (`electron-rare@…`)
- **Apertus PII/copyright:** `llm-privacy-requests@swiss-ai.org`
- **Issues / audit requests:** GitHub Issues on this repo

## 11. Changelog

| Date | Version | Change |
|---|---|---|
| 2026-04-28 | 0.3.0 | Provenance remediation, license normalization (transparency.md) |
| 2026-05-05 | 0.4.0 | First model card; full benchmark suite published (HumanEval+, MT-Bench, GSM8K, KIKI-DSL v3); cross-bench transfer analysis added; known-limitations section consolidates v3 taxonomy revision |
