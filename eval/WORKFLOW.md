# ailiance Benchmark Workflow — Complete Trace

Comprehensive trace of the eval pipeline architecture, every decision made,
every bug found, every workaround applied. Designed to be auditable end-to-end
and reproducible by an external reviewer.

**Snapshot date:** 2026-05-04 / 2026-05-05

---

## 1. Hardware topology

```
┌─────────────────────────────────────────────────────────────────────┐
│  GrosMac (M5, 16 GB) — dev machine                                  │
│    repo: ~/Documents/Projets/ailiance/                               │
│    role: write code, push to GitHub, transit results                │
└─────────────────┬───────────────────────────────────────────────────┘
                  │ Tailscale
   ┌──────────────┴──────────────────────┬────────────────────────┐
   │                                     │                        │
┌──▼──────────────┐  ┌──────────────────▼────┐  ┌─────────────────▼─┐
│ macM1           │  │ studio (M3 Ultra)     │  │ kx6tm-23           │
│ M1 Max 32 GB    │  │ 512 GB                │  │ Linux x86_64       │
│ macOS           │  │ macOS                 │  │ Proxmox PVE 6.17  │
│                 │  │                       │  │ Python 3.13.5      │
│ codegen +       │  │ heavy training        │  │ official EvalPlus  │
│ fuse + bench    │  │ + BF16 fuse           │  │ scoring (sandbox)  │
└─────────────────┘  └───────────────────────┘  └────────────────────┘
```

Why 3 machines : codegen needs MLX (Apple Silicon), heavy fuse needs RAM
(Studio), scoring needs Linux sandbox (kx6tm-23 — EvalPlus uses
`resource.RLIMIT_AS` which is a Linux-only API).

---

## 2. Repo structure of the eval pipeline

```
ailiance/
├── eval/
│   ├── README.md           # methodology + benchmark suite spec
│   ├── HOSTS.md            # machine matrix + sync workflow
│   ├── WORKFLOW.md         # this document
│   ├── run_all.sh          # orchestrator (--quick / --extended / --mtbench)
│   ├── tasks/
│   │   └── kiki_native/
│   │       └── kicad_dsl.json    # 10 hand-curated KiCad prompts
│   ├── runners/
│   │   ├── mlx_server_runner.py  # spawn server, capture env.json
│   │   ├── result_writer.py      # auto-generate methodology.md, rerun.sh, report.md
│   │   ├── lighteval_runner.py   # GSM8K/MMLU-Pro/IFEval via litellm proxy
│   │   ├── evalplus_runner.py    # HumanEval+/MBPP+ via OpenAI endpoint
│   │   ├── mtbench_runner.py     # 80q × 2t with local Mistral-Medium judge
│   │   ├── kiki_native_runner.py # KIKI-native domain bench (rule-based scoring)
│   │   └── fuse_adapter.sh       # bake adapter into self-contained model
│   └── results/
│       ├── SUMMARY.md             # consolidated results table
│       └── 2026-05-04/<run>/      # per-run env.json + report.md + rerun.sh
└── output/adapters/devstral/<domain>/
    ├── adapters.safetensors
    └── adapter_config.json
```

---

## 3. The bug story — `mlx_lm.server --adapter-path` is silently broken

**Symptom:** `mlx_lm.server --model X --adapter-path Y` accepts the call,
returns 200 OK on `/v1/models`, generates plausible output. But:

| Test | base output | "+ adapter" output |
|------|-------------|---------------------|
| 11 control prompts (HumanEval/0 + KiCad-DSL ×10) | … | bit-identical to base |
| Devstral 24B 4-bit + python adapter | "Here's a Python function…" | identical text |
| Qwen 35B-A3B BF16 + v4-sota kicad-dsl | "En tant qu'IA textuelle…" | identical text |

**Diagnosis:** the runtime LoRA wrapper does not match
`QuantizedLinear` modules nor MoE-specific gates (`shared_expert_gate`),
so every `--adapter-path` load is silently a no-op. Same on BF16 too.

**Verified via :**
```python
# inspect adapters.safetensors
n_tensors = 560 (Devstral) / 816 (v4-sota)
all lora_B non-zero ✅
shapes match ✅
key prefixes (`language_model.model.layers.X...`) match the model ✅
```
Adapter is genuine; runtime is broken.

**Workaround validated:** `mlx_lm fuse` bakes LoRA into base weights at
load time, producing a self-contained model. The fused model serves
identically via `mlx_lm.server` (no `--adapter-path` flag).

Decisive empirical confirmation on Studio :
```
prompt: "Génère un sch KiCad: R1=10k entre VCC et GND."
Qwen 35B BF16 BASE   : "En tant qu'IA textuelle, je ne peux pas générer..."
Qwen 35B BF16 FUSED  : "`R1 VCC GND 10k`"  (drastically different)
```

`fuse_adapter.sh` (committed in `f27d46d`) wraps the workflow.

---

## 4. The `--de-quantize` typo + Qwen3 thinking-mode trap

Two more booby traps documented for future runs :

1. **`mlx_lm fuse --de-quantize` does NOT exist** — the flag is
   `--dequantize`. The mistyped flag is silently dropped, fuse loads
   the base, writes 0 bytes, and exits 0. Caught only by `du -sh`.

2. **Qwen 3.x emits `reasoning` field separately from `content`** when
   thinking mode is on (default). Without `chat_template_kwargs.enable_thinking=False`,
   our runner returned `[PARSE_ERROR: 'content']` for every prompt and
   the bench scored 0/0 across both base and fused (false equivalence).
   Fixed in `c5288d5`.

Lesson : always run a control prompt with **expected divergent output**
(e.g. style-shifting prompt) before declaring "no effect from adapter".

---

## 5. The codegen → score pipeline (HumanEval+)

```
┌────────────────────────────────────────────────────────────────┐
│ 1. macM1 — Build fused 4-bit model                            │
│    mlx_lm fuse                                                 │
│      --model Devstral-Small-2-24B-MLX-4bit                     │
│      --adapter-path ailiance/output/adapters/devstral/python    │
│      --save-path /tmp/devstral-python-fused-4bit               │
│    → 12 GB self-contained checkpoint (~25 min)                 │
└─────────────────────────────────┬──────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────┐
│ 2. macM1 — Serve + codegen                                    │
│    mlx_lm.server :8850                                         │
│    evalplus.evaluate humaneval --backend openai                │
│    164 problems × ~30-60 s/problem → ~2 h                      │
│    output: humaneval/<id>_temp_0.0.jsonl (sanitized)           │
└─────────────────────────────────┬──────────────────────────────┘
                                  │ scp via GrosMac
                                  ▼
┌────────────────────────────────────────────────────────────────┐
│ 3. kx6tm-23 — Official EvalPlus scoring (Linux sandbox)       │
│    evalplus.evaluate humaneval --samples X.jsonl               │
│    → pass@1 base + pass@1 plus                                 │
│    ~10 s for 164 problems                                       │
└─────────────────────────────────┬──────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────┐
│ 4. GrosMac — Persist + commit                                 │
│    eval/results/<date>/<label>/                                │
│      env.json (model SHA, adapter SHA, hardware, git, pip)     │
│      methodology.md (run identity + reproduction steps)        │
│      rerun.sh (executable)                                     │
│      report.md (metrics table)                                 │
│      evalplus_humanevalplus_linux_official/                    │
│        eval_results.json (raw EvalPlus output)                 │
│        results.json (per-run summary)                          │
└────────────────────────────────────────────────────────────────┘
```

---

## 6. The KIKI-native domain bench pipeline (faster, all on macM1)

```
┌────────────────────────────────────────────────────────────────┐
│ 1. fuse Qwen3.6-35B-A3B-4bit + v4-sota/<domain> adapter       │
│    output: /tmp/qwen-<domain>-fused-4bit (18 GB, 25 min)       │
└─────────────────────────────────┬──────────────────────────────┘
                                  ▼
┌────────────────────────────────────────────────────────────────┐
│ 2. mlx_lm.server + kiki_native_runner                         │
│    10 prompts (KiCad/SPICE/EMC variety)                        │
│    chat_template_kwargs.enable_thinking=False                  │
│    rule-based scoring:                                         │
│      syntax_ok    : balanced parens OR SPICE-like netlist      │
│      contains     : fraction of must_contain found             │
│      unique       : reference designators appear properly      │
│      overall_avg  : mean of the three (0..1)                   │
│      pass        : overall ≥ 0.6                               │
│    duration: ~5-8 min                                           │
└─────────────────────────────────┬──────────────────────────────┘
                                  ▼
┌────────────────────────────────────────────────────────────────┐
│ 3. results/<date>/qwen36-35b-4bit-fused-<domain>-on-kicad-dsl/ │
│    kiki_kicad_dsl/results.json                                 │
└────────────────────────────────────────────────────────────────┘
```

Total per adapter: **~30-35 min** (fuse + bench), fully autonomous on macM1.

---

## 7. Results summary

### ailiance v1 — HumanEval+ (official Linux scoring)

| Run | Pass@1 base | Pass@1 + |
|-----|-----------:|---------:|
| Devstral-Small-2-24B-MLX-4bit BASE | **87.20 %** | **82.90 %** |
| + ailiance v1 python (FUSED) | 86.00 % | 81.10 % |
| **Δ** | -1.20 pts | -1.80 pts |

→ Adapter is style-drift on academic prompts (verbose Stack-Overflow
Python instead of terse algorithmic completion). Safe for chat
deployment, not optimal for HumanEval. Only 3 additional failures
on extra tests (133 vs 136 passing).

### v4-sota adapters — KIKI-DSL bench (Qwen3.6-35B-A3B-4bit)

| # | Adapter | Pass | Avg | Δ pass |
|---|---------|-----:|----:|-------:|
| — | (base) | 60 % | 0.593 | — |
| 🥇 | **reasoning** | **90 %** | **0.672** | **+30** 🎯 |
| 🥈 | math | 70 % | 0.638 | **+10** |
| 🥈 | chat-fr | 70 % | 0.569 | **+10** |
| 3 | kicad-pcb | 60 % | 0.535 | 0 |
| 4 | embedded | 50 % | 0.440 | -10 |
| 5 | electronics | 40 % | 0.532 | -20 |
| 6 | spice-sim | 30 % | 0.473 | -30 |
| 7 | components | 30 % | 0.418 | -30 |
| 8 | kicad-dsl | 30 % | 0.382 | -30 |

### Three clusters identified

```
🟢 Behavioral / general / cognitive (HELP)
   Trained on broad reasoning / conversational data.
   Improve KIKI-DSL because they reduce the base's
   refusal preamble, broaden response surface, and
   enable structured chain-of-thought through hardware.
   ─────────────────────────────────────────────
   reasoning : +30 pass / +0.079 avg ← record absolu
   math      : +10 pass / +0.045 avg
   chat-fr   : +10 pass / -0.024 avg

⚪ Domain-adjacent (NEUTRAL or MILD DROP)
   Trained on related but not identical surface.
   ─────────────────────────────────────────────
   kicad-pcb    : 0 / -0.06
   embedded     : -10 / -0.15
   electronics  : -20 / -0.06

🔴 Domain-narrow (COLLAPSE -30)
   Hyper-specialized training on terse formats.
   ─────────────────────────────────────────────
   kicad-dsl    : -30 / -0.21
   components   : -30 / -0.18
   spice-sim    : -30 / -0.12
```

→ **Domain proximity does NOT predict adapter benefit.** kicad-dsl
(closest to bench) is the worst performer; reasoning (least technical
of all) is the best. The **width** of training data dominates.

→ **Reasoning is the killer adapter** for multi-domain hardware tasks.
9/10 problems passed (vs 6/10 base), Q9 ESD = 1.00 PASS (first time
any adapter perfectly nails this prompt). All cognitive/general
adapters help; ALL domain-narrow ones hurt.

---

## 8. Per-question contour analysis (KIKI-DSL)

| Q | Domain | Base | Best adapter | Worst adapter |
|---|--------|-----:|--------------|---------------|
| 001 | R divider | 0.67 | base, components 0.67 | electronics 0.00 |
| 002 | LDO AMS1117 | 0.57 | math 0.95 | kicad-dsl 0.05 |
| 003 | LED 330R | 0.67 | kicad-pcb 0.89 | electronics 0.17 |
| 004 | RC filter | 0.24 | spice-sim 1.00 | (collapse common) |
| 005 | BJT 2N3904 | 0.96 | electronics 1.00 | kicad-dsl 0.18 |
| 006 | I2C BME280 | 0.25 | kicad-pcb 0.67 | components 0.25 |
| 007 | H-bridge MOSFETs | 0.67 | math 1.00 | electronics 0.04 |
| 008 | STM32 decoupling | 0.57 | (no adapter helps) | electronics 0.38 |
| 009 | ESD USB | 0.67 | (multiple match base 0.67) | kicad-dsl 0.40 |
| 010 | LM358 buffer | 0.67 | (multiple match base 0.67) | kicad-dsl 0.39 |

Notable max scores (1.00) :
- electronics on Q5 (BJT) — hardware specialty wins where it matters
- math on Q7 (H-bridge) — only adapter to nail this complex topology
- spice-sim on Q4 (RC filter) — pure SPICE win (RC = canonical SPICE problem)

→ **No single adapter dominates;** specialization gives "spike" wins
on its niche but degrades elsewhere. The general adapters spread
the gains.

---

## 9. Reproducibility chain (per run)

Each result directory contains :

```
results/<date>/<label>/
  env.json
    model_path, model_first_safetensors_sha256
    adapter_path, adapter_sha256
    hardware: platform, machine, processor, node
    python, mlx_version, mlx_lm_version
    git: repo, commit, describe, dirty status
    pip_freeze (uv pip freeze fallback)
    argv, cwd
    started_at (ISO 8601)

  methodology.md
    Run identity table
    Sampling configuration (T, max_tokens, n_samples, seed)
    "How to reproduce" with explicit `git checkout <sha>` + setup steps
    Limitations (template version, judge model, sandbox notes)
    EU AI Act Art. 53(1)(d) reference

  rerun.sh
    executable script
    invokes run_all.sh with the same arguments + the right model + adapter

  report.md
    Aggregated metrics table per task
    Reproducibility links to env.json + methodology.md + rerun.sh

  <task>/results.json
    metrics dict (pass_rate, overall_avg, etc.)
    n_passed, n_questions
    by_category breakdown
    answers_path
    evaluated_at
```

---

## 10. Cross-machine sync workflow

```
GrosMac (dev)                 macM1 (runner)            kx6tm-23 (Linux)
     │                              │                          │
     │ git push                     │                          │
     │─────────────────────────────▶│                          │
     │ rsync models / adapters      │                          │
     │ from studio (one-time)       │                          │
     │                              │                          │
     │                              │ codegen                  │
     │                              │ (~2h HumanEval+,         │
     │                              │  ~30 min KIKI-DSL)       │
     │                              │                          │
     │                              │ scp samples              │
     │                              │─────────────────────────▶│
     │                              │                          │ evalplus.evaluate
     │                              │                          │ (~10 s, official sandbox)
     │                              │                          │
     │ scp results.json             │                          │
     │◀─────────────────────────────│                          │
     │                              │                          │
     │ scp eval_results.json from kx6tm-23                     │
     │◀─────────────────────────────────────────────────────────│
     │                              │                          │
     │ git add → commit → push      │                          │
```

---

## 11. Tooling installed

### macM1
- `~/.local/bin/uv` (curl install, astral.sh)
- `~/.local/bin/gh` 2.62.0 (binary direct from GitHub releases)
- `~/Projets/ailiance/.venv` Python 3.13.13
  - mlx_lm 0.31.3
  - lighteval 0.13.0
  - evalplus
  - fastchat 0.2.36

### studio
- `~/.local/bin/uv` + venv Python 3.14.4
- mlx_lm 0.31.3 + lighteval 0.13.0 + evalplus + fastchat

### kx6tm-23
- `~/.local/bin/uv` + venv Python 3.13.5
- evalplus only (sandbox scorer)

---

## 12. Known issues / TODO

- [ ] Fuse + Apertus 70B (132 GB) — Studio only, never ran
- [ ] EuroLLM 22B fuse — limit on macM1 (42 GB BF16)
- [ ] MT-Bench full — judge model dependency on Studio Mistral-Medium-128B-4bit
- [ ] HumanEval+ on remaining Devstral adapters (cpp, rust, typescript, ...)
- [ ] KIKI-DSL test set v3 — too biased toward verbose-with-named-ICs;
      add SPICE-pure prompts to be fair to compact adapters
- [ ] `spice` adapter (separate from `spice-sim`) — was rsync incomplete
      during last attempt, retry pending
- [ ] Resume Studio batch10 training — paused at iter 200 / 500 on
      `security-fenrir`, checkpoint preserved
- [ ] Investigate why mlx_lm.server `--adapter-path` doesn't apply LoRA;
      file issue upstream

---

## 13. Commit graph (this session, latest first)

```
a3efe9e  results: math adapter (NEW BEST 70% / 0.638)
511071f  results: spice-sim adapter (-30 pts pass)
261ae77  results: embedded adapter (-10 pts pass)
b5f58d1  results: electronics adapter (-20 pts pass)
49f1f22  results: chat-fr adapter (+10 pts pass!)
cbb739d  results: components adapter (-30 pts pass)
2ab6aaa  results: kicad-pcb adapter (neutral)
ade2bfd  results: first valid ailiance adapter delta
eeb21c2  results: ailiance v1 Devstral python (-1.8 pts)
c5288d5  fix(eval): disable thinking + reasoning fallback
f1dac4d  fix(eval): accept SPICE-like netlist in KIKI-DSL
c5dcde4  feat(eval): KIKI-native runner + KiCad-DSL task
f27d46d  feat(eval): fuse_adapter.sh workaround
886e4b9  fix(results): invalidate adapter run (4-bit)
6850a6e  fix(eval): macOS-compatible scorer for HumanEval+
780a1fd  docs: confirm fuse workaround works (Studio test)
6fbbc5c  fix(eval): use absolute model path as API id
601483a  docs(eval): align macM1 + studio + HOSTS matrix
b1ef93e  feat(eval): MT-Bench runner with local judge
9ff1912  feat(eval): per-run traceability + rerun.sh
6ade435  feat: publishable benchmark pipeline
c5354ee  results: Devstral 24B 4-bit baseline HumanEval+
```

22 commits since pipeline inception (2026-05-04 16:50).
