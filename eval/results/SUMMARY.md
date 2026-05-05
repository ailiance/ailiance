# eu-kiki Benchmark Results — Summary

Aggregated table of all publishable benchmark runs. Each result entry
links to its self-contained directory with `env.json`, `methodology.md`,
`rerun.sh`, and the official scoring artifacts.

## External validation — GSM8K 5-shot (n=200)

Custom runner (Lighteval/LiteLLM bypass — both kept hitting HuggingFace
tokenizer 404 even with `HF_HUB_OFFLINE=1`). Direct call to OpenAI-compat
endpoint, last-numeric-token answer extraction, exact-match scoring.

| Model | Accuracy | Δ vs base |
|---|---:|---:|
| Qwen3.6-35B-A3B-MLX-4bit (base) | **94.5 %** (189/200) | ref |
| + reasoning v4-sota fused | 94.5 % (189/200) | **0** |
| + math v4-sota fused | 90.0 % (180/200) | **−4.5** |

### Cross-bench reading

| Adapter | KIKI-DSL v3 (Δ pass) | GSM8K (Δ acc) |
|---|---:|---:|
| reasoning | +13.4 | 0 |
| math | +6.7 | -4.5 |

Two findings:

1. **Base Qwen 35B-A3B-4bit already saturates GSM8K** at 94.5 % — there is
   essentially no headroom for an adapter to add on top. Adding +reasoning
   keeps performance identical; adding +math actually loses 4.5 pts.

2. **The KIKI-DSL v3 wins of `reasoning` (+13) and `math` (+7) do not
   transfer to general GSM8K**. The wins were specific to the
   electronics-DSL surface, where the cognitive scaffolding helps
   structure outputs. On a saturated math benchmark, the adapter adds
   noise (math case) or is neutral (reasoning case).

This rules out the strong claim "reasoning/math adapters are general
cognitive winners." The honest claim is: on KIKI-DSL v3, where the base
is at 73 % and the prompts have format complexity, cognitive adapters
help. On a benchmark where the base is already near-ceiling, they don't.

[`qwen36-base-gsm8k/`](2026-05-04/qwen36-base-gsm8k/) ·
[`qwen36-reasoning-gsm8k/`](2026-05-04/qwen36-reasoning-gsm8k/) ·
[`qwen36-math-gsm8k/`](2026-05-04/qwen36-math-gsm8k/)

## MT-Bench — Devstral-Small-2-24B-MLX-4bit (base, no adapter)

| Metric | Value |
|---|---|
| Overall score | **8.892 / 10** |
| Judged turns | 37 / 160 (parseable `[[rating]]`) |
| Judge | Mistral-Medium-3.5-128B-MLX-4bit (local, port 8500) |
| Turn 1 avg | 9.421 |
| Turn 2 avg | 8.333 |
| Categories ≥ 9.0 | writing 9.33, math 10.0, coding 10.0, stem 10.0, humanities 9.0 |
| Categories < 9.0 | extraction 8.86, reasoning 7.83, roleplay 7.40 |

⚠️ **Caveat** : seuls 37/160 turns ont produit un `[[rating]]` extractible
par regex (bug parsing du runner sur réponses verboses). Score biaisé
vers les catégories writing/math/coding où le judge formate proprement.
Useful as smoke + qualitative read, not as final ranking.

[`devstral-base-mtbench-full/results.json`](2026-05-04/devstral-base-mtbench-full/results.json)

## KIKI-DSL v3 — REVISED TAXONOMY (balanced test set)

The v1 test set (10 prompts) was biased toward named-IC requirements
(LM358, AMS1117, BME280, 2N3904, IRF540, PESD5V0L1BA, STM32F103,
LDO AMS1117). The v3 set (15 prompts) adds 5 SPICE-pure prompts using
abstract net labels (A, B, N1, N2, BASE, EMITTER, OUT). Results below
use `kicad_dsl_v3.json`.

### v3 ranking (5 adapters benched)

| Adapter | Pass | Avg | Δ pass | Δ avg | v1 → v3 |
|---------|-----:|----:|-------:|------:|---------|
| 🥇 reasoning | **86.7 %** | 0.705 | +13.4 | +0.001 | (90% v1) confirmed cognitive winner |
| 🥈 math | 80.0 % | **0.726** | +6.7 | +0.022 | (70% v1) confirmed, BEST avg |
| ⚖️ base v3 | 73.3 % | 0.704 | ref | ref | reference |
| ⚖️ chat-fr | 73.3 % | 0.615 | 0.0 | -0.089 | LOST v1 advantage (+10 → 0) |
| 📉 spice-sim | 53.3 % | 0.614 | -20.0 | -0.090 | (-30 v1 → -20 v3) recovers somewhat |
| ❌ kicad-dsl | 46.7 % | 0.477 | -26.6 | -0.227 | (-30 v1 → -27 v3) still bad |

### Findings (HONEST, post-v3 revision)

1. **True cognitive adapters help robustly** — `reasoning` and `math` win on both v1 and v3. The cluster is real but smaller (reasoning's +30 on v1 → +13 on v3; math +10 → +7).

2. **Stylistic adapters were oversold** — `chat-fr` looked +10 on v1 but is **NEUTRAL on v3**. The v1 win was largely a function of v1 prompts triggering refusal preambles which chat-fr suppressed.

3. **Domain-narrow adapters do degrade — but less than v1 suggested** — kicad-dsl/spice-sim drop ~25 points (not 30) on the fair test. They have real wins on their niche prompts (`spice-sim` Q14 bias network = 1.00, Q15 LC tank = 1.00) but lose elsewhere.

4. **Base Qwen 3.6-35B-A3B-4bit is already strong on SPICE-pure prompts** — 5/5 PASS on Q11-Q15 (4 perfect 1.00). Specialized adapters for compact netlist format are largely redundant.

### v1 vs v3 — what was an artifact

| Adapter | v1 Δ pass | v3 Δ pass | Verdict |
|---------|----------:|----------:|---------|
| reasoning | +30 | +13 | win confirmed, magnitude inflated on v1 |
| math | +10 | +7 | win confirmed |
| chat-fr | +10 | 0 | **artifact**: v1 win was test-set-specific |
| security | +10 | +6.7 | win confirmed, partly test-set-amplified |
| kicad-pcb | 0 | -6.7 | hidden negative-transfer revealed on SPICE-pure |
| embedded | -10 | (TBD) | retest pending |
| electronics | -20 | (TBD) | retest pending |
| spice-sim | -30 | -20 | partial artifact, real degradation reduced |
| components | -30 | -20 | partial artifact, real degradation reduced |
| kicad-dsl | -30 | -27 | confirmed, slight reduction |

→ The v1 test set inflated both wins and losses. v3 is more honest.

## KIKI-DSL v1 (10 prompts, biased — historical reference)

First valid measurement of an eu-kiki adapter delta on a custom KIKI-native bench. Workflow: `mlx_lm fuse` produces a self-contained 4-bit model with adapter weights baked in (workaround for `--adapter-path` runtime bug), evaluated by [`runners/kiki_native_runner.py`](../runners/kiki_native_runner.py) on 10 hand-curated KiCad prompts.

| Run | Base model | Adapter | Pass rate | Avg score | Notes |
|-----|------------|---------|----------:|----------:|-------|
| [`qwen36-35b-4bit-base-kicad-dsl`](2026-05-04/qwen36-35b-4bit-base-kicad-dsl/) | Qwen3.6-35B-A3B-MLX-4bit | — | **60.0 %** | **0.593** | baseline reference |
| [`qwen36-35b-4bit-fused-kicad-dsl`](2026-05-04/qwen36-35b-4bit-fused-kicad-dsl/) | Qwen3.6-35B-A3B-MLX-4bit | v4-sota kicad-dsl | **30.0 %** | **0.382** | -30 pts pass, SPICE-compact over-specialization |
| [`qwen36-35b-4bit-fused-kicad-pcb-on-kicad-dsl`](2026-05-04/qwen36-35b-4bit-fused-kicad-pcb-on-kicad-dsl/) | Qwen3.6-35B-A3B-MLX-4bit | v4-sota kicad-pcb | **60.0 %** | **0.535** | 0 pts pass, -0.06 avg, redistribution |
| [`qwen36-35b-4bit-fused-components-on-kicad-dsl`](2026-05-04/qwen36-35b-4bit-fused-components-on-kicad-dsl/) | Qwen3.6-35B-A3B-MLX-4bit | v4-sota components | **30.0 %** | **0.418** | -30 pts pass, similar to kicad-dsl pattern |
| [`qwen36-35b-4bit-fused-chat-fr-on-kicad-dsl`](2026-05-04/qwen36-35b-4bit-fused-chat-fr-on-kicad-dsl/) | Qwen3.6-35B-A3B-MLX-4bit | v4-sota chat-fr | **70.0 %** | **0.569** | **+10 pts pass**, only positive delta so far |
| [`qwen36-35b-4bit-fused-electronics-on-kicad-dsl`](2026-05-04/qwen36-35b-4bit-fused-electronics-on-kicad-dsl/) | Qwen3.6-35B-A3B-MLX-4bit | v4-sota electronics | **40.0 %** | **0.532** | -20 pts pass, mid-range |
| [`qwen36-35b-4bit-fused-embedded-on-kicad-dsl`](2026-05-04/qwen36-35b-4bit-fused-embedded-on-kicad-dsl/) | Qwen3.6-35B-A3B-MLX-4bit | v4-sota embedded | **50.0 %** | **0.440** | -10 pts pass, mid-range |
| [`qwen36-35b-4bit-fused-spice-sim-on-kicad-dsl`](2026-05-04/qwen36-35b-4bit-fused-spice-sim-on-kicad-dsl/) | Qwen3.6-35B-A3B-MLX-4bit | v4-sota spice-sim | **30.0 %** | **0.473** | -30 pts pass, narrow style trap |
| [`qwen36-35b-4bit-fused-math-on-kicad-dsl`](2026-05-04/qwen36-35b-4bit-fused-math-on-kicad-dsl/) | Qwen3.6-35B-A3B-MLX-4bit | v4-sota math | **70.0 %** | **0.638** | **+10 pts pass / +0.045 avg**, general/cognitive |
| [`qwen36-35b-4bit-fused-reasoning-on-kicad-dsl`](2026-05-04/qwen36-35b-4bit-fused-reasoning-on-kicad-dsl/) | Qwen3.6-35B-A3B-MLX-4bit | v4-sota reasoning | **90.0 %** | **0.672** | **+30 pts pass / +0.079 avg** 🥇 best-of-session |
| [`devstral-base-kicad-dsl`](2026-05-04/devstral-base-kicad-dsl/) | Devstral-Small-2-24B-MLX-4bit | — | 80.0 % | 0.750 | different base for reference |

### Per-question delta — Qwen base vs each fused adapter

| Q | Domain | Base | +kicad-dsl | +kicad-pcb | +components |
|---|--------|-----:|----------:|----------:|------------:|
| 001 | passive R/R divider | 0.67 P | 0.06 | 0.00 | **0.67 P** |
| 002 | LDO AMS1117 | 0.57 | 0.05 | **0.62 P** | 0.05 |
| 003 | LED + 330R | 0.67 P | 0.67 P | **0.89 P** | 0.67 P |
| 004 | RC filter | 0.24 | **0.67 P** | **0.91 P** | 0.48 |
| 005 | BJT amp 2N3904 | **0.96 P** | 0.18 | 0.63 P | 0.26 |
| 006 | I2C pull-ups + BME280 | 0.25 | 0.21 | **0.67 P** | 0.25 |
| 007 | H-bridge MOSFETs | 0.67 P | 0.63 P | 0.04 | 0.07 |
| 008 | STM32 decoupling | 0.57 | 0.57 | 0.38 | 0.57 |
| 009 | ESD USB PESD5V0 | 0.67 P | 0.40 | 0.67 P | 0.67 P |
| 010 | LM358 buffer | 0.67 P | 0.39 | 0.56 | 0.50 |
| **PASS count** | | **6/10** | 3/10 | 6/10 | 3/10 |
| **avg overall** | | **0.593** | 0.382 | 0.535 | 0.418 |

### Insights from cross-adapter comparison

- **kicad-dsl** over-specializes on SPICE-compact format → hostile to named ICs (LM358, AMS1117, BME280, 2N3904).
- **kicad-pcb** redistributes strengths: gains on prompts with explicit IC references (LDO, LED, RC filter, I2C+BME280), loses on structural designs (H-bridge, decoupling).
- **No uniform v4-sota pattern** — each adapter has its own personality. Can't extrapolate from one to all.
- The kicad-dsl test set may not be the right bench for a kicad-pcb adapter (different sub-domain). A kicad-pcb-specific test set (PCB layout, footprints) would likely show more positive delta.

### Reading

- The **base Qwen3.6-35B-A3B** is already strong on KiCad prompts in French (60 % pass).
- The **v4-sota kicad-dsl adapter** pushes towards a SPICE-compact netlist style (`R1 10k VCC GND`) — verified identical adapter behavior on Studio BF16 fused and macM1 4-bit fused runs.
- This style hurts our test set, which expects rich content (named ICs like LM358, AMS1117, BME280; explicit value tokens like `100n`, `4.7u`). The adapter omits these to be terse.
- Net effect: -30 pts pass-rate, -0.21 avg score. Honest negative result documented for publication.

### Per-question delta (Qwen base → fused)

| ID | Domain | Base | Fused | Δ |
|----|--------|------|-------|---|
| 001 | passive (R/R divider) | 0.67 PASS | 0.06 FAIL | -0.61 |
| 002 | regulator (LDO AMS1117) | 0.57 FAIL | 0.05 FAIL | -0.52 |
| 003 | led (LED + 330R) | 0.67 PASS | 0.67 PASS | 0.00 |
| 004 | rc-filter | 0.24 FAIL | 0.67 **PASS** | **+0.43** |
| 005 | transistor amp (2N3904) | 0.96 PASS | 0.18 FAIL | -0.78 |
| 006 | i2c-pullup (BME280) | 0.25 FAIL | 0.21 FAIL | -0.04 |
| 007 | h-bridge (IRF540 ×4) | 0.67 PASS | 0.63 PASS | -0.04 |
| 008 | STM32 decoupling | 0.57 FAIL | 0.57 FAIL | 0.00 |
| 009 | ESD USB (PESD5V0) | 0.67 PASS | 0.40 FAIL | -0.27 |
| 010 | opamp buffer (LM358) | 0.67 PASS | 0.39 FAIL | -0.28 |

→ Adapter helps RC filter (component-light, format-driven) but hurts every prompt requiring named ICs.

### Implications for eu-kiki adapters

- v4-sota training likely used SPICE-style netlist as canonical output → over-specialized
- Future adapter retraining should preserve verbose KiCad sch format OR provide format-mode controls
- **All 38 other v4-sota adapters likely share this pattern** — should be eval'd before release
- Pipeline now reproducible end-to-end on macM1 alone (fuse + serve + bench in 30 min for any adapter)

## MT-Bench (LLM-as-judge, local Mistral-Medium-128B)

**First MT-Bench score** for eu-kiki, fully local pipeline. Subject and judge both run on Studio (M3 Ultra 512 GB). No external API dependency — judge is Mistral-Medium-3.5-128B-MLX-4bit (~73 GB), reproducible.

| Run | Subject | Judge | Score | Turns judged |
|-----|---------|-------|------:|-------------:|
| [`devstral-base-mtbench-smoke`](2026-05-04/devstral-base-mtbench-smoke/) | Devstral-Small-2-24B-MLX-4bit (BASE) | Mistral-Medium-3.5-128B-MLX-4bit | **9.2 / 10** | 5 (smoke) |

### Caveats (smoke run, not publishable as-is)

- Only **5 questions × 1+ turn** judged (smoke run, validates pipeline)
- All 5 in category **"writing"** — first 5 of MT-Bench `question.jsonl`
- Turn 2 only successfully judged for 1 of the 5 questions (4 timeouts caused by GPU contention with a concurrent training process on Studio)
- **Mistral-Medium may be a lenient judge** — score 9.2 is on par with GPT-4 (~9.0) on the original MT-Bench paper, which is suspiciously high for a 24B 4-bit model. Calibration with a stronger reference judge needed for publication.

### Pipeline validated end-to-end

- mtbench_runner: spawn subject server, fetch MT-Bench questions from FastChat repo, generate, judge, aggregate
- Bug fixes during validation (5 attempts):
  1. `--max-questions` not in CLI parser
  2. `fastchat` hard import dependency
  3. Qwen3.x thinking mode → reasoning vs content (KeyError)
  4. macM1 OOM Metal (Qwen 35B fused + 800-token chat KV cache)
  5. Studio not synced with latest fixes
- Final commit: `593302e` — pipeline portable, runs all-on-Studio.

### Next

- MT-Bench **full** (80 questions × 2 turns × 8 categories) for publishable result
- A/B: Devstral 24B base vs Devstral 24B + eu-kiki v1 python adapter (fused)
- Consider stronger judge (GPT-4 or human) for calibration of Mistral-Medium leniency

## eu-kiki v1 Devstral on HumanEval+ (FIRST VALID adapter delta)

Replaces the invalidated `devstral-python-adapter-2026-05-04` run (which silently used the base via the broken `--adapter-path`). This time the adapter is **fused** into the base via `mlx_lm fuse` on macM1, producing a self-contained 4-bit checkpoint where the LoRA contributions are guaranteed to be active.

| Run | Setup | HumanEval base | HumanEval+ |
|-----|-------|---------------:|-----------:|
| [`devstral-base-baseline-2026-05-04-v2`](2026-05-04/devstral-base-baseline-2026-05-04-v2/) | Devstral-Small-2-24B-MLX-4bit BASE | **87.20 %** | **82.90 %** |
| [`devstral-python-fused-humanevalplus`](2026-05-04/devstral-python-fused-humanevalplus/) | + eu-kiki v1 python adapter (FUSED) | **86.00 %** | **81.10 %** |
| **Δ** | | **-1.20 pts** | **-1.80 pts** |

→ Adapter slightly degrades HumanEval+ — matches the initial prediction (scenario A "maintain or slight regression"). The adapter targets verbose chat-assistant Python (Stack-Overflow-style instructional code), not the terse algorithmic completion HumanEval expects. Net effect: 3 additional failures on extra tests (133 vs 136 passing).

Safe to deploy for chat/assistant; should not be relied on for code-generation benchmarks. For a positive delta the right surface is **MT-Bench** (chat) or **AlpacaEval 2.0** (LLM-judge).

## HumanEval / HumanEval+ (EvalPlus)

| Run | Model | Adapter | HumanEval base | HumanEval+ | Notes |
|-----|-------|---------|---------------:|-----------:|-------|
| [`devstral-base-baseline-2026-05-04-v2`](2026-05-04/devstral-base-baseline-2026-05-04-v2/) | Devstral-Small-2-24B-MLX-4bit | — | **87.20 %** | **82.90 %** | Valid baseline |
| [`devstral-python-adapter-2026-05-04`](2026-05-04/devstral-python-adapter-2026-05-04/) | Devstral-Small-2-24B-MLX-4bit | python (eu-kiki) | 87.20 % | 82.90 % | ⚠️ **INVALIDATED** — adapter silently NOT applied (mlx_lm `load_adapters` skips QuantizedLinear modules; 11/11 outputs bit-identical to base on a control test). Run actually measures base again. |

**Lesson learned (2026-05-04, confirmed via Studio test):** mlx_lm.server `--adapter-path` succeeds without error but does NOT apply LoRA weights — neither on 4-bit MLX models NOR on BF16. The bug is widespread: tested both Devstral 2 24B 4-bit + python adapter and Qwen 35B-A3B BF16 + v4-sota kicad-dsl adapter — in both cases outputs are bit-identical to base.

**Workaround validated:** `mlx_lm fuse --save-path <fused-model>` successfully bakes the adapter into a self-contained model. Decisive test on Studio:

| Setup | Same prompt: "Génère un sch KiCad: R1=10k entre VCC et GND." |
|-------|---|
| Qwen 35B-A3B BASE | "En tant qu'IA textuelle, je ne peux pas générer..." (refuses, suggests manual) |
| Qwen 35B-A3B + v4-sota kicad-dsl (via fuse) | "`R1 VCC GND 10k`" (direct netlist output) |

→ Fuse produces a **drastically different**, adapter-influenced output. The adapter is genuinely functional; the runtime loader is the problem.

**New methodology for adapter benchmarks:**
1. `bash eval/runners/fuse_adapter.sh --base <bf16-model> --adapter <path> --out-name <model+adapter>`
2. Optionally quantize the fused model to 4-bit
3. Serve the fused/quantized checkpoint via `mlx_lm.server --model <fused-model>` (no `--adapter-path` needed)
4. Run benchmarks normally

All previously committed "adapter" results that used `--adapter-path` are invalidated. This includes `devstral-python-adapter-2026-05-04` and any KiCad-DSL adapter run. Only base-model benchmarks remain valid (Devstral 24B 4-bit base 87.2 / 82.9 % HumanEval).

### Comparison context (HumanEval+)

| Model | HumanEval base | HumanEval+ |
|-------|---------------:|-----------:|
| Qwen2.5-Coder-32B-Instruct | 92.7 % | 87.2 % |
| Claude 3.5 Sonnet | 92.0 % | 84.1 % |
| GPT-4o | 90.2 % | 86.6 % |
| **Devstral 2 24B 4-bit (this run)** | **87.2 %** | **82.9 %** |
| DeepSeek-Coder-V2.5 | 89.0 % | 80.0 % |

→ Devstral 2 24B in 4-bit MLX on M1 Max is **competitive with GPT-4o** on HumanEval+ despite 25× fewer parameters and aggressive quantization.

### Adapter delta interpretation

The eu-kiki Python adapter does **not degrade** the base model on HumanEval, confirming **safe deployment in production**. It also does not improve (the base is already saturated on this benchmark family). Real adapter impact will be measured on:

- **MT-Bench** (chat, LLM-as-judge) — where the adapter's chat/instructional training shines
- **AlpacaEval 2.0** — win-rate vs reference
- **KIKI-native benchmarks** (kicad-dsl, spice, emc, misra-c) — where the adapter's domain knowledge is tested

## Methodology fingerprint

Every result above is fully reproducible from its `env.json` + `rerun.sh`:

- **Model SHA** (first chunk safetensors) and **adapter SHA** logged
- **eu-kiki git commit** locked at run time
- **Hardware** (`MacBookPro.lan`, `MacStudio-de-MonsieurB.local`, `kx6tm-23`) recorded
- **MLX version**, **Python version**, **pip freeze** captured
- **Sampling config** (T=0, greedy, n=1, seed=42)
- **Sandbox**: official EvalPlus on Linux (kx6tm-23, Python 3.13.5) — bypasses macOS `RLIMIT_AS` incompatibility

## Pipeline cross-platform

```
Codegen (macM1 M1 Max 32 GB)
  └─ mlx_lm.server :8801 / :8802
  └─ EvalPlus codegen via OpenAI-compat API
                ↓
        ~/Projets/eu-kiki/eval/results/<run>/
        evalplus_humanevalplus/humaneval/*.jsonl
                ↓
Transit (GrosMac, scp two-hop)
                ↓
Scoring (kx6tm-23, Linux x86_64, Python 3.13.5)
  └─ evalplus.evaluate humaneval --samples ...
                ↓
        eval_results.json + pass@1 official
                ↓
Tracked artifacts back in eu-kiki/eval/results/.../
        evalplus_humanevalplus_linux_official/results.json
```

## EU AI Act compliance

These benchmarks support Article 53(1)(d) technical documentation requirements. See [`docs/eu-ai-act-transparency.md`](../../docs/eu-ai-act-transparency.md) and [`eval/README.md`](../README.md) for the broader transparency framework.
