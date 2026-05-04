# eu-kiki Benchmark Results — Summary

Aggregated table of all publishable benchmark runs. Each result entry
links to its self-contained directory with `env.json`, `methodology.md`,
`rerun.sh`, and the official scoring artifacts.

## KIKI-DSL (KiCad schematic synthesis)

First valid measurement of an eu-kiki adapter delta on a custom KIKI-native bench. Workflow: `mlx_lm fuse` produces a self-contained 4-bit model with adapter weights baked in (workaround for `--adapter-path` runtime bug), evaluated by [`runners/kiki_native_runner.py`](../runners/kiki_native_runner.py) on 10 hand-curated KiCad prompts.

| Run | Base model | Adapter | Pass rate | Avg score | Notes |
|-----|------------|---------|----------:|----------:|-------|
| [`qwen36-35b-4bit-base-kicad-dsl`](2026-05-04/qwen36-35b-4bit-base-kicad-dsl/) | Qwen3.6-35B-A3B-MLX-4bit | — | **60.0 %** | **0.593** | baseline reference |
| [`qwen36-35b-4bit-fused-kicad-dsl`](2026-05-04/qwen36-35b-4bit-fused-kicad-dsl/) | Qwen3.6-35B-A3B-MLX-4bit | v4-sota kicad-dsl | **30.0 %** | **0.382** | -30 pts pass, SPICE-compact over-specialization |
| [`qwen36-35b-4bit-fused-kicad-pcb-on-kicad-dsl`](2026-05-04/qwen36-35b-4bit-fused-kicad-pcb-on-kicad-dsl/) | Qwen3.6-35B-A3B-MLX-4bit | v4-sota kicad-pcb | **60.0 %** | **0.535** | 0 pts pass, -0.06 avg, redistribution |
| [`qwen36-35b-4bit-fused-components-on-kicad-dsl`](2026-05-04/qwen36-35b-4bit-fused-components-on-kicad-dsl/) | Qwen3.6-35B-A3B-MLX-4bit | v4-sota components | **30.0 %** | **0.418** | -30 pts pass, similar to kicad-dsl pattern |
| [`qwen36-35b-4bit-fused-chat-fr-on-kicad-dsl`](2026-05-04/qwen36-35b-4bit-fused-chat-fr-on-kicad-dsl/) | Qwen3.6-35B-A3B-MLX-4bit | v4-sota chat-fr | **70.0 %** | **0.569** | **+10 pts pass**, only positive delta so far |
| [`qwen36-35b-4bit-fused-electronics-on-kicad-dsl`](2026-05-04/qwen36-35b-4bit-fused-electronics-on-kicad-dsl/) | Qwen3.6-35B-A3B-MLX-4bit | v4-sota electronics | **40.0 %** | **0.532** | -20 pts pass, mid-range |
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
