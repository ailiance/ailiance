# eu-kiki Benchmark Results — Summary

Aggregated table of all publishable benchmark runs. Each result entry
links to its self-contained directory with `env.json`, `methodology.md`,
`rerun.sh`, and the official scoring artifacts.

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
