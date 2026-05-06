# ailiance — Evaluation Suite

Reproducible, publishable benchmark pipeline for the ailiance adapters
(Apertus 70B, Devstral 24B, EuroLLM 22B) and KIKI-Mac_tunner models
(Mistral Large, Qwen3.5-122B/35B, Sonnet-Devstral 123B, Brainstacks).

## Goals

1. **Reproducibility** — every result tagged with model SHA, adapter SHA,
   dataset commit, prompt template, hyperparams, hardware, MLX version.
2. **Standard benchmarks** — recognized public suites (Lighteval, EvalPlus,
   bigcode-eval) for cross-model comparison.
3. **KIKI-native bench** — original benchmarks (KiCad-DSL, SPICE, EMC, MISRA-C,
   router accuracy) defending the project's IP.
4. **EU AI Act Art. 53(1)(d)** — full transparency on methodology.

## Suite

### Standard benchmarks (Lighteval / EvalPlus)

| Benchmark | Domain | Metric | Notes |
|-----------|--------|--------|-------|
| HumanEval+ | Code Python | pass@1, pass@10 | EvalPlus extended tests |
| MBPP+ | Code Python | pass@1, pass@10 | EvalPlus extended tests |
| MultiPL-E | Code Rust/TS/C++/Go | pass@1 | Lighteval |
| BigCodeBench | Code multi-step | pass@1 | bigcode-eval |
| GSM8K | Math word | exact match | Lighteval, 5-shot |
| MATH | Math advanced | exact match | Lighteval, 4-shot |
| MMLU-Pro | Knowledge | accuracy | Lighteval, 5-shot |
| BBH | Reasoning | accuracy | Lighteval, 3-shot CoT |
| BBEH | Hard reasoning | accuracy | Lighteval (successor to BBH) |
| IFEval | Instruction following | strict-prompt accuracy | Lighteval |
| HellaSwag | Common sense | accuracy | Lighteval, 0-shot |
| TruthfulQA | Truthfulness | MC2 | Lighteval, 0-shot |
| MT-Bench | Chat | judge score / 10 | fastchat, judge = local Mistral-Medium-128B |
| AlpacaEval 2.0 | Chat win-rate | LC-WR vs reference | judge = local Mistral-Medium-128B |

### KIKI-native benchmarks (custom IP)

| Benchmark | Description | Size | Judge |
|-----------|-------------|------|-------|
| **kiki-kicad-dsl** | Synthesize KiCad schematics from text | 50 | Auto (KiCad CLI parse) |
| **kiki-spice-eval** | Generate SPICE netlists, run sim | 30 | Auto (ngspice exit code) |
| **kiki-emc-qa** | EMC normative Q&A (IEC 61000, CISPR) | 40 | LLM judge |
| **kiki-misra-c** | MISRA-C violation detection | 60 | Rule-based + LLM |
| **kiki-router-acc** | 32-domain router accuracy | 1280 (40/dom) | top-1 / top-3 / multi-label F1 |

## Folder layout

```
eval/
├── README.md            # this file
├── run_all.sh           # entrypoint
├── tasks/               # task definitions (YAML/JSON)
│   ├── lighteval_*.yaml
│   ├── kiki_*.json      # custom
│   └── prompts/
├── runners/             # Python runners
│   ├── mlx_server_runner.py    # mlx_lm.server + adapter swap
│   ├── lighteval_runner.py
│   ├── evalplus_runner.py
│   └── kiki_judge_runner.py
├── results/             # per-run artifacts (gitignored except summaries)
│   └── YYYY-MM-DD/
│       └── <model>-<version>-<domain>/
│           ├── results.json
│           ├── report.md
│           ├── methodology.md
│           └── env.json
└── publish/
    ├── hf_model_card.md.j2
    ├── arxiv_table.tex.j2
    └── push_to_hf.py
```

## Methodology (publishable)

All benchmark runs MUST log :

| Field | Source |
|-------|--------|
| `model_id` | HF repo or local path |
| `model_sha` | `git rev-parse HEAD` of base model dir or `sha256sum *.safetensors` first chunk |
| `adapter_sha` | `sha256sum adapters.safetensors` |
| `dataset_id` + `dataset_revision` | Lighteval config |
| `prompt_template` | exact template hash |
| `temperature`, `top_p`, `max_tokens`, `seed` | sampling config |
| `hardware` | CPU model, RAM, OS, MLX version, machine ID |
| `lighteval_version` / `evalplus_version` | `pip freeze` snapshot |
| `start_time`, `end_time`, `elapsed_s` | ISO 8601 |
| `n_samples_seen`, `n_passes`, `n_failures` | counters |

→ Stored in `env.json` next to each `results.json`.

## Reference judge (local)

For LLM-as-judge benchmarks (MT-Bench, AlpacaEval, kiki-emc-qa) :
- **Primary** : Mistral-Medium-3.5-128B-MLX-4bit on Studio (port :8500), cost-free, fully reproducible
- **Backup** : Claude Opus 4.7 via Anthropic API (paid, only for arbitration)

The judge model SHA is logged for each run. **Never use a closed-source-only judge for publication** — reproducibility requires the judge to be re-runnable.

## Publication targets

1. **HuggingFace model cards** with eval table (auto-generated via `publish/hf_model_card.md.j2`)
2. **HuggingFace Open LLM Leaderboard 2** (eligible: Apertus 70B + adapters, EuroLLM 22B, Devstral fine-tunes)
3. **Papers** : "EU-sovereign LLM stack — first Apertus 70B fine-tunes" + "Brainstacks 32-expert MoE-LoRA on Apple Silicon"
4. **Arxiv table** : LaTeX table auto-generated from `results/*.json`

## Hardware allocation

| Bench | Studio M3 Ultra 512 GB | macM1 32 GB |
|-------|------------------------|-------------|
| HumanEval / MBPP on Devstral 24B-4bit | ✅ | ✅ |
| GSM8K / MMLU-Pro / BBH on Apertus 70B | ✅ | ❌ (model too big) |
| BBEH / MATH on Mistral-Medium 128B | ✅ | ❌ |
| MT-Bench / AlpacaEval (judge: local) | ✅ | ✅ (judge over Tailscale) |
| kiki-native | ✅ | ✅ |

Default policy : Studio runs heavy benchmarks (≥ 24B BF16, ≥ 70B 4-bit), macM1 runs eval ≤ 24B 4-bit. Both stream results to a shared rsync target.
