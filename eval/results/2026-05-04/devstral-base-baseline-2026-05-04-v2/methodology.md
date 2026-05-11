# Methodology — devstral-base-baseline-2026-05-04-v2

**Generated:** 2026-05-04T18:32:39+0200
**Schema:** ailiance-eval-result/1.0

## Run identity

| Field | Value |
|-------|-------|
| Label | `devstral-base-baseline-2026-05-04-v2` |
| Model | `/Users/electron/Projets/ailiance-mac-tuner/models/Devstral-Small-2-24B-MLX-4bit` |
| Model SHA-256 (first chunk) | `662d914f8f30ca78f687c96fb7599d4ac25409fa02cd02c3540a1d280bd17f86` |
| Adapter | `(none)` |
| Adapter SHA-256 | `(none)` |
| ailiance git commit | `fd120b337f624bee2ebb932c9e22c19637bd1ba2` |
| ailiance git describe | `fd120b3` |
| ailiance dirty? | `no` |
| Hardware | `MacBookPro.lan` (arm64, arm) |
| Python | `3.13.13` |
| MLX | `unknown` |
| MLX-LM | `0.31.3` |
| Date | `2026-05-04` |

## Benchmarks executed

### Lighteval

```
lighteval|humaneval|0|0,lighteval|gsm8k|5|0
```

### EvalPlus

```
humanevalplus
```

## Sampling configuration

| Param | Value |
|-------|-------|
| Temperature | `0.0` |
| max_tokens | `1024` |
| n_samples per problem | `1` |
| Seed | `42` |

## How to reproduce

1. Check out the ailiance repo at the commit above:

   ```bash
   git checkout fd120b337f624bee2ebb932c9e22c19637bd1ba2
   ```

2. Set up the environment (Python 3.13, MLX-LM, Lighteval, EvalPlus):

   ```bash
   uv venv && uv pip install -e '.[dev]'
   uv pip install 'lighteval[extended_tasks]' evalplus
   ```

3. Verify the model + adapter SHAs match (otherwise results won't reproduce):

   ```bash
   sha256sum /Users/electron/Projets/ailiance-mac-tuner/models/Devstral-Small-2-24B-MLX-4bit/*.safetensors | head -1
   sha256sum (none)/adapters.safetensors
   ```

4. Run:

   ```bash
   bash /Users/electron/Projets/ailiance/eval/results/2026-05-04/devstral-base-baseline-2026-05-04-v2/rerun.sh
   ```

## Limitations

- Lighteval / EvalPlus use their default prompt templates at the version pinned
  in this run's `pip_freeze` (see `env.json`). Future versions of these tools
  may change templates and produce different scores.
- The mlx_lm server applies the model's chat template; this is not currently
  hashed, but is determined by the model directory contents.
- LLM-as-judge benchmarks (MT-Bench, AlpacaEval) when present log the judge
  model SHA separately.

## EU AI Act Art. 53(1)(d)

This methodology document is part of the technical documentation maintained
under EU AI Act Art. 53(1)(d) for the ailiance system. See
`docs/eu-ai-act-transparency.md` for the broader transparency framework.
