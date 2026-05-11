# Methodology — devstral-python-adapter-2026-05-04

**Generated:** 2026-05-04T19:51:40+0200
**Schema:** ailiance-eval-result/1.0

## Run identity

| Field | Value |
|-------|-------|
| Label | `devstral-python-adapter-2026-05-04` |
| Model | `/Users/electron/Projets/ailiance-mac-tuner/models/Devstral-Small-2-24B-MLX-4bit` |
| Model SHA-256 (first chunk) | `662d914f8f30ca78f687c96fb7599d4ac25409fa02cd02c3540a1d280bd17f86` |
| Adapter | `/Users/electron/Projets/ailiance/output/adapters/devstral/python` |
| Adapter SHA-256 | `88a0038f4fb0d331d66807b188f6692c41f9c537c92f5c57286bb896d33f086a` |
| ailiance git commit | `b1ef93e2a142dfc706117139fd3b95fc6afa1dac` |
| ailiance git describe | `b1ef93e` |
| ailiance dirty? | `no` |
| Hardware | `MacBookPro.lan` (arm64, arm) |
| Python | `3.13.13` |
| MLX | `unknown` |
| MLX-LM | `0.31.3` |
| Date | `2026-05-04` |

## Benchmarks executed

### Lighteval

```
gsm8k|5
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
   git checkout b1ef93e2a142dfc706117139fd3b95fc6afa1dac
   ```

2. Set up the environment (Python 3.13, MLX-LM, Lighteval, EvalPlus):

   ```bash
   uv venv && uv pip install -e '.[dev]'
   uv pip install 'lighteval[extended_tasks]' evalplus
   ```

3. Verify the model + adapter SHAs match (otherwise results won't reproduce):

   ```bash
   sha256sum /Users/electron/Projets/ailiance-mac-tuner/models/Devstral-Small-2-24B-MLX-4bit/*.safetensors | head -1
   sha256sum /Users/electron/Projets/ailiance/output/adapters/devstral/python/adapters.safetensors
   ```

4. Run:

   ```bash
   bash /Users/electron/Projets/ailiance/eval/results/2026-05-04/devstral-python-adapter-2026-05-04/rerun.sh
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
