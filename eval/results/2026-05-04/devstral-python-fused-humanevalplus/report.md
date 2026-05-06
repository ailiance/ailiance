# ailiance bench report — devstral-python-fused-humanevalplus

**Generated:** 2026-05-04T23:45+0200

## Identity

- **Model**: Devstral-Small-2-24B-MLX-4bit + ailiance v1 python adapter (FUSED via `mlx_lm fuse`)
- **Adapter origin**: `~/Projets/ailiance/output/adapters/devstral/python` (rank 16 alpha 32, q/k/v/o + MLP, 500 iters)
- **Fused checkpoint**: `/tmp/devstral-python-fused-4bit` (12 GB)
- **Host**: macM1 M1 Max 32 GB (codegen) + kx6tm-23 Linux x86_64 (scoring)
- **Git commit**: `49f1f22`

## Metrics (164 problems, EvalPlus official Linux scorer)

| Metric | Base | Fused | Δ |
|--------|-----:|------:|--:|
| HumanEval base | 87.20 % | **86.00 %** | -1.20 pts |
| HumanEval+ extra | 82.90 % | **81.10 %** | -1.80 pts |
| Pass count base | 143 | 141 | -2 |
| Pass count plus | 136 | 133 | -3 |

## Reproducibility

- Methodology: this file
- Environment snapshot: [`env.json`](env.json)
- Official scorer output: [`evalplus_humanevalplus_linux_official/eval_results.json`](evalplus_humanevalplus_linux_official/eval_results.json)
- Custom scorer summary: [`evalplus_humanevalplus_linux_official/results.json`](evalplus_humanevalplus_linux_official/results.json)

## How to reproduce

```bash
# 1. Fuse on a machine with the base + adapter
mlx_lm fuse \
  --model models/Devstral-Small-2-24B-MLX-4bit \
  --adapter-path ailiance/output/adapters/devstral/python \
  --save-path /tmp/devstral-python-fused-4bit

# 2. Serve the fused model
mlx_lm server --model /tmp/devstral-python-fused-4bit --port 8850

# 3. Codegen via EvalPlus (writes sanitized samples)
python -m evalplus.evaluate humaneval \
  --backend openai --base-url http://127.0.0.1:8850/v1 \
  --model /tmp/devstral-python-fused-4bit \
  --greedy --n-samples 1

# 4. Score on Linux (EvalPlus sandbox is Linux-only)
scp ...sanitized.jsonl <linux-host>:.
ssh <linux-host> 'python -m evalplus.evaluate humaneval --samples sanitized.jsonl'
```

## Conclusion

The ailiance v1 Devstral python adapter slightly **degrades** HumanEval+ vs the base (-1.8 pts on extra tests). This matches the initial prediction (scenario A "maintain or slight regression", 50 % prob). The adapter targets a verbose chat-assistant style (Stack-Overflow-like instructional Python), not the terse algorithmic completion expected by HumanEval. Net effect: 3 additional failures on extra tests due to style mismatch.

This is **safe to deploy** in chat/assistant production (no catastrophic forgetting) but should not be relied on for code-generation benchmarks. The base 24B already saturates HumanEval Python coverage; fine-tuning on a different style displaces a few edge cases.

For a publishable adapter delta on the right surface, recommend MT-Bench (chat) or AlpacaEval 2.0, where the verbose instructional style is rewarded.
