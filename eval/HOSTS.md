# Hosts — eval pipeline machine matrix

Reproducible eval runs on a single machine produce a deterministic
fingerprint (model SHA + adapter SHA + git commit + pip freeze + hardware).
Cross-machine eval is allowed but each result file documents its own host.

## Machines

| Host | Tailscale | Hardware | OS | Python | venv path | Repo path |
|------|-----------|----------|----|--------|-----------|-----------|
| **studio** | studio M3 Ultra 512 GB | macOS 26+ | 3.14.4 | `~/ailiance/.venv` | `~/ailiance` |
| **macM1** | `macm1.tail78ae15.ts.net` | M1 Max 32 GB | macOS 26+ | 3.13.13 | `~/Projets/ailiance/.venv` | `~/Projets/ailiance` |
| **GrosMac** | `100.123.239.46` | M5 16 GB | macOS 26+ | (n/a, dev only) | (n/a) | `~/Documents/Projets/ailiance` |

GrosMac is dev-only (writing code, pushing to GitHub). It does NOT run benchmarks.

## Allocation policy

| Bench | studio | macM1 |
|-------|--------|-------|
| Devstral 24B 4-bit (ailiance/devstral adapters) | ✅ | ✅ |
| Apertus 70B (ailiance/apertus adapters) | ✅ | ❌ (model >32 GB) |
| EuroLLM 22B (ailiance/eurollm adapters) | ✅ | borderline (BF16 = 42 GB, MLX-4bit ~11 GB OK) |
| Mistral-Medium-3.5-128B BF16 | ✅ (training, eval) | ❌ |
| Brainstacks Qwen3.5-4B + adapters | ✅ | ✅ |
| Brainstacks Qwen3.5-35B-A3B + adapters | ✅ | borderline |
| KIKI-native (KiCad/SPICE/EMC/MISRA) | ✅ | ✅ |
| MT-Bench / AlpacaEval (judge: Mistral-Medium-128B local on studio) | ✅ | ✅ (judge over Tailscale) |

Heuristic: if `(model + adapter + KV cache) > 28 GB`, run on studio.

## Sync workflow (preferred: git, fallback: rsync)

### When eval/ code changes

1. Edit on GrosMac (`~/Documents/Projets/ailiance/eval/`)
2. Commit + push to `L-electron-Rare/ailiance`
3. Pull on each runner:

   ```bash
   ssh studio "cd ~/ailiance && git pull"
   ssh macM1  "cd ~/Projets/ailiance && git pull"
   ```

### When models/adapters change

Models & adapters live OUTSIDE the repo. Copy with rsync :

```bash
# Studio is the canonical source for models + adapters.
ssh macM1 "rsync -avzP studio:~/KIKI-Mac_tunner/models/<MODEL>/ \\
                       ~/Projets/KIKI-Mac_tunner/models/<MODEL>/"
ssh macM1 "rsync -avzP studio:~/ailiance/output/adapters/<MODEL>/<DOMAIN>/ \\
                       ~/Projets/ailiance/output/adapters/<MODEL>/<DOMAIN>/"
```

### When results/ changes

Results contain reproducible fingerprints. Aggregate by syncing back to GrosMac
for comparison + publication :

```bash
# Pull results from runners to GrosMac for aggregation
rsync -avz studio:~/ailiance/eval/results/ ~/Documents/Projets/ailiance/eval/results/
rsync -avz macM1:~/Projets/ailiance/eval/results/ ~/Documents/Projets/ailiance/eval/results/
```

`results/` is `.gitignore`d to keep repo size sane.

## Standard run command

Always invoke from the repo's `eval/` directory so `python -m runners.X`
finds the package :

```bash
cd ~/ailiance/eval        # studio
cd ~/Projets/ailiance/eval # macM1

../.venv/bin/python -m runners.mlx_server_runner --help
bash run_all.sh --model <model_path> --adapter <adapter_path> --label <label>
```

## Verifying alignment

Run on each machine, compare git commits + pip versions :

```bash
cd <repo>/eval && git rev-parse HEAD && \
  ../.venv/bin/python -c 'import lighteval, evalplus, mlx_lm; \
    print(lighteval.__version__, mlx_lm.__version__)'
```

Both runners should be on the same git commit when running comparable
benchmarks. Otherwise the methodology.md will flag the divergence via
the embedded `git_commit` and `pip_freeze` fields.

## Troubleshooting

- **`No module named 'runners'`** — invoke from `eval/` cwd or use full
  paths in `python -m`.
- **`pip_freeze` empty in env.json** — venv lacks `pip`; install-time
  result_writer will use `uv pip freeze` as fallback.
- **`mlx_lm.server` won't start** — check `--port` not in use, model path
  exists, adapter dir contains `adapters.safetensors`.
- **Lighteval task name unknown** — verify with
  `python -m lighteval tasks list | grep <task>`.
