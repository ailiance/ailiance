#!/usr/bin/env python3
"""result_writer — generate per-run methodology.md, rerun.sh, and report.md.

Each eval run produces a self-contained directory with everything needed to
reproduce: env.json (model/adapter SHAs, hardware, pip freeze, git state),
methodology.md (human-readable), rerun.sh (executable), and report.md (table).

Usage (from run_all.sh):
    python -m runners.result_writer \\
        --output-dir results/2026-05-04/devstral-v1-python \\
        --model-path models/Devstral-Small-2-24B-MLX-4bit \\
        --adapter-path output/adapters/devstral/python \\
        --label devstral-v1-python \\
        --tasks "lighteval|humaneval|0|0,extended|bbeh|0|0" \\
        --evalplus-tasks humanevalplus,mbppplus
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


METHODOLOGY_TEMPLATE = """# Methodology — {label}

**Generated:** {generated_at}
**Schema:** eu-kiki-eval-result/1.0

## Run identity

| Field | Value |
|-------|-------|
| Label | `{label}` |
| Model | `{model_path}` |
| Model SHA-256 (first chunk) | `{model_sha}` |
| Adapter | `{adapter_path}` |
| Adapter SHA-256 | `{adapter_sha}` |
| eu-kiki git commit | `{git_commit}` |
| eu-kiki git describe | `{git_describe}` |
| eu-kiki dirty? | `{git_dirty}` |
| Hardware | `{hardware_node}` ({machine}, {processor}) |
| Python | `{python_version}` |
| MLX | `{mlx_version}` |
| MLX-LM | `{mlx_lm_version}` |
| Date | `{date}` |

## Benchmarks executed

### Lighteval

```
{lighteval_tasks}
```

### EvalPlus

```
{evalplus_tasks}
```

## Sampling configuration

| Param | Value |
|-------|-------|
| Temperature | `{temperature}` |
| max_tokens | `{max_tokens}` |
| n_samples per problem | `{n_samples}` |
| Seed | `{seed}` |

## How to reproduce

1. Check out the eu-kiki repo at the commit above:

   ```bash
   git checkout {git_commit}
   ```

2. Set up the environment (Python 3.13, MLX-LM, Lighteval, EvalPlus):

   ```bash
   uv venv && uv pip install -e '.[dev]'
   uv pip install 'lighteval[extended_tasks]' evalplus
   ```

3. Verify the model + adapter SHAs match (otherwise results won't reproduce):

   ```bash
   sha256sum {model_path}/*.safetensors | head -1
   sha256sum {adapter_path}/adapters.safetensors
   ```

4. Run:

   ```bash
   bash {rerun_script}
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
under EU AI Act Art. 53(1)(d) for the eu-kiki system. See
`docs/eu-ai-act-transparency.md` for the broader transparency framework.
"""


RERUN_TEMPLATE = """#!/usr/bin/env bash
# Auto-generated rerun script for eval result {label}
# Generated at {generated_at}
#
# This re-runs the SAME benchmarks against the SAME model+adapter.
# To reproduce identically, check out git commit {git_commit} first.

set -euo pipefail

LABEL="{label}"
MODEL="{model_path}"
ADAPTER="{adapter_path}"
PORT={port}

EVAL_DIR="$(cd "$(dirname "$0")/../../.." && pwd)/eval"
cd "$EVAL_DIR/.."   # eu-kiki root

bash eval/run_all.sh \\
    --model "$MODEL" \\
    {adapter_arg} \\
    --label "$LABEL-rerun-$(date +%Y%m%d-%H%M)" \\
    --port "$PORT" \\
    {extra_flags}
"""


def _safe_get(d: dict, *path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def write_methodology(output_dir: Path, env_path: Path, args: dict) -> Path:
    env = json.loads(env_path.read_text())
    rerun_path = output_dir / "rerun.sh"

    md = METHODOLOGY_TEMPLATE.format(
        label=args.get("label", "unknown"),
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        model_path=env.get("model_path", "?"),
        model_sha=env.get("model_first_safetensors_sha256", "?") or "(none)",
        adapter_path=env.get("adapter_path") or "(none)",
        adapter_sha=env.get("adapter_sha256") or "(none)",
        git_commit=_safe_get(env, "git", "commit", default="?") or "?",
        git_describe=_safe_get(env, "git", "describe", default="?") or "?",
        git_dirty="yes" if _safe_get(env, "git", "status_short", default="") else "no",
        hardware_node=_safe_get(env, "hardware", "node", default="?"),
        machine=_safe_get(env, "hardware", "machine", default="?"),
        processor=_safe_get(env, "hardware", "processor", default="?"),
        python_version=_safe_get(env, "hardware", "python", default="?"),
        mlx_version=_safe_get(env, "hardware", "mlx_version", default="?"),
        mlx_lm_version=_safe_get(env, "hardware", "mlx_lm_version", default="?"),
        date=time.strftime("%Y-%m-%d"),
        lighteval_tasks=args.get("lighteval_tasks") or "(none)",
        evalplus_tasks=args.get("evalplus_tasks") or "(none)",
        temperature=args.get("temperature", "0.0"),
        max_tokens=args.get("max_tokens", 1024),
        n_samples=args.get("n_samples", 1),
        seed=args.get("seed", 42),
        rerun_script=str(rerun_path),
    )
    methodology_path = output_dir / "methodology.md"
    methodology_path.write_text(md)
    return methodology_path


def write_rerun(output_dir: Path, env_path: Path, args: dict) -> Path:
    env = json.loads(env_path.read_text())
    rerun_path = output_dir / "rerun.sh"
    extra_flags = []
    if args.get("quick"):
        extra_flags.append("--quick")
    if args.get("extended"):
        extra_flags.append("--extended")
    rerun = RERUN_TEMPLATE.format(
        label=args.get("label", "unknown"),
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        git_commit=_safe_get(env, "git", "commit", default="?") or "?",
        model_path=env.get("model_path", "?"),
        adapter_path=env.get("adapter_path") or "",
        adapter_arg=f'--adapter "$ADAPTER"' if env.get("adapter_path") else "",
        port=args.get("port", 8000),
        extra_flags=" ".join(extra_flags),
    )
    rerun_path.write_text(rerun)
    rerun_path.chmod(0o755)
    return rerun_path


def write_report(output_dir: Path) -> Path:
    """Aggregate metrics from results.json files into a single table."""
    metrics: dict[str, float | str] = {}
    for f in output_dir.rglob("results.json"):
        rel = f.relative_to(output_dir).parent
        try:
            d = json.loads(f.read_text())
        except Exception as e:
            metrics[f"{rel}/__error__"] = str(e)
            continue
        if isinstance(d.get("metrics"), dict):
            for k, v in d["metrics"].items():
                metrics[f"{rel}/{k}"] = v
        if isinstance(d.get("pass_at_k"), dict):
            for k, v in d["pass_at_k"].items():
                metrics[f"{rel}/{k}"] = v

    env_path = output_dir / "env.json"
    env = json.loads(env_path.read_text()) if env_path.exists() else {}
    label = output_dir.name

    lines = [
        f"# eu-kiki bench report — {label}",
        "",
        f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
        "",
        "## Identity",
        "",
        f"- **Model**: `{env.get('model_path', '?')}`",
        f"- **Adapter**: `{env.get('adapter_path') or '(none)'}`",
        f"- **Adapter SHA**: `{env.get('adapter_sha256') or '(none)'}`",
        f"- **Git commit**: `{_safe_get(env, 'git', 'commit', default='?') or '?'}`",
        f"- **Hardware**: `{_safe_get(env, 'hardware', 'node', default='?')}`",
        "",
        "## Metrics",
        "",
        "| Task | Value |",
        "|------|-------|",
    ]
    for k, v in sorted(metrics.items()):
        val = f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
        lines.append(f"| `{k}` | {val} |")
    lines.append("")
    lines.append("## Reproducibility")
    lines.append("")
    lines.append("- Methodology: [`methodology.md`](methodology.md)")
    lines.append("- Environment snapshot: [`env.json`](env.json)")
    lines.append("- Rerun script: [`rerun.sh`](rerun.sh)")
    lines.append("")

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines))
    return report_path


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--label", required=True)
    p.add_argument("--lighteval-tasks", default="")
    p.add_argument("--evalplus-tasks", default="")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--temperature", default="0.0")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--n-samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--extended", action="store_true")
    args = vars(p.parse_args())

    out = args["output_dir"]
    env_path = out / "env.json"
    if not env_path.exists():
        raise SystemExit(f"env.json not found in {out} — run mlx_server_runner --env-out first")

    methodology = write_methodology(out, env_path, args)
    rerun = write_rerun(out, env_path, args)
    report = write_report(out)
    print(f"  methodology.md → {methodology}")
    print(f"  rerun.sh       → {rerun}")
    print(f"  report.md      → {report}")


if __name__ == "__main__":
    _cli()
