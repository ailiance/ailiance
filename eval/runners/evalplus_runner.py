#!/usr/bin/env python3
"""evalplus_runner — run HumanEval+ / MBPP+ against an MLX-LM server.

Wraps EvalPlus (https://github.com/evalplus/evalplus) for the eu-kiki suite.
Hits the local mlx_lm server (OpenAI-compatible) and produces a publishable
JSON+Markdown report.

Usage:
    python -m runners.evalplus_runner \\
        --base-url http://localhost:8000/v1 \\
        --model devstral-python \\
        --task humaneval \\
        --output-dir results/2026-05-04/devstral-v1-python/humaneval

Tasks: humaneval, mbpp, humanevalplus, mbppplus
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _ensure_evalplus() -> None:
    try:
        import evalplus  # noqa: F401
    except ImportError:
        sys.exit(
            "evalplus not installed. "
            "Install with: uv pip install evalplus"
        )


def _run_subprocess(cmd: list[str], log_path: Path, env: dict | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as fp:
        proc = subprocess.run(
            cmd, stdout=fp, stderr=subprocess.STDOUT, env={**os.environ, **(env or {})},
        )
    return proc.returncode


def run_evalplus(
    *,
    base_url: str,
    model_name: str,
    task: str,
    output_dir: Path,
    n_samples: int = 1,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    seed: int = 42,
) -> dict:
    """Run EvalPlus task against an OpenAI-compatible endpoint."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_log = output_dir / "evalplus.log"

    # 1) Generate samples
    samples_path = output_dir / "samples.jsonl"
    gen_cmd = [
        sys.executable, "-m", "evalplus.evaluate",
        "--dataset", task,
        "--samples", str(samples_path),
        "--base_url", base_url,
        "--model", model_name,
        "--temperature", str(temperature),
        "--n_samples", str(n_samples),
        "--max_tokens", str(max_tokens),
    ]
    print(f"[evalplus] Generating {task} samples → {samples_path}")
    rc = _run_subprocess(gen_cmd, raw_log)
    if rc != 0:
        raise RuntimeError(f"evalplus generation failed (rc={rc}); see {raw_log}")

    # 2) Score (EvalPlus auto-runs scoring after generation if --evaluate is set,
    # but for explicit control we re-invoke)
    eval_results = output_dir / "evalplus_results.json"
    eval_cmd = [
        sys.executable, "-m", "evalplus.evaluate",
        "--dataset", task,
        "--samples", str(samples_path),
    ]
    print(f"[evalplus] Scoring {task}")
    _run_subprocess(eval_cmd, raw_log)

    # 3) Parse results — EvalPlus writes to stdout. Read the log.
    pass_at_k = _parse_evalplus_log(raw_log)

    summary = {
        "task": task,
        "model_name": model_name,
        "base_url": base_url,
        "n_samples_per_problem": n_samples,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
        "pass_at_k": pass_at_k,
        "samples_jsonl": str(samples_path),
        "log": str(raw_log),
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (output_dir / "results.json").write_text(json.dumps(summary, indent=2))
    return summary


def _parse_evalplus_log(log_path: Path) -> dict:
    """Extract pass@k metrics from evalplus stdout log."""
    if not log_path.exists():
        return {}
    text = log_path.read_text()
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        # EvalPlus lines look like: "pass@1: 0.756" or "pass@1 (base tests): 0.756"
        line = line.strip()
        if line.startswith("pass@"):
            try:
                key_part, val_part = line.split(":", 1)
                key = key_part.strip()
                val = float(val_part.strip())
                metrics[key] = val
            except (ValueError, IndexError):
                continue
    return metrics


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True, help="Model name to send in API calls")
    parser.add_argument("--task", choices=["humaneval", "mbpp", "humanevalplus", "mbppplus"], required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _ensure_evalplus()
    summary = run_evalplus(
        base_url=args.base_url,
        model_name=args.model,
        task=args.task,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    _cli()
