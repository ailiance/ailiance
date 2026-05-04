#!/usr/bin/env python3
"""lighteval_runner — run Lighteval tasks against an MLX-LM server.

Lighteval (HF) v0.13+ supports OpenAI-compatible endpoints. We use this to
evaluate MLX models served via mlx_lm server.

Usage:
    python -m runners.lighteval_runner \\
        --base-url http://localhost:8000/v1 \\
        --model devstral-python \\
        --tasks "lighteval|gsm8k|5|0,lighteval|mmlu_pro|5|0" \\
        --output-dir results/2026-05-04/devstral-v1-python/lighteval

Common task strings:
    lighteval|humaneval|0|0          - HumanEval, 0-shot
    lighteval|mbpp|3|0               - MBPP, 3-shot
    lighteval|gsm8k|5|0              - GSM8K, 5-shot
    lighteval|mmlu_pro|5|0           - MMLU-Pro, 5-shot
    lighteval|hellaswag|10|0         - HellaSwag, 10-shot
    lighteval|truthfulqa_mc2|0|0     - TruthfulQA MC2, 0-shot
    lighteval|ifeval|0|0             - IFEval, 0-shot
    lighteval|big_bench_hard|3|1     - BBH, 3-shot CoT (CoT enabled by trailing "1")
    extended|bbeh|0|0                - BBEH (extended_tasks extra)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _ensure_lighteval() -> None:
    try:
        import lighteval  # noqa: F401
    except ImportError:
        sys.exit(
            "lighteval not installed. "
            "Install with: uv pip install 'lighteval[extended_tasks]'"
        )


def run_lighteval(
    *,
    base_url: str,
    model_name: str,
    tasks: str,
    output_dir: Path,
    api_key: str = "EMPTY",
    max_samples: int | None = None,
) -> dict:
    """Run lighteval tasks against an OpenAI-compatible endpoint via LiteLLM.

    Lighteval 0.13 dropped the standalone `openai` endpoint subcommand. We use
    `endpoint litellm` instead, which proxies to any OpenAI-compatible API
    when given `model_name=openai/<id>,base_url=<url>` (the `openai/` prefix
    tells LiteLLM to use its OpenAI client, not actually contact OpenAI).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_log = output_dir / "lighteval.log"

    # LiteLLM expects the model_name to be `openai/<arbitrary-id>` for
    # OpenAI-compatible endpoints. The base_url override redirects to localhost.
    litellm_model_args = (
        f"model_name=openai/{model_name},"
        f"base_url={base_url},"
        f"api_key={api_key}"
    )

    cmd = [
        sys.executable, "-m", "lighteval", "endpoint", "litellm",
        litellm_model_args,
        tasks,
        "--output-dir", str(output_dir),
    ]
    if max_samples is not None:
        cmd.extend(["--max-samples", str(max_samples)])

    print(f"[lighteval] {' '.join(cmd)}")
    raw_log.parent.mkdir(parents=True, exist_ok=True)
    with raw_log.open("w") as fp:
        proc = subprocess.run(cmd, stdout=fp, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"lighteval failed (rc={proc.returncode}); see {raw_log}")

    # Lighteval writes results.json under output_dir/<model>/results_<timestamp>.json
    results_files = sorted(output_dir.rglob("results_*.json"), key=os.path.getmtime)
    last = results_files[-1] if results_files else None
    parsed = json.loads(last.read_text()) if last else {}

    summary = {
        "tasks_requested": tasks,
        "model_name": model_name,
        "base_url": base_url,
        "max_samples": max_samples,
        "lighteval_results_path": str(last) if last else None,
        "metrics": _flatten_metrics(parsed.get("results", {})),
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (output_dir / "results.json").write_text(json.dumps(summary, indent=2))
    return summary


def _flatten_metrics(results: dict) -> dict[str, float]:
    """Turn lighteval nested results into flat task→metric→value."""
    flat: dict[str, float] = {}
    for task_name, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                flat[f"{task_name}.{metric_name}"] = float(value)
    return flat


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True, help="Model name to send in API calls")
    parser.add_argument("--tasks", required=True,
                        help="Comma-separated lighteval task strings (e.g. 'lighteval|gsm8k|5|0')")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit samples per task (for smoke tests)")
    args = parser.parse_args()

    _ensure_lighteval()
    summary = run_lighteval(
        base_url=args.base_url,
        model_name=args.model,
        tasks=args.tasks,
        output_dir=args.output_dir,
        api_key=args.api_key,
        max_samples=args.max_samples,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    _cli()
