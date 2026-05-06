#!/usr/bin/env python3
"""evalplus_runner — run HumanEval+ / MBPP+ against an MLX-LM server.

Wraps EvalPlus (https://github.com/evalplus/evalplus) for the ailiance suite.
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
    seed: int = 42,
) -> dict:
    """Run EvalPlus task against an OpenAI-compatible endpoint.

    EvalPlus's `evaluate` entrypoint chains codegen + scoring when given a
    DATASET positional and `--backend openai --base_url <url> --model <id>`.
    We pass `greedy=True` for deterministic baselines (temperature=0).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_log = output_dir / "evalplus.log"

    # EvalPlus expects DATASET as a positional argument, then flags via Fire.
    # `dataset` task names: humaneval, mbpp (the +/plus tests are auto-included).
    dataset = task.replace("plus", "")  # humanevalplus -> humaneval
    samples_path = output_dir / "samples.jsonl"
    gen_cmd = [
        sys.executable, "-m", "evalplus.evaluate",
        dataset,
        "--samples", str(samples_path),
        "--root", str(output_dir),
        "--backend", "openai",
        "--base-url", base_url,
        "--model", model_name,
        "--n-samples", str(n_samples),
    ]
    if temperature == 0.0:
        gen_cmd.append("--greedy")
    else:
        gen_cmd.extend(["--temperature", str(temperature)])

    print(f"[evalplus] Generating {dataset} samples → {samples_path}")
    rc = _run_subprocess(gen_cmd, raw_log)
    if rc != 0:
        raise RuntimeError(f"evalplus generation failed (rc={rc}); see {raw_log}")

    # 2) Score using custom scorer (EvalPlus sandbox uses resource.RLIMIT_AS
    # which is incompatible with macOS — every test sandbox subprocess crashes
    # during init, returning "timeout" status for all problems regardless
    # of correctness). The custom scorer uses subprocess.run with a 30s
    # timeout, sandbox-free. See evalplus issue #N on GitHub for details.
    sanitized_glob = sorted(
        (output_dir / dataset).glob("*temp_*.jsonl")
    )
    sanitized = next(
        (p for p in sanitized_glob if not p.name.endswith(".raw.jsonl")),
        None,
    )
    if sanitized is None:
        print(f"[evalplus] WARN: no sanitized samples found in {output_dir / dataset}")
        pass_at_k = {}
    else:
        pass_at_k = _score_with_custom(sanitized, dataset)
        print(f"[evalplus] {dataset} (custom scorer): pass@1 = {pass_at_k}")

    summary = {
        "task": task,
        "dataset": dataset,
        "model_name": model_name,
        "base_url": base_url,
        "n_samples_per_problem": n_samples,
        "temperature": temperature,
        "seed": seed,
        "pass_at_k": pass_at_k,
        "samples_jsonl": str(samples_path),
        "log": str(raw_log),
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (output_dir / "results.json").write_text(json.dumps(summary, indent=2))
    return summary


def _score_with_custom(sanitized_path: Path, dataset: str, timeout_s: int = 30) -> dict:
    """Custom subprocess-based scorer (sandbox-free, macOS-compatible).

    EvalPlus's official sandbox uses resource.RLIMIT_AS which raises
    `ValueError: current limit exceeds maximum limit` on macOS, marking
    every problem as "timeout" regardless of correctness.

    This scorer runs `python3 -c <code>` in a plain subprocess with a
    timeout. Less rigorous than EvalPlus's sandbox (no memory cap, no
    syscall isolation) but produces accurate pass/fail for the test
    suite. Trade-off documented in eval/README.md "Limitations".
    """
    if dataset != "humaneval":
        # MBPP scoring would need its own loader; skip for now.
        return {}

    try:
        from evalplus.data import get_human_eval_plus
        problems = get_human_eval_plus()
    except Exception as e:
        return {"_error": f"could not load evalplus dataset: {e}"}

    solutions = {}
    for line in sanitized_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        solutions[d["task_id"]] = d.get("solution", "")

    base_pass = plus_pass = 0
    n = 0
    for task_id in sorted(problems.keys(), key=lambda x: int(x.split("/")[1])):
        if task_id not in solutions:
            continue
        n += 1
        sol = solutions[task_id]
        p = problems[task_id]
        ep = p["entry_point"]
        base_code = sol + "\n\n" + p["test"] + f"\n\ncheck({ep})\n"
        plus_test = p.get("test_plus") or p["test"]
        plus_code = sol + "\n\n" + plus_test + f"\n\ncheck({ep})\n"

        for code, kind in [(base_code, "base"), (plus_code, "plus")]:
            try:
                r = subprocess.run(
                    [sys.executable, "-c", code],
                    capture_output=True, text=True, timeout=timeout_s,
                )
                if r.returncode == 0:
                    if kind == "base":
                        base_pass += 1
                    else:
                        plus_pass += 1
            except subprocess.TimeoutExpired:
                pass
            except Exception:
                pass

    if n == 0:
        return {}
    return {
        "humaneval_base_pass_at_1": round(base_pass / n, 4),
        "humaneval_plus_pass_at_1": round(plus_pass / n, 4),
        "n_problems": n,
        "scorer": "custom-subprocess (macOS-compatible, EvalPlus sandbox bypassed)",
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True, help="Model name to send in API calls")
    parser.add_argument("--task", choices=["humaneval", "mbpp", "humanevalplus", "mbppplus"], required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
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
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    _cli()
