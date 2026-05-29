#!/usr/bin/env python3
"""Bench base models (no LoRA) on all v1+v2 domains.

Produces the "BEFORE training" half of the comparison matrix that
pairs with eval_framework.py's --v1-only / sequential-strict output
(the "AFTER training" half).

Usage:
    .venv/bin/python scripts/bench_base.py
    .venv/bin/python scripts/bench_base.py --max-samples 5  # quick mode
    .venv/bin/python scripts/bench_base.py --models apertus eurollm

Reads model paths and methodology from eval_framework.py to ensure
the perplexity numbers are directly comparable to the tuned matrix.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# Import eval_framework's machinery so methodology stays in sync.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_framework as ef  # noqa: E402

import mlx.core as mx  # noqa: E402,F401

AILIANCE = Path(os.environ.get("AILIANCE_HOME", Path(__file__).resolve().parent.parent))
HF_DATA = AILIANCE / "data" / "hf-traced"
RAW_OUTPUT = AILIANCE / "output" / "eval" / "raw"


def list_domains_with_valid_jsonl() -> list[str]:
    """Return the sorted list of domains that have a usable valid.jsonl."""
    domains = []
    for d in sorted(HF_DATA.iterdir()):
        if d.is_dir() and (d / "valid.jsonl").exists():
            domains.append(d.name)
    return domains


def bench_base_for_model(model_key: str, domains: list[str], max_samples: int) -> list[dict]:
    """Load model_key once, eval base PPL on each domain. Returns rows."""
    model_info = ef.MODELS[model_key]
    model_path = model_info["path"]
    short = model_info["short"]

    print(f"\n=== Loading base {model_key} ({short}) from {model_path} ===", flush=True)
    t0 = time.time()
    # Reuse eval_framework's loader so the same memory caps + path validation apply.
    model, tokenizer = ef.load_model_and_tokenizer(model_path, adapter_path=None)
    load_s = time.time() - t0
    print(f"Loaded in {load_s:.1f}s", flush=True)

    rows = []
    for domain in domains:
        valid_path = HF_DATA / domain / "valid.jsonl"
        if not valid_path.exists():
            continue
        print(f"  Eval base {model_key}/{domain} ...", flush=True)
        t0 = time.time()
        try:
            avg_loss, n_samples = ef.compute_perplexity(
                model, tokenizer, valid_path, max_samples=max_samples
            )
            elapsed = time.time() - t0
            ppl = math.exp(avg_loss) if not math.isnan(avg_loss) else float("nan")
            print(
                f"    loss={avg_loss:.4f} ppl={ppl:.2f} n={n_samples} t={elapsed:.1f}s",
                flush=True,
            )
            rows.append({
                "model_key": model_key,
                "model_name": short,
                "domain": domain,
                "version": "base",
                "val_loss": round(avg_loss, 4),
                "perplexity": round(ppl, 4),
                "n_samples": n_samples,
                "eval_time_s": round(elapsed, 1),
            })
        except Exception as exc:
            print(f"    ERROR: {exc}", flush=True)
            rows.append({
                "model_key": model_key,
                "model_name": short,
                "domain": domain,
                "version": "base",
                "error": str(exc)[:200],
            })

    # Free the model before loading the next one.
    print(f"=== Unloading {model_key} ===", flush=True)
    del model, tokenizer
    ef.unload_model()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["apertus", "devstral", "eurollm"],
        help="model keys to bench (subset of eval_framework.MODELS)",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="restrict to these domains (default: all with valid.jsonl)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output JSON path (default: output/eval/raw/perplexity_base_<stamp>.json)",
    )
    args = parser.parse_args()

    domains = args.domains or list_domains_with_valid_jsonl()
    print(f"Domains to eval ({len(domains)}): {domains}", flush=True)
    print(f"Models: {args.models}", flush=True)
    print(f"Max samples per cell: {args.max_samples}", flush=True)

    all_rows = []
    for model_key in args.models:
        if model_key not in ef.MODELS:
            print(f"  SKIP {model_key} — not in eval_framework.MODELS", flush=True)
            continue
        rows = bench_base_for_model(model_key, domains, args.max_samples)
        all_rows.extend(rows)

    RAW_OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.out:
        out_path = Path(args.out)
    else:
        stamp = time.strftime("%Y%m%d_%H%M")
        out_path = RAW_OUTPUT / f"perplexity_base_{stamp}.json"
    out_path.write_text(json.dumps(all_rows, indent=2))
    print(f"\nWrote {len(all_rows)} rows to {out_path}", flush=True)


if __name__ == "__main__":
    main()
