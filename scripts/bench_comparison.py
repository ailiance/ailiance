#!/usr/bin/env python3
"""Join base PPL (no LoRA) with tuned PPL (with LoRA) → comparison matrix.

Reads:
- output/eval/raw/perplexity_base_*.json (from scripts/bench_base.py)
- output/eval/raw/perplexity_v1-only_*.json (from scripts/eval_framework.py)

Writes:
- output/eval/comparison_<stamp>.json
- output/eval/comparison_<stamp>.md

Computes per (model_key, domain) cell:
- base_ppl: model without adapter
- tuned_ppl: model + LoRA
- lift_pct: (base - tuned) / base * 100  (positive = training improved)
- lift_log: ln(base) - ln(tuned)         (log-space lift, scale-invariant)

CLI:
    .venv/bin/python scripts/bench_comparison.py
    .venv/bin/python scripts/bench_comparison.py --base path/to/base.json --tuned path/to/tuned.json
"""
import argparse
import glob
import json
import math
import statistics
import sys
import time
from pathlib import Path

EU_KIKI = Path.home() / "eu-kiki"
RAW = EU_KIKI / "output" / "eval" / "raw"
COMPARISON_DIR = EU_KIKI / "output" / "eval"


def latest(pattern: str) -> Path | None:
    files = sorted(glob.glob(str(RAW / pattern)), reverse=True)
    return Path(files[0]) if files else None


def load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=None,
                        help="path to perplexity_base_*.json (default: latest)")
    parser.add_argument("--tuned", default=None,
                        help="path to perplexity_v1-only_*.json (default: latest)")
    parser.add_argument("--out-prefix", default=None,
                        help="output prefix (default: output/eval/comparison_<stamp>)")
    args = parser.parse_args()

    base_path = Path(args.base) if args.base else latest("perplexity_base_*.json")
    tuned_path = Path(args.tuned) if args.tuned else latest("perplexity_v1-only_*.json")
    if not base_path or not base_path.exists():
        sys.exit(f"No base PPL JSON found (looked for {RAW}/perplexity_base_*.json)")
    if not tuned_path or not tuned_path.exists():
        sys.exit(f"No tuned PPL JSON found (looked for {RAW}/perplexity_v1-only_*.json)")
    print(f"Base PPL:  {base_path}")
    print(f"Tuned PPL: {tuned_path}")

    base_rows = load_rows(base_path)
    tuned_rows = load_rows(tuned_path)

    # Index base by (model_key, domain)
    base_idx = {(r["model_key"], r["domain"]): r for r in base_rows if "perplexity" in r}
    tuned_idx = {(r["model_key"], r["domain"]): r for r in tuned_rows if "perplexity" in r}

    keys = sorted(set(base_idx) | set(tuned_idx))
    rows = []
    for key in keys:
        b = base_idx.get(key)
        t = tuned_idx.get(key)
        row = {
            "model_key": key[0],
            "domain": key[1],
            "base_ppl": round(b["perplexity"], 4) if b else None,
            "tuned_ppl": round(t["perplexity"], 4) if t else None,
            "base_n_samples": b.get("n_samples") if b else None,
            "tuned_n_samples": t.get("n_samples") if t else None,
        }
        if b and t and b["perplexity"] > 0 and t["perplexity"] > 0:
            base_p = b["perplexity"]
            tuned_p = t["perplexity"]
            row["lift_pct"] = round((base_p - tuned_p) / base_p * 100, 2)
            row["lift_log"] = round(math.log(base_p) - math.log(tuned_p), 4)
        rows.append(row)

    # Output JSON
    if args.out_prefix:
        out_prefix = Path(args.out_prefix)
    else:
        stamp = time.strftime("%Y%m%d_%H%M")
        out_prefix = COMPARISON_DIR / f"comparison_{stamp}"
    json_path = Path(str(out_prefix) + ".json")
    md_path = Path(str(out_prefix) + ".md")
    json_path.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {json_path}")

    # Output markdown
    lines = [
        "# Bench comparison — base vs tuned",
        "",
        f"- Base: `{base_path.name}` ({len(base_rows)} rows)",
        f"- Tuned: `{tuned_path.name}` ({len(tuned_rows)} rows)",
        f"- Joined cells: {sum(1 for r in rows if r.get('lift_pct') is not None)}",
        "",
    ]
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model_key"], []).append(r)
    for model in sorted(by_model):
        cells = by_model[model]
        joined = [c for c in cells if c.get("lift_pct") is not None]
        joined.sort(key=lambda c: -c["lift_pct"])  # highest lift first
        lines.append(f"## {model} ({len(joined)}/{len(cells)} joined)")
        lines.append("")
        lines.append("| domain | base_ppl | tuned_ppl | lift_pct | lift_log | base_n | tuned_n |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for c in cells:
            lines.append(
                f"| {c['domain']} | "
                f"{c['base_ppl'] if c['base_ppl'] is not None else '–'} | "
                f"{c['tuned_ppl'] if c['tuned_ppl'] is not None else '–'} | "
                f"{c.get('lift_pct', '–')} | "
                f"{c.get('lift_log', '–')} | "
                f"{c.get('base_n_samples', '–')} | "
                f"{c.get('tuned_n_samples', '–')} |"
            )
        if joined:
            lifts = [c["lift_pct"] for c in joined]
            lines.append("")
            lines.append(
                f"**{model} stats**: median lift = {statistics.median(lifts):.2f}%, "
                f"min = {min(lifts):.2f}%, max = {max(lifts):.2f}%, "
                f"cells where adapter HURT (negative lift): {sum(1 for x in lifts if x < 0)}"
            )
        lines.append("")
    md_path.write_text("\n".join(lines))
    print(f"Wrote {md_path}")

    # Stdout summary
    all_joined = [r for r in rows if r.get("lift_pct") is not None]
    if all_joined:
        all_lifts = [r["lift_pct"] for r in all_joined]
        print()
        print(f"=== Aggregate ({len(all_joined)} joined cells) ===")
        print(f"Median lift: {statistics.median(all_lifts):.2f}%")
        print(f"Min: {min(all_lifts):.2f}% ({[r for r in all_joined if r['lift_pct']==min(all_lifts)][0]['model_key']}/{[r for r in all_joined if r['lift_pct']==min(all_lifts)][0]['domain']})")
        print(f"Max: {max(all_lifts):.2f}% ({[r for r in all_joined if r['lift_pct']==max(all_lifts)][0]['model_key']}/{[r for r in all_joined if r['lift_pct']==max(all_lifts)][0]['domain']})")
        print(f"Adapter HURT cells: {sum(1 for x in all_lifts if x < 0)}")


if __name__ == "__main__":
    main()
