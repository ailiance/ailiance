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

Headline metric is lift_log (scale-invariant across baseline strengths).
Cells with n_samples < MIN_SAMPLES_FOR_MEDIAN are flagged and excluded from
the median computation, but still appear in raw rows + markdown table.

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

MIN_SAMPLES_FOR_MEDIAN = 25


def latest(pattern: str) -> Path | None:
    files = sorted(glob.glob(str(RAW / pattern)), reverse=True)
    return Path(files[0]) if files else None


def load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def _has_enough_samples(c: dict) -> bool:
    return ((c.get("base_n_samples") or 0) >= MIN_SAMPLES_FOR_MEDIAN
            and (c.get("tuned_n_samples") or 0) >= MIN_SAMPLES_FOR_MEDIAN)


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
        f"- Joined cells: {sum(1 for r in rows if r.get('lift_log') is not None)}",
        f"- Headline metric: **lift_log** (scale-invariant); cells with n<{MIN_SAMPLES_FOR_MEDIAN} flagged ⚠️ and excluded from median.",
        "",
    ]
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model_key"], []).append(r)
    for model in sorted(by_model):
        cells = by_model[model]
        joined = [c for c in cells if c.get("lift_log") is not None]
        joined.sort(key=lambda c: -c["lift_log"])  # highest log-lift first
        joined_eligible = [c for c in joined if _has_enough_samples(c)]
        flagged_count = len(joined) - len(joined_eligible)
        lines.append(f"## {model} ({len(joined)}/{len(cells)} joined, {flagged_count} flagged n<{MIN_SAMPLES_FOR_MEDIAN})")
        lines.append("")
        lines.append("| flag | domain | base_ppl | tuned_ppl | lift_log | lift_pct | base_n | tuned_n |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for c in cells:
            flag = "" if _has_enough_samples(c) else f"⚠️ n<{MIN_SAMPLES_FOR_MEDIAN}"
            lines.append(
                f"| {flag} | {c['domain']} | "
                f"{c['base_ppl'] if c['base_ppl'] is not None else '–'} | "
                f"{c['tuned_ppl'] if c['tuned_ppl'] is not None else '–'} | "
                f"{c.get('lift_log', '–')} | "
                f"{c.get('lift_pct', '–')} | "
                f"{c.get('base_n_samples', '–')} | "
                f"{c.get('tuned_n_samples', '–')} |"
            )
        if joined_eligible:
            log_lifts = [c["lift_log"] for c in joined_eligible]
            pct_lifts = [c["lift_pct"] for c in joined_eligible]
            med_log = statistics.median(log_lifts)
            eff_pct = (math.exp(med_log) - 1) * 100
            lines.append("")
            lines.append(
                f"**{model} stats** (n≥{MIN_SAMPLES_FOR_MEDIAN}, {len(joined_eligible)} cells): "
                f"median log-lift = {med_log:.4f} (e{eff_pct:.2f}% effective), "
                f"min = {min(log_lifts):.4f}, max = {max(log_lifts):.4f}, "
                f"median lift_pct (legacy) = {statistics.median(pct_lifts):.2f}%, "
                f"cells where adapter HURT (negative log-lift): {sum(1 for x in log_lifts if x < 0)}, "
                f"flagged (n<{MIN_SAMPLES_FOR_MEDIAN}): {flagged_count}"
            )
        elif joined:
            lines.append("")
            lines.append(
                f"**{model} stats**: all {len(joined)} joined cells have n<{MIN_SAMPLES_FOR_MEDIAN}; median suppressed."
            )
        lines.append("")
    md_path.write_text("\n".join(lines))
    print(f"Wrote {md_path}")

    # Stdout summary
    all_joined = [r for r in rows if r.get("lift_log") is not None]
    all_eligible = [r for r in all_joined if _has_enough_samples(r)]
    filtered_count = len(all_joined) - len(all_eligible)
    if all_eligible:
        log_lifts = [r["lift_log"] for r in all_eligible]
        pct_lifts = [r["lift_pct"] for r in all_eligible]
        med_log = statistics.median(log_lifts)
        eff_pct = (math.exp(med_log) - 1) * 100
        min_log = min(log_lifts)
        max_log = max(log_lifts)
        min_cell = [r for r in all_eligible if r["lift_log"] == min_log][0]
        max_cell = [r for r in all_eligible if r["lift_log"] == max_log][0]
        print()
        print(f"=== Aggregate ({len(all_eligible)} eligible cells, {filtered_count} flagged n<{MIN_SAMPLES_FOR_MEDIAN}) ===")
        print("Headline metric: lift_log (scale-invariant)")
        print(f"Median log-lift: {med_log:.4f} (e{eff_pct:.2f}% effective gain)")
        print(f"Median lift_pct: {statistics.median(pct_lifts):.2f}% (legacy, for cross-reference)")
        print(f"Cells filtered (n<{MIN_SAMPLES_FOR_MEDIAN}): {filtered_count}")
        print(f"Min log-lift: {min_log:.4f} ({min_cell['model_key']}/{min_cell['domain']})")
        print(f"Max log-lift: {max_log:.4f} ({max_cell['model_key']}/{max_cell['domain']})")
        print(f"Adapter HURT cells (log-lift<0): {sum(1 for x in log_lifts if x < 0)}")
    elif all_joined:
        print()
        print(f"=== Aggregate ({len(all_joined)} joined, ALL flagged n<{MIN_SAMPLES_FOR_MEDIAN}) ===")
        print("Median suppressed: no eligible cells.")


if __name__ == "__main__":
    main()
