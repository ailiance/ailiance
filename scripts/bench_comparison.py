#!/usr/bin/env python3
"""Join base PPL (no LoRA) with tuned PPL (with LoRA) -> comparison matrix.

Reads:
- output/eval/raw/perplexity_base_*.json (from scripts/bench_base.py)
- output/eval/raw/perplexity_v1-only_*.json (from scripts/eval_framework.py)

Optionally overlays a 3rd axis from iact-bench functional validator runs via
--validator-base / --validator-tuned. Each is a JSON list of cells:
    {"model_key": "...", "domain": "...", "pass_rate": 0.85, "n_samples": 50}
(extra fields like validator_name/passed/failed are ignored).

Writes:
- output/eval/comparison_<stamp>.json
- output/eval/comparison_<stamp>.md

Computes per (model_key, domain) cell:
- base_ppl, tuned_ppl, lift_pct, lift_log
- (optional) base_validator_rate, tuned_validator_rate, validator_lift (pp)

Headline metric is lift_log (scale-invariant). When validator data is present,
also detects "silent catastrophic forgetting": lift_log > 0.1 AND
validator_lift < -5pp (PPL improves but functional capability regresses).
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


def load_validator(path: Path):
    """Load validator JSON.

    Returns (pass_rate_idx, axes_idx) where:
      pass_rate_idx: {(model_key, domain): pass_rate}
      axes_idx:     {(model_key, domain): {axis_name: float, ...}}
    """
    data = json.loads(path.read_text())
    pr_idx: dict[tuple[str, str], float] = {}
    ax_idx: dict[tuple[str, str], dict[str, float]] = {}
    for r in data:
        mk = r.get("model_key")
        dom = r.get("domain")
        if mk is None or dom is None:
            continue
        if r.get("pass_rate") is not None:
            pr_idx[(mk, dom)] = float(r["pass_rate"])
        axes = {k[len("axis_"):]: float(v)
                for k, v in r.items()
                if k.startswith("axis_") and v is not None}
        if axes:
            ax_idx[(mk, dom)] = axes
    return pr_idx, ax_idx


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
    parser.add_argument("--validator-base", default=None,
                        help="optional iact-bench validator JSON for base run "
                             "(adds 3rd axis: validator pass_rate)")
    parser.add_argument("--validator-tuned", default=None,
                        help="optional iact-bench validator JSON for tuned run")
    parser.add_argument("--validator-min-cells", type=int, default=5,
                        help="minimum cells with validator data to surface overlay "
                             "(default: 5)")
    parser.add_argument(
        "--metric-axes", default=None,
        help="Comma-sep axis names to surface as extra MD columns "
             "(reads axis_<name> fields from --validator-tuned JSON). "
             "Example: parse_ok,erc_clean,sch_render,drc_clean,sem_equiv",
    )
    args = parser.parse_args()

    metric_axes: list[str] = []
    if args.metric_axes:
        metric_axes = [a.strip() for a in args.metric_axes.split(",")
                       if a.strip()]

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

    val_base_idx: dict[tuple[str, str], float] = {}
    val_tuned_idx: dict[tuple[str, str], float] = {}
    ax_base_idx: dict[tuple[str, str], dict[str, float]] = {}
    ax_tuned_idx: dict[tuple[str, str], dict[str, float]] = {}
    if args.validator_base:
        vb_path = Path(args.validator_base)
        if not vb_path.exists():
            sys.exit(f"--validator-base not found: {vb_path}")
        val_base_idx, ax_base_idx = load_validator(vb_path)
        print(f"Validator base:  {vb_path} ({len(val_base_idx)} cells)")
    if args.validator_tuned:
        vt_path = Path(args.validator_tuned)
        if not vt_path.exists():
            sys.exit(f"--validator-tuned not found: {vt_path}")
        val_tuned_idx, ax_tuned_idx = load_validator(vt_path)
        print(f"Validator tuned: {vt_path} ({len(val_tuned_idx)} cells)")

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

        # Validator overlay (always include fields when at least one validator
        # source is provided, so downstream consumers see a consistent schema)
        if val_base_idx or val_tuned_idx:
            vb = val_base_idx.get(key)
            vt = val_tuned_idx.get(key)
            row["base_validator_rate"] = round(vb, 4) if vb is not None else None
            row["tuned_validator_rate"] = round(vt, 4) if vt is not None else None
            if vb is not None and vt is not None:
                # Pass rates are in [0,1]; lift is in percentage points (signed)
                row["validator_lift"] = round((vt - vb) * 100, 2)
            else:
                row["validator_lift"] = None
        if metric_axes:
            base_axes = ax_base_idx.get(key, {})
            tuned_axes = ax_tuned_idx.get(key, {})
            for axis in metric_axes:
                # Prefer tuned (post-training) for headline; fallback to base.
                v = tuned_axes.get(axis, base_axes.get(axis))
                row[f"axis_{axis}"] = round(v, 4) if v is not None else None
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

    has_validator = any(r.get("validator_lift") is not None for r in rows)

    # Output markdown
    joined_count = sum(1 for r in rows if r.get("lift_log") is not None)
    lines = [
        "# Bench comparison -- base vs tuned",
        "",
        f"- Base: `{base_path.name}` ({len(base_rows)} rows)",
        f"- Tuned: `{tuned_path.name}` ({len(tuned_rows)} rows)",
        f"- Joined cells: {joined_count}",
        f"- Headline metric: **lift_log** (scale-invariant); cells with n<{MIN_SAMPLES_FOR_MEDIAN} flagged and excluded from median.",
    ]
    if has_validator:
        v_count = sum(1 for r in rows if r.get("validator_lift") is not None)
        lines.append(
            f"- Validator overlay: {v_count} cells with paired base+tuned "
            f"validator pass_rate (3rd axis)."
        )
    lines.append("")

    by_model: dict[str, list[dict]] = {}
    for r in rows:
        by_model.setdefault(r["model_key"], []).append(r)
    for model in sorted(by_model):
        cells = by_model[model]
        joined = [c for c in cells if c.get("lift_log") is not None]
        joined.sort(key=lambda c: -c["lift_log"])  # highest log-lift first
        joined_eligible = [c for c in joined if _has_enough_samples(c)]
        flagged_count = len(joined) - len(joined_eligible)
        lines.append(
            f"## {model} ({len(joined)}/{len(cells)} joined, "
            f"{flagged_count} flagged n<{MIN_SAMPLES_FOR_MEDIAN})"
        )
        lines.append("")
        model_has_val = any(
            c.get("validator_lift") is not None
            or c.get("base_validator_rate") is not None
            or c.get("tuned_validator_rate") is not None
            for c in cells
        )
        if model_has_val:
            lines.append(
                "| flag | domain | base_ppl | tuned_ppl | lift_log | lift_pct "
                "| base_n | tuned_n | base_val | tuned_val | val_lift_pp |"
            )
            lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        else:
            lines.append(
                "| flag | domain | base_ppl | tuned_ppl | lift_log | lift_pct "
                "| base_n | tuned_n |"
            )
            lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        if metric_axes:
            axis_header = " " + " | ".join(metric_axes) + " |"
            axis_sep = "---:|" * len(metric_axes)
            # Append to last header line + separator line.
            lines[-2] = lines[-2].rstrip("|") + " | " + axis_header
            lines[-1] = lines[-1].rstrip("|") + "|" + axis_sep
        for c in cells:
            flag = "" if _has_enough_samples(c) else f"n<{MIN_SAMPLES_FOR_MEDIAN}"
            base_ppl = c["base_ppl"] if c["base_ppl"] is not None else "-"
            tuned_ppl = c["tuned_ppl"] if c["tuned_ppl"] is not None else "-"
            lift_log = c.get("lift_log", "-")
            lift_pct = c.get("lift_pct", "-")
            base_n = c.get("base_n_samples", "-")
            tuned_n = c.get("tuned_n_samples", "-")
            row_str = (
                f"| {flag} | {c['domain']} | {base_ppl} | {tuned_ppl} | "
                f"{lift_log} | {lift_pct} | {base_n} | {tuned_n} |"
            )
            if model_has_val:
                bv = c.get("base_validator_rate")
                tv = c.get("tuned_validator_rate")
                vl = c.get("validator_lift")
                bv_s = bv if bv is not None else "-"
                tv_s = tv if tv is not None else "-"
                vl_s = vl if vl is not None else "-"
                row_str += f" {bv_s} | {tv_s} | {vl_s} |"
            if metric_axes:
                for axis in metric_axes:
                    v = c.get(f"axis_{axis}")
                    row_str += f" {v if v is not None else '-'} |"
            lines.append(row_str)
        if joined_eligible:
            log_lifts = [c["lift_log"] for c in joined_eligible]
            pct_lifts = [c["lift_pct"] for c in joined_eligible]
            med_log = statistics.median(log_lifts)
            eff_pct = (math.exp(med_log) - 1) * 100
            lines.append("")
            lines.append(
                f"**{model} stats** (n>={MIN_SAMPLES_FOR_MEDIAN}, "
                f"{len(joined_eligible)} cells): "
                f"median log-lift = {med_log:.4f} (e{eff_pct:.2f}% effective), "
                f"min = {min(log_lifts):.4f}, max = {max(log_lifts):.4f}, "
                f"median lift_pct (legacy) = {statistics.median(pct_lifts):.2f}%, "
                f"cells where adapter HURT (negative log-lift): "
                f"{sum(1 for x in log_lifts if x < 0)}, "
                f"flagged (n<{MIN_SAMPLES_FOR_MEDIAN}): {flagged_count}"
            )
        elif joined:
            lines.append("")
            lines.append(
                f"**{model} stats**: all {len(joined)} joined cells have "
                f"n<{MIN_SAMPLES_FOR_MEDIAN}; median suppressed."
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
        print(
            f"=== Aggregate ({len(all_eligible)} eligible cells, "
            f"{filtered_count} flagged n<{MIN_SAMPLES_FOR_MEDIAN}) ==="
        )
        print("Headline metric: lift_log (scale-invariant)")
        print(f"Median log-lift: {med_log:.4f} (e{eff_pct:.2f}% effective gain)")
        print(f"Median lift_pct: {statistics.median(pct_lifts):.2f}% (legacy, for cross-reference)")
        print(f"Cells filtered (n<{MIN_SAMPLES_FOR_MEDIAN}): {filtered_count}")
        print(f"Min log-lift: {min_log:.4f} ({min_cell['model_key']}/{min_cell['domain']})")
        print(f"Max log-lift: {max_log:.4f} ({max_cell['model_key']}/{max_cell['domain']})")
        print(f"Adapter HURT cells (log-lift<0): {sum(1 for x in log_lifts if x < 0)}")
    elif all_joined:
        print()
        print(
            f"=== Aggregate ({len(all_joined)} joined, "
            f"ALL flagged n<{MIN_SAMPLES_FOR_MEDIAN}) ==="
        )
        print("Median suppressed: no eligible cells.")

    # Validator overlay summary
    val_cells = [r for r in rows if r.get("validator_lift") is not None]
    if val_cells:
        val_lifts = [r["validator_lift"] for r in val_cells]
        med_vl = statistics.median(val_lifts)
        # Silent catastrophic forgetting: PPL log-lift > 0.1 AND validator < -5pp
        silent_forget = [
            r for r in val_cells
            if r.get("lift_log") is not None
            and r["lift_log"] > 0.1
            and r["validator_lift"] < -5.0
        ]
        print()
        print(f"Validator overlay: {len(val_cells)} cells with validator data")
        print(f"  Median validator lift: {med_vl:.2f} pp")
        print(
            f"  Cells where PPL improves but validator REGRESSES: "
            f"{len(silent_forget)} (silent catastrophic forgetting)"
        )
        for r in silent_forget:
            print(
                f"    - {r['model_key']}/{r['domain']}: "
                f"lift_log={r['lift_log']:+.4f}, "
                f"validator_lift={r['validator_lift']:+.2f}pp"
            )
        if len(val_cells) < args.validator_min_cells:
            print(
                f"  Note: below --validator-min-cells={args.validator_min_cells} "
                f"threshold; treat overlay as indicative only."
            )


if __name__ == "__main__":
    main()
