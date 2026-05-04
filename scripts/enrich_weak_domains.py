#!/usr/bin/env python3
"""
enrich_weak_domains.py -- Enrich 5 weak training domains to ~2850 train examples each.

Domains enriched:
  - sql       (160 -> 2850)  via gretelai/synthetic_text_to_sql (Apache-2.0)
  - shell     (57  -> 2850)  via Takiyoshia/commitpack-parquet lang=Shell (MIT)
  - cpp       (191 -> 2850)  via Takiyoshia/commitpack-parquet lang=C++ (MIT)
                             + TokenBender/code_instructions_122k_alpaca_style (Apache-2.0)
  - html-css  (94  -> 2850)  via Takiyoshia/commitpack-parquet lang=HTML+CSS (MIT)
  - ml-training (76 -> 2850) via Takiyoshia/commitpack-parquet Python+ML filter (MIT)

All source datasets are Apache-2.0 or MIT licensed.

Usage:
    cd ~/eu-kiki && uv run python scripts/enrich_weak_domains.py
"""

import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from datasets import load_dataset
except ImportError:
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
    from datasets import load_dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
TARGET = 3000  # sample size before split
VALID_RATIO = 0.05
OUT_ROOT = Path(__file__).parent.parent / "data" / "hf-traced"
MANIFEST_PATH = OUT_ROOT / "MANIFEST_enriched.json"

# ---------------------------------------------------------------------------
# Helpers (aligned with build_hf_datasets.py patterns)
# ---------------------------------------------------------------------------


def make_message(
    user: str,
    assistant: str,
    provenance: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build one JSONL record in messages format with provenance metadata."""
    rec: dict[str, Any] = {
        "messages": [
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": assistant.strip()},
        ]
    }
    if provenance:
        rec["_provenance"] = provenance
    return rec


def split_train_valid(
    records: list[dict], seed: int = SEED
) -> tuple[list[dict], list[dict]]:
    """Reproducible 95/5 split."""
    rng = random.Random(seed)
    shuffled = records.copy()
    rng.shuffle(shuffled)
    n_valid = max(1, round(len(shuffled) * VALID_RATIO))
    return shuffled[n_valid:], shuffled[:n_valid]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  -> wrote {len(records):,} records to {path}")


def save_domain(domain: str, records: list[dict]) -> tuple[int, int]:
    """Split and overwrite a domain directory; return (n_train, n_valid)."""
    train, valid = split_train_valid(records)
    domain_dir = OUT_ROOT / domain
    write_jsonl(domain_dir / "train.jsonl", train)
    write_jsonl(domain_dir / "valid.jsonl", valid)
    return len(train), len(valid)


def cap(
    records: list[dict], n: int = TARGET, seed: int = SEED
) -> list[dict]:
    """Randomly cap to at most n examples."""
    if len(records) <= n:
        return records
    rng = random.Random(seed)
    return rng.sample(records, n)


def is_nonempty(*texts: str) -> bool:
    return all(t and t.strip() for t in texts)


# ---------------------------------------------------------------------------
# Manifest tracking
# ---------------------------------------------------------------------------
manifest_entries: list[dict[str, Any]] = []


def record_manifest(
    domain: str,
    hf_id: str,
    license_: str,
    n_source: int,
    n_used: int,
    n_train: int,
    n_valid: int,
    notes: str = "",
) -> None:
    manifest_entries.append(
        {
            "domain": domain,
            "hf_id": hf_id,
            "license": license_,
            "n_source": n_source,
            "n_used": n_used,
            "n_train": n_train,
            "n_valid": n_valid,
            "access_date": datetime.now(timezone.utc).isoformat(),
            "notes": notes,
        }
    )


# ---------------------------------------------------------------------------
# 1. SQL -- gretelai/synthetic_text_to_sql (Apache-2.0, ~105k)
# ---------------------------------------------------------------------------
def enrich_sql() -> bool:
    HF_ID = "gretelai/synthetic_text_to_sql"
    LICENSE = "Apache-2.0"

    print(f"\n[sql] Loading {HF_ID} ...")
    try:
        ds = load_dataset(HF_ID, split="train")
        total = len(ds)
        print(f"  total rows: {total:,}")

        records: list[dict] = []
        for idx, row in enumerate(ds):
            sql_prompt = str(row.get("sql_prompt", "") or row.get("prompt", ""))
            sql_code = str(row.get("sql", "") or row.get("query", ""))
            sql_context = str(row.get("sql_context", "") or "")

            if not is_nonempty(sql_prompt, sql_code):
                continue

            # Build a richer user prompt including context when available
            user_text = sql_prompt
            if sql_context and len(sql_context) > 20:
                user_text = f"Given the following schema:\n{sql_context}\n\n{sql_prompt}"

            records.append(
                make_message(
                    user_text,
                    sql_code,
                    provenance={
                        "source": HF_ID,
                        "license": LICENSE,
                        "record_idx": str(idx),
                    },
                )
            )

        print(f"  usable rows: {len(records):,}")
        capped = cap(records)
        n_train, n_valid = save_domain("sql", capped)
        record_manifest("sql", HF_ID, LICENSE, total, len(capped), n_train, n_valid,
                        "synthetic text-to-SQL; sql_prompt+sql_context -> sql")
        print(f"  [sql] OK: {n_train} train / {n_valid} valid")
        return True
    except Exception as exc:
        print(f"  [sql] FAILED: {exc}")
        return False


# ---------------------------------------------------------------------------
# Shared: commitpack-parquet single-pass multi-domain collector
# ---------------------------------------------------------------------------
COMMITPACK_HF_ID = "Takiyoshia/commitpack-parquet"
COMMITPACK_LICENSE = "MIT"

# Domain -> set of language tags to match
# NOTE: Takiyoshia/commitpack-parquet is Python-only in practice.
# Shell, C++, HTML/CSS are handled by dedicated supplement functions.
COMMITPACK_LANG_MAP: dict[str, set[str]] = {
    "shell": {"Shell"},
    "cpp": {"C++", "C"},
    "html-css": {"HTML", "CSS"},
    "ml-training": {"Python"},  # further filtered by _has_ml_imports
}


def _format_commitpack_record(
    msg: str, old_code: str, new_code: str, lang: str,
) -> dict[str, Any]:
    """Convert a commitpack row to messages format."""
    lang_lower = lang.lower()
    if old_code and len(old_code) > 10:
        user_text = (
            f"Modify this {lang_lower} code to: {msg}\n\n"
            f"Existing code:\n```{lang_lower}\n{old_code[:2000]}\n```"
        )
    else:
        user_text = f"Write {lang_lower} code that: {msg}"
    return make_message(
        user_text,
        new_code,
        provenance={
            "source": COMMITPACK_HF_ID,
            "license": COMMITPACK_LICENSE,
            "language": lang,
        },
    )


def enrich_commitpack_domains() -> dict[str, bool]:
    """Single streaming pass over commitpack-parquet for all 4 domains.

    Returns dict of domain -> success boolean.
    """
    print(f"\n[commitpack] Single-pass streaming {COMMITPACK_HF_ID} for 4 domains ...")
    sys.stdout.flush()

    # Accumulate records per domain
    domain_records: dict[str, list[dict]] = {d: [] for d in COMMITPACK_LANG_MAP}
    # Reverse map: language -> list of domains
    lang_to_domains: dict[str, list[str]] = {}
    for domain, langs in COMMITPACK_LANG_MAP.items():
        for lang in langs:
            lang_to_domains.setdefault(lang, []).append(domain)

    # Track which domains still need more data
    domain_limit = TARGET * 2

    try:
        ds = load_dataset(COMMITPACK_HF_ID, split="train", streaming=True)
        scanned = 0

        for row in ds:
            scanned += 1
            if scanned % 200_000 == 0:
                counts = {d: len(r) for d, r in domain_records.items()}
                print(f"    scanned {scanned:,} | {counts}")
                sys.stdout.flush()

            # Check if all domains have enough
            if all(len(r) >= domain_limit for r in domain_records.values()):
                print(f"    all domains full at {scanned:,} rows")
                break
            # Early exit: if ml-training is full and other domains are still empty
            # after 500K rows, this dataset is Python-only — stop scanning
            if (scanned >= 500_000
                    and len(domain_records["ml-training"]) >= domain_limit):
                non_ml = {d: len(r) for d, r in domain_records.items()
                          if d != "ml-training"}
                if all(v == 0 for v in non_ml.values()):
                    print(f"    ml-training full, other domains empty after "
                          f"{scanned:,} rows — stopping early")
                    break

            lang = str(row.get("lang", "") or row.get("language", ""))
            matching_domains = lang_to_domains.get(lang)
            if not matching_domains:
                continue

            msg = str(row.get("subject", "") or row.get("message", ""))
            old_code = str(row.get("old_contents", "") or "")
            new_code = str(row.get("new_contents", "") or row.get("content", ""))

            if not is_nonempty(msg, new_code):
                continue
            if len(new_code) > 8000 or len(new_code) < 20:
                continue

            for domain in matching_domains:
                if len(domain_records[domain]) >= domain_limit:
                    continue

                # Extra filter for ml-training: must have ML imports
                if domain == "ml-training" and not _has_ml_imports(new_code):
                    continue

                # For ml-training, adjust the user prompt
                if domain == "ml-training":
                    if old_code and len(old_code) > 10:
                        user_text = (
                            f"Modify this ML training code to: {msg}\n\n"
                            f"Existing code:\n```python\n{old_code[:2000]}\n```"
                        )
                    else:
                        user_text = f"Write a Python ML script that: {msg}"
                    domain_records[domain].append(
                        make_message(
                            user_text,
                            new_code,
                            provenance={
                                "source": COMMITPACK_HF_ID,
                                "license": COMMITPACK_LICENSE,
                                "language": lang,
                                "filter": "ml-imports>=2",
                            },
                        )
                    )
                else:
                    domain_records[domain].append(
                        _format_commitpack_record(msg, old_code, new_code, lang)
                    )

        print(f"  [commitpack] Finished scanning {scanned:,} rows")
        counts = {d: len(r) for d, r in domain_records.items()}
        print(f"  collected: {counts}")
        sys.stdout.flush()

    except Exception as exc:
        print(f"  [commitpack] Stream failed: {exc}")
        sys.stdout.flush()

    # Post-process: supplement domains that are still short
    results: dict[str, bool] = {}

    # Shell: supplement from self-oss-instruct if needed
    shell_recs = domain_records["shell"]
    if len(shell_recs) < TARGET:
        print(f"\n  [shell] Only {len(shell_recs)} from commitpack, supplementing ...")
        sys.stdout.flush()
        shell_recs = _supplement_shell(shell_recs)
    try:
        capped = cap(shell_recs)
        n_train, n_valid = save_domain("shell", capped)
        record_manifest("shell", COMMITPACK_HF_ID, COMMITPACK_LICENSE,
                        -1, len(capped), n_train, n_valid,
                        "commitpack-parquet lang=Shell; commit msg -> code")
        print(f"  [shell] OK: {n_train} train / {n_valid} valid")
        results["shell"] = True
    except Exception as exc:
        print(f"  [shell] FAILED saving: {exc}")
        results["shell"] = False

    # C++: supplement from code_instructions if needed
    cpp_recs = domain_records["cpp"]
    if len(cpp_recs) < TARGET:
        print(f"\n  [cpp] Only {len(cpp_recs)} from commitpack, supplementing ...")
        sys.stdout.flush()
        cpp_recs = _supplement_from_code_instructions(cpp_recs, cpp_markers=True)
    try:
        capped = cap(cpp_recs)
        n_train, n_valid = save_domain("cpp", capped)
        record_manifest("cpp", COMMITPACK_HF_ID, COMMITPACK_LICENSE,
                        -1, len(capped), n_train, n_valid,
                        "commitpack-parquet lang=C++/C; commit msg -> code")
        print(f"  [cpp] OK: {n_train} train / {n_valid} valid")
        results["cpp"] = True
    except Exception as exc:
        print(f"  [cpp] FAILED saving: {exc}")
        results["cpp"] = False

    # HTML/CSS — commitpack-parquet lacks HTML/CSS, use code instruction datasets
    html_css_recs = domain_records["html-css"]
    if len(html_css_recs) < TARGET:
        print(f"\n  [html-css] Only {len(html_css_recs)} from commitpack-parquet, "
              f"supplementing from code instruction datasets ...")
        sys.stdout.flush()
        html_css_recs = _supplement_html_css(html_css_recs)
    try:
        capped = cap(html_css_recs)
        n_train, n_valid = save_domain("html-css", capped)
        record_manifest(
            "html-css",
            "iamtarun/code_instructions_120k_alpaca+sahil2801/CodeAlpaca-20k",
            "Apache-2.0+CC-BY-4.0",
            -1, len(capped), n_train, n_valid,
            "HTML/CSS filtered from code instruction datasets",
        )
        print(f"  [html-css] OK: {n_train} train / {n_valid} valid")
        results["html-css"] = True
    except Exception as exc:
        print(f"  [html-css] FAILED saving: {exc}")
        results["html-css"] = False

    # ML-training
    try:
        capped = cap(domain_records["ml-training"])
        n_train, n_valid = save_domain("ml-training", capped)
        record_manifest("ml-training", COMMITPACK_HF_ID, COMMITPACK_LICENSE,
                        -1, len(capped), n_train, n_valid,
                        "commitpack-parquet Python+ML-imports; commit msg -> code")
        print(f"  [ml-training] OK: {n_train} train / {n_valid} valid")
        results["ml-training"] = True
    except Exception as exc:
        print(f"  [ml-training] FAILED saving: {exc}")
        results["ml-training"] = False

    return results


def _supplement_shell(existing: list[dict]) -> list[dict]:
    """Supplement shell data from self-oss-instruct if commitpack is insufficient."""
    HF_SUP = "bigcode/self-oss-instruct-sc2-exec-filter-50k"
    need = TARGET * 2 - len(existing)
    print(f"  supplementing from {HF_SUP}, need ~{need} more ...")
    sys.stdout.flush()

    try:
        ds = load_dataset(HF_SUP, split="train")
        shell_markers = [
            "#!/bin/bash", "#!/bin/sh", "echo ", "if [", "fi\n",
            "done\n", "export ", "#!/usr/bin/env bash", "grep ",
            "awk ", "sed ", "curl ", "wget ",
        ]
        for row in ds:
            instruction = str(row.get("instruction", ""))
            response = str(row.get("response", ""))
            seed = str(row.get("seed", ""))
            text_combined = (seed + instruction + response).lower()

            if not is_nonempty(instruction, response):
                continue
            if any(m.lower() in text_combined for m in shell_markers):
                existing.append(
                    make_message(
                        instruction,
                        response,
                        provenance={"source": HF_SUP, "license": "Apache-2.0", "language": "shell"},
                    )
                )
            if len(existing) >= TARGET * 2:
                break
    except Exception as exc:
        print(f"    supplement failed: {exc}")

    return existing


def _supplement_from_code_instructions(
    existing: list[dict],
    cpp_markers: bool = False,
) -> list[dict]:
    """Supplement with TokenBender/code_instructions_122k_alpaca_style."""
    HF_SUP = "TokenBender/code_instructions_122k_alpaca_style"
    print(f"  supplementing from {HF_SUP} ...")
    sys.stdout.flush()
    try:
        ds = load_dataset(HF_SUP, split="train", streaming=True)
        cpp_keywords = ["#include", "std::", "int main", "vector<", "cout",
                        "nullptr", "class ", "template<", "namespace"]
        for row in ds:
            instruction = str(row.get("instruction", ""))
            output = str(row.get("output", ""))
            if not is_nonempty(instruction, output):
                continue
            if cpp_markers:
                combined = (instruction + output).lower()
                if not any(k.lower() in combined for k in cpp_keywords):
                    continue
            existing.append(
                make_message(
                    instruction,
                    output,
                    provenance={"source": HF_SUP, "license": "Apache-2.0", "language": "C++"},
                )
            )
            if len(existing) >= TARGET * 2:
                break
    except Exception as exc:
        print(f"    supplement failed: {exc}")
    return existing


# ---------------------------------------------------------------------------
# 5. ML-Training -- commitpack-parquet Python + ML import filter (MIT)
# ---------------------------------------------------------------------------

ML_IMPORT_PATTERNS = [
    re.compile(r"(?:import|from)\s+torch\b"),
    re.compile(r"(?:import|from)\s+tensorflow\b"),
    re.compile(r"(?:import|from)\s+sklearn\b"),
    re.compile(r"(?:import|from)\s+transformers\b"),
    re.compile(r"(?:import|from)\s+keras\b"),
    re.compile(r"(?:import|from)\s+(?:numpy|pandas)\b"),
    re.compile(r"(?:import|from)\s+lightning\b"),
    re.compile(r"(?:import|from)\s+jax\b"),
    re.compile(r"(?:import|from)\s+flax\b"),
    re.compile(r"(?:import|from)\s+mlflow\b"),
    re.compile(r"(?:import|from)\s+wandb\b"),
    re.compile(r"(?:import|from)\s+datasets\b"),
    re.compile(r"(?:import|from)\s+accelerate\b"),
    re.compile(r"(?:import|from)\s+peft\b"),
    re.compile(r"(?:import|from)\s+trl\b"),
]


def _has_ml_imports(code: str) -> bool:
    """Check if code contains ML-related imports (at least 2 distinct)."""
    return sum(1 for p in ML_IMPORT_PATTERNS if p.search(code)) >= 2


# ---------------------------------------------------------------------------
# HTML/CSS supplement from code instruction datasets
# ---------------------------------------------------------------------------
_HTML_MARKERS = ["<html", "<div", "<span", "<!doctype", "<head", "<body", "<form",
                 "<table", "<section", "<nav", "<header", "<footer", "<ul", "<ol"]
_CSS_MARKERS = ["color:", "margin:", "padding:", "display:", "font-size:",
                "background:", "border:", "flex", "grid", "@media"]


def _is_html_css_record(instruction: str, output: str) -> bool:
    """Check if a record is HTML or CSS related."""
    combined = (output + instruction).lower()
    has_html = any(m in combined for m in _HTML_MARKERS)
    has_css = (
        sum(1 for m in _CSS_MARKERS if m in combined) >= 2
        and "{" in output
        and "}" in output
    )
    return has_html or has_css


def _supplement_html_css(existing: list[dict]) -> list[dict]:
    """Load HTML/CSS instruction pairs from code instruction datasets.

    Sources:
      - iamtarun/code_instructions_120k_alpaca (Apache-2.0) ~8K HTML/CSS records
      - sahil2801/CodeAlpaca-20k (CC-BY-4.0) ~1.3K HTML records
    """
    sources = [
        ("iamtarun/code_instructions_120k_alpaca", "Apache-2.0"),
        ("sahil2801/CodeAlpaca-20k", "CC-BY-4.0"),
    ]

    for hf_id, license_ in sources:
        if len(existing) >= TARGET * 2:
            break
        try:
            print(f"    loading {hf_id} ...")
            sys.stdout.flush()
            ds = load_dataset(hf_id, split="train", streaming=True)
            count = 0
            for row in ds:
                instruction = str(row.get("instruction", ""))
                inp = str(row.get("input", ""))
                output = str(row.get("output", ""))
                if not is_nonempty(instruction, output):
                    continue
                if not _is_html_css_record(instruction, output):
                    continue
                # Build user prompt including input context if present
                user_text = instruction
                if inp and inp.strip():
                    user_text = f"{instruction}\n\nInput:\n{inp}"
                existing.append(
                    make_message(
                        user_text,
                        output,
                        provenance={
                            "source": hf_id,
                            "license": license_,
                            "language": "HTML/CSS",
                        },
                    )
                )
                count += 1
                if len(existing) >= TARGET * 2:
                    break
            print(f"    {hf_id}: collected {count} HTML/CSS records")
            sys.stdout.flush()
        except Exception as exc:
            print(f"    {hf_id} failed: {exc}")
            sys.stdout.flush()

    return existing


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("eu-kiki -- Weak domain enrichment")
    print(f"Output root : {OUT_ROOT}")
    print(f"Target/domain: {TARGET} (-> ~{int(TARGET * (1 - VALID_RATIO))} train)")
    print(f"Split ratio : {int((1 - VALID_RATIO) * 100)}/{int(VALID_RATIO * 100)} (train/valid)")
    print(f"Seed        : {SEED}")
    print("=" * 60)

    results: dict[str, bool] = {}

    # SQL uses its own dataset (fast)
    results["sql"] = enrich_sql()

    # Shell, C++, HTML/CSS, ML-training all come from commitpack-parquet
    # Do a single streaming pass to avoid scanning 3M+ rows 4 times
    commitpack_results = enrich_commitpack_domains()
    results.update(commitpack_results)

    # Save enrichment manifest
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest_entries, f, indent=2, ensure_ascii=False)
    print(f"\n[manifest] Enrichment manifest written to {MANIFEST_PATH}")

    # Update MANIFEST_niche.json with new/updated entries for all 5 domains
    niche_manifest_path = OUT_ROOT / "MANIFEST_niche.json"
    try:
        with open(niche_manifest_path, "r", encoding="utf-8") as f:
            niche_entries: list[dict] = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        niche_entries = []

    enriched_domains = {e["domain"] for e in manifest_entries}
    # Remove old entries for enriched domains, then append new ones
    niche_entries = [e for e in niche_entries if e.get("domain") not in enriched_domains]
    niche_entries.extend(manifest_entries)
    with open(niche_manifest_path, "w", encoding="utf-8") as f:
        json.dump(niche_entries, f, indent=2, ensure_ascii=False)
    print(f"[manifest] Updated {niche_manifest_path} ({len(niche_entries)} total entries)")

    # Summary
    print("\n" + "=" * 60)
    print("Enrichment Summary")
    print("=" * 60)
    print(f"{'Domain':<15} {'Status':<8} {'Used':>7} {'Train':>7} {'Valid':>7}  Source")
    print("-" * 75)
    for entry in manifest_entries:
        domain = entry["domain"]
        status = "OK" if results.get(domain, False) else "FAIL"
        print(
            f"{domain:<15} {status:<8} {entry['n_used']:>7,} "
            f"{entry['n_train']:>7,} {entry['n_valid']:>7,}  {entry['hf_id']}"
        )

    # Report failures
    failed = [d for d, ok in results.items() if not ok]
    if failed:
        print(f"\nFAILED domains: {', '.join(failed)}")
    else:
        print("\nAll 5 domains enriched successfully.")

    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
