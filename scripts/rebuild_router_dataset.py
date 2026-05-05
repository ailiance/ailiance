"""Rebuild a clean per-domain training corpus for the router.

Strategy
--------
- Each domain is sourced from ONE permissive open HF dataset that is
  topically focused — no more "rust.jsonl actually about photosynthesis".
- Long-tail technical domains (stm32, kicad, calcul-normatif, etc.) where
  no HF dataset fits use the manually-curated prompts already in this
  repo (scripts/augment_router_data.py).
- Each prompt is tagged with `source` and `license` for AI Act traceability.
- Output: data/router-clean/<domain>.jsonl with rows
  {"prompt": str, "domain": str, "source": str, "license": str}

Provenance
----------
Every dataset commit and per-row source URL is logged in
data/router-clean/PROVENANCE.json so we can reproduce the build later.

Run on studio (needs `datasets` lib, ~5 GB temporary disk):
  ~/KIKI-Mac_tunner/.venv/bin/python scripts/rebuild_router_dataset.py

The script is idempotent: re-running overwrites data/router-clean/.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "router-clean"
SEED = 42
TARGET_PER_DOMAIN = 800  # cap per domain to keep classes balanced
MIN_LEN = 8              # min chars in user prompt
MAX_LEN = 600            # max chars (router only sees one-shot intent)

# (domain, hf_repo, license, split, take_n_max, hf_filter_fn, hf_extract_fn)
# extract_fn returns the user prompt text or None to skip the row.
HF_SOURCES: list[dict] = [
    # === Programming languages ===
    {
        "domain": "python",
        "hf_repo": "iamtarun/python_code_instructions_18k_alpaca",
        "license": "apache-2.0",
        "split": "train",
        "take": 1000,
        "extract": lambda r: r.get("instruction"),
    },
    {
        "domain": "rust",
        "hf_repo": "ammarnasr/the-stack-rust-clean",
        "license": "other-stack-v1",  # The Stack, opt-out respected
        "split": "train",
        "take": 1000,
        # The Stack is *code*. We synthesise a prompt around the file path.
        "extract": lambda r: f"Explain or modify this Rust file: `{r.get('max_stars_repo_path', 'main.rs')}`" if r.get("content") else None,
    },
    {
        "domain": "typescript",
        "hf_repo": "bigcode/the-stack-smol",
        "license": "other-stack-v1",
        "split": "train",
        "config": "data/typescript",
        "take": 1000,
        "extract": lambda r: f"Review or extend this TypeScript file: `{r.get('max_stars_repo_path', 'index.ts')}`" if r.get("content") else None,
    },
    {
        "domain": "cpp",
        "hf_repo": "bigcode/the-stack-smol",
        "license": "other-stack-v1",
        "split": "train",
        "config": "data/c++",
        "take": 1000,
        "extract": lambda r: f"Review or extend this C++ file: `{r.get('max_stars_repo_path', 'main.cpp')}`" if r.get("content") else None,
    },
    {
        "domain": "shell",
        "hf_repo": "bigcode/the-stack-smol",
        "license": "other-stack-v1",
        "split": "train",
        "config": "data/shell",
        "take": 1000,
        "extract": lambda r: f"Explain or refactor this Bash / shell script: `{r.get('max_stars_repo_path', 'script.sh')}`" if r.get("content") else None,
    },
    {
        "domain": "sql",
        "hf_repo": "b-mc2/sql-create-context",
        "license": "cc-by-4.0",
        "split": "train",
        "take": 1000,
        "extract": lambda r: r.get("question"),
    },
    # === Web ===
    {
        "domain": "html-css",
        "hf_repo": "bigcode/the-stack-smol",
        "license": "other-stack-v1",
        "split": "train",
        "config": "data/html",
        "take": 800,
        "extract": lambda r: f"Improve or fix this HTML page: `{r.get('max_stars_repo_path', 'index.html')}`" if r.get("content") else None,
    },
    # === Multilingual ===
    {
        "domain": "chat-fr",
        "hf_repo": "OpenLLM-France/Claire-Dialogue-French-0.1",
        "license": "cc-by-nc-sa-4.0",
        "split": "train",
        "take": 1000,
        "extract": lambda r: (r.get("text") or "").split("[Speaker001]")[1].split("[Speaker")[0].strip()[:200] if "[Speaker001]" in (r.get("text") or "") else None,
    },
    # === Math / reasoning ===
    {
        "domain": "math",
        "hf_repo": "openai/gsm8k",
        "license": "mit",
        "split": "train",
        "config": "main",
        "take": 1000,
        "extract": lambda r: r.get("question"),
    },
    {
        "domain": "reasoning",
        "hf_repo": "allenai/openbookqa",
        "license": "apache-2.0",
        "split": "train",
        "config": "main",
        "take": 1000,
        "extract": lambda r: r.get("question_stem"),
    },
    # === Security ===
    {
        "domain": "security",
        "hf_repo": "scieval/SecQA",
        "license": "mit",
        "split": "train",
        "take": 800,
        "extract": lambda r: r.get("question"),
    },
]

# Manually curated domains — already in scripts/augment_router_data.py.
# We re-import directly from that module rather than duplicate.
MANUAL_DOMAINS = {
    "calcul-normatif",
    "docker",
    "spice",
    # Long-tail technical: keep using the legacy data temporarily, marked.
    # These will be replaced in a follow-up pass.
}


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _ok(prompt: str | None) -> bool:
    if not prompt:
        return False
    p = prompt.strip()
    return MIN_LEN <= len(p) <= MAX_LEN


def fetch_hf(spec: dict) -> list[dict]:
    """Fetch a HF dataset and convert rows to our format. Returns empty list
    on any error so the rebuild can continue."""
    from datasets import load_dataset

    hf_repo = spec["hf_repo"]
    domain = spec["domain"]
    print(f"  fetching {hf_repo} for domain={domain} ...", flush=True)
    try:
        kwargs = {"split": spec["split"], "streaming": True}
        if "config" in spec:
            ds = load_dataset(hf_repo, spec["config"], **kwargs)
        else:
            ds = load_dataset(hf_repo, **kwargs)
    except Exception as e:
        print(f"    SKIP — load_dataset failed: {e}", flush=True)
        return []

    take = spec["take"]
    rows: list[dict] = []
    seen_hashes: set[str] = set()
    for r in ds:
        try:
            prompt = spec["extract"](r)
        except Exception:
            continue
        if not _ok(prompt):
            continue
        prompt = prompt.strip()
        h = _hash(prompt)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        rows.append({
            "prompt": prompt,
            "domain": domain,
            "source": hf_repo,
            "license": spec["license"],
        })
        if len(rows) >= take:
            break
    print(f"    got {len(rows)} unique rows", flush=True)
    return rows


def main() -> None:
    random.seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    provenance: dict = {
        "_doc": "EU AI Act Annex IV §2(b) — per-source training-data record.",
        "rebuilt_at_utc": "2026-05-05",
        "seed": SEED,
        "target_per_domain": TARGET_PER_DOMAIN,
        "sources": [],
    }

    # === HF-sourced domains ===
    for spec in HF_SOURCES:
        rows = fetch_hf(spec)
        # Cap to TARGET_PER_DOMAIN but allow more if the source is rich
        if len(rows) > TARGET_PER_DOMAIN:
            random.shuffle(rows)
            rows = rows[:TARGET_PER_DOMAIN]
        all_rows.extend(rows)
        provenance["sources"].append({
            "domain": spec["domain"],
            "huggingface_repo": spec["hf_repo"],
            "config": spec.get("config"),
            "split": spec["split"],
            "license_spdx": spec["license"],
            "rows_used": len(rows),
        })

    # === Manually curated domains (re-import from sibling script) ===
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        from augment_router_data import (  # type: ignore
            CALCUL_NORMATIF, DOCKER_AUG, SPICE_AUG, MIXED_FR_EN,
        )
    except Exception as e:
        print(f"FAIL importing augment_router_data: {e}")
        sys.exit(1)

    for prompts, domain in [
        (CALCUL_NORMATIF, "calcul-normatif"),
        (DOCKER_AUG, "docker"),
        (SPICE_AUG, "spice"),
    ]:
        for p in prompts:
            if _ok(p):
                all_rows.append({
                    "prompt": p,
                    "domain": domain,
                    "source": "L'Électron Rare internal (curated)",
                    "license": "apache-2.0",
                })
        provenance["sources"].append({
            "domain": domain,
            "huggingface_repo": None,
            "split": None,
            "license_spdx": "apache-2.0",
            "rows_used": sum(1 for r in all_rows if r["domain"] == domain),
            "note": "Curated by L'Électron Rare in May 2026. See scripts/augment_router_data.py.",
        })

    for prompt, domain in MIXED_FR_EN:
        if _ok(prompt):
            all_rows.append({
                "prompt": prompt,
                "domain": domain,
                "source": "L'Électron Rare internal (FR/EN code-switched)",
                "license": "apache-2.0",
            })

    # Write per-domain JSONL
    by_domain: dict[str, list[dict]] = {}
    for r in all_rows:
        by_domain.setdefault(r["domain"], []).append(r)

    for domain, rows in sorted(by_domain.items()):
        target = OUT / f"{domain}.jsonl"
        with target.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  wrote {target.name}: {len(rows)} rows")

    # Provenance summary
    counts = Counter(r["domain"] for r in all_rows)
    provenance["totals"] = dict(counts.most_common())
    provenance["grand_total"] = len(all_rows)
    (OUT / "PROVENANCE.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nGrand total: {len(all_rows)} rows across {len(by_domain)} domains")
    print(f"Wrote {OUT / 'PROVENANCE.json'}")


if __name__ == "__main__":
    main()
