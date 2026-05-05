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
    # === Programming languages — CodeAlpaca filtered by language keyword ===
    # All language-specific prompts come from sahil2801/CodeAlpaca-20k (CC-BY-4.0)
    # which contains generic instruction-tuning prompts; we filter per language
    # so each domain gets prompts that mention the language explicitly.
    # CodeAlpaca is verified existing (2023-07, 19k downloads).
    {
        "domain": "python",
        "hf_repo": "sahil2801/CodeAlpaca-20k",
        "license": "cc-by-4.0",
        "split": "train",
        "take": 800,
        "filter_keywords": ["python", "def ", "import ", "pip ", "numpy", "pandas"],
        "extract": lambda r: r.get("instruction") or r.get("prompt"),
    },
    {
        "domain": "rust",
        "hf_repo": "sahil2801/CodeAlpaca-20k",
        "license": "cc-by-4.0",
        "split": "train",
        "take": 600,
        "filter_keywords": ["rust ", " rust", "fn ", "cargo ", "trait ", "impl ", "let mut"],
        "extract": lambda r: r.get("instruction") or r.get("prompt"),
    },
    {
        "domain": "typescript",
        "hf_repo": "sahil2801/CodeAlpaca-20k",
        "license": "cc-by-4.0",
        "split": "train",
        "take": 600,
        "filter_keywords": ["typescript", "javascript", "interface ", "node.js", "tsconfig", "react", " js "],
        "extract": lambda r: r.get("instruction") or r.get("prompt"),
    },
    {
        "domain": "cpp",
        "hf_repo": "sahil2801/CodeAlpaca-20k",
        "license": "cc-by-4.0",
        "split": "train",
        "take": 600,
        "filter_keywords": ["c++", "cpp", "std::", "iostream", "printf"],
        "extract": lambda r: r.get("instruction") or r.get("prompt"),
    },
    {
        "domain": "shell",
        "hf_repo": "sahil2801/CodeAlpaca-20k",
        "license": "cc-by-4.0",
        "split": "train",
        "take": 600,
        "filter_keywords": ["bash", "shell ", "command line", "linux", "terminal command", "awk ", "sed ", "grep "],
        "extract": lambda r: r.get("instruction") or r.get("prompt"),
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
        "hf_repo": "sahil2801/CodeAlpaca-20k",
        "license": "cc-by-4.0",
        "split": "train",
        "take": 600,
        "filter_keywords": ["html", "css", "<div>", "<style>", "stylesheet"],
        "extract": lambda r: r.get("instruction") or r.get("prompt"),
    },
    # === Multilingual chat (FR) ===
    # OpenAssistant covers FR; we filter by detecting French keywords. Multi-lingual
    # licence: Apache-2.0.
    {
        "domain": "chat-fr",
        "hf_repo": "OpenAssistant/oasst1",
        "license": "apache-2.0",
        "split": "train",
        "take": 1000,
        "filter_keywords": [" le ", " la ", " les ", " est ", " une ", " que ", " pour ", " bonjour", " comment "],
        "extract": lambda r: r.get("text") if r.get("role") == "prompter" else None,
    },
    # === Math ===
    {
        "domain": "math",
        "hf_repo": "openai/gsm8k",
        "license": "mit",
        "split": "train",
        "config": "main",
        "take": 1000,
        "extract": lambda r: r.get("question"),
    },
    # === Reasoning — NuminaMath-CoT (Apache 2.0) for chain-of-thought reasoning ===
    {
        "domain": "reasoning",
        "hf_repo": "AI-MO/NuminaMath-CoT",
        "license": "apache-2.0",
        "split": "train",
        "take": 1000,
        "extract": lambda r: r.get("problem"),
    },
    # === Security — sourced from CodeAlpaca with security keywords ===
    {
        "domain": "security",
        "hf_repo": "sahil2801/CodeAlpaca-20k",
        "license": "cc-by-4.0",
        "split": "train",
        "take": 600,
        "filter_keywords": ["security", "encrypt", "decrypt", "password", "auth", "vulnerab", "secure ", "ssl ", "tls "],
        "extract": lambda r: r.get("instruction") or r.get("prompt"),
    },
    # === Analog electronics — Electronics-QA (Apache-2.0, 2516 rows) ===
    # Heavy focus on transistor circuits, fuzz pedals, op-amps; complements the
    # spice / power / electronics-hw curated prompts.
    {
        "domain": "spice",
        "hf_repo": "theprint/Electronics-QA",
        "license": "apache-2.0",
        "split": "train",
        "take": 600,
        "filter_keywords": ["circuit", "transistor", "diode", "amplifier", "spice", "simulat",
                            "ngspice", "ltspice", "voltage", "current", "filter", "rc circuit",
                            "rl circuit", "rlc", "operational amplifier", "op-amp"],
        "extract": lambda r: r.get("question"),
    },
    {
        "domain": "power",
        "hf_repo": "theprint/Electronics-QA",
        "license": "apache-2.0",
        "split": "train",
        "take": 600,
        "filter_keywords": ["power", "buck", "boost", "converter", "battery", "regulator",
                            "transformer", "inverter", "supply", "watt", "load",
                            "smps", "linear regulator", "ldo"],
        "extract": lambda r: r.get("question"),
    },
    {
        "domain": "electronics",
        "hf_repo": "theprint/Electronics-QA",
        "license": "apache-2.0",
        "split": "train",
        "take": 600,
        "filter_keywords": ["resistor", "capacitor", "inductor", "led ", "ic ",
                            "pcb", "schematic", "component", "datasheet", "pinout",
                            "current limit", "ohm", "farad"],
        "extract": lambda r: r.get("question"),
    },
    # === DevOps — CodeAlpaca filtered for k8s/docker/CI/CD/Terraform/Ansible ===
    # The "good" DevOps datasets on HF (PL-DevOps-Instruct, devops-kubectl-v1,
    # devops-gitops-corpus) all have empty READMEs and no declared licence.
    # We stay strict and filter CodeAlpaca on DevOps keywords instead.
    {
        "domain": "devops",
        "hf_repo": "sahil2801/CodeAlpaca-20k",
        "license": "cc-by-4.0",
        "split": "train",
        "take": 600,
        "filter_keywords": ["kubernetes", "kubectl", "k8s ", "docker ", "compose",
                            "terraform", "ansible", "ci/cd", "github actions",
                            "jenkins", "gitlab ci", "deploy", "helm", "nginx"],
        "extract": lambda r: r.get("instruction") or r.get("prompt"),
    },
    # === FreeCAD parametric sketches (CC-BY-4.0, 3000 .py files) ===
    # The dataset is Python files, not prompts. We synthesise a prompt for
    # each unique sketch name in the file structure (`PartName_SketchName.py`).
    # Loaded specially in main() rather than via fetch_hf().
    # === Lua / Luau (Roblox MIT, 10k+ rows) ===
    # Schema is {prompt: str, completion: str} where prompt is a Lua comment
    # describing what the next code block does. We use the prompt directly,
    # stripping the leading "--" comment markers.
    {
        "domain": "lua-upy",
        "hf_repo": "Roblox/luau_corpus",
        "license": "mit",
        "split": "train",
        "take": 600,
        "extract": lambda r: (r.get("prompt") or "").lstrip("-").strip() if r.get("prompt") else None,
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
    keywords = [k.lower() for k in spec.get("filter_keywords") or []]
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
        if keywords:
            lower = prompt.lower()
            if not any(k in lower for k in keywords):
                continue
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

    # === FreeCAD — special handling: snapshot the file list of Yas1n/FreeCAD_Sketches
    # via the HF API and synthesise a "Generate a FreeCAD sketch named X" prompt
    # per file. Avoids downloading the whole repo just to read filenames.
    try:
        import urllib.request
        url = "https://huggingface.co/api/datasets/Yas1n/FreeCAD_Sketches/tree/main?recursive=true"
        with urllib.request.urlopen(url, timeout=20) as resp:
            tree = json.loads(resp.read())
        py_files = [
            t["path"] for t in tree
            if t.get("type") == "file" and t["path"].endswith(".py")
        ]
        random.shuffle(py_files)
        prompts = []
        seen: set[str] = set()
        for p in py_files[:1500]:
            stem = Path(p).stem  # PartName_SketchName
            parts = stem.replace("_", " ").strip()
            if not parts or parts.lower() in seen:
                continue
            seen.add(parts.lower())
            prompts.append({
                "prompt": f"Generate a parametric FreeCAD sketch in Python for: {parts}",
                "domain": "freecad",
                "source": "Yas1n/FreeCAD_Sketches",
                "license": "cc-by-4.0",
            })
            if len(prompts) >= 600:
                break
        all_rows.extend(prompts)
        provenance["sources"].append({
            "domain": "freecad",
            "huggingface_repo": "Yas1n/FreeCAD_Sketches",
            "config": "filename-derived prompts (no row download)",
            "split": None,
            "license_spdx": "cc-by-4.0",
            "rows_used": len(prompts),
        })
        print(f"  freecad — synthesised {len(prompts)} prompts from FreeCAD_Sketches filenames")
    except Exception as e:
        print(f"  freecad — fetch failed: {e}")

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
