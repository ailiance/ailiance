#!/usr/bin/env python3
"""
build_hf_datasets.py — Download and convert HuggingFace datasets into LoRA training format.

Usage:
    cd ~/eu-kiki && uv run python scripts/build_hf_datasets.py

Output format (JSONL):
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Output layout:
    data/hf-traced/<domain>/train.jsonl
    data/hf-traced/<domain>/valid.jsonl
    data/hf-traced/MANIFEST.json
"""

import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Dependency check — install `datasets` if missing
# ---------------------------------------------------------------------------
try:
    from datasets import load_dataset, Dataset
except ImportError:
    import subprocess

    print("[setup] Installing `datasets` library …")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
    from datasets import load_dataset, Dataset  # type: ignore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
MAX_PER_DOMAIN = 3000
VALID_RATIO = 0.05
OUT_ROOT = Path(__file__).parent.parent / "data" / "hf-traced"

# ISO 639-3 codes for EU official languages (used for aya filtering)
EU_LANGUAGE_CODES = {
    "bul",  # Bulgarian
    "hrv",  # Croatian
    "ces",  # Czech
    "dan",  # Danish
    "nld",  # Dutch
    "eng",  # English
    "est",  # Estonian
    "fin",  # Finnish
    "fra",  # French
    "deu",  # German
    "ell",  # Greek
    "hun",  # Hungarian
    "gle",  # Irish
    "ita",  # Italian
    "lav",  # Latvian
    "lit",  # Lithuanian
    "mlt",  # Maltese
    "pol",  # Polish
    "por",  # Portuguese
    "ron",  # Romanian
    "slk",  # Slovak
    "slv",  # Slovenian
    "spa",  # Spanish
    "swe",  # Swedish
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_message(user: str, assistant: str) -> dict[str, Any]:
    """Build one JSONL record in the canonical messages format."""
    return {
        "messages": [
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": assistant.strip()},
        ]
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  → wrote {len(records):,} records to {path}")


def split_train_valid(records: list[dict], seed: int = SEED) -> tuple[list, list]:
    """Reproducible 95/5 split."""
    rng = random.Random(seed)
    shuffled = records.copy()
    rng.shuffle(shuffled)
    n_valid = max(1, round(len(shuffled) * VALID_RATIO))
    return shuffled[n_valid:], shuffled[:n_valid]


def save_domain(domain: str, records: list[dict]) -> tuple[int, int]:
    """Split and save a domain; return (n_train, n_valid)."""
    train, valid = split_train_valid(records)
    domain_dir = OUT_ROOT / domain
    write_jsonl(domain_dir / "train.jsonl", train)
    write_jsonl(domain_dir / "valid.jsonl", valid)
    return len(train), len(valid)


def cap(records: list[dict], n: int = MAX_PER_DOMAIN, seed: int = SEED) -> list[dict]:
    """Randomly cap to at most n examples."""
    if len(records) <= n:
        return records
    rng = random.Random(seed)
    return rng.sample(records, n)


def is_nonempty(*texts: str) -> bool:
    return all(t and t.strip() for t in texts)


# ---------------------------------------------------------------------------
# Dataset converters
# ---------------------------------------------------------------------------


def convert_self_oss_instruct(
    hf_dataset,
    lang_filter: str | list[str],
    domain: str,
) -> list[dict]:
    """
    bigcode/self-oss-instruct-sc2-exec-filter-50k
    Columns: instruction, response, lang (e.g. Python, Rust, …)
    """
    if isinstance(lang_filter, str):
        lang_filter = [lang_filter]
    lang_lower = [lf.lower() for lf in lang_filter]

    records: list[dict] = []
    for row in hf_dataset:
        lang = str(row.get("lang", "")).lower()
        instruction = str(row.get("instruction", ""))
        response = str(row.get("response", ""))
        if lang in lang_lower and is_nonempty(instruction, response):
            records.append(make_message(instruction, response))
    return records


def convert_aya(hf_dataset, lang_codes: set[str]) -> list[dict]:
    """
    CohereForAI/aya_dataset
    Columns: inputs, targets, language_code (ISO 639-3)
    """
    records: list[dict] = []
    for row in hf_dataset:
        code = str(row.get("language_code", ""))
        if code not in lang_codes:
            continue
        user = str(row.get("inputs", ""))
        assistant = str(row.get("targets", ""))
        if is_nonempty(user, assistant):
            records.append(make_message(user, assistant))
    return records


def convert_gsm8k(hf_dataset) -> list[dict]:
    """
    openai/gsm8k  (config="main")
    Columns: question, answer
    """
    records: list[dict] = []
    for row in hf_dataset:
        q = str(row.get("question", ""))
        a = str(row.get("answer", ""))
        if is_nonempty(q, a):
            records.append(make_message(q, a))
    return records


def convert_math_instruct(hf_dataset) -> list[dict]:
    """
    TIGER-Lab/MathInstruct
    Columns: instruction, output  (or query / response)
    """
    records: list[dict] = []
    for row in hf_dataset:
        user = str(row.get("instruction", "") or row.get("query", ""))
        assistant = str(row.get("output", "") or row.get("response", ""))
        if is_nonempty(user, assistant):
            records.append(make_message(user, assistant))
    return records


def convert_multilingual_thinking(hf_dataset, lang_codes: set[str]) -> list[dict]:
    """
    HuggingFaceH4/Multilingual-Thinking
    Columns vary; try prompt/completion, instruction/response, messages list.
    """
    records: list[dict] = []
    for row in hf_dataset:
        lang = str(row.get("language", row.get("lang", ""))).lower()[:3]
        if lang_codes and lang not in lang_codes:
            continue
        # Try messages list first
        messages = row.get("messages", [])
        if messages and isinstance(messages, list):
            user = next(
                (m.get("content", "") for m in messages if m.get("role") == "user"), ""
            )
            assistant = next(
                (
                    m.get("content", "")
                    for m in messages
                    if m.get("role") == "assistant"
                ),
                "",
            )
            if is_nonempty(user, assistant):
                records.append(make_message(user, assistant))
            continue
        # Fallback to flat columns
        user = str(
            row.get("prompt", "")
            or row.get("instruction", "")
            or row.get("question", "")
        )
        assistant = str(
            row.get("completion", "")
            or row.get("response", "")
            or row.get("answer", "")
        )
        if is_nonempty(user, assistant):
            records.append(make_message(user, assistant))
    return records


# ---------------------------------------------------------------------------
# Domain build plans
# ---------------------------------------------------------------------------

MANIFEST: dict[str, Any] = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "seed": SEED,
    "max_per_domain": MAX_PER_DOMAIN,
    "valid_ratio": VALID_RATIO,
    "domains": {},
}


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
    MANIFEST["domains"][domain] = {
        "hf_dataset_id": hf_id,
        "license": license_,
        "download_date": datetime.now(timezone.utc).date().isoformat(),
        "n_source_rows": n_source,
        "n_used": n_used,
        "n_train": n_train,
        "n_valid": n_valid,
        "notes": notes,
    }


def build_code_domains(split: str = "train") -> None:
    """
    bigcode/self-oss-instruct-sc2-exec-filter-50k  — Apache 2.0
    50 K code instruction pairs across many languages.
    """
    HF_ID = "bigcode/self-oss-instruct-sc2-exec-filter-50k"
    LICENSE = "Apache-2.0"

    print(f"\n[code] Loading {HF_ID} …")
    ds = load_dataset(HF_ID, split=split)
    total = len(ds)
    print(f"  total rows: {total:,}")

    code_domains: dict[str, list[str]] = {
        "python": ["python"],
        "rust": ["rust"],
        "typescript": ["typescript", "javascript"],
        "cpp": ["c++", "c", "cpp"],
        "shell": ["shell", "bash", "sh"],
        "sql": ["sql"],
        "html-css": ["html", "css"],
    }

    for domain, langs in code_domains.items():
        print(f"\n  [{domain}] filtering for langs={langs} …")
        raw = convert_self_oss_instruct(ds, langs, domain)
        print(f"    matched: {len(raw):,}")
        capped = cap(raw)
        n_train, n_valid = save_domain(domain, capped)
        record_manifest(
            domain=domain,
            hf_id=HF_ID,
            license_=LICENSE,
            n_source=total,
            n_used=len(capped),
            n_train=n_train,
            n_valid=n_valid,
            notes=f"filtered lang in {langs}",
        )


def build_chat_fr() -> None:
    """
    CohereForAI/aya_dataset  — Apache 2.0
    Multilingual instruction/response pairs; filter language_code == 'fra'.
    """
    HF_ID = "CohereForAI/aya_dataset"
    LICENSE = "Apache-2.0"
    LANG = {"fra"}

    print(f"\n[chat-fr] Loading {HF_ID} …")
    ds = load_dataset(HF_ID, split="train")
    total = len(ds)
    print(f"  total rows: {total:,}")

    raw = convert_aya(ds, LANG)
    print(f"  French rows: {len(raw):,}")
    capped = cap(raw)
    n_train, n_valid = save_domain("chat-fr", capped)
    record_manifest(
        domain="chat-fr",
        hf_id=HF_ID,
        license_=LICENSE,
        n_source=total,
        n_used=len(capped),
        n_train=n_train,
        n_valid=n_valid,
        notes="language_code == 'fra'",
    )


def build_multilingual_eu() -> None:
    """
    Try HuggingFaceH4/Multilingual-Thinking first (Apache 2.0).
    Fall back to CohereForAI/aya_dataset filtered for all EU languages.
    """
    PRIMARY_HF_ID = "HuggingFaceH4/Multilingual-Thinking"
    FALLBACK_HF_ID = "CohereForAI/aya_dataset"
    LICENSE = "Apache-2.0"

    # Map ISO 639-3 codes to short 2-char for Multilingual-Thinking
    eu_3char = EU_LANGUAGE_CODES
    eu_2char = {c[:2] for c in eu_3char}

    print(f"\n[multilingual-eu] Trying {PRIMARY_HF_ID} …")
    try:
        ds = load_dataset(PRIMARY_HF_ID, split="train")
        total = len(ds)
        print(f"  rows: {total:,}")
        raw = convert_multilingual_thinking(ds, eu_2char | eu_3char)
        if not raw:
            raise ValueError("No matching rows found in primary dataset")
        print(f"  EU-language rows: {len(raw):,}")
        capped = cap(raw)
        n_train, n_valid = save_domain("multilingual-eu", capped)
        record_manifest(
            domain="multilingual-eu",
            hf_id=PRIMARY_HF_ID,
            license_=LICENSE,
            n_source=total,
            n_used=len(capped),
            n_train=n_train,
            n_valid=n_valid,
            notes=f"filtered for EU language codes",
        )
    except Exception as exc:
        print(f"  Primary failed ({exc}), falling back to {FALLBACK_HF_ID} …")
        ds = load_dataset(FALLBACK_HF_ID, split="train")
        total = len(ds)
        # Exclude French (already in chat-fr)
        eu_without_fr = eu_3char - {"fra"}
        raw = convert_aya(ds, eu_without_fr)
        print(f"  EU rows (excl. French): {len(raw):,}")
        capped = cap(raw)
        n_train, n_valid = save_domain("multilingual-eu", capped)
        record_manifest(
            domain="multilingual-eu",
            hf_id=FALLBACK_HF_ID,
            license_=LICENSE,
            n_source=total,
            n_used=len(capped),
            n_train=n_train,
            n_valid=n_valid,
            notes=f"fallback from {PRIMARY_HF_ID}; EU languages excl. fra",
        )


def build_math_gsm8k() -> None:
    """
    openai/gsm8k  — MIT
    Grade-school math word problems with step-by-step solutions.
    """
    HF_ID = "openai/gsm8k"
    LICENSE = "MIT"

    print(f"\n[math-gsm8k] Loading {HF_ID} …")
    ds = load_dataset(HF_ID, "main", split="train")
    total = len(ds)
    print(f"  rows: {total:,}")

    raw = convert_gsm8k(ds)
    capped = cap(raw)
    n_train, n_valid = save_domain("math-gsm8k", capped)
    record_manifest(
        domain="math-gsm8k",
        hf_id=HF_ID,
        license_=LICENSE,
        n_source=total,
        n_used=len(capped),
        n_train=n_train,
        n_valid=n_valid,
    )


def build_math_reasoning() -> None:
    """
    TIGER-Lab/MathInstruct  — Apache-2.0 (most subsets)
    Mixed math instruction/response pairs; diverse reasoning formats.
    """
    HF_ID = "TIGER-Lab/MathInstruct"
    LICENSE = "Apache-2.0 (check individual source attributions)"

    print(f"\n[math-reasoning] Loading {HF_ID} …")
    ds = load_dataset(HF_ID, split="train")
    total = len(ds)
    print(f"  rows: {total:,}")

    raw = convert_math_instruct(ds)
    capped = cap(raw)
    n_train, n_valid = save_domain("math-reasoning", capped)
    record_manifest(
        domain="math-reasoning",
        hf_id=HF_ID,
        license_=LICENSE,
        n_source=total,
        n_used=len(capped),
        n_train=n_train,
        n_valid=n_valid,
        notes="diverse math instruction sets; verify per-source licenses before redistribution",
    )


def build_rust_domain() -> None:
    """
    Fortytwo-Network Strand-Rust dataset  — Open
    191K peer-reviewed synthetic Rust training pairs.
    Falls back to codeparrot/github-code filtered for Rust when unavailable.
    """
    PRIMARY_HF_ID = "Fortytwo-Network/Strand-Rust-Coder"
    FALLBACK_HF_ID = "codeparrot/github-code"
    PRIMARY_LICENSE = "Open"
    FALLBACK_LICENSE = "Apache-2.0"

    print(f"\n[rust-strand] Trying {PRIMARY_HF_ID} …")
    try:
        ds = load_dataset(PRIMARY_HF_ID, split="train")
        total = len(ds)
        print(f"  rows: {total:,}")

        records: list[dict] = []
        for row in ds:
            user = str(
                row.get("instruction", "")
                or row.get("prompt", "")
                or row.get("question", "")
            )
            assistant = str(
                row.get("response", "")
                or row.get("output", "")
                or row.get("completion", "")
            )
            if is_nonempty(user, assistant):
                records.append(make_message(user, assistant))

        if not records:
            raise ValueError("No usable rows found in primary Strand-Rust dataset")

        print(f"  usable rows: {len(records):,}")
        capped = cap(records)
        n_train, n_valid = save_domain("rust-strand", capped)
        record_manifest(
            domain="rust-strand",
            hf_id=PRIMARY_HF_ID,
            license_=PRIMARY_LICENSE,
            n_source=total,
            n_used=len(capped),
            n_train=n_train,
            n_valid=n_valid,
            notes=(
                "Fortytwo-Network Strand-Rust-Coder; peer-reviewed synthetic Rust corpus"
            ),
        )
    except Exception as exc:
        print(f"  Primary failed ({exc}), falling back to {FALLBACK_HF_ID} …")
        ds = load_dataset(
            FALLBACK_HF_ID,
            split="train",
            streaming=True,
        )
        records = []
        for row in ds:
            if str(row.get("language", "")).lower() not in ("rust",):
                continue
            code = str(row.get("code", ""))
            repo = str(row.get("repo_name", "unknown"))
            if not code.strip():
                continue
            user = f"Show me a Rust code example from the repository `{repo}`."
            assistant = code
            records.append(make_message(user, assistant))
            if len(records) >= MAX_PER_DOMAIN * 2:
                break

        total = len(records)
        print(f"  Rust rows collected from fallback: {total:,}")
        capped = cap(records)
        n_train, n_valid = save_domain("rust-strand", capped)
        record_manifest(
            domain="rust-strand",
            hf_id=FALLBACK_HF_ID,
            license_=FALLBACK_LICENSE,
            n_source=total,
            n_used=len(capped),
            n_train=n_train,
            n_valid=n_valid,
            notes=(
                f"Fallback from {PRIMARY_HF_ID}; "
                "codeparrot/github-code filtered for language==Rust"
            ),
        )


def build_security_domain() -> None:
    """
    AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.0  — Apache 2.0
    83.9K OWASP / MITRE ATT&CK / NIST-aligned synthetic security Q&A pairs.
    """
    HF_ID = "AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.0"
    LICENSE = "Apache-2.0"

    print(f"\n[security-fenrir] Loading {HF_ID} …")
    ds = load_dataset(HF_ID, split="train")
    total = len(ds)
    print(f"  rows: {total:,}")

    records: list[dict] = []
    for row in ds:
        # Dataset columns vary; try common patterns
        user = str(
            row.get("instruction", "")
            or row.get("prompt", "")
            or row.get("question", "")
            or row.get("input", "")
        )
        assistant = str(
            row.get("output", "")
            or row.get("response", "")
            or row.get("answer", "")
            or row.get("completion", "")
        )
        if is_nonempty(user, assistant):
            records.append(make_message(user, assistant))

    print(f"  usable rows: {len(records):,}")
    capped = cap(records)
    n_train, n_valid = save_domain("security-fenrir", capped)
    record_manifest(
        domain="security-fenrir",
        hf_id=HF_ID,
        license_=LICENSE,
        n_source=total,
        n_used=len(capped),
        n_train=n_train,
        n_valid=n_valid,
        notes="OWASP / MITRE ATT&CK / NIST-aligned synthetic cybersecurity Q&A",
    )


def build_misra_domain() -> None:
    """
    wuog/CertiCoder  — Research license
    37.4K MISRA C:2012-aware certified embedded coding Q&A pairs.
    """
    HF_ID = "wuog/CertiCoder"
    LICENSE = "Research"

    print(f"\n[misra-certicoder] Loading {HF_ID} …")
    ds = load_dataset(HF_ID, split="train")
    total = len(ds)
    print(f"  rows: {total:,}")

    records: list[dict] = []
    for row in ds:
        user = str(
            row.get("instruction", "")
            or row.get("prompt", "")
            or row.get("question", "")
            or row.get("input", "")
        )
        assistant = str(
            row.get("output", "")
            or row.get("response", "")
            or row.get("answer", "")
            or row.get("completion", "")
        )
        if is_nonempty(user, assistant):
            records.append(make_message(user, assistant))

    print(f"  usable rows: {len(records):,}")
    capped = cap(records)
    n_train, n_valid = save_domain("misra-certicoder", capped)
    record_manifest(
        domain="misra-certicoder",
        hf_id=HF_ID,
        license_=LICENSE,
        n_source=total,
        n_used=len(capped),
        n_train=n_train,
        n_valid=n_valid,
        notes=(
            "MISRA C:2012-aware certified coding Q&A; research license — "
            "do not redistribute without author permission"
        ),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("eu-kiki  —  HuggingFace dataset builder")
    print(f"Output root : {OUT_ROOT}")
    print(f"Max/domain  : {MAX_PER_DOMAIN}")
    print(f"Split ratio : {int((1 - VALID_RATIO)*100)}/{int(VALID_RATIO*100)} (train/valid)")
    print(f"Seed        : {SEED}")
    print("=" * 60)

    # Code domains (all from same HF dataset — load once)
    build_code_domains(split="train")

    # French conversational
    build_chat_fr()

    # Multilingual EU
    build_multilingual_eu()

    # Math — GSM8K
    build_math_gsm8k()

    # Math — MathInstruct
    build_math_reasoning()

    # Rust — Strand-Rust-Coder (with codeparrot fallback)
    build_rust_domain()

    # Security — Cybersecurity Fenrir v2 (OWASP/MITRE/NIST)
    build_security_domain()

    # Safety-critical — CertiCoder (MISRA C:2012)
    build_misra_domain()

    # Write provenance manifest
    manifest_path = OUT_ROOT / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(MANIFEST, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[manifest] Written to {manifest_path}")

    # Summary table
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Domain':<20} {'Used':>7} {'Train':>7} {'Valid':>7}  HF Dataset")
    print("-" * 60)
    for domain, info in MANIFEST["domains"].items():
        print(
            f"{domain:<20} {info['n_used']:>7,} {info['n_train']:>7,} {info['n_valid']:>7,}  {info['hf_dataset_id']}"
        )
    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
