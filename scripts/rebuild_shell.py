#!/usr/bin/env python3
"""Rebuild shell domain with clean, domain-pure data.

Sources (priority order):
  A) NickIBrody/linux-commands-ru-en  (CC-BY-4.0, ~25K)
  B) bigcode/commitpackft lang=Shell  (MIT, backup)

Usage:
    cd ~/ailiance && uv run python scripts/rebuild_shell.py
"""
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
    from datasets import load_dataset

SEED = 42
TARGET = 3000
VALID_RATIO = 0.05
OUT = Path("data/hf-traced")
MANIFEST_PATH = OUT / "MANIFEST_niche.json"

SOURCE_A_ID = "NickIBrody/linux-commands-ru-en"
SOURCE_A_LICENSE = "CC-BY-4.0"


def make_msg(user: str, assistant: str, provenance: dict) -> dict:
    return {
        "messages": [
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": assistant.strip()},
        ],
        "_provenance": provenance,
    }


def _is_english(text: str) -> bool:
    """Return True if text is predominantly ASCII/Latin (English)."""
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return False
    ascii_alpha = sum(1 for c in alpha if c.isascii())
    return ascii_alpha / len(alpha) > 0.7


def load_source_a() -> list[dict]:
    """Load NickIBrody/linux-commands-ru-en, filter English, convert.

    Dataset format: each row has 'messages' list with system/user/assistant.
    Pairs alternate RU/EN — we keep only English user prompts.
    We drop the system message (bilingual boilerplate) and keep user→assistant.
    """
    print(f"Loading {SOURCE_A_ID} ...")
    ds = load_dataset(SOURCE_A_ID, split="train")
    print(f"  Raw records: {len(ds)}")

    records: list[dict] = []
    for idx, row in enumerate(ds):
        msgs = row.get("messages", [])

        # Extract user and assistant content
        user_content = ""
        assistant_content = ""
        for m in msgs:
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "user":
                user_content = content
            elif role == "assistant":
                assistant_content = content

        if not user_content.strip() or not assistant_content.strip():
            continue

        # Keep only English user prompts
        if not _is_english(user_content):
            continue

        provenance = {
            "source": SOURCE_A_ID,
            "license": SOURCE_A_LICENSE,
            "record_id": str(idx),
        }
        records.append(make_msg(user_content, assistant_content, provenance))

    print(f"  After English filter: {len(records)}")
    return records


def save_split(records: list[dict], domain: str = "shell") -> tuple[int, int]:
    """Shuffle, split, and write train/valid JSONL."""
    rng = random.Random(SEED)
    rng.shuffle(records)

    n_val = max(1, round(len(records) * VALID_RATIO))
    train, valid = records[n_val:], records[:n_val]

    d = OUT / domain
    d.mkdir(parents=True, exist_ok=True)

    for name, data in [("train.jsonl", train), ("valid.jsonl", valid)]:
        with open(d / name, "w") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"  → {domain}: {len(train)} train / {len(valid)} valid")
    return len(train), len(valid)


def update_manifest(
    hf_id: str,
    license_: str,
    n_source: int,
    n_used: int,
    n_train: int,
    n_valid: int,
    notes: str,
) -> None:
    """Update the shell entry in MANIFEST_niche.json."""
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
    else:
        manifest = []

    # Remove existing shell entry
    manifest = [e for e in manifest if e.get("domain") != "shell"]

    manifest.append({
        "domain": "shell",
        "hf_id": hf_id,
        "license": license_,
        "n_source": n_source,
        "n_used": n_used,
        "n_train": n_train,
        "n_valid": n_valid,
        "access_date": datetime.now(timezone.utc).isoformat(),
        "notes": notes,
    })

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"  → MANIFEST_niche.json updated (shell entry)")


def print_samples(records: list[dict], n: int = 3) -> None:
    """Print sample records for quality validation."""
    print(f"\n{'='*60}")
    print(f"Sample records ({n}):")
    print(f"{'='*60}")
    for i, rec in enumerate(records[:n]):
        user = rec["messages"][0]["content"]
        assistant = rec["messages"][1]["content"]
        print(f"\n--- Sample {i+1} ---")
        print(f"USER: {user[:200]}")
        print(f"ASSISTANT: {assistant[:300]}")
        print(f"PROVENANCE: {rec['_provenance']}")


def main() -> None:
    # --- Source A ---
    records = load_source_a()

    if len(records) < TARGET:
        print(f"  WARNING: Only {len(records)} records, target is {TARGET}")
        print(f"  Using all available records")

    # Cap to TARGET
    if len(records) > TARGET:
        records = random.Random(SEED).sample(records, TARGET)
        print(f"  Capped to {TARGET}")

    n_source = len(records)

    # Save
    n_train, n_valid = save_split(records)

    # Update manifest
    update_manifest(
        hf_id=SOURCE_A_ID,
        license_=SOURCE_A_LICENSE,
        n_source=n_source,
        n_used=len(records),
        n_train=n_train,
        n_valid=n_valid,
        notes="Linux commands EN instruction→command; domain-pure shell/bash",
    )

    # Validate samples
    print_samples(records)

    print(f"\nDone. Source: {SOURCE_A_ID}, "
          f"total available: {n_source}, "
          f"train: {n_train}, valid: {n_valid}")


if __name__ == "__main__":
    main()
