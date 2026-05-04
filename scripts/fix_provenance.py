"""Add _provenance to all HF-sourced domains that lack it.

EU AI Act compliance: every training record must carry per-record provenance
(source dataset, SPDX license, record index, access date).

Usage:
    uv run python scripts/fix_provenance.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "hf-traced"

# Domains that already have _provenance — skip them
SKIP_DOMAINS = frozenset({
    "cpp",
    "freecad",
    "html-css",
    "kicad-dsl",
    "kicad-pcb",
    "ml-training",
    "rust-embedded",
    "shell",
    "sql",
})


def load_manifest_main() -> dict[str, dict]:
    """Load MANIFEST.json and index by domain."""
    path = DATA_ROOT / "MANIFEST.json"
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    result: dict[str, dict] = {}
    for domain, info in data.get("domains", {}).items():
        result[domain] = {
            "source": info.get("hf_dataset_id", "unknown"),
            "license": info.get("license", "unknown"),
            "access_date": info.get("download_date", "2026-04-28"),
        }
    return result


def load_manifest_niche() -> dict[str, dict]:
    """Load MANIFEST_niche.json and index by domain."""
    path = DATA_ROOT / "MANIFEST_niche.json"
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    result: dict[str, dict] = {}
    for entry in data:
        domain = entry.get("domain", "")
        # For multi-source domains, use hf_id or first source
        source = entry.get("hf_id", "unknown")
        if "sources" in entry and not source:
            source = entry["sources"][0].get("id", "unknown")
        license_val = entry.get("license", "unknown")
        access_date = entry.get("access_date", "2026-04-28")
        # Normalize access_date to date-only
        if "T" in str(access_date):
            access_date = str(access_date).split("T")[0]
        result[domain] = {
            "source": source,
            "license": license_val,
            "access_date": access_date,
        }
    return result


def load_manifest_enriched() -> dict[str, dict]:
    """Load MANIFEST_enriched.json and index by domain."""
    path = DATA_ROOT / "MANIFEST_enriched.json"
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    result: dict[str, dict] = {}
    for entry in data:
        domain = entry.get("domain", "")
        source = entry.get("hf_id", "unknown")
        license_val = entry.get("license", "unknown")
        access_date = entry.get("access_date", "2026-04-28")
        if "T" in str(access_date):
            access_date = str(access_date).split("T")[0]
        result[domain] = {
            "source": source,
            "license": license_val,
            "access_date": access_date,
        }
    return result


def resolve_provenance(domain: str, manifests: list[dict[str, dict]]) -> dict:
    """Find best source/license for a domain across all manifests.

    Niche manifest takes priority over enriched, which takes priority over main.
    """
    # Reverse order: last wins (niche is first in list = highest priority)
    info: dict = {"source": "unknown", "license": "unknown", "access_date": "2026-04-28"}
    for manifest in reversed(manifests):
        if domain in manifest:
            info = manifest[domain]
    return info


def add_provenance_to_file(filepath: Path, source: str, license_id: str, access_date: str) -> int:
    """Add _provenance to each record in a JSONL file. Returns record count."""
    records: list[dict] = []
    with open(filepath, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    updated = 0
    for idx, record in enumerate(records):
        if "_provenance" not in record:
            record["_provenance"] = {
                "source": source,
                "license": license_id,
                "record_idx": idx,
                "access_date": access_date,
            }
            updated += 1

    with open(filepath, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    return updated


def main() -> None:
    print("=" * 70)
    print("EU-KIKI Provenance Fixer")
    print("=" * 70)

    # Load all manifests
    manifest_main = load_manifest_main()
    manifest_niche = load_manifest_niche()
    manifest_enriched = load_manifest_enriched()
    manifests = [manifest_niche, manifest_enriched, manifest_main]

    # Discover domains
    domains = sorted(
        d.name
        for d in DATA_ROOT.iterdir()
        if d.is_dir() and (d / "train.jsonl").exists()
    )

    total_updated = 0
    total_skipped = 0

    for domain in domains:
        if domain in SKIP_DOMAINS:
            print(f"  SKIP {domain} (already has _provenance)")
            total_skipped += 1
            continue

        info = resolve_provenance(domain, manifests)
        source = info["source"]
        license_id = info["license"]
        access_date = info["access_date"]

        print(f"\n  {domain}:")
        print(f"    source={source}, license={license_id}, date={access_date}")

        domain_updated = 0
        for filename in ["train.jsonl", "valid.jsonl"]:
            filepath = DATA_ROOT / domain / filename
            if filepath.exists():
                count = add_provenance_to_file(filepath, source, license_id, access_date)
                domain_updated += count
                print(f"    {filename}: {count} records updated")
            else:
                print(f"    {filename}: not found")

        total_updated += domain_updated

    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {total_updated} records updated, {total_skipped} domains skipped")
    print(f"{'=' * 70}")

    if total_updated == 0:
        print("Nothing to do — all records already have _provenance.")
    else:
        print(f"Done. {total_updated} records now have per-record provenance.")


if __name__ == "__main__":
    main()
