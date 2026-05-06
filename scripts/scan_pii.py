"""PII scanner for AILIANCE training data using Microsoft Presidio.

Scans all train.jsonl files in scraped/ and hf-traced/ directories,
extracts text from message content fields, and reports PII entities found.
"""

from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
REPORT_PATH = DATA_ROOT / "pii-scan-report.json"

MAX_RECORDS_PER_FILE = 500

PII_ENTITY_TYPES = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "PERSON",
    "CREDIT_CARD",
    "IP_ADDRESS",
    "IBAN_CODE",
    "US_SSN",
    "US_PASSPORT",
    "UK_NHS",
    "CRYPTO",
    "MEDICAL_LICENSE",
    "URL",
    "DATE_TIME",
    "NRP",           # nationality / religious / political group
    "LOCATION",
]

# Minimum score threshold to count as a real detection
SCORE_THRESHOLD = 0.6

# Max example snippets to keep per entity type per file
MAX_EXAMPLES = 3

# Context chars around a detection for the snippet
SNIPPET_CONTEXT = 40


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_text(record: dict[str, Any]) -> str:
    """Concatenate all message content fields into a single text block."""
    messages = record.get("messages", [])
    parts: list[str] = []
    for msg in messages:
        content = msg.get("content", "")
        if content:
            parts.append(content)
    return "\n".join(parts)


def snippet_around(text: str, start: int, end: int, context: int = SNIPPET_CONTEXT) -> str:
    """Return a short snippet around the detected span."""
    s = max(0, start - context)
    e = min(len(text), end + context)
    prefix = "..." if s > 0 else ""
    suffix = "..." if e < len(text) else ""
    detected = text[start:end]
    return f"{prefix}{text[s:start]}[{detected}]{text[end:e]}{suffix}"


def discover_files() -> list[Path]:
    """Find all train.jsonl files under data/scraped and data/hf-traced."""
    files: list[Path] = []
    for subdir in ["scraped", "hf-traced"]:
        base = DATA_ROOT / subdir
        if base.exists():
            files.extend(sorted(base.rglob("train.jsonl")))
    return files


def load_records(path: Path, max_records: int) -> tuple[list[dict], int]:
    """Load up to max_records from a JSONL file. Returns (sampled_records, total_count)."""
    all_records: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                all_records.append(json.loads(line))

    total = len(all_records)
    if total <= max_records:
        return all_records, total

    sampled = random.sample(all_records, max_records)
    return sampled, total


# ---------------------------------------------------------------------------
# Main scan
# ---------------------------------------------------------------------------

def main() -> None:
    random.seed(42)

    print("=" * 70)
    print("AILIANCE PII Scanner — Presidio + en_core_web_lg")
    print("=" * 70)

    # Build analyzer
    print("\nLoading NLP engine and Presidio analyzer...")
    t0 = time.time()
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
    }
    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
    print(f"  Analyzer ready in {time.time() - t0:.1f}s")

    files = discover_files()
    print(f"\nFound {len(files)} train.jsonl files to scan.\n")

    report: dict[str, Any] = {
        "scan_date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "score_threshold": SCORE_THRESHOLD,
        "max_records_per_file": MAX_RECORDS_PER_FILE,
        "files": {},
        "global_summary": {},
    }

    global_entity_counts: Counter = Counter()
    total_files_with_pii = 0

    for filepath in files:
        rel = str(filepath.relative_to(DATA_ROOT))
        print(f"Scanning: {rel}")

        records, total_count = load_records(filepath, MAX_RECORDS_PER_FILE)
        scanned = len(records)
        print(f"  Total records: {total_count}, sampled: {scanned}")

        file_entity_counts: Counter = Counter()
        file_examples: dict[str, list[str]] = defaultdict(list)
        records_with_pii = 0

        for rec in records:
            text = extract_text(rec)
            if not text:
                continue

            results = analyzer.analyze(
                text=text,
                entities=PII_ENTITY_TYPES,
                language="en",
                score_threshold=SCORE_THRESHOLD,
            )

            if results:
                records_with_pii += 1

            for result in results:
                entity_type = result.entity_type
                file_entity_counts[entity_type] += 1
                global_entity_counts[entity_type] += 1

                if len(file_examples[entity_type]) < MAX_EXAMPLES:
                    snip = snippet_around(text, result.start, result.end)
                    file_examples[entity_type].append(
                        f"(score={result.score:.2f}) {snip}"
                    )

        has_pii = records_with_pii > 0
        if has_pii:
            total_files_with_pii += 1

        # Exclude low-signal entity types from the "concerning" report
        # PERSON, LOCATION, DATE_TIME, NRP, URL are often false positives in technical text
        high_signal_types = {
            "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD",
            "IP_ADDRESS", "IBAN_CODE", "US_SSN", "US_PASSPORT",
            "UK_NHS", "CRYPTO", "MEDICAL_LICENSE",
        }
        concerning_counts = {
            k: v for k, v in file_entity_counts.items() if k in high_signal_types
        }

        file_report = {
            "total_records": total_count,
            "records_scanned": scanned,
            "records_with_pii": records_with_pii,
            "entity_counts": dict(file_entity_counts.most_common()),
            "concerning_entity_counts": concerning_counts,
            "examples": {k: v for k, v in file_examples.items()},
        }
        report["files"][rel] = file_report

        # Print summary for this file
        if file_entity_counts:
            print(f"  PII detections: {dict(file_entity_counts.most_common())}")
            if concerning_counts:
                print(f"  ** CONCERNING: {concerning_counts}")
        else:
            print("  No PII detected.")
        print()

    # Global summary
    report["global_summary"] = {
        "total_files": len(files),
        "files_with_pii": total_files_with_pii,
        "entity_counts": dict(global_entity_counts.most_common()),
    }

    # Write JSON report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    # Print global summary
    print("=" * 70)
    print("GLOBAL SUMMARY")
    print("=" * 70)
    print(f"Files scanned:       {len(files)}")
    print(f"Files with PII:      {total_files_with_pii}")
    print(f"\nEntity counts across all files:")
    for entity_type, count in global_entity_counts.most_common():
        marker = " **" if entity_type in high_signal_types else ""
        print(f"  {entity_type:25s} {count:6d}{marker}")

    print(f"\n** = high-signal PII type (not a common false positive)")
    print(f"\nDetailed report written to: {REPORT_PATH}")

    # Exit code: 1 if any high-signal PII found
    high_signal_total = sum(
        v for k, v in global_entity_counts.items() if k in high_signal_types
    )
    if high_signal_total > 0:
        print(f"\n⚠ WARNING: {high_signal_total} high-signal PII detections found!")
        sys.exit(1)
    else:
        print("\n✓ No high-signal PII detected. Low-signal entities (PERSON, LOCATION, etc.) may be false positives in technical text.")
        sys.exit(0)


if __name__ == "__main__":
    main()
