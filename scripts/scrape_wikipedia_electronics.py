#!/usr/bin/env python3
"""Extract electronics articles from Wikipedia for LoRA training.

Source: Wikipedia dumps (dumps.wikimedia.org)
License: CC-BY-SA 3.0
EU AI Act: Article 53 compliant — official bulk download, documented.

Usage:
    # First download the dump:
    # wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
    # Then run:
    uv run python scripts/scrape_wikipedia_electronics.py --dump-path /path/to/dump.xml.bz2
"""
import argparse, json, re
from pathlib import Path
from datetime import datetime, timezone

OUTPUT = Path("data/scraped/wikipedia-electronics")

# Articles to extract (by title pattern)
ELECTRONICS_PATTERNS = [
    r"electromagnetic compatibility", r"EMC", r"SPICE",
    r"circuit.*simulation", r"printed circuit board", r"PCB",
    r"signal integrity", r"power supply", r"voltage regulator",
    r"operational amplifier", r"transistor", r"diode",
    r"capacitor", r"inductor", r"resistor", r"filter.*circuit",
    r"analog.*circuit", r"digital.*circuit", r"microcontroller",
    r"embedded system", r"FPGA", r"ASIC", r"KiCad",
    r"creepage", r"clearance.*voltage", r"IEC 60950",
    r"IEC 61508", r"functional safety", r"MISRA",
]

def extract_from_dump(dump_path: str):
    """Extract matching articles from Wikipedia XML dump."""
    # This uses mwxml for parsing - install with: pip install mwxml
    try:
        import mwxml
    except ImportError:
        print("Install mwxml: pip install mwxml")
        print("Alternatively, use the HuggingFace wikipedia dataset:")
        print("  from datasets import load_dataset")
        print("  ds = load_dataset('wikipedia', '20231101.en', split='train')")
        return extract_from_hf()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    patterns = [re.compile(p, re.IGNORECASE) for p in ELECTRONICS_PATTERNS]
    records = []

    dump = mwxml.Dump.from_file(open(dump_path, 'rb'))
    for page in dump:
        title = page.title
        if not any(p.search(title) for p in patterns):
            continue

        for revision in page:
            text = revision.text or ""
            if len(text) < 200:
                continue

            # Clean wiki markup (basic)
            clean = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
            clean = re.sub(r'\{\{[^}]+\}\}', '', clean)
            clean = re.sub(r'<[^>]+>', '', clean)
            clean = clean[:3000]  # Truncate

            records.append({
                "messages": [
                    {"role": "user", "content": f"Explain {title} in detail."},
                    {"role": "assistant", "content": clean},
                ],
                "_provenance": {
                    "source": "wikipedia",
                    "title": title,
                    "license": "CC-BY-SA-3.0",
                    "access_method": "bulk XML dump",
                    "access_date": datetime.now(timezone.utc).isoformat(),
                }
            })
            break  # Only latest revision

    save_records(records)


def extract_from_hf():
    """Fallback: use HuggingFace wikipedia dataset."""
    from datasets import load_dataset

    OUTPUT.mkdir(parents=True, exist_ok=True)
    patterns = [re.compile(p, re.IGNORECASE) for p in ELECTRONICS_PATTERNS]

    print("Loading Wikipedia from HuggingFace (streaming)...")
    ds = load_dataset("wikipedia", "20231101.en", split="train", streaming=True)

    records = []
    checked = 0
    for row in ds:
        checked += 1
        title = row.get("title", "")
        text = row.get("text", "")

        if not any(p.search(title) for p in patterns):
            continue

        if len(text) < 200:
            continue

        records.append({
            "messages": [
                {"role": "user", "content": f"Explain {title} in detail."},
                {"role": "assistant", "content": text[:3000]},
            ],
            "_provenance": {
                "source": "wikipedia (via HuggingFace)",
                "title": title,
                "license": "CC-BY-SA-3.0",
                "access_method": "HuggingFace datasets streaming",
                "access_date": datetime.now(timezone.utc).isoformat(),
            }
        })

        if len(records) >= 500:
            break
        if checked % 100000 == 0:
            print(f"  Checked {checked:,} articles, found {len(records)}")

    save_records(records)


def save_records(records):
    if records:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT / "train.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved {len(records)} electronics articles to {OUTPUT}/train.jsonl")
    else:
        print("No matching articles found.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dump-path", help="Path to enwiki XML dump (optional, uses HF fallback)")
    args = p.parse_args()

    if args.dump_path:
        extract_from_dump(args.dump_path)
    else:
        extract_from_hf()
