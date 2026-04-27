#!/usr/bin/env python3
"""Extract electrical engineering papers from arXiv for training.

Source: arXiv eess.* (Electrical Engineering and Systems Science)
License: arXiv non-exclusive license (papers are author-copyrighted, arXiv has distribution rights)
EU AI Act: Article 53 compliant — official bulk access, documented.

Note: arXiv papers are typically author-owned. The arXiv license grants
non-exclusive distribution rights. For AI training under EU TDM exception
(Art. 4 DSM Directive), this is permissible as long as:
- No opt-out has been expressed
- Access is lawful
- Usage is documented
"""
import json, re
from pathlib import Path
from datetime import datetime, timezone

OUTPUT = Path("data/scraped/arxiv-eess")

# Categories of interest
CATEGORIES = ["eess.SP", "eess.SY", "eess.AS"]  # Signal Processing, Systems & Control, Audio

def extract_from_hf():
    """Use the arXiv abstracts dataset from HuggingFace."""
    from datasets import load_dataset

    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("Loading arXiv dataset from HuggingFace...")
    # Use the abstracts dataset (smaller, faster)
    try:
        ds = load_dataset("gfissore/arxiv-abstracts-2021", split="train", streaming=True)
    except Exception:
        ds = load_dataset("Cornell-University/arxiv", split="train", streaming=True)

    records = []
    checked = 0

    for row in ds:
        checked += 1
        categories = str(row.get("categories", ""))

        # Filter for electrical engineering
        if not any(cat in categories for cat in CATEGORIES):
            # Also check for EMC/signal integrity keywords in other categories
            title = str(row.get("title", "")).lower()
            abstract = str(row.get("abstract", "")).lower()
            keywords = ["electromagnetic compatibility", "emc", "signal integrity",
                       "power electronics", "circuit design", "pcb", "spice"]
            if not any(kw in title or kw in abstract for kw in keywords):
                continue

        title = str(row.get("title", "")).strip()
        abstract = str(row.get("abstract", "")).strip()

        if not title or not abstract or len(abstract) < 100:
            continue

        records.append({
            "messages": [
                {"role": "user", "content": f"Summarize the research paper: {title}"},
                {"role": "assistant", "content": abstract},
            ],
            "_provenance": {
                "source": "arXiv",
                "arxiv_id": str(row.get("id", "")),
                "categories": categories,
                "license": "arXiv non-exclusive (author-owned)",
                "access_method": "HuggingFace dataset (official mirror)",
                "access_date": datetime.now(timezone.utc).isoformat(),
                "tdm_exception": "EU DSM Directive Art. 3-4",
            }
        })

        if len(records) >= 3000:
            break
        if checked % 50000 == 0:
            print(f"  Checked {checked:,} papers, found {len(records)} relevant")

    # Save
    if records:
        with open(OUTPUT / "train.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved {len(records)} eess papers to {OUTPUT}/train.jsonl")
    else:
        print("No matching papers found.")


if __name__ == "__main__":
    extract_from_hf()
