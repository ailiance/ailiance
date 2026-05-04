#!/usr/bin/env python3
"""Scrape ESP-IDF examples from a local sparse checkout for embedded C/C++ training data.

Source: https://github.com/espressif/esp-idf/tree/master/examples
License: Apache-2.0
EU AI Act: Article 53 compliant — open-source, permissive license, public repository.
"""
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

EXAMPLES_ROOT = Path("/tmp/esp-idf-examples/examples")
OUTPUT = Path("data/scraped/espidf-examples")
MIN_LINES = 20
MAX_LINES = 500


def find_example_root(file_path: Path) -> Path:
    """Walk up from a source file to find the example root (directory containing CMakeLists.txt with project())."""
    current = file_path.parent
    while current != EXAMPLES_ROOT and current != EXAMPLES_ROOT.parent:
        cmakelists = current / "CMakeLists.txt"
        if cmakelists.exists():
            try:
                content = cmakelists.read_text(errors="replace")
                if "project(" in content:
                    return current
            except OSError:
                pass
        current = current.parent
    return file_path.parent


def find_readme(example_dir: Path) -> str | None:
    """Find and read the README.md in an example directory."""
    for name in ("README.md", "readme.md", "README.rst"):
        readme = example_dir / name
        if readme.exists():
            try:
                text = readme.read_text(errors="replace")
                # Strip badges, HTML tags, and excessive whitespace
                text = re.sub(r"<[^>]+>", "", text)
                text = re.sub(r"\[!\[.*?\]\(.*?\)\]\(.*?\)", "", text)
                text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
                # Trim to first ~2000 chars to keep prompts reasonable
                if len(text) > 2000:
                    text = text[:2000] + "..."
                return text.strip()
            except OSError:
                pass
    return None


def extract_category(rel_path: str) -> str:
    """Extract the top-level category from the relative path."""
    parts = rel_path.split("/")
    return parts[0] if parts else "unknown"


def build_user_prompt(example_name: str, readme_text: str | None, rel_path: str) -> str:
    """Create a user prompt combining context and instruction."""
    category = extract_category(rel_path)
    base = f"Write ESP-IDF firmware code for: {example_name} (category: {category})"
    if readme_text:
        # Use first paragraph or first 500 chars of README as context
        lines = readme_text.split("\n")
        # Skip title lines (starting with #)
        content_lines = [l for l in lines if l.strip() and not l.strip().startswith("#")]
        context = "\n".join(content_lines[:10])
        if len(context) > 500:
            context = context[:500] + "..."
        if context.strip():
            return f"Context from documentation:\n{context}\n\n{base}"
    return base


def main() -> None:
    if not EXAMPLES_ROOT.exists():
        print(f"ERROR: {EXAMPLES_ROOT} not found.")
        print("Run: cd /tmp && git clone --depth 1 --filter=blob:none --sparse "
              "https://github.com/espressif/esp-idf.git esp-idf-examples")
        print("     cd esp-idf-examples && git sparse-checkout set examples")
        return

    OUTPUT.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    skipped_short = 0
    skipped_long = 0
    skipped_read_error = 0
    category_counts: Counter[str] = Counter()

    source_files = sorted(
        p for p in EXAMPLES_ROOT.rglob("*")
        if p.suffix in (".c", ".cpp") and p.is_file()
    )
    print(f"Found {len(source_files)} .c/.cpp files in {EXAMPLES_ROOT}")

    for src_file in source_files:
        try:
            code = src_file.read_text(errors="replace")
        except OSError:
            skipped_read_error += 1
            continue

        line_count = len(code.splitlines())
        if line_count < MIN_LINES:
            skipped_short += 1
            continue
        if line_count > MAX_LINES:
            skipped_long += 1
            continue

        rel_path = str(src_file.relative_to(EXAMPLES_ROOT))
        example_dir = find_example_root(src_file)
        example_name = example_dir.name
        readme_text = find_readme(example_dir)
        category = extract_category(rel_path)

        user_prompt = build_user_prompt(example_name, readme_text, rel_path)

        records.append({
            "messages": [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": code},
            ],
            "_provenance": {
                "source": "espressif/esp-idf/examples",
                "license": "Apache-2.0",
                "file_path": rel_path,
                "domain_tag": "embedded-mcu",
                "access_date": datetime.now(timezone.utc).isoformat(),
                "category": category,
            },
        })
        category_counts[category] += 1

    # Save
    out_file = OUTPUT / "train.jsonl"
    with open(out_file, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Report
    print(f"\n{'=' * 60}")
    print(f"ESP-IDF Examples Scrape Report")
    print(f"{'=' * 60}")
    print(f"Total source files found:   {len(source_files)}")
    print(f"Skipped (< {MIN_LINES} lines):     {skipped_short}")
    print(f"Skipped (> {MAX_LINES} lines):    {skipped_long}")
    print(f"Skipped (read errors):      {skipped_read_error}")
    print(f"Training records saved:     {len(records)}")
    print(f"Output: {out_file}")

    print(f"\nDistribution by category:")
    for cat, count in category_counts.most_common():
        bar = "█" * (count // 2) if count > 1 else "█"
        print(f"  {cat:<25} {count:>4}  {bar}")

    # Sample records
    if records:
        print(f"\n--- Sample record (first) ---")
        sample = records[0]
        print(f"  User prompt:  {sample['messages'][0]['content'][:120]}...")
        print(f"  Code length:  {len(sample['messages'][1]['content'])} chars")
        print(f"  Provenance:   {sample['_provenance']}")

        if len(records) > 1:
            mid = records[len(records) // 2]
            print(f"\n--- Sample record (middle) ---")
            print(f"  User prompt:  {mid['messages'][0]['content'][:120]}...")
            print(f"  Code length:  {len(mid['messages'][1]['content'])} chars")
            print(f"  Provenance:   {mid['_provenance']}")


if __name__ == "__main__":
    main()
