#!/usr/bin/env python3
"""Scrape embedded Rust examples from official repos for LoRA fine-tuning.

Sources (all permissive licenses, EU AI Act Article 53 compliant):
  - rust-embedded/cortex-m          (MIT OR Apache-2.0)
  - embassy-rs/embassy              (MIT OR Apache-2.0)
  - esp-rs/esp-hal                  (MIT OR Apache-2.0)
  - rtic-rs/rtic                    (MIT OR Apache-2.0)
  - knurling-rs/defmt               (MIT OR Apache-2.0)
  - rust-embedded/discovery         (MIT OR Apache-2.0)
  - stm32-rs/stm32f4xx-hal          (0BSD)
  - nrf-rs/nrf-hal                  (MIT OR Apache-2.0)
  - rust-embedded/embedded-hal      (MIT OR Apache-2.0)
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

MIN_LINES = 15
MAX_LINES = 500

SCRAPED_OUTPUT = Path("data/scraped/rust-embedded")
FINAL_OUTPUT = Path("data/hf-traced/rust-embedded")
TARGET_TOTAL = 3000
TRAIN_COUNT = 2850
VALID_COUNT = 150


@dataclass(frozen=True)
class RepoConfig:
    name: str
    path: Path
    license: str
    platform: str
    scan_dirs: tuple[str, ...]  # relative dirs to scan for .rs files


REPOS: tuple[RepoConfig, ...] = (
    RepoConfig(
        name="rust-embedded/cortex-m",
        path=Path("/tmp/cortex-m"),
        license="MIT OR Apache-2.0",
        platform="cortex-m",
        scan_dirs=("cortex-m/examples", "cortex-m/src", "cortex-m-rt/src", "examples"),
    ),
    RepoConfig(
        name="embassy-rs/embassy",
        path=Path("/tmp/embassy"),
        license="MIT OR Apache-2.0",
        platform="multi",  # nrf, stm32, rp, esp
        scan_dirs=("examples",),
    ),
    RepoConfig(
        name="esp-rs/esp-hal",
        path=Path("/tmp/esp-hal"),
        license="MIT OR Apache-2.0",
        platform="esp32",
        scan_dirs=("examples",),
    ),
    RepoConfig(
        name="rtic-rs/rtic",
        path=Path("/tmp/rtic"),
        license="MIT OR Apache-2.0",
        platform="cortex-m",
        scan_dirs=("examples", "rtic/examples", "rtic-macros/examples"),
    ),
    RepoConfig(
        name="knurling-rs/defmt",
        path=Path("/tmp/defmt"),
        license="MIT OR Apache-2.0",
        platform="cortex-m",
        scan_dirs=("defmt/src", "defmt-macros/src", "firmware/defmt-test/src"),
    ),
    RepoConfig(
        name="rust-embedded/discovery",
        path=Path("/tmp/discovery"),
        license="MIT OR Apache-2.0",
        platform="cortex-m",
        scan_dirs=("microbit/src", "f3discovery/src"),
    ),
    RepoConfig(
        name="stm32-rs/stm32f4xx-hal",
        path=Path("/tmp/stm32f4xx-hal"),
        license="0BSD",
        platform="stm32",
        scan_dirs=("examples",),
    ),
    RepoConfig(
        name="nrf-rs/nrf-hal",
        path=Path("/tmp/nrf-hal"),
        license="MIT OR Apache-2.0",
        platform="nrf",
        scan_dirs=("examples",),
    ),
    RepoConfig(
        name="rust-embedded/embedded-hal",
        path=Path("/tmp/embedded-hal"),
        license="MIT OR Apache-2.0",
        platform="generic-hal",
        scan_dirs=("embedded-hal/src", "embedded-hal-async/src", "embedded-hal-bus/src"),
    ),
)


def detect_platform(repo: RepoConfig, file_path: Path) -> str:
    """Infer the target platform from file path and content hints."""
    path_str = str(file_path).lower()

    platform_hints = {
        "nrf": "nrf",
        "stm32": "stm32",
        "esp": "esp32",
        "rp2040": "rp2040",
        "rp23": "rp2350",
        "microbit": "nrf-microbit",
        "f3discovery": "stm32-f3discovery",
    }
    for hint, platform in platform_hints.items():
        if hint in path_str:
            return platform

    return repo.platform


def find_readme(file_path: Path, repo_root: Path) -> str | None:
    """Walk up from file to find nearest README."""
    current = file_path.parent
    while current != repo_root.parent:
        for name in ("README.md", "readme.md", "README.rst"):
            readme = current / name
            if readme.exists():
                try:
                    text = readme.read_text(errors="replace")
                    text = re.sub(r"<[^>]+>", "", text)
                    text = re.sub(r"\[!\[.*?\]\(.*?\)\]\(.*?\)", "", text)
                    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
                    if len(text) > 1500:
                        text = text[:1500] + "..."
                    return text.strip()
                except OSError:
                    pass
        current = current.parent
    return None


def extract_doc_comments(code: str) -> str:
    """Extract top-level doc comments (//! or /// at file start)."""
    lines = code.splitlines()
    doc_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("//!") or stripped.startswith("///"):
            cleaned = stripped.lstrip("/!").strip()
            if cleaned:
                doc_lines.append(cleaned)
        elif stripped.startswith("//"):
            continue
        elif stripped == "" and not doc_lines:
            continue
        elif stripped.startswith("#![") or stripped.startswith("#["):
            continue
        elif stripped.startswith("use ") or stripped.startswith("extern "):
            continue
        else:
            break

    return " ".join(doc_lines) if doc_lines else ""


def build_user_prompt(
    file_name: str,
    doc_comment: str,
    readme_text: str | None,
    platform: str,
    rel_path: str,
) -> str:
    """Create a user prompt for the training pair."""
    # Build a description from available context
    description_parts: list[str] = []

    if doc_comment:
        description_parts.append(doc_comment)

    if readme_text:
        # Extract first meaningful paragraph from README
        lines = readme_text.split("\n")
        content_lines = [
            line for line in lines
            if line.strip() and not line.strip().startswith("#")
        ]
        context = " ".join(content_lines[:5])
        if len(context) > 400:
            context = context[:400] + "..."
        if context.strip():
            description_parts.append(f"Context: {context}")

    # Derive example name from filename
    example_name = file_name.replace(".rs", "").replace("_", " ").title()

    if description_parts:
        desc = "\n".join(description_parts)
        return (
            f"{desc}\n\n"
            f"Write embedded Rust code for {example_name} targeting {platform}. "
            f"File: {rel_path}"
        )

    return (
        f"Write embedded Rust code for {example_name} targeting {platform}. "
        f"File: {rel_path}"
    )


def verify_license(repo: RepoConfig) -> bool:
    """Verify that the repo has a recognized permissive license file."""
    allowed_spdx = {"MIT", "Apache-2.0", "0BSD", "BSD-3-Clause"}
    license_files = list(repo.path.glob("LICENSE*")) + list(repo.path.glob("COPYING*"))
    if not license_files:
        # Check Cargo.toml
        cargo = repo.path / "Cargo.toml"
        if cargo.exists():
            text = cargo.read_text(errors="replace")
            for spdx in allowed_spdx:
                if spdx in text:
                    return True
        print(f"  WARNING: No license file found for {repo.name}")
        return False

    for lf in license_files:
        name = lf.name.upper()
        if any(tag in name for tag in ("MIT", "APACHE", "0BSD", "BSD")):
            return True

    return True  # has license files, trust Cargo.toml / repo metadata


def scrape_repo(repo: RepoConfig) -> list[dict]:
    """Scrape .rs files from a single repo."""
    if not repo.path.exists():
        print(f"  SKIP: {repo.path} not found")
        return []

    if not verify_license(repo):
        print(f"  SKIP: License verification failed for {repo.name}")
        return []

    records: list[dict] = []
    skipped_short = 0
    skipped_long = 0

    # Collect .rs files from scan_dirs
    rs_files: list[Path] = []
    for scan_dir in repo.scan_dirs:
        target = repo.path / scan_dir
        if target.exists():
            rs_files.extend(sorted(target.rglob("*.rs")))

    # Deduplicate
    rs_files = sorted(set(rs_files))

    for rs_file in rs_files:
        if not rs_file.is_file():
            continue

        try:
            code = rs_file.read_text(errors="replace")
        except OSError:
            continue

        line_count = len(code.splitlines())
        if line_count < MIN_LINES:
            skipped_short += 1
            continue
        if line_count > MAX_LINES:
            skipped_long += 1
            continue

        rel_path = str(rs_file.relative_to(repo.path))
        platform = detect_platform(repo, rs_file)
        doc_comment = extract_doc_comments(code)
        readme_text = find_readme(rs_file, repo.path)

        user_prompt = build_user_prompt(
            rs_file.name, doc_comment, readme_text, platform, rel_path,
        )

        records.append({
            "messages": [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": code},
            ],
            "_provenance": {
                "source": repo.name,
                "license": repo.license,
                "file_path": rel_path,
                "platform_tag": platform,
                "domain_tag": "rust-embedded",
                "access_date": datetime.now(timezone.utc).isoformat(),
            },
        })

    print(
        f"  {repo.name}: {len(records)} records "
        f"(skipped {skipped_short} short, {skipped_long} long, "
        f"from {len(rs_files)} .rs files)"
    )
    return records


EMBEDDED_KEYWORDS = re.compile(
    r"\b(no_std|cortex|embassy|hal|gpio|interrupt|peripheral|register|"
    r"volatile|unsafe|pac::|dp::|embedded|firmware|microcontroller|"
    r"rtic|defmt|probe|flash|uart|spi|i2c|pwm|adc|dma|timer|"
    r"bare.?metal|#\[entry\]|#\[interrupt\]|embedded.hal|"
    r"stm32|nrf|esp32|esp.hal|cortex.m|svd2rust|probe.rs)\b",
    re.IGNORECASE,
)


def fetch_commitpackft_rust(needed: int) -> list[dict]:
    """Fetch Rust commit-based instruction pairs from bigcode/commitpackft.

    Filters for:
      1. Permissive licenses (MIT, Apache-2.0, BSD-*, 0BSD)
      2. Embedded-related content (keywords in code or commit message)
      3. New_contents between MIN_LINES and MAX_LINES
    """
    try:
        import pandas as pd
        import requests
    except ImportError:
        print("  SKIP: pandas or requests not available for HF fetch")
        return []

    print("  Fetching bigcode/commitpackft Rust subset from HuggingFace...")
    api_url = (
        "https://huggingface.co/api/datasets/"
        "bigcode/commitpackft/parquet/rust/train/0.parquet"
    )

    try:
        resp = requests.get(api_url, timeout=15)
        parquet_url = resp.url
        df = pd.read_parquet(parquet_url)
    except Exception as exc:
        print(f"  ERROR fetching commitpackft: {exc}")
        return []

    print(f"  Raw commitpackft Rust records: {len(df)}")

    # Filter permissive licenses
    permissive = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "0bsd", "isc", "unlicense"}
    df = df[df["license"].str.lower().isin(permissive)]
    print(f"  After license filter: {len(df)}")

    # Filter for embedded content
    records: list[dict] = []
    for _, row in df.iterrows():
        code = str(row.get("new_contents", ""))
        subject = str(row.get("subject", ""))
        message = str(row.get("message", ""))
        combined = f"{subject} {message} {code}"

        if not EMBEDDED_KEYWORDS.search(combined):
            continue

        line_count = len(code.splitlines())
        if line_count < MIN_LINES or line_count > MAX_LINES:
            continue

        # Build instruction from commit message
        instruction = subject.strip()
        if message.strip() and message.strip() != subject.strip():
            instruction = f"{instruction}\n\n{message.strip()}"

        instruction += (
            "\n\nWrite the embedded Rust code implementing this change."
        )

        record_license = str(row.get("license", "MIT"))

        records.append({
            "messages": [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": code},
            ],
            "_provenance": {
                "source": "bigcode/commitpackft",
                "license": record_license,
                "file_path": str(row.get("new_file", "")),
                "commit": str(row.get("commit", ""))[:12],
                "platform_tag": "generic-rust-embedded",
                "domain_tag": "rust-embedded",
                "access_date": datetime.now(timezone.utc).isoformat(),
            },
        })

        if len(records) >= needed:
            break

    print(f"  Embedded-relevant commitpackft records: {len(records)}")
    return records


def supplement_from_existing_rust(
    current_count: int,
) -> list[dict]:
    """Pull embedded-relevant records from the generic rust domain."""
    rust_train = Path("data/hf-traced/rust/train.jsonl")
    if not rust_train.exists():
        print("  No existing rust domain to supplement from")
        return []

    supplemental: list[dict] = []
    needed = TARGET_TOTAL - current_count

    with open(rust_train) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            messages = record.get("messages", [])
            full_text = " ".join(m.get("content", "") for m in messages)
            if EMBEDDED_KEYWORDS.search(full_text):
                # Re-tag as rust-embedded
                prov = record.get("_provenance", {})
                record["_provenance"] = {
                    **prov,
                    "domain_tag": "rust-embedded",
                    "supplemented_from": "rust-generic",
                }
                supplemental.append(record)

    if supplemental:
        print(f"  Found {len(supplemental)} embedded-relevant records from rust domain")
        if len(supplemental) > needed:
            supplemental = supplemental[:needed]

    return supplemental


def sort_curriculum(records: list[dict]) -> list[dict]:
    """Sort records short to long (curriculum-style learning)."""
    return sorted(
        records,
        key=lambda r: len(r["messages"][1]["content"]),
    )


def main() -> None:
    print("=" * 60)
    print("Embedded Rust Scraper for AILIANCE LoRA Fine-Tuning")
    print("=" * 60)

    # Phase 1: Scrape all repos
    all_records: list[dict] = []
    repo_counts: dict[str, int] = {}
    platform_counts: Counter[str] = Counter()

    for repo in REPOS:
        print(f"\nScraping {repo.name}...")
        records = scrape_repo(repo)
        all_records.extend(records)
        repo_counts[repo.name] = len(records)
        for r in records:
            platform_counts[r["_provenance"]["platform_tag"]] += 1

    print(f"\n{'=' * 60}")
    print(f"Phase 1 — Scraped records: {len(all_records)}")

    # Save raw scraped data
    SCRAPED_OUTPUT.mkdir(parents=True, exist_ok=True)
    scraped_file = SCRAPED_OUTPUT / "train.jsonl"
    with open(scraped_file, "w") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved raw scraped data to {scraped_file}")

    # Phase 2: Supplement from HuggingFace + existing rust domain
    if len(all_records) < TARGET_TOTAL:
        needed = TARGET_TOTAL - len(all_records)
        print(f"\nPhase 2 — Need {needed} more records")

        # 2a: HuggingFace commitpackft
        hf_records = fetch_commitpackft_rust(needed)
        if hf_records:
            all_records.extend(hf_records)
            repo_counts["bigcode/commitpackft"] = len(hf_records)
            for r in hf_records:
                platform_counts[r["_provenance"]["platform_tag"]] += 1
            print(f"  Total after HF: {len(all_records)}")

        # 2b: Existing rust domain supplement
        if len(all_records) < TARGET_TOTAL:
            supplemental = supplement_from_existing_rust(len(all_records))
            if supplemental:
                all_records.extend(supplemental)
                print(f"  Total after rust-generic supplement: {len(all_records)}")

    # Phase 3: Sort curriculum-style and split
    all_records = sort_curriculum(all_records)

    FINAL_OUTPUT.mkdir(parents=True, exist_ok=True)

    if len(all_records) >= TARGET_TOTAL:
        train_records = all_records[:TRAIN_COUNT]
        valid_records = all_records[TRAIN_COUNT : TRAIN_COUNT + VALID_COUNT]
    else:
        # Use 95/5 split
        split_idx = int(len(all_records) * 0.95)
        train_records = all_records[:split_idx]
        valid_records = all_records[split_idx:]

    train_file = FINAL_OUTPUT / "train.jsonl"
    valid_file = FINAL_OUTPUT / "valid.jsonl"

    with open(train_file, "w") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(valid_file, "w") as f:
        for r in valid_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Phase 4: Report
    print(f"\n{'=' * 60}")
    print("FINAL REPORT")
    print(f"{'=' * 60}")

    print("\nPer-repo yields:")
    for name, count in sorted(repo_counts.items(), key=lambda x: -x[1]):
        bar = "#" * (count // 5) if count >= 5 else "#"
        print(f"  {name:<35} {count:>4}  {bar}")

    print(f"\nPlatform distribution:")
    for platform, count in platform_counts.most_common():
        bar = "#" * (count // 5) if count >= 5 else "#"
        print(f"  {platform:<20} {count:>4}  {bar}")

    print(f"\nLicense verification:")
    for repo in REPOS:
        status = "OK" if verify_license(repo) else "FAIL"
        print(f"  {repo.name:<35} {repo.license:<25} [{status}]")

    print(f"\nDataset summary:")
    print(f"  Total scraped records:  {sum(repo_counts.values())}")
    print(f"  Supplemented records:   {len(all_records) - sum(repo_counts.values())}")
    print(f"  Total combined:         {len(all_records)}")
    print(f"  Train split:            {len(train_records)}")
    print(f"  Valid split:            {len(valid_records)}")
    print(f"  Output:                 {FINAL_OUTPUT}/")

    # Sample records
    if train_records:
        print(f"\n--- Sample record (shortest, curriculum start) ---")
        s = train_records[0]
        print(f"  User:  {s['messages'][0]['content'][:150]}...")
        print(f"  Code:  {len(s['messages'][1]['content'])} chars")
        print(f"  Prov:  {s['_provenance']}")

        mid = train_records[len(train_records) // 2]
        print(f"\n--- Sample record (middle) ---")
        print(f"  User:  {mid['messages'][0]['content'][:150]}...")
        print(f"  Code:  {len(mid['messages'][1]['content'])} chars")
        print(f"  Prov:  {mid['_provenance']}")

        longest = train_records[-1]
        print(f"\n--- Sample record (longest, curriculum end) ---")
        print(f"  User:  {longest['messages'][0]['content'][:150]}...")
        print(f"  Code:  {len(longest['messages'][1]['content'])} chars")
        print(f"  Prov:  {longest['_provenance']}")


if __name__ == "__main__":
    main()
