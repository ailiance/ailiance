"""Convert extracted PDF text into instruction-response training pairs.

Splits text by section headings, generates instruction pairs,
adds provenance metadata, and runs a lightweight PII scan.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import HF_TRACED_DIR, PDF_EXTRACTED_DIR, PDF_RAW_DIR


# ---------------------------------------------------------------------------
# Section splitting
# ---------------------------------------------------------------------------

# Match common section patterns:
#   "1.2 Title", "1.2.3 Title", "Section 3: Title", "## Title"
_SECTION_RE = re.compile(
    r"^(?:"
    r"(?:\d+\.)+\d*\s+"              # 1.2 or 1.2.3
    r"|Section\s+\d+[.:]\s+"         # Section 3:
    r"|Chapter\s+\d+[.:]\s+"         # Chapter 2:
    r"|#{1,4}\s+"                    # Markdown headings
    r"|[A-Z][A-Z ]{3,}\n"           # ALL-CAPS HEADING
    r")"
    r"(.+)",
    re.MULTILINE,
)

MIN_SECTION_LEN = 200
MAX_SECTION_LEN = 4000


@dataclass(frozen=True)
class Section:
    title: str
    content: str
    start_line: int


def split_into_sections(text: str) -> list[Section]:
    """Split text into sections based on heading patterns."""
    lines = text.split("\n")
    sections: list[Section] = []

    # Find all heading positions
    headings: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        match = _SECTION_RE.match(stripped)
        if match:
            headings.append((i, stripped))
        elif stripped.isupper() and len(stripped) > 5 and len(stripped) < 80:
            headings.append((i, stripped))

    if not headings:
        # No headings found — treat the whole text as one section
        if len(text) >= MIN_SECTION_LEN:
            return [Section(title="Document Content", content=text[:MAX_SECTION_LEN], start_line=0)]
        return []

    # Extract sections between headings
    for idx, (line_num, title) in enumerate(headings):
        start = line_num + 1
        end = headings[idx + 1][0] if idx + 1 < len(headings) else len(lines)
        content = "\n".join(lines[start:end]).strip()

        if len(content) < MIN_SECTION_LEN:
            continue
        if len(content) > MAX_SECTION_LEN:
            content = content[:MAX_SECTION_LEN]

        clean_title = re.sub(r"^[\d.]+\s*", "", title).strip()
        clean_title = re.sub(r"^(?:Section|Chapter)\s+\d+[.:]\s*", "", clean_title).strip()
        clean_title = re.sub(r"^#+\s*", "", clean_title).strip()

        if not clean_title:
            clean_title = title

        sections.append(Section(title=clean_title, content=content, start_line=line_num))

    return sections


# ---------------------------------------------------------------------------
# Instruction generation
# ---------------------------------------------------------------------------

def _generate_instruction(section: Section, source_name: str) -> str:
    """Generate a natural instruction from the section title."""
    title = section.title

    # Technical patterns
    if any(kw in title.lower() for kw in ("configuration", "setup", "getting started")):
        return f"How do I configure {title.lower().replace('configuration', '').strip()}?"
    if any(kw in title.lower() for kw in ("overview", "introduction", "description")):
        return f"Explain {title.lower().replace('overview', '').replace('introduction to', '').strip()}."
    if any(kw in title.lower() for kw in ("schematic", "circuit", "layout", "design")):
        return f"Describe the {title.lower()} considerations."
    if any(kw in title.lower() for kw in ("example", "application", "use case")):
        return f"Show an {title.lower()}."
    if any(kw in title.lower() for kw in ("troubleshoot", "debug", "error", "issue")):
        return f"How to troubleshoot {title.lower().replace('troubleshooting', '').strip()}?"

    return f"Explain the following technical topic: {title}"


# ---------------------------------------------------------------------------
# PII scanning (lightweight, reuses patterns from scan_pii.py)
# ---------------------------------------------------------------------------

_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "phone": re.compile(r"\b(?:\+\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b"),
    "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}


def _scan_pii(text: str) -> list[dict[str, str]]:
    """Quick regex-based PII scan. Returns list of findings."""
    findings: list[dict[str, str]] = []
    for pii_type, pattern in _PII_PATTERNS.items():
        for match in pattern.finditer(text):
            # Filter out common false positives
            value = match.group()
            if pii_type == "ip_address" and value.startswith(("0.", "1.", "255.", "127.", "192.168.")):
                continue  # Likely technical content, not real PII
            if pii_type == "phone" and len(value.replace(" ", "").replace("-", "")) < 7:
                continue
            findings.append({"type": pii_type, "value": value[:6] + "***"})
    return findings


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------

def convert_source(
    source_name_safe: str,
    source_name: str,
    domains: tuple[str, ...],
    legal_basis: str,
    robots_status: str,
    robots_check_date: str,
    download_date: str,
) -> dict[str, int]:
    """Convert extracted texts for a source into training pairs."""
    extracted_dir = PDF_EXTRACTED_DIR / source_name_safe
    if not extracted_dir.exists():
        print(f"  No extracted texts for {source_name}")
        return {"pairs": 0, "pii_findings": 0}

    # Load extraction metadata for file hashes
    manifest_path = PDF_RAW_DIR / source_name_safe / "manifest.json"
    file_hashes: dict[str, str] = {}
    file_urls: dict[str, str] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for f in manifest.get("files", []):
            stem = f["filename"].replace(".pdf", "")
            file_hashes[stem] = f.get("sha256", "")
            file_urls[stem] = f.get("url", "")

    txt_files = sorted(extracted_dir.glob("*.txt"))
    if not txt_files:
        print(f"  No text files in {extracted_dir}")
        return {"pairs": 0, "pii_findings": 0}

    total_pairs = 0
    total_pii = 0

    # Write to primary domain
    primary_domain = domains[0] if domains else "general"

    output_dir = HF_TRACED_DIR / primary_domain
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "pdf_supplement.jsonl"

    records: list[dict] = []

    for txt_path in txt_files:
        stem = txt_path.stem
        text = txt_path.read_text(encoding="utf-8")
        sections = split_into_sections(text)

        print(f"    {txt_path.name}: {len(sections)} sections")

        for section in sections:
            instruction = _generate_instruction(section, source_name)
            pii_findings = _scan_pii(section.content)
            total_pii += len(pii_findings)

            record = {
                "messages": [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": section.content},
                ],
                "_provenance": {
                    "source": source_name,
                    "url": file_urls.get(stem, ""),
                    "file_hash": file_hashes.get(stem, ""),
                    "legal_basis": legal_basis,
                    "robots_status": robots_status,
                    "robots_check_date": robots_check_date,
                    "download_date": download_date,
                    "extraction_method": "pymupdf",
                    "section": section.title,
                    "page_range": "",
                    "domain_tag": primary_domain,
                    "pipeline_version": "0.2",
                },
            }

            if pii_findings:
                record["_pii_findings"] = pii_findings

            records.append(record)
            total_pairs += 1

    # Write training pairs (append mode for supplementary data)
    with output_path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Also write to additional domains
    for domain in domains[1:]:
        domain_dir = HF_TRACED_DIR / domain
        domain_dir.mkdir(parents=True, exist_ok=True)
        domain_path = domain_dir / "pdf_supplement.jsonl"
        with domain_path.open("a", encoding="utf-8") as f:
            for record in records:
                tagged = {**record}
                tagged["_provenance"] = {**record["_provenance"], "domain_tag": domain}
                f.write(json.dumps(tagged, ensure_ascii=False) + "\n")

    return {"pairs": total_pairs, "pii_findings": total_pii}
