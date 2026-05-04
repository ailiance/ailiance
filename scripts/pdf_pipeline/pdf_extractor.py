"""PDF text extraction using pymupdf (fitz).

Extracts text from digital PDFs. Falls back gracefully when
page text is too sparse (likely scanned), logging the issue
without requiring OCR dependencies.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import fitz  # pymupdf

from .config import PDF_EXTRACTED_DIR, PDF_RAW_DIR


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class ExtractionResult:
    source_file: str
    output_file: str
    method: str  # "pymupdf" | "pymupdf_sparse"
    page_count: int
    char_count: int
    detected_language: str
    sparse_pages: list[int]
    extraction_date: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    error: str = ""


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

_HEADER_FOOTER_RE = re.compile(
    r"^(?:"
    r"(?:Page\s+)?\d+(?:\s+of\s+\d+)?"
    r"|AN\d+"
    r"|Rev\.?\s*\d+"
    r"|Doc\s*ID\s*\d+"
    r"|www\.\S+"
    r")\s*$",
    re.MULTILINE | re.IGNORECASE,
)

_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r" {2,}")


def clean_text(raw: str) -> str:
    """Clean extracted PDF text: fix hyphenation, headers/footers, whitespace."""
    text = _HEADER_FOOTER_RE.sub("", raw)
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def _detect_language(text: str) -> str:
    """Very rough language detection based on common words."""
    sample = text[:3000].lower()
    fr_words = sum(1 for w in ("le ", "la ", "les ", "des ", "est ", "une ") if w in sample)
    en_words = sum(1 for w in ("the ", "is ", "are ", "for ", "and ", "this ") if w in sample)
    de_words = sum(1 for w in ("der ", "die ", "das ", "und ", "ist ", "ein ") if w in sample)

    if fr_words > en_words and fr_words > de_words:
        return "fr"
    if de_words > en_words:
        return "de"
    return "en"


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

MIN_CHARS_PER_PAGE = 100


def extract_pdf(pdf_path: Path, output_dir: Path) -> ExtractionResult:
    """Extract text from a single PDF file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = pdf_path.stem
    output_path = output_dir / f"{stem}.txt"

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        return ExtractionResult(
            source_file=pdf_path.name,
            output_file="",
            method="error",
            page_count=0,
            char_count=0,
            detected_language="",
            sparse_pages=[],
            error=str(exc),
        )

    pages_text: list[str] = []
    sparse_pages: list[int] = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if len(text.strip()) < MIN_CHARS_PER_PAGE:
            sparse_pages.append(page_num + 1)
        pages_text.append(text)

    doc.close()

    raw_text = "\n\n".join(pages_text)
    cleaned = clean_text(raw_text)

    if not cleaned:
        return ExtractionResult(
            source_file=pdf_path.name,
            output_file="",
            method="pymupdf_empty",
            page_count=len(pages_text),
            char_count=0,
            detected_language="",
            sparse_pages=sparse_pages,
            error="No text extracted — likely scanned PDF, OCR not available",
        )

    method = "pymupdf_sparse" if len(sparse_pages) > len(pages_text) / 2 else "pymupdf"
    output_path.write_text(cleaned, encoding="utf-8")

    return ExtractionResult(
        source_file=pdf_path.name,
        output_file=output_path.name,
        method=method,
        page_count=len(pages_text),
        char_count=len(cleaned),
        detected_language=_detect_language(cleaned),
        sparse_pages=sparse_pages,
    )


def extract_source(source_name_safe: str) -> list[ExtractionResult]:
    """Extract all PDFs for a given source directory name."""
    source_dir = PDF_RAW_DIR / source_name_safe
    output_dir = PDF_EXTRACTED_DIR / source_name_safe

    if not source_dir.exists():
        print(f"  Source directory not found: {source_dir}")
        return []

    pdf_files = sorted(source_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"  No PDFs found in {source_dir}")
        return []

    results: list[ExtractionResult] = []
    for pdf_path in pdf_files:
        print(f"    Extracting {pdf_path.name}...", end=" ", flush=True)
        result = extract_pdf(pdf_path, output_dir)
        print(f"{result.char_count} chars, {result.page_count} pages" if not result.error else f"ERROR: {result.error}")
        results.append(result)

    # Save extraction metadata
    meta_path = output_dir / "extraction_metadata.json"
    meta_path.write_text(
        json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False) + "\n",
    )

    return results
