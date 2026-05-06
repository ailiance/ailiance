"""Main orchestrator for the PDF scraping and conversion pipeline.

Usage:
    uv run python -m scripts.pdf_pipeline.pipeline --source "ST Application Notes" --max-pdfs 3
    uv run python -m scripts.pdf_pipeline.pipeline --all --max-pdfs-per-source 5
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

from .compliance_report import save_report
from .config import SOURCES, PdfSource, get_source
from .pdf_downloader import download_source
from .pdf_extractor import extract_source
from .pdf_to_training import convert_source
from .robots_checker import check_source


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _safe_dir_name(source: PdfSource) -> str:
    return source.name.lower().replace(" ", "_")


def run_source(source: PdfSource, max_pdfs: int = 20) -> dict[str, int | str]:
    """Run the full pipeline for a single source."""
    print(f"\n{'='*60}")
    print(f"  Source: {source.name}")
    print(f"  Legal basis: {source.legal_basis}")
    print(f"  Domains: {', '.join(source.domains)}")
    print(f"{'='*60}\n")

    # Step 1: Verify robots.txt
    print("[1/5] Checking robots.txt and TDM opt-out...")
    robots = check_source(source)
    print(f"  Status: {robots.status}")
    print(f"  Details: {robots.details}")

    if robots.status == "BLOCKED":
        print(f"\n  *** BLOCKED — skipping {source.name} ***")
        return {
            "source": source.name,
            "status": "BLOCKED",
            "reason": robots.details,
            "pdfs": 0,
            "pairs": 0,
        }

    # Step 2: Download PDFs
    print(f"\n[2/5] Downloading PDFs (max {max_pdfs})...")
    urls = list(source.example_urls)
    if not urls:
        print("  No example URLs configured — skipping download.")
        return {
            "source": source.name,
            "status": "NO_URLS",
            "reason": "No example_urls defined in config",
            "pdfs": 0,
            "pairs": 0,
        }

    manifest = download_source(
        source=source,
        urls=urls,
        robots_status=robots.status,
        max_pdfs=max_pdfs,
    )
    pdf_count = len(manifest.files)
    print(f"  Downloaded {pdf_count} PDFs")

    if pdf_count == 0:
        return {
            "source": source.name,
            "status": "NO_PDFS",
            "pdfs": 0,
            "pairs": 0,
        }

    # Step 3: Extract text
    safe_name = _safe_dir_name(source)
    print(f"\n[3/5] Extracting text...")
    extraction_results = extract_source(safe_name)
    successful = [r for r in extraction_results if not r.error]
    print(f"  Extracted {len(successful)}/{len(extraction_results)} PDFs")

    # Step 4: Convert to training pairs
    print(f"\n[4/5] Generating training pairs...")
    now = datetime.now(timezone.utc).isoformat()
    stats = convert_source(
        source_name_safe=safe_name,
        source_name=source.name,
        domains=source.domains,
        legal_basis=source.legal_basis,
        robots_status=robots.status,
        robots_check_date=robots.checked_at,
        download_date=now,
    )
    print(f"  Generated {stats['pairs']} training pairs")
    if stats["pii_findings"]:
        print(f"  PII findings: {stats['pii_findings']}")

    # Step 5: Update compliance report
    print(f"\n[5/5] Updating compliance report...")
    report_path = save_report()
    print(f"  Report: {report_path}")

    return {
        "source": source.name,
        "status": "OK",
        "pdfs": pdf_count,
        "pairs": stats["pairs"],
        "pii_findings": stats["pii_findings"],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="AILIANCE PDF Pipeline — scrape, extract, convert with DSM compliance",
    )
    parser.add_argument("--source", type=str, help="Process a specific source by name")
    parser.add_argument("--all", action="store_true", help="Process all sources")
    parser.add_argument("--max-pdfs", type=int, default=20, help="Max PDFs per source (default: 20)")
    parser.add_argument("--max-pdfs-per-source", type=int, help="Alias for --max-pdfs when using --all")
    args = parser.parse_args(argv)

    if not args.all and not args.source:
        parser.print_help()
        sys.exit(1)

    max_pdfs = args.max_pdfs_per_source or args.max_pdfs

    print("=" * 60)
    print("  AILIANCE PDF Pipeline v0.2")
    print("  EU AI Act & DSM Directive Compliant")
    print("=" * 60)

    if args.all:
        sources = list(SOURCES)
    else:
        sources = [get_source(args.source)]

    results: list[dict] = []
    for source in sources:
        result = run_source(source, max_pdfs=max_pdfs)
        results.append(result)

    # Final summary
    print("\n" + "=" * 60)
    print("  PIPELINE SUMMARY")
    print("=" * 60)
    total_pdfs = sum(r.get("pdfs", 0) for r in results)
    total_pairs = sum(r.get("pairs", 0) for r in results)

    for r in results:
        status_icon = {"OK": "+", "BLOCKED": "X", "NO_URLS": "-", "NO_PDFS": "-"}
        icon = status_icon.get(str(r["status"]), "?")
        print(f"  [{icon}] {r['source']}: {r['status']} — {r.get('pdfs', 0)} PDFs, {r.get('pairs', 0)} pairs")

    print(f"\n  Total: {total_pdfs} PDFs, {total_pairs} training pairs")
    print("=" * 60)


if __name__ == "__main__":
    main()
