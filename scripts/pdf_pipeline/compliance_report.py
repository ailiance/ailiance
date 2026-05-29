"""Generate a markdown compliance audit trail for the PDF pipeline.

Output: $AILIANCE/docs/pdf-compliance-report.md
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .config import HF_TRACED_DIR, PDF_EXTRACTED_DIR, PDF_RAW_DIR, PROJECT_ROOT


def _load_json(path: Path) -> dict | list:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def generate_report() -> str:
    """Build the compliance report as a markdown string."""
    lines: list[str] = []
    now = datetime.now(timezone.utc).isoformat()

    lines.append("# AILIANCE PDF Pipeline — Compliance Audit Trail")
    lines.append(f"\nGenerated: {now}\n")
    lines.append("## Legal Framework\n")
    lines.append("- **EU Digital Single Market Directive, Article 4**: Text and Data Mining")
    lines.append("  for research purposes is permitted when lawful access is available")
    lines.append("  and the rights holder has not expressly reserved TDM rights.")
    lines.append("- **EU AI Act**: Training data provenance must be documented and auditable.\n")

    # Robots check results
    robots_path = PDF_RAW_DIR / "robots_check_results.json"
    robots_results = _load_json(robots_path)

    lines.append("## Robots.txt Verification\n")
    if isinstance(robots_results, list):
        lines.append("| Source | Status | TDM Opt-Out | Checked At |")
        lines.append("|--------|--------|-------------|------------|")
        for r in robots_results:
            lines.append(
                f"| {r['source_name']} | {r['status']} | "
                f"{'Yes' if r['tdm_opt_out'] else 'No'} | {r['checked_at'][:10]} |"
            )
    else:
        lines.append("_No robots.txt check results found._\n")

    # Per-source details
    lines.append("\n## Source Details\n")

    source_dirs = sorted(PDF_RAW_DIR.iterdir()) if PDF_RAW_DIR.exists() else []
    total_pdfs = 0
    total_pairs = 0
    total_pii = 0

    for source_dir in source_dirs:
        if not source_dir.is_dir():
            continue

        manifest_path = source_dir / "manifest.json"
        if not manifest_path.exists():
            continue

        manifest = _load_json(manifest_path)
        source_name = manifest.get("source_name", source_dir.name)
        files = manifest.get("files", [])
        total_pdfs += len(files)

        lines.append(f"### {source_name}\n")
        lines.append(f"- **Legal basis**: {manifest.get('legal_basis', 'N/A')}")
        lines.append(f"- **License note**: {manifest.get('license_note', 'N/A')}")
        lines.append(f"- **Robots status**: {manifest.get('robots_status', 'N/A')}")
        lines.append(f"- **PDFs downloaded**: {len(files)}\n")

        if files:
            lines.append("| File | SHA-256 | Size | Date |")
            lines.append("|------|---------|------|------|")
            for f in files:
                sha_short = f.get("sha256", "")[:20] + "..."
                size_kb = f.get("file_size", 0) // 1024
                date = f.get("download_date", "")[:10]
                lines.append(f"| {f['filename']} | `{sha_short}` | {size_kb}KB | {date} |")

        # Extraction metadata
        ext_dir = PDF_EXTRACTED_DIR / source_dir.name
        ext_meta_path = ext_dir / "extraction_metadata.json"
        if ext_meta_path.exists():
            ext_meta = _load_json(ext_meta_path)
            lines.append(f"\n**Extraction**: {len(ext_meta)} files processed")
            for em in ext_meta:
                if em.get("error"):
                    lines.append(f"  - {em['source_file']}: ERROR — {em['error']}")

        lines.append("")

    # Training pairs summary
    lines.append("## Training Data Output\n")

    supplement_files = list(HF_TRACED_DIR.rglob("pdf_supplement.jsonl")) if HF_TRACED_DIR.exists() else []
    domain_counts: dict[str, int] = {}
    for sf in supplement_files:
        domain = sf.parent.name
        count = sum(1 for _ in sf.open())
        domain_counts[domain] = count
        total_pairs += count

    if domain_counts:
        lines.append("| Domain | Training Pairs |")
        lines.append("|--------|---------------|")
        for domain, count in sorted(domain_counts.items()):
            lines.append(f"| {domain} | {count} |")
    else:
        lines.append("_No training pairs generated yet._")

    # Overall summary
    lines.append("\n## Summary\n")
    lines.append(f"- **Total PDFs downloaded**: {total_pdfs}")
    lines.append(f"- **Total training pairs**: {total_pairs}")
    lines.append(f"- **PII findings**: {total_pii} (inline in JSONL `_pii_findings`)")
    lines.append(f"- **Report date**: {now[:10]}")

    return "\n".join(lines)


def save_report() -> Path:
    """Generate and save the compliance report."""
    docs_dir = PROJECT_ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    report_path = docs_dir / "pdf-compliance-report.md"
    report_path.write_text(generate_report() + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    report_path = save_report()
    print(f"Compliance report saved to {report_path}")


if __name__ == "__main__":
    main()
