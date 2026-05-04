"""PDF downloader with full traceability and rate limiting.

Downloads PDFs to data/pdf-raw/{source}/ with a manifest.json
recording URL, timestamp, hash, HTTP headers, and legal basis.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

from .config import PDF_RAW_DIR, RATE_LIMIT_SECONDS, USER_AGENT, PdfSource


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class DownloadRecord:
    url: str
    filename: str
    sha256: str
    file_size: int
    download_date: str
    http_status: int
    content_type: str
    legal_basis: str
    license_note: str
    robots_status: str
    source_name: str


@dataclass
class DownloadManifest:
    source_name: str
    legal_basis: str
    license_note: str
    robots_status: str
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    files: list[DownloadRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------

def _source_dir(source: PdfSource) -> Path:
    """Derive a safe directory name from the source name."""
    safe = source.name.lower().replace(" ", "_")
    return PDF_RAW_DIR / safe


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download_pdf(
    url: str,
    dest_dir: Path,
) -> tuple[Path | None, int, str, str]:
    """Download a single PDF. Returns (path, status, content_type, sha256)."""
    try:
        resp = httpx.get(
            url,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
            timeout=60.0,
        )
    except httpx.HTTPError as exc:
        print(f"    ERROR downloading {url}: {exc}")
        return None, 0, "", ""

    if resp.status_code != 200:
        print(f"    HTTP {resp.status_code} for {url}")
        return None, resp.status_code, "", ""

    content_type = resp.headers.get("content-type", "")
    if "pdf" not in content_type.lower() and not url.lower().endswith(".pdf"):
        print(f"    WARNING: Content-Type is {content_type!r}, may not be PDF")

    # Derive filename from URL
    filename = url.rstrip("/").split("/")[-1]
    if not filename.endswith(".pdf"):
        filename += ".pdf"

    dest = dest_dir / filename
    pdf_bytes = resp.content
    dest.write_bytes(pdf_bytes)

    file_hash = _sha256(pdf_bytes)
    return dest, resp.status_code, content_type, file_hash


def _load_manifest(manifest_path: Path) -> DownloadManifest | None:
    """Load existing manifest if present."""
    if not manifest_path.exists():
        return None
    raw = json.loads(manifest_path.read_text())
    files = [DownloadRecord(**f) for f in raw.get("files", [])]
    return DownloadManifest(
        source_name=raw["source_name"],
        legal_basis=raw["legal_basis"],
        license_note=raw.get("license_note", ""),
        robots_status=raw.get("robots_status", "UNKNOWN"),
        created_at=raw.get("created_at", ""),
        files=files,
    )


def _save_manifest(manifest: DownloadManifest, manifest_path: Path) -> None:
    data = asdict(manifest)
    manifest_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def download_source(
    source: PdfSource,
    urls: list[str],
    robots_status: str = "UNKNOWN",
    max_pdfs: int = 50,
) -> DownloadManifest:
    """Download PDFs for a source, respecting rate limits."""
    dest_dir = _source_dir(source)
    dest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dest_dir / "manifest.json"

    manifest = _load_manifest(manifest_path) or DownloadManifest(
        source_name=source.name,
        legal_basis=source.legal_basis,
        license_note=source.license_note,
        robots_status=robots_status,
    )
    manifest.robots_status = robots_status

    existing_urls = {r.url for r in manifest.files}
    to_download = [u for u in urls if u not in existing_urls][:max_pdfs]

    if not to_download:
        print(f"  No new PDFs to download for {source.name}")
        return manifest

    print(f"  Downloading {len(to_download)} PDFs for {source.name}...")

    for i, url in enumerate(to_download, 1):
        print(f"    [{i}/{len(to_download)}] {url.split('/')[-1]}")
        dest, status, ct, sha = download_pdf(url, dest_dir)

        if dest is not None:
            record = DownloadRecord(
                url=url,
                filename=dest.name,
                sha256=f"sha256:{sha}",
                file_size=dest.stat().st_size,
                download_date=datetime.now(timezone.utc).isoformat(),
                http_status=status,
                content_type=ct,
                legal_basis=source.legal_basis,
                license_note=source.license_note,
                robots_status=robots_status,
                source_name=source.name,
            )
            manifest.files.append(record)

        _save_manifest(manifest, manifest_path)

        if i < len(to_download):
            time.sleep(RATE_LIMIT_SECONDS)

    return manifest
