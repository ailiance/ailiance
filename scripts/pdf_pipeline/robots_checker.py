"""Robots.txt and TDM opt-out verification for PDF sources.

Checks:
1. robots.txt rules for the target paths
2. TDM-specific opt-out headers (X-Robots-Tag: noai, noml)
3. Logs ALLOWED / BLOCKED / UNKNOWN per source
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

from .config import PDF_RAW_DIR, SOURCES, USER_AGENT, PdfSource


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class RobotsResult:
    source_name: str
    robots_url: str
    status: str  # ALLOWED | BLOCKED | UNKNOWN | NO_ROBOTS_TXT | ERROR
    tdm_opt_out: bool
    details: str
    checked_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    raw_robots: str = ""


# ---------------------------------------------------------------------------
# Checker
# ---------------------------------------------------------------------------

def _fetch_robots_txt(url: str) -> tuple[str, int]:
    """Fetch robots.txt content. Returns (body, status_code)."""
    try:
        resp = httpx.get(
            url,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
            timeout=15.0,
        )
        return resp.text, resp.status_code
    except httpx.HTTPError as exc:
        return str(exc), 0


def _check_tdm_headers(base_url: str) -> tuple[bool, str]:
    """HEAD request to detect TDM opt-out headers."""
    try:
        resp = httpx.head(
            base_url,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
            timeout=15.0,
        )
        headers_lower = {k.lower(): v.lower() for k, v in resp.headers.items()}

        x_robots = headers_lower.get("x-robots-tag", "")
        tdm_reservation = headers_lower.get("tdm-reservation", "")

        opt_out_signals: list[str] = []
        if "noai" in x_robots or "noml" in x_robots:
            opt_out_signals.append(f"X-Robots-Tag: {x_robots}")
        if tdm_reservation:
            opt_out_signals.append(f"TDM-Reservation: {tdm_reservation}")

        if opt_out_signals:
            return True, "; ".join(opt_out_signals)
        return False, "No TDM opt-out headers found"
    except httpx.HTTPError as exc:
        return False, f"Header check failed: {exc}"


def check_source(source: PdfSource) -> RobotsResult:
    """Check robots.txt and TDM opt-out for a single source."""
    if not source.robots_txt:
        tdm_out, tdm_detail = _check_tdm_headers(source.base_url)
        return RobotsResult(
            source_name=source.name,
            robots_url="",
            status="BLOCKED" if tdm_out else "ALLOWED",
            tdm_opt_out=tdm_out,
            details=f"No robots.txt configured. {tdm_detail}",
        )

    body, status_code = _fetch_robots_txt(source.robots_txt)

    if status_code == 0:
        return RobotsResult(
            source_name=source.name,
            robots_url=source.robots_txt,
            status="ERROR",
            tdm_opt_out=False,
            details=f"Failed to fetch robots.txt: {body}",
        )

    if status_code == 404:
        tdm_out, tdm_detail = _check_tdm_headers(source.base_url)
        return RobotsResult(
            source_name=source.name,
            robots_url=source.robots_txt,
            status="BLOCKED" if tdm_out else "ALLOWED",
            tdm_opt_out=tdm_out,
            details=f"robots.txt not found (404). {tdm_detail}",
            raw_robots="",
        )

    # Parse robots.txt
    rp = RobotFileParser()
    rp.parse(body.splitlines())

    parsed_base = urlparse(source.base_url)
    path = parsed_base.path or "/"
    allowed = rp.can_fetch(USER_AGENT, path)

    # Also check with wildcard user-agent
    allowed_star = rp.can_fetch("*", path)

    # Check TDM headers
    tdm_out, tdm_detail = _check_tdm_headers(source.base_url)

    if tdm_out:
        status = "BLOCKED"
        details = f"TDM opt-out detected: {tdm_detail}"
    elif not allowed and not allowed_star:
        status = "BLOCKED"
        details = f"robots.txt disallows crawling path {path!r}"
    elif allowed or allowed_star:
        status = "ALLOWED"
        details = f"robots.txt allows path {path!r}. {tdm_detail}"
    else:
        status = "UNKNOWN"
        details = f"Ambiguous robots.txt for path {path!r}. {tdm_detail}"

    return RobotsResult(
        source_name=source.name,
        robots_url=source.robots_txt,
        status=status,
        tdm_opt_out=tdm_out,
        details=details,
        raw_robots=body[:2000],
    )


def check_all() -> list[RobotsResult]:
    """Check all registered sources."""
    results: list[RobotsResult] = []
    for source in SOURCES:
        print(f"  Checking {source.name}...", end=" ", flush=True)
        result = check_source(source)
        print(result.status)
        results.append(result)
        time.sleep(1)
    return results


def save_results(results: list[RobotsResult]) -> Path:
    """Persist results to JSON."""
    out_dir = PDF_RAW_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "robots_check_results.json"
    payload = [asdict(r) for r in results]
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Check robots.txt for PDF sources")
    parser.add_argument("--all", action="store_true", help="Check all sources")
    parser.add_argument("--source", type=str, help="Check a specific source by name")
    args = parser.parse_args(argv)

    if not args.all and not args.source:
        parser.print_help()
        sys.exit(1)

    print("=== AILIANCE PDF Pipeline — Robots.txt Verification ===\n")

    if args.all:
        results = check_all()
    else:
        from .config import get_source
        src = get_source(args.source)
        print(f"  Checking {src.name}...", end=" ", flush=True)
        result = check_source(src)
        print(result.status)
        results = [result]

    # Summary
    print("\n--- Summary ---")
    for r in results:
        icon = {"ALLOWED": "+", "BLOCKED": "X", "UNKNOWN": "?", "ERROR": "!"}
        print(f"  [{icon.get(r.status, '?')}] {r.source_name}: {r.status}")
        print(f"      {r.details}")
        if r.tdm_opt_out:
            print(f"      *** TDM OPT-OUT DETECTED — DO NOT SCRAPE ***")

    out_path = save_results(results)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
