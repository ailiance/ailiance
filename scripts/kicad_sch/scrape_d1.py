"""D1: scrape .kicad_sch from GitHub, license-filter, normalize, dedupe.

Output: hash-named files in ``~/eu-kiki-data/kicad-sch-scraped/`` plus
an Annex-IV manifest (D1 split) and an NDJSON audit log.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from scripts.kicad_sch.audit_log import AuditLogger
from scripts.kicad_sch.manifest import DatasetManifest

_UUID_RE = re.compile(
    r'"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"',
    re.IGNORECASE,
)

# Default keyword shards: gh search code caps at 1000 hits per query, so
# we fan out across hardware-relevant keywords to grow the corpus.
DEFAULT_QUERIES = [
    "extension:kicad_sch resistor",
    "extension:kicad_sch capacitor",
    "extension:kicad_sch led",
    "extension:kicad_sch transistor",
    "extension:kicad_sch power",
    "extension:kicad_sch microcontroller",
    "extension:kicad_sch sensor",
    "extension:kicad_sch amplifier",
    "extension:kicad_sch filter",
    "extension:kicad_sch oscillator",
    "extension:kicad_sch motor",
    "extension:kicad_sch battery",
    "extension:kicad_sch usb",
    "extension:kicad_sch esp32",
    "extension:kicad_sch stm32",
    "extension:kicad_sch arduino",
    "extension:kicad_sch raspberry",
    "extension:kicad_sch audio",
    "extension:kicad_sch power supply",
    "extension:kicad_sch analog",
]

# Hardware-community canonical license set: CERN-OHL dominates KiCad
# projects; without it we starve D1.
DEFAULT_LICENSE_ALLOWLIST = ",".join(
    [
        "MIT",
        "Apache-2.0",
        "CC0-1.0",
        "GPL-3.0",
        "CERN-OHL-S-2.0",
        "CERN-OHL-P-2.0",
        "CERN-OHL-W-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "ISC",
        "Unlicense",
        "GPL-2.0",
        "LGPL-3.0",
        "LGPL-2.1",
        "MPL-2.0",
        "CC-BY-4.0",
        "CC-BY-SA-4.0",
    ]
)


def canonical_hash(text: str) -> str:
    """Hash schematic text after UUID normalization (placement-stable)."""
    canon = _UUID_RE.sub(
        '"00000000-0000-0000-0000-000000000000"', text
    )
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def license_allowed(spdx: str | None, allow: set[str]) -> bool:
    """Case-insensitive SPDX membership test."""
    if not spdx:
        return False
    norm = {x.upper() for x in allow}
    return spdx.upper() in norm


def _fetch_raw(url: str) -> str:
    # gh search code returns blob HTML URLs; rewrite to raw.githubusercontent.com.
    raw_url = url.replace(
        "https://github.com/", "https://raw.githubusercontent.com/"
    ).replace("/blob/", "/")
    r = subprocess.run(
        ["curl", "-fsSL", raw_url],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if r.returncode != 0:
        raise RuntimeError(f"fetch failed: {url}")
    return r.stdout


def _kicad_update(path: Path) -> int:
    # KiCad 10.0.x: subcommand is `upgrade` (not `update`).
    r = subprocess.run(
        ["kicad-cli", "sch", "upgrade", str(path)],
        capture_output=True,
        timeout=120,
    )
    return r.returncode


def _gh_search_one(query: str, max_files: int) -> list[dict]:
    r = subprocess.run(
        [
            "gh",
            "search",
            "code",
            query,
            "--limit",
            str(max_files),
            "--json",
            "repository,path,url,sha",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if r.returncode != 0:
        return []
    return json.loads(r.stdout or "[]")


def _gh_search(queries: list[str], max_files: int) -> list[dict]:
    """Fan-out search across keyword shards and dedupe by url."""
    per_query_cap = min(max_files, 1000)
    seen: set[str] = set()
    out: list[dict] = []
    for q in queries:
        if len(out) >= max_files:
            break
        hits = _gh_search_one(q, per_query_cap)
        for h in hits:
            key = h.get("url") or ""
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(h)
            if len(out) >= max_files:
                break
    return out


def _repo_license(name_with_owner: str) -> str | None:
    # REST API returns spdx_id; GraphQL/gh repo view --json licenseInfo lacks it.
    r = subprocess.run(
        ["gh", "api", f"repos/{name_with_owner}", "--jq", ".license.spdx_id"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if r.returncode != 0:
        return None
    val = (r.stdout or "").strip()
    if not val or val == "null":
        return None
    return val


def download_and_normalize(
    repo: str,
    path: str,
    url: str,
    commit: str,
    license_spdx: str,
    out_dir: Path,
) -> Path | None:
    """Fetch a raw file, run kicad-cli upgrade, dedupe-rename.

    Returns the final ``Path`` or ``None`` when kicad-cli rejects the
    input (parse error on legacy v5/v6 schemas).
    """
    raw = _fetch_raw(url)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f".tmp-{os.getpid()}.kicad_sch"
    tmp.write_text(raw, encoding="utf-8")
    if _kicad_update(tmp) != 0:
        tmp.unlink(missing_ok=True)
        return None
    text = tmp.read_text(encoding="utf-8")
    h = canonical_hash(text)
    final = out_dir / f"{h}.kicad_sch"
    if final.exists():
        tmp.unlink(missing_ok=True)
        return final
    tmp.rename(final)
    return final


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-files", type=int, default=10000)
    p.add_argument(
        "--license-allowlist",
        default=DEFAULT_LICENSE_ALLOWLIST,
        help="Comma-separated SPDX IDs to accept.",
    )
    p.add_argument(
        "--queries",
        default=",".join(DEFAULT_QUERIES),
        help=(
            "Comma-separated gh-search-code queries. Each query is "
            "capped at 1000 hits by gh; shard across keywords to grow "
            "the corpus."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path.home() / "eu-kiki-data/kicad-sch-scraped",
    )
    p.add_argument(
        "--audit-dir",
        type=Path,
        default=Path.home() / "eu-kiki/output/audit/kicad-sch-2026-05-11",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path.home() / "eu-kiki/output/audit/kicad-sch-2026-05-11/d1_manifest.csv",
    )
    a = p.parse_args(argv)
    allow = set(a.license_allowlist.split(","))
    queries = [q.strip() for q in a.queries.split(",") if q.strip()]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    audit_path = a.audit_dir / f"d1-{run_stamp}.ndjson"
    log = AuditLogger(audit_path)
    manifest = DatasetManifest(a.manifest, split="D1")
    hits = _gh_search(queries, a.max_files)
    log.log("d1_search_done", n=len(hits), n_queries=len(queries))
    n_ok = 0
    for h in hits:
        repo = h["repository"]["nameWithOwner"]
        spdx = _repo_license(repo)
        if not license_allowed(spdx, allow):
            log.log("d1_license_skip", repo=repo, spdx=spdx)
            continue
        try:
            out = download_and_normalize(
                repo=repo,
                path=h["path"],
                url=h["url"],
                commit=h.get("sha", ""),
                license_spdx=spdx,
                out_dir=a.out_dir,
            )
        except Exception as exc:
            log.log("d1_fetch_fail", repo=repo, err=str(exc))
            continue
        if out is None:
            log.log("d1_update_fail", repo=repo, path=h["path"])
            continue
        manifest.add(
            source_type="github_scrape",
            source_url=h["url"],
            commit_sha=h.get("sha", ""),
            license_spdx=spdx or "",
            dedup_hash=out.stem,
            file_size_bytes=out.stat().st_size,
            kicad_version_before="unknown",
            kicad_version_after="10.0.2",
        )
        n_ok += 1
    manifest.write()
    log.log("d1_done", accepted=n_ok)
    print(f"D1: {n_ok} files written to {a.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
