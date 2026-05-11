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
    r = subprocess.run(
        ["curl", "-fsSL", url],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if r.returncode != 0:
        raise RuntimeError(f"fetch failed: {url}")
    return r.stdout


def _kicad_update(path: Path) -> int:
    r = subprocess.run(
        ["kicad-cli", "sch", "update", str(path)],
        capture_output=True,
        timeout=120,
    )
    return r.returncode


def _gh_search(max_files: int) -> list[dict]:
    r = subprocess.run(
        [
            "gh",
            "search",
            "code",
            "extension:kicad_sch",
            "--limit",
            str(max_files),
            "--json",
            "repository,path,url,sha",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if r.returncode != 0:
        return []
    return json.loads(r.stdout or "[]")


def _repo_license(name_with_owner: str) -> str | None:
    r = subprocess.run(
        ["gh", "repo", "view", name_with_owner, "--json", "licenseInfo"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if r.returncode != 0:
        return None
    info = json.loads(r.stdout or "{}").get("licenseInfo") or {}
    return info.get("spdxId")


def download_and_normalize(
    repo: str,
    path: str,
    url: str,
    commit: str,
    license_spdx: str,
    out_dir: Path,
) -> Path | None:
    """Fetch a raw file, run kicad-cli update, dedupe-rename.

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
        default="MIT,Apache-2.0,CC0-1.0,GPL-3.0",
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
    a.out_dir.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    audit_path = a.audit_dir / f"d1-{run_stamp}.ndjson"
    log = AuditLogger(audit_path)
    manifest = DatasetManifest(a.manifest, split="D1")
    hits = _gh_search(a.max_files)
    log.log("d1_search_done", n=len(hits))
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
