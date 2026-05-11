"""D1 v3: repo-clone scraper for ``.kicad_sch`` files.

Code-search D1 (PR #45/#46) caps at ~33 unique URLs because GitHub
does not tokenize ``.kicad_sch`` s-expression bodies. This module
pivots to repo discovery via ``gh search repos`` + shallow ``git
clone`` + filesystem walk, lifting the ceiling to thousands of real
schemas across hardware-licensed repos.

Pipeline:

1. ``gh search repos`` sharded by topic (kicad, kicad-pcb, eda, ...).
2. Repo-level license filter via the JSON metadata.
3. Shallow ``git clone --depth 1`` into a transient temp dir.
4. Walk for ``*.kicad_sch``.
5. ``kicad-cli sch upgrade`` to v10 (skip on parse fail).
6. SHA256 UUID-stripped hash dedupe against existing D1 manifest.
7. Append rows to manifest CSV + NDJSON audit trail.
8. ``rm -rf`` per-repo clone to bound disk usage.

The repo-level license is *inferred* and recorded as such in the
audit log (``license_inferred_from: repo_metadata``); per-file
vendored licences are not introspected.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from scripts.kicad_sch.audit_log import AuditLogger
from scripts.kicad_sch.manifest import DatasetManifest, FIELDNAMES
from scripts.kicad_sch.scrape_d1 import (
    canonical_hash,
    license_allowed,
    DEFAULT_LICENSE_ALLOWLIST,
)

DEFAULT_TOPIC_SHARDS = [
    "kicad",
    "kicad-pcb",
    "kicad-project",
    "kicad-library",
    "pcb-design",
    "open-hardware",
    "open-source-hardware",
    "eda",
    "schematic",
    "electronics",
    "hardware",
]


def _gh_search_repos(
    topic: str, license_csv: str, limit: int
) -> list[dict]:
    """Return repo dicts (fullName, license, defaultBranch, ...) for one topic."""
    r = subprocess.run(
        [
            "gh",
            "search",
            "repos",
            "--topic",
            topic,
            "--license",
            license_csv,
            "--limit",
            str(limit),
            "--json",
            "fullName,license,defaultBranch,stargazersCount,description",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if r.returncode != 0:
        return []
    try:
        return json.loads(r.stdout or "[]")
    except json.JSONDecodeError:
        return []


def discover_repos(
    topics: list[str], license_csv: str, per_topic: int
) -> list[dict]:
    """Fan-out across topics, dedupe by ``fullName``."""
    seen: set[str] = set()
    out: list[dict] = []
    for topic in topics:
        for r in _gh_search_repos(topic, license_csv, per_topic):
            name = r.get("fullName") or ""
            if not name or name in seen:
                continue
            seen.add(name)
            out.append(r)
    return out


def _sanitize(full_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", full_name)


def shallow_clone(full_name: str, dest: Path, timeout: int = 300) -> bool:
    """Run ``git clone --depth 1`` into ``dest``. Return True on success."""
    url = f"https://github.com/{full_name}.git"
    r = subprocess.run(
        ["git", "clone", "--depth", "1", "--quiet", url, str(dest)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return r.returncode == 0


def _git_head_sha(repo_dir: Path) -> str:
    r = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    return (r.stdout or "").strip()


def _kicad_upgrade(path: Path) -> int:
    r = subprocess.run(
        ["kicad-cli", "sch", "upgrade", str(path)],
        capture_output=True,
        timeout=120,
    )
    return r.returncode


def walk_kicad_sch(repo_dir: Path) -> list[Path]:
    """Return all ``*.kicad_sch`` files under ``repo_dir``."""
    return [p for p in repo_dir.rglob("*.kicad_sch") if p.is_file()]


def load_existing_hashes(manifest_path: Path) -> set[str]:
    """Read prior dedup_hash values from an existing manifest CSV, if any."""
    if not manifest_path.exists():
        return set()
    seen: set[str] = set()
    with manifest_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            h = (row.get("dedup_hash") or "").strip()
            if h:
                seen.add(h)
    return seen


def load_existing_rows(manifest_path: Path) -> list[dict]:
    """Read all prior manifest rows so we can rewrite without data loss."""
    if not manifest_path.exists():
        return []
    with manifest_path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_manifest(manifest_path: Path, rows: list[dict]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(FIELDNAMES))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDNAMES})


def process_repo(
    repo: dict,
    clone_root: Path,
    out_dir: Path,
    seen_hashes: set[str],
    log: AuditLogger,
    max_files_per_repo: int,
) -> list[dict]:
    """Clone one repo, harvest, normalize. Return manifest rows added."""
    full_name = repo["fullName"]
    spdx = ((repo.get("license") or {}).get("key") or "").upper()
    # gh search repos returns license.key like 'mit', 'apache-2.0'; normalize
    # for comparison against SPDX upper-case.
    spdx_norm_map = {
        "MIT": "MIT",
        "APACHE-2.0": "Apache-2.0",
        "CC0-1.0": "CC0-1.0",
        "GPL-3.0": "GPL-3.0",
        "GPL-2.0": "GPL-2.0",
        "LGPL-3.0": "LGPL-3.0",
        "LGPL-2.1": "LGPL-2.1",
        "BSD-2-CLAUSE": "BSD-2-Clause",
        "BSD-3-CLAUSE": "BSD-3-Clause",
        "ISC": "ISC",
        "UNLICENSE": "Unlicense",
        "MPL-2.0": "MPL-2.0",
        "CC-BY-4.0": "CC-BY-4.0",
        "CC-BY-SA-4.0": "CC-BY-SA-4.0",
        "CERN-OHL-S-2.0": "CERN-OHL-S-2.0",
        "CERN-OHL-P-2.0": "CERN-OHL-P-2.0",
        "CERN-OHL-W-2.0": "CERN-OHL-W-2.0",
    }
    spdx_canon = spdx_norm_map.get(spdx, spdx) or ""

    safe = _sanitize(full_name)
    dest = clone_root / safe
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)

    if not shallow_clone(full_name, dest):
        log.log("repo_clone_fail", repo=full_name)
        shutil.rmtree(dest, ignore_errors=True)
        return []

    log.log("repo_clone_ok", repo=full_name, spdx=spdx_canon)
    head = _git_head_sha(dest)
    rows: list[dict] = []
    try:
        files = walk_kicad_sch(dest)
        for src in files[:max_files_per_repo]:
            try:
                rel = src.relative_to(dest)
            except ValueError:
                continue
            try:
                rc = _kicad_upgrade(src)
            except subprocess.TimeoutExpired:
                log.log(
                    "file_skipped",
                    repo=full_name,
                    path=str(rel),
                    reason="kicad_cli_timeout",
                )
                continue
            if rc != 0:
                log.log(
                    "file_skipped",
                    repo=full_name,
                    path=str(rel),
                    reason="kicad_cli_upgrade_failed",
                )
                continue
            try:
                text = src.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                log.log(
                    "file_skipped",
                    repo=full_name,
                    path=str(rel),
                    reason=f"read_err:{exc}",
                )
                continue
            h = canonical_hash(text)
            if h in seen_hashes:
                log.log(
                    "file_skipped",
                    repo=full_name,
                    path=str(rel),
                    reason="dedup_hash_collision",
                    dedup_hash=h,
                )
                continue
            seen_hashes.add(h)
            final = out_dir / f"{h}.kicad_sch"
            try:
                final.write_text(text, encoding="utf-8")
            except OSError as exc:
                log.log(
                    "file_skipped",
                    repo=full_name,
                    path=str(rel),
                    reason=f"write_err:{exc}",
                )
                continue
            size = final.stat().st_size
            row = {
                "source_type": "github_repo_clone",
                "source_url": (
                    f"https://github.com/{full_name}/blob/{head}/{rel}"
                ),
                "commit_sha": head,
                "license_spdx": spdx_canon,
                "dedup_hash": h,
                "file_size_bytes": size,
                "kicad_version_before": "unknown",
                "kicad_version_after": "10.0.2",
            }
            rows.append(row)
            log.log(
                "file_accepted",
                repo=full_name,
                path=str(rel),
                dedup_hash=h,
                bytes=size,
                license_inferred_from="repo_metadata",
            )
    finally:
        shutil.rmtree(dest, ignore_errors=True)
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-repos", type=int, default=1000)
    p.add_argument("--max-files", type=int, default=5000)
    p.add_argument("--max-files-per-repo", type=int, default=200)
    p.add_argument("--per-topic-limit", type=int, default=200)
    p.add_argument(
        "--topics", default=",".join(DEFAULT_TOPIC_SHARDS)
    )
    p.add_argument(
        "--license-allowlist",
        default=DEFAULT_LICENSE_ALLOWLIST,
    )
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path.home() / "eu-kiki-data/kicad-sch-scraped",
    )
    p.add_argument(
        "--clone-root",
        type=Path,
        default=Path(tempfile.gettempdir()) / "d1_repos",
    )
    p.add_argument(
        "--audit-dir",
        type=Path,
        default=Path.home() / "eu-kiki/output/audit/kicad-sch-2026-05-11",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path.home()
        / "eu-kiki/output/audit/kicad-sch-2026-05-11/d1_manifest.csv",
    )
    a = p.parse_args(argv)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    a.clone_root.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    audit_path = a.audit_dir / f"d1-repos-{run_stamp}.ndjson"
    log = AuditLogger(audit_path)

    topics = [t.strip() for t in a.topics.split(",") if t.strip()]
    log.log("repo_search_start", topics=topics, license=a.license_allowlist)
    repos = discover_repos(topics, a.license_allowlist, a.per_topic_limit)
    repos = repos[: a.max_repos]
    log.log("repo_search_done", n_repos=len(repos))

    # Preserve prior rows + hashes so we extend, not overwrite.
    prior_rows = load_existing_rows(a.manifest)
    seen_hashes = {r.get("dedup_hash", "") for r in prior_rows if r.get("dedup_hash")}
    # Also fold any files already on disk into the dedupe set.
    for p_ in a.out_dir.glob("*.kicad_sch"):
        seen_hashes.add(p_.stem)
    log.log("dedup_init", n_prior_rows=len(prior_rows), n_prior_hashes=len(seen_hashes))

    new_rows: list[dict] = []
    files_written = 0
    write_lock_rows: list[dict] = []

    def _worker(r: dict) -> list[dict]:
        return process_repo(
            r,
            a.clone_root,
            a.out_dir,
            seen_hashes,
            log,
            a.max_files_per_repo,
        )

    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        futs = {ex.submit(_worker, r): r for r in repos}
        for fut in as_completed(futs):
            try:
                rows = fut.result()
            except Exception as exc:  # noqa: BLE001
                repo_name = futs[fut].get("fullName", "?")
                log.log("repo_worker_error", repo=repo_name, err=str(exc))
                continue
            new_rows.extend(rows)
            files_written += len(rows)
            if files_written >= a.max_files:
                break

    all_rows = prior_rows + new_rows
    write_manifest(a.manifest, all_rows)
    log.log(
        "manifest_written",
        path=str(a.manifest),
        n_prior=len(prior_rows),
        n_new=len(new_rows),
        n_total=len(all_rows),
    )
    print(
        f"D1 v3: {len(new_rows)} new files (total {len(all_rows)}) "
        f"written to {a.out_dir}"
    )
    # final cleanup of clone root
    shutil.rmtree(a.clone_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
