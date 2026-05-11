"""Tests for scripts.kicad_sch.scrape_d1_repos (D1 v3 repo-clone)."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.kicad_sch import scrape_d1_repos
from scripts.kicad_sch.audit_log import AuditLogger
from scripts.kicad_sch.scrape_d1_repos import (
    discover_repos,
    load_existing_hashes,
    load_existing_rows,
    process_repo,
    walk_kicad_sch,
    write_manifest,
)


_SCH = (
    '(kicad_sch (version 20240101) (generator eeschema)\n'
    '  (uuid "11111111-2222-3333-4444-555555555555"))\n'
)


def test_walk_kicad_sch_finds_nested(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.kicad_sch").write_text(_SCH)
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "y.kicad_sch").write_text(_SCH)
    (tmp_path / "ignore.txt").write_text("nope")
    out = walk_kicad_sch(tmp_path)
    names = sorted(p.name for p in out)
    assert names == ["x.kicad_sch", "y.kicad_sch"]


def test_load_existing_hashes_empty(tmp_path: Path) -> None:
    assert load_existing_hashes(tmp_path / "missing.csv") == set()


def test_load_and_write_manifest_roundtrip(tmp_path: Path) -> None:
    manifest = tmp_path / "d1_manifest.csv"
    rows = [
        {
            "source_type": "github_repo_clone",
            "source_url": "https://github.com/foo/bar/blob/abc/x.kicad_sch",
            "commit_sha": "abc",
            "license_spdx": "MIT",
            "dedup_hash": "deadbeef",
            "file_size_bytes": 123,
            "kicad_version_before": "unknown",
            "kicad_version_after": "10.0.2",
        }
    ]
    write_manifest(manifest, rows)
    read = load_existing_rows(manifest)
    assert len(read) == 1
    assert read[0]["dedup_hash"] == "deadbeef"
    assert load_existing_hashes(manifest) == {"deadbeef"}


def test_discover_repos_dedupes_across_topics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake(topic: str, license_csv: str, limit: int) -> list[dict]:
        calls.append(topic)
        if topic == "kicad":
            return [{"fullName": "a/x"}, {"fullName": "a/y"}]
        if topic == "eda":
            return [{"fullName": "a/y"}, {"fullName": "a/z"}]
        return []

    monkeypatch.setattr(scrape_d1_repos, "_gh_search_repos", fake)
    out = discover_repos(["kicad", "eda"], "MIT", 200)
    assert [r["fullName"] for r in out] == ["a/x", "a/y", "a/z"]
    assert calls == ["kicad", "eda"]


def test_process_repo_writes_row_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clone_root = tmp_path / "clones"
    out_dir = tmp_path / "out"
    clone_root.mkdir()
    out_dir.mkdir()
    audit = tmp_path / "audit.ndjson"
    log = AuditLogger(audit)

    def fake_clone(full_name: str, dest: Path, timeout: int = 300) -> bool:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "schematic.kicad_sch").write_text(_SCH)
        (dest / "README.md").write_text("# repo")
        return True

    monkeypatch.setattr(scrape_d1_repos, "shallow_clone", fake_clone)
    monkeypatch.setattr(
        scrape_d1_repos, "_git_head_sha", lambda d: "feedface"
    )
    monkeypatch.setattr(
        scrape_d1_repos, "_kicad_upgrade", lambda p: 0
    )

    seen: set[str] = set()
    rows = process_repo(
        repo={"fullName": "foo/bar", "license": {"key": "mit"}},
        clone_root=clone_root,
        out_dir=out_dir,
        seen_hashes=seen,
        log=log,
        max_files_per_repo=10,
    )
    assert len(rows) == 1
    r = rows[0]
    assert r["source_type"] == "github_repo_clone"
    assert r["commit_sha"] == "feedface"
    assert r["license_spdx"] == "MIT"
    assert r["dedup_hash"] in seen
    assert r["source_url"].startswith(
        "https://github.com/foo/bar/blob/feedface/"
    )
    # File written
    written = out_dir / f"{r['dedup_hash']}.kicad_sch"
    assert written.exists()
    # Clone dir cleaned up
    assert not (clone_root / "foo_bar").exists()
    # Audit has accepted event
    lines = audit.read_text().splitlines()
    assert any('"file_accepted"' in ln for ln in lines)
    assert any('"repo_clone_ok"' in ln for ln in lines)


def test_process_repo_skips_on_upgrade_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clone_root = tmp_path / "clones"
    out_dir = tmp_path / "out"
    clone_root.mkdir()
    out_dir.mkdir()
    log = AuditLogger(tmp_path / "audit.ndjson")

    def fake_clone(full_name: str, dest: Path, timeout: int = 300) -> bool:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "bad.kicad_sch").write_text("garbage")
        return True

    monkeypatch.setattr(scrape_d1_repos, "shallow_clone", fake_clone)
    monkeypatch.setattr(scrape_d1_repos, "_git_head_sha", lambda d: "0" * 7)
    monkeypatch.setattr(scrape_d1_repos, "_kicad_upgrade", lambda p: 1)

    rows = process_repo(
        repo={"fullName": "foo/bad", "license": {"key": "mit"}},
        clone_root=clone_root,
        out_dir=out_dir,
        seen_hashes=set(),
        log=log,
        max_files_per_repo=10,
    )
    assert rows == []
    assert list(out_dir.glob("*.kicad_sch")) == []


def test_process_repo_dedup_skips_known_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clone_root = tmp_path / "clones"
    out_dir = tmp_path / "out"
    clone_root.mkdir()
    out_dir.mkdir()
    log = AuditLogger(tmp_path / "audit.ndjson")

    def fake_clone(full_name: str, dest: Path, timeout: int = 300) -> bool:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "x.kicad_sch").write_text(_SCH)
        return True

    monkeypatch.setattr(scrape_d1_repos, "shallow_clone", fake_clone)
    monkeypatch.setattr(scrape_d1_repos, "_git_head_sha", lambda d: "abc")
    monkeypatch.setattr(scrape_d1_repos, "_kicad_upgrade", lambda p: 0)

    from scripts.kicad_sch.scrape_d1 import canonical_hash

    known = canonical_hash(_SCH)
    rows = process_repo(
        repo={"fullName": "foo/dup", "license": {"key": "mit"}},
        clone_root=clone_root,
        out_dir=out_dir,
        seen_hashes={known},
        log=log,
        max_files_per_repo=10,
    )
    assert rows == []
