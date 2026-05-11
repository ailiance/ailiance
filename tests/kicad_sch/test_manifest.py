"""Tests for the dataset manifest CSV writer (EU AI Act Annex IV §2.b)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.kicad_sch.manifest import DatasetManifest


HEADER = (
    "source_type,source_url,commit_sha,license_spdx,dedup_hash,"
    "file_size_bytes,kicad_version_before,kicad_version_after"
)


def test_manifest_writes_csv_with_header(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    m = DatasetManifest(path, split="D1")
    m.add(
        source_type="scraped",
        source_url="https://github.com/foo/bar",
        commit_sha="abc",
        license_spdx="MIT",
        dedup_hash="def",
        file_size_bytes=1024,
        kicad_version_before="v6",
        kicad_version_after="v10",
    )
    m.write()
    content = path.read_text()
    assert HEADER in content
    assert "scraped,https://github.com/foo/bar,abc,MIT,def,1024,v6,v10" in content


def test_manifest_multiple_rows(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    m = DatasetManifest(path, split="D2")
    for i in range(3):
        m.add(
            source_type="synth",
            source_url=f"seed={i}",
            commit_sha="zzz",
            license_spdx="CC0-1.0",
            dedup_hash=f"hash{i}",
            file_size_bytes=2048,
            kicad_version_before="v10",
            kicad_version_after="v10",
        )
    m.write()
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 4  # header + 3 rows
    assert lines[0] == HEADER


def test_manifest_rejects_invalid_split(tmp_path: Path) -> None:
    with pytest.raises((ValueError, TypeError)):
        DatasetManifest(tmp_path / "m.csv", split="D9")  # type: ignore[arg-type]
