"""Tests for the dataset manifest CSV writer (EU AI Act Annex IV §2.b)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.kicad_sch.manifest import DatasetManifest


HEADER = (
    "source_type,source_url,commit_sha,license_spdx,dedup_hash,"
    "file_size_bytes,kicad_version_before,kicad_version_after"
)

# ---------------------------------------------------------------------------
# Helper — resolved split-encoded path (mirrors manifest.output_path logic)
# ---------------------------------------------------------------------------

def _resolved(base: Path, split: str) -> Path:
    """Return the split-encoded path expected by write()."""
    stem = base.stem
    if stem.endswith(f"_{split}"):
        return base
    return base.with_name(f"{stem}_{split}{base.suffix}")


# ---------------------------------------------------------------------------
# Existing tests — updated to read from the split-encoded filename (#37)
# ---------------------------------------------------------------------------

def test_manifest_writes_csv_with_header(tmp_path: Path) -> None:
    base = tmp_path / "manifest.csv"
    m = DatasetManifest(base, split="D1")
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
    out = _resolved(base, "D1")
    content = out.read_text()
    assert HEADER in content
    assert "scraped,https://github.com/foo/bar,abc,MIT,def,1024,v6,v10" in content


def test_manifest_multiple_rows(tmp_path: Path) -> None:
    base = tmp_path / "manifest.csv"
    m = DatasetManifest(base, split="D2")
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
    out = _resolved(base, "D2")
    lines = out.read_text().strip().split("\n")
    assert len(lines) == 4  # header + 3 rows
    assert lines[0] == HEADER


def test_manifest_rejects_invalid_split(tmp_path: Path) -> None:
    with pytest.raises((ValueError, TypeError)):
        DatasetManifest(tmp_path / "m.csv", split="D9")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# #40 — New tests
# ---------------------------------------------------------------------------

def _sample_row(**overrides: object) -> dict[str, object]:
    """Return a valid add() kwargs dict, with optional field overrides."""
    base: dict[str, object] = dict(
        source_type="scraped",
        source_url="https://example.com/repo",
        commit_sha="abc123",
        license_spdx="Apache-2.0",
        dedup_hash="deadbeef",
        file_size_bytes=512,
        kicad_version_before="v8",
        kicad_version_after="v10",
    )
    base.update(overrides)
    return base


# --- CSV escape correctness ------------------------------------------------

def test_csv_escape_roundtrip(tmp_path: Path) -> None:
    """A field with comma, double-quote, and newline must roundtrip exactly."""
    tricky = 'a,b"c\nd'
    base = tmp_path / "manifest.csv"
    m = DatasetManifest(base, split="D1")
    m.add(**_sample_row(source_url=tricky))  # type: ignore[arg-type]
    m.write()

    out = _resolved(base, "D1")
    with out.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["source_url"] == tricky


# --- Zero rows -------------------------------------------------------------

def test_write_with_no_rows(tmp_path: Path) -> None:
    """write() with no add() calls → header line only, 0 data rows."""
    base = tmp_path / "manifest.csv"
    m = DatasetManifest(base, split="D3")
    m.write()

    out = _resolved(base, "D3")
    with out.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    assert rows == []
    # Header is present (DictReader populated fieldnames)
    assert reader.fieldnames is not None
    assert "source_type" in reader.fieldnames


# --- Write twice (overwrite, not append) -----------------------------------

def test_write_twice_overwrites(tmp_path: Path) -> None:
    """Second write() must overwrite, not append."""
    base = tmp_path / "manifest.csv"
    m = DatasetManifest(base, split="D1")
    m.add(**_sample_row(commit_sha="first"))  # type: ignore[arg-type]
    m.write()

    m.add(**_sample_row(commit_sha="second"))  # type: ignore[arg-type]
    m.write()

    out = _resolved(base, "D1")
    with out.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    # Both rows (first + second) are in the final file — but header appears only once
    assert len(rows) == 2
    commit_shas = {r["commit_sha"] for r in rows}
    assert commit_shas == {"first", "second"}

    # Assert no duplicate header (overwrite, not append)
    raw_lines = out.read_text().splitlines()
    header_count = sum(1 for line in raw_lines if line.startswith("source_type,"))
    assert header_count == 1


# --- Split-encoded filename (#37) -----------------------------------------

def test_split_filename_convention(tmp_path: Path) -> None:
    """write() must produce a split-encoded filename (e.g. manifest_D1.csv)."""
    base = tmp_path / "manifest.csv"
    m = DatasetManifest(base, split="D1")
    m.add(**_sample_row())  # type: ignore[arg-type]
    m.write()

    expected = tmp_path / "manifest_D1.csv"
    assert expected.exists(), f"Expected {expected} but it was not created"
    # The bare path must NOT exist (split is encoded in the name)
    assert not base.exists(), "Bare manifest.csv must not be created — split goes in filename"


def test_split_filename_idempotent(tmp_path: Path) -> None:
    """If caller already passes the split-encoded name, do not double-encode."""
    base = tmp_path / "manifest_D2.csv"
    m = DatasetManifest(base, split="D2")
    m.add(**_sample_row())  # type: ignore[arg-type]
    m.write()

    assert (tmp_path / "manifest_D2.csv").exists()
    assert not (tmp_path / "manifest_D2_D2.csv").exists()


def test_split_recoverable_from_filename(tmp_path: Path) -> None:
    """An auditor can recover the split from the written filename stem."""
    for split in ("D1", "D2", "D3"):
        base = tmp_path / "manifest.csv"
        m = DatasetManifest(base, split=split)  # type: ignore[arg-type]
        m.add(**_sample_row())  # type: ignore[arg-type]
        m.write()
        out = _resolved(base, split)
        assert out.stem.endswith(f"_{split}"), (
            f"Split {split!r} not recoverable from stem {out.stem!r}"
        )


# --- #38 runtime assertions ------------------------------------------------

def test_add_rejects_string_file_size(tmp_path: Path) -> None:
    """file_size_bytes must be int, not str."""
    m = DatasetManifest(tmp_path / "m.csv", split="D1")
    with pytest.raises(TypeError):
        m.add(**_sample_row(file_size_bytes="1024"))  # type: ignore[arg-type]


def test_add_rejects_negative_file_size(tmp_path: Path) -> None:
    """file_size_bytes must be >= 0."""
    m = DatasetManifest(tmp_path / "m.csv", split="D1")
    with pytest.raises(ValueError):
        m.add(**_sample_row(file_size_bytes=-1))  # type: ignore[arg-type]


def test_add_rejects_bool_file_size(tmp_path: Path) -> None:
    """bool is a subclass of int — must be explicitly rejected."""
    m = DatasetManifest(tmp_path / "m.csv", split="D1")
    with pytest.raises(TypeError):
        m.add(**_sample_row(file_size_bytes=True))  # type: ignore[arg-type]


def test_add_rejects_empty_dedup_hash(tmp_path: Path) -> None:
    """dedup_hash must be a non-empty str."""
    m = DatasetManifest(tmp_path / "m.csv", split="D1")
    with pytest.raises(ValueError):
        m.add(**_sample_row(dedup_hash=""))  # type: ignore[arg-type]


def test_add_accepts_empty_commit_sha_and_license(tmp_path: Path) -> None:
    """commit_sha and license_spdx MAY be empty: D3 mix rows inherit
    lineage from source rows. Policy is enforced by lineage_validator,
    not the manifest writer."""
    m = DatasetManifest(tmp_path / "m.csv", split="D3")
    m.add(**_sample_row(commit_sha="", license_spdx=""))  # type: ignore[arg-type]
    assert m.rows[0]["commit_sha"] == ""
    assert m.rows[0]["license_spdx"] == ""


def test_add_rejects_empty_source_url(tmp_path: Path) -> None:
    """source_url must be a non-empty str."""
    m = DatasetManifest(tmp_path / "m.csv", split="D1")
    with pytest.raises(ValueError):
        m.add(**_sample_row(source_url=""))  # type: ignore[arg-type]
