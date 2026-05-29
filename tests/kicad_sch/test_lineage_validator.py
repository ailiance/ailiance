"""Tests for the lineage validator — EU AI Act Annex IV §2.b per-split gate.

TDD: tests written before implementation.  Covers:
  - D1 clean row (valid SPDX + valid source_type) → no violations
  - D1 bad SPDX (BSD-3-Clause not in D1 allowlist) → 1 violation
  - D1 bad source_type → 1 violation
  - D1 row with both bad license AND bad source_type → 2 violations
  - validate_manifest() convenience wrapper (DatasetManifest in, violations out)
  - Unknown split → ValueError
  - assert_clean() raises LineageError on violations, silent when clean
  - All four D1 canonical SPDX IDs accepted (MIT/Apache-2.0/CC0-1.0/GPL-3.0)
"""

from __future__ import annotations

import pytest

from scripts.kicad_sch.lineage_validator import (
    LineageError,
    LineageViolation,
    assert_clean,
    validate_manifest,
    validate_rows,
)
from scripts.kicad_sch.manifest import DatasetManifest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_D1_GOOD_ROW = {
    "source_type": "github_scrape",
    "source_url": "https://github.com/example/repo/blob/abc/board.kicad_sch",
    "commit_sha": "abc123",
    "license_spdx": "MIT",
    "dedup_hash": "deadbeef",
    "file_size_bytes": 1024,
    "kicad_version_before": "7.0.0",
    "kicad_version_after": "10.0.3",
}


def _make_manifest(split: str) -> DatasetManifest:
    """Return an in-memory DatasetManifest with no path side-effects."""
    from pathlib import Path

    return DatasetManifest(Path(f"/tmp/manifest_{split}.csv"), split)


def _add_row(m: DatasetManifest, **overrides) -> None:
    base = {
        "source_type": "github_scrape",
        "source_url": "https://github.com/x/y/blob/abc/x.kicad_sch",
        "commit_sha": "abc123",
        "license_spdx": "MIT",
        "dedup_hash": "cafebabe",
        "file_size_bytes": 512,
        "kicad_version_before": "7.0.0",
        "kicad_version_after": "10.0.3",
    }
    base.update(overrides)
    m.add(**base)


# ---------------------------------------------------------------------------
# 1. D1 clean row → no violations
# ---------------------------------------------------------------------------


def test_d1_clean_row_no_violations():
    violations = validate_rows([_D1_GOOD_ROW], split="D1")
    assert violations == []


# ---------------------------------------------------------------------------
# 2. D1 bad SPDX (BSD-3-Clause not in D1 allowlist) → 1 violation on license_spdx
# ---------------------------------------------------------------------------


def test_d1_bad_spdx_one_violation():
    row = {**_D1_GOOD_ROW, "license_spdx": "BSD-3-Clause"}
    violations = validate_rows([row], split="D1")

    assert len(violations) == 1
    v = violations[0]
    assert v.row_index == 0
    assert v.field == "license_spdx"
    assert v.value == "BSD-3-Clause"
    assert "D1" in v.reason
    assert "BSD-3-Clause" in v.reason


# ---------------------------------------------------------------------------
# 3. D1 bad source_type → 1 violation on source_type
# ---------------------------------------------------------------------------


def test_d1_bad_source_type_one_violation():
    row = {**_D1_GOOD_ROW, "source_type": "unknown_origin"}
    violations = validate_rows([row], split="D1")

    assert len(violations) == 1
    v = violations[0]
    assert v.row_index == 0
    assert v.field == "source_type"
    assert v.value == "unknown_origin"
    assert "source_type" in v.reason.lower() or "D1" in v.reason


# ---------------------------------------------------------------------------
# 4. Both bad license AND bad source_type → 2 violations
# ---------------------------------------------------------------------------


def test_d1_both_bad_two_violations():
    row = {**_D1_GOOD_ROW, "license_spdx": "GPL-2.0", "source_type": "scraped_elsewhere"}
    violations = validate_rows([row], split="D1")

    assert len(violations) == 2
    fields = {v.field for v in violations}
    assert fields == {"license_spdx", "source_type"}
    for v in violations:
        assert v.row_index == 0


# ---------------------------------------------------------------------------
# 5. validate_manifest() — wrapper over DatasetManifest
# ---------------------------------------------------------------------------


def test_validate_manifest_mixed_rows():
    m = _make_manifest("D1")
    _add_row(m, license_spdx="Apache-2.0")          # clean
    _add_row(m, license_spdx="CERN-OHL-S-2.0")      # dirty (not in D1 strict set)

    violations = validate_manifest(m)

    # row 0 is clean, row 1 has 1 violation on license_spdx
    assert len(violations) == 1
    v = violations[0]
    assert v.row_index == 1
    assert v.field == "license_spdx"
    assert v.value == "CERN-OHL-S-2.0"


# ---------------------------------------------------------------------------
# 6. Unknown split → ValueError
# ---------------------------------------------------------------------------


def test_unknown_split_raises_value_error():
    with pytest.raises(ValueError, match="D9"):
        validate_rows([_D1_GOOD_ROW], split="D9")


# ---------------------------------------------------------------------------
# 7. assert_clean — raises LineageError on violations, silent when clean
# ---------------------------------------------------------------------------


def test_assert_clean_raises_on_violations():
    row = {**_D1_GOOD_ROW, "license_spdx": "BSD-3-Clause"}
    with pytest.raises(LineageError):
        assert_clean([row], split="D1")


def test_assert_clean_silent_when_clean():
    # Must not raise
    assert_clean([_D1_GOOD_ROW], split="D1")


# ---------------------------------------------------------------------------
# 8. All four D1 canonical SPDX IDs are accepted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spdx", ["MIT", "Apache-2.0", "CC0-1.0", "GPL-3.0"])
def test_d1_canonical_spdx_accepted(spdx: str):
    row = {**_D1_GOOD_ROW, "license_spdx": spdx}
    violations = validate_rows([row], split="D1")
    license_violations = [v for v in violations if v.field == "license_spdx"]
    assert license_violations == [], f"Expected {spdx!r} to be accepted for D1"


# ---------------------------------------------------------------------------
# Bonus: multiple rows — row_index reported correctly
# ---------------------------------------------------------------------------


def test_row_index_reported_correctly():
    rows = [
        _D1_GOOD_ROW,                                         # index 0 — clean
        {**_D1_GOOD_ROW, "license_spdx": "GPL-2.0"},         # index 1 — dirty
        {**_D1_GOOD_ROW, "license_spdx": "GPL-3.0"},         # index 2 — clean
        {**_D1_GOOD_ROW, "license_spdx": "LGPL-2.1"},        # index 3 — dirty
    ]
    violations = validate_rows(rows, split="D1")
    dirty_indices = {v.row_index for v in violations}
    assert dirty_indices == {1, 3}


# ---------------------------------------------------------------------------
# D2 clean row: source_type="synth", license_spdx="CC0-1.0"
# ---------------------------------------------------------------------------


def test_d2_clean_row_no_violations():
    row = {
        **_D1_GOOD_ROW,
        "source_type": "synth",
        "license_spdx": "CC0-1.0",
    }
    violations = validate_rows([row], split="D2")
    assert violations == []


# ---------------------------------------------------------------------------
# D3 clean row: source_type="mix"
# ---------------------------------------------------------------------------


def test_d3_clean_row_no_violations():
    row = {
        **_D1_GOOD_ROW,
        "source_type": "mix",
        "license_spdx": "MIT",
    }
    violations = validate_rows([row], split="D3")
    assert violations == []
