"""Red tests for scripts.kicad_sch.scrape_d1 (TDD C3)."""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.kicad_sch.scrape_d1 import (
    canonical_hash,
    download_and_normalize,
    license_allowed,
)


def test_license_allowlist() -> None:
    al = {"MIT", "Apache-2.0", "CC0-1.0", "GPL-3.0"}
    assert license_allowed("MIT", al)
    assert license_allowed("apache-2.0", al)
    assert not license_allowed("AGPL-3.0", al)
    assert not license_allowed(None, al)


def test_canonical_hash_strips_uuids() -> None:
    a = '(uuid "11111111-2222-3333-4444-555555555555") (rest 1)'
    b = '(uuid "99999999-aaaa-bbbb-cccc-dddddddddddd") (rest 1)'
    assert canonical_hash(a) == canonical_hash(b)


def test_download_and_normalize_writes_dedup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src_text = "(kicad_sch (version 20240101) (generator eeschema))"
    monkeypatch.setattr(
        "scripts.kicad_sch.scrape_d1._fetch_raw", lambda url: src_text
    )
    monkeypatch.setattr(
        "scripts.kicad_sch.scrape_d1._kicad_update", lambda p: 0
    )
    out = download_and_normalize(
        repo="foo/bar",
        path="x.kicad_sch",
        url="https://x",
        commit="abc",
        license_spdx="MIT",
        out_dir=tmp_path,
    )
    assert out is not None
    assert out.exists()
    assert out.suffix == ".kicad_sch"
