"""Red tests for scripts.kicad_sch.strip_lib_symbols (TDD C1)."""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.kicad_sch.strip_lib_symbols import strip_lib_symbols


def test_strip_preserves_lib_id_references(
    tmp_path: Path, min_sch_with_lib: str
) -> None:
    src = tmp_path / "in.kicad_sch"
    src.write_text(min_sch_with_lib)
    out = tmp_path / "out.kicad_sch"
    rc = strip_lib_symbols(src, out)
    assert rc == 0
    content = out.read_text()
    assert "(pin passive line" not in content
    assert '(lib_id "Device:R")' in content


def test_strip_returns_nonzero_on_unparseable(tmp_path: Path) -> None:
    src = tmp_path / "bad.kicad_sch"
    src.write_text("(((not balanced")
    out = tmp_path / "out.kicad_sch"
    rc = strip_lib_symbols(src, out)
    assert rc != 0


def test_strip_idempotent_when_no_lib_symbols(tmp_path: Path) -> None:
    src = tmp_path / "in.kicad_sch"
    src.write_text(
        "(kicad_sch (version 20240101) (generator eeschema))"
    )
    out = tmp_path / "out.kicad_sch"
    rc = strip_lib_symbols(src, out)
    assert rc == 0
    assert "(kicad_sch" in out.read_text()
