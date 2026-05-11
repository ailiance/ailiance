"""Shared fixtures for Track C kicad-sch tests."""
from __future__ import annotations

import shutil

import pytest

MIN_SCH_WITH_LIB = """(kicad_sch (version 20240101) (generator eeschema)
  (uuid \"00000000-0000-0000-0000-000000000001\")
  (paper \"A4\")
  (lib_symbols
    (symbol \"Device:R\"
      (pin passive line (at 0 0 0) (name \"~\") (number \"1\"))
      (pin passive line (at 0 -10 0) (name \"~\") (number \"2\"))))
  (symbol (lib_id \"Device:R\") (at 100 100 0)
    (uuid \"00000000-0000-0000-0000-000000000002\")))"""


@pytest.fixture
def min_sch_with_lib() -> str:
    return MIN_SCH_WITH_LIB


@pytest.fixture
def kicad_cli_available() -> bool:
    if shutil.which("kicad-cli") is None:
        pytest.skip("kicad-cli not on PATH")
    return True
