"""Tests for scripts.kicad_sch.eval_n3 (5-axis evaluator)."""
import shutil
from pathlib import Path

import pytest

from scripts.kicad_sch.eval_n3 import eval_parse_ok  # noqa: F401


REF_SCH = Path.home() / "eu-kiki-data/kicad-sch-refs/spi_bus_4devices.kicad_sch"
HAS_CLI = shutil.which("kicad-cli") is not None


def make_broken_sch(tmp_path: Path) -> Path:
    """Emit a kicad_sch missing the (version ...) header and unbalanced parens."""
    bad = tmp_path / "broken.kicad_sch"
    bad.write_text("(kicad_sch broken")
    return bad


def test_module_imports():
    """eval_n3 module must be importable from scripts.kicad_sch package."""
    from scripts.kicad_sch import eval_n3
    assert hasattr(eval_n3, "eval_parse_ok")


@pytest.mark.skipif(not REF_SCH.exists() or not HAS_CLI,
                    reason="ref fixture or kicad-cli missing")
def test_parse_ok_returns_1_for_valid_sch():
    score = eval_parse_ok(REF_SCH)
    assert score == 1


def test_parse_ok_returns_0_for_broken_sch(tmp_path):
    bad = make_broken_sch(tmp_path)
    score = eval_parse_ok(bad)
    assert score == 0
