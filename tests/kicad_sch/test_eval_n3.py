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


from scripts.kicad_sch.eval_n3 import eval_erc_clean


@pytest.mark.skipif(not REF_SCH.exists() or not HAS_CLI,
                    reason="ref fixture or kicad-cli missing")
def test_erc_clean_returns_1_when_no_errors():
    score = eval_erc_clean(REF_SCH, Path("kicad-cli"))
    assert score == 1


def test_erc_clean_returns_0_when_parse_fails(tmp_path):
    bad = make_broken_sch(tmp_path)
    assert eval_erc_clean(bad, Path("kicad-cli")) == 0


def test_erc_clean_parses_violations_count(monkeypatch, tmp_path):
    """Stub subprocess to return synthetic ERC output with 2 errors."""
    fake_sch = tmp_path / "x.kicad_sch"
    fake_sch.write_text("(kicad_sch)")

    class FakeProc:
        returncode = 0
        stdout = "ERC report\nViolations: 2 errors, 0 warnings\n"
        stderr = ""

    def fake_run(*a, **kw):
        return FakeProc()

    monkeypatch.setattr("subprocess.run", fake_run)
    assert eval_erc_clean(fake_sch, Path("kicad-cli")) == 0


def test_erc_clean_zero_errors(monkeypatch, tmp_path):
    fake_sch = tmp_path / "x.kicad_sch"
    fake_sch.write_text("(kicad_sch)")

    class FakeProc:
        returncode = 0
        stdout = "ERC report\nViolations: 0 errors, 0 warnings\n"
        stderr = ""

    monkeypatch.setattr("subprocess.run", lambda *a, **kw: FakeProc())
    assert eval_erc_clean(fake_sch, Path("kicad-cli")) == 1


from scripts.kicad_sch.eval_n3 import eval_sch_render


@pytest.mark.skipif(not REF_SCH.exists() or not HAS_CLI,
                    reason="ref fixture or kicad-cli missing")
def test_sch_render_returns_1_for_valid_sch():
    assert eval_sch_render(REF_SCH, Path("kicad-cli")) == 1


def test_sch_render_returns_0_for_broken_sch(tmp_path):
    bad = make_broken_sch(tmp_path)
    assert eval_sch_render(bad, Path("kicad-cli")) == 0

