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


from scripts.kicad_sch.eval_n3 import eval_drc_clean


def test_drc_clean_returns_0_when_pcbnew_missing(monkeypatch, tmp_path):
    fake = tmp_path / "x.kicad_sch"
    fake.write_text("(kicad_sch)")
    monkeypatch.setattr("shutil.which", lambda x: None)
    # No pcbnew & no kicad-cli pcb subcommand path -> 0.
    assert eval_drc_clean(fake, Path("kicad-cli")) == 0


def test_drc_clean_returns_1_when_drc_passes(monkeypatch, tmp_path):
    fake = tmp_path / "x.kicad_sch"
    fake.write_text("(kicad_sch)")

    class FakeProc:
        returncode = 0
        stdout = "DRC report\n0 errors\n"
        stderr = ""

    # eval_drc_clean checks shutil.which AND produces a pcb file via netlist
    # export — we monkeypatch shutil.which AND subprocess.run AND make sure a
    # fake pcb appears in the tmpdir. Easier: patch Path.exists to True.
    monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/kicad-cli")
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: FakeProc())
    # Force the pcb-existence check to True so we exercise the drc branch.
    import scripts.kicad_sch.eval_n3 as m
    real_path_exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda self: True)
    try:
        assert eval_drc_clean(fake, Path("kicad-cli")) == 1
    finally:
        monkeypatch.setattr(Path, "exists", real_path_exists)


from scripts.kicad_sch.eval_n3 import eval_sem_equiv


def test_sem_equiv_returns_1_for_identical(tmp_path):
    # Two identical kicad_sch files -> sem_equiv == 1.0
    src = tmp_path / "a.kicad_sch"
    dst = tmp_path / "b.kicad_sch"
    content = "(kicad_sch (version 20240101) (symbol U1) (symbol R1) (net N1 U1 R1))"
    src.write_text(content)
    dst.write_text(content)
    score = eval_sem_equiv(src, dst)
    assert abs(score - 1.0) < 1e-6


def test_sem_equiv_returns_0_for_empty_vs_full(tmp_path):
    src = tmp_path / "a.kicad_sch"
    dst = tmp_path / "b.kicad_sch"
    src.write_text("(kicad_sch)")
    dst.write_text("(kicad_sch (symbol U1) (symbol R1) (net N1 U1 R1))")
    score = eval_sem_equiv(src, dst)
    assert 0.0 <= score < 0.5


def test_sem_equiv_returns_float_in_unit_interval(tmp_path):
    src = tmp_path / "a.kicad_sch"
    dst = tmp_path / "b.kicad_sch"
    src.write_text("(kicad_sch (symbol U1) (symbol R1))")
    dst.write_text("(kicad_sch (symbol U1) (symbol C1))")
    score = eval_sem_equiv(src, dst)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

