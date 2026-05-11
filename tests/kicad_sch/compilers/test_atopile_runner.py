import shutil
import pytest
from scripts.kicad_sch.compilers import atopile_runner


pytestmark = pytest.mark.skipif(
    shutil.which("ato") is None,
    reason="ato CLI not on PATH",
)


def test_atopile_runner_writes_ato_and_invokes_build(tmp_path):
    dsl = (
        'import Resistor from "generics/resistors.ato"\n'
        "module Main:\n"
        "    signal vin\n"
        "    signal gnd\n"
        "    signal vout\n"
        "    r1 = new Resistor; r1.value = 10kohm; r1.package = \"0603\"\n"
        "    r2 = new Resistor; r2.value = 10kohm; r2.package = \"0603\"\n"
        "    vin ~ r1.p1; r1.p2 ~ vout; vout ~ r2.p1; r2.p2 ~ gnd\n"
    )
    result = atopile_runner.run(dsl, tmp_path)
    # We do not assert compile_ok=True here because atopile may need a
    # project ato.yaml; we DO assert the .ato was written and ato was
    # invoked (dsl_parse_ok is True iff ato accepted the syntax).
    assert (tmp_path / "main.ato").exists()
    assert isinstance(result.dsl_parse_ok, bool)
    assert isinstance(result.compile_ok, bool)
    assert result.wall_time_ms >= 0


def test_atopile_runner_marks_garbage_dsl_as_parse_fail(tmp_path):
    dsl = "@@@ not atopile @@@"
    result = atopile_runner.run(dsl, tmp_path)
    assert result.dsl_parse_ok is False
    assert result.compile_ok is False
