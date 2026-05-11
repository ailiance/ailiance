import shutil
import pytest
from scripts.kicad_sch.compilers import tscircuit_runner


pytestmark = pytest.mark.skipif(
    shutil.which("npx") is None,
    reason="npx not on PATH",
)


def test_tscircuit_runner_writes_tsx_and_invokes_tsci(tmp_path):
    dsl = (
        'import { Board } from "@tscircuit/core"\n'
        "export default () => (\n"
        '  <board width="10mm" height="10mm">\n'
        '    <resistor name="R1" resistance="10k" footprint="0603" />\n'
        '    <resistor name="R2" resistance="10k" footprint="0603" />\n'
        "  </board>\n"
        ")\n"
    )
    result = tscircuit_runner.run(dsl, tmp_path)
    assert (tmp_path / "circuit.tsx").exists()
    assert isinstance(result.dsl_parse_ok, bool)
    assert isinstance(result.compile_ok, bool)


def test_tscircuit_runner_marks_garbage_as_parse_fail(tmp_path):
    dsl = "<<< not tsx >>>"
    result = tscircuit_runner.run(dsl, tmp_path)
    assert result.dsl_parse_ok is False
    assert result.compile_ok is False
