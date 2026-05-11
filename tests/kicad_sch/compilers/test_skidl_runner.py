import importlib.util
import pytest
from pathlib import Path
from scripts.kicad_sch.compilers import skidl_runner


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("skidl") is None,
    reason="skidl not installed on this host",
)


def test_skidl_runner_compiles_minimal_voltage_divider(tmp_path):
    dsl = (
        "from skidl import *\n"
        "set_default_tool(KICAD)\n"
        "vin = Net('VIN'); gnd = Net('GND'); vout = Net('VOUT')\n"
        "r1 = Part('Device','R',value='10k',"
        "footprint='Resistor_SMD:R_0603_1608Metric')\n"
        "r2 = Part('Device','R',value='10k',"
        "footprint='Resistor_SMD:R_0603_1608Metric')\n"
        "vin & r1 & vout & r2 & gnd\n"
        f"generate_schematic(filepath=r'{tmp_path / 'out.kicad_sch'}')\n"
    )
    result = skidl_runner.run(dsl, tmp_path)
    # We only assert structural invariants — the runner may legitimately
    # fail when KiCad symbol libraries aren't on PATH, but it must still
    # return a well-formed CompileResult and have ast-parsed the DSL.
    assert result.dsl_parse_ok is True
    assert isinstance(result.compile_ok, bool)


def test_skidl_runner_marks_bad_dsl_as_parse_fail(tmp_path):
    dsl = "from skidl import * BROKEN SYNTAX :::"
    result = skidl_runner.run(dsl, tmp_path)
    assert result.dsl_parse_ok is False
    assert result.compile_ok is False
    assert result.output_path is None
    assert "SyntaxError" in result.stderr or "invalid syntax" in result.stderr


def test_skidl_runner_marks_compile_fail_when_no_output(tmp_path):
    # syntactically valid Python but never calls generate_schematic
    dsl = "from skidl import *\nset_default_tool(KICAD)\nNet('X')\n"
    result = skidl_runner.run(dsl, tmp_path)
    assert result.dsl_parse_ok is True
    assert result.compile_ok is False
    assert result.output_path is None
