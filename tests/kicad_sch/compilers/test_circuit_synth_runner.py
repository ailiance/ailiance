import importlib.util
import pytest
from scripts.kicad_sch.compilers import circuit_synth_runner


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("circuit_synth") is None,
    reason="circuit_synth not installed",
)


def test_circuit_synth_runner_writes_script(tmp_path):
    dsl = (
        "from circuit_synth import Circuit, Component, Net\n"
        "def build() -> Circuit:\n"
        "    c = Circuit('divider')\n"
        "    vin, gnd, vout = Net('VIN'), Net('GND'), Net('VOUT')\n"
        "    r1 = Component(symbol='Device:R', value='10k',"
        "                   footprint='Resistor_SMD:R_0603_1608Metric')\n"
        "    r2 = Component(symbol='Device:R', value='10k',"
        "                   footprint='Resistor_SMD:R_0603_1608Metric')\n"
        "    c.connect(vin, r1[1]); c.connect(r1[2], vout)\n"
        "    c.connect(vout, r2[1]); c.connect(r2[2], gnd)\n"
        "    return c\n"
    )
    result = circuit_synth_runner.run(dsl, tmp_path)
    assert (tmp_path / "circuit.py").exists()
    assert isinstance(result.dsl_parse_ok, bool)


def test_circuit_synth_runner_marks_bad_dsl_as_parse_fail(tmp_path):
    dsl = "from circuit_synth import *** broken"
    result = circuit_synth_runner.run(dsl, tmp_path)
    assert result.dsl_parse_ok is False
    assert result.compile_ok is False
