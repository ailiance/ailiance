from pathlib import Path
from scripts.kicad_sch.compilers.result import CompileResult


def test_compile_result_defaults_to_all_false():
    r = CompileResult()
    assert r.dsl_parse_ok is False
    assert r.compile_ok is False
    assert r.output_path is None
    assert r.stderr == ""
    assert r.wall_time_ms == 0


def test_compile_result_accepts_full_payload(tmp_path):
    out = tmp_path / "x.kicad_sch"
    out.write_text("(kicad_sch)")
    r = CompileResult(
        dsl_parse_ok=True,
        compile_ok=True,
        output_path=out,
        stderr="warn: foo",
        wall_time_ms=842,
    )
    assert r.compile_ok and r.output_path.exists()
    assert r.wall_time_ms == 842


def test_compile_result_as_dict_serialises_path_to_str(tmp_path):
    out = tmp_path / "x.kicad_sch"
    r = CompileResult(dsl_parse_ok=True, compile_ok=True, output_path=out)
    d = r.as_dict()
    assert d["output_path"] == str(out)
    assert d["dsl_parse_ok"] is True
