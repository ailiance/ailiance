import json
import pytest
from pathlib import Path

pytest.importorskip("mlx")  # mlx is Apple-Silicon only; skip on CI/linux

from scripts.kicad_sch.audit_log import AuditLogger
from scripts.kicad_sch import hybrid_pipeline


class _StubModel:
    """Stand-in for an MLX model: just a tag we can recognise."""
    def __init__(self, key): self.key = key


class _StubTok:
    def __init__(self): pass


def _stub_load(model_path, adapter_path=None):
    return _StubModel(model_path), _StubTok()


def _stub_generate(model, tok, prompt, max_tokens, temperature, seed):
    return (
        "from skidl import *\n"
        "set_default_tool(KICAD)\n"
        "vin=Net('VIN'); gnd=Net('GND'); vout=Net('VOUT')\n"
        "r1=Part('Device','R',value='10k',"
        "footprint='Resistor_SMD:R_0603_1608Metric')\n"
        "r2=Part('Device','R',value='10k',"
        "footprint='Resistor_SMD:R_0603_1608Metric')\n"
        "vin & r1 & vout & r2 & gnd\n"
        "generate_schematic(filepath='out.kicad_sch')\n"
    )


class _StubRunner:
    def __init__(self): self.calls = []
    def run(self, dsl, out_dir, **_):
        from scripts.kicad_sch.compilers.result import CompileResult
        self.calls.append((dsl, Path(out_dir)))
        sch = Path(out_dir) / "out.kicad_sch"
        sch.parent.mkdir(parents=True, exist_ok=True)
        sch.write_text("(kicad_sch (version 20240101))")
        return CompileResult(
            dsl_parse_ok=True, compile_ok=True, output_path=sch,
            stderr="", wall_time_ms=42,
        )


def test_run_cell_logs_each_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(hybrid_pipeline, "load_model_and_tokenizer", _stub_load)
    monkeypatch.setattr(hybrid_pipeline, "generate_sample", _stub_generate)
    stub_runner = _StubRunner()
    monkeypatch.setitem(hybrid_pipeline.RUNNERS, "skidl", stub_runner)

    audit_path = tmp_path / "audit.ndjson"
    logger = AuditLogger(audit_path)
    out = hybrid_pipeline.run_cell(
        base_model_key="qwen36",
        compiler="skidl",
        prompt="voltage divider 10k 10k",
        seeds=[42],
        n_samples=1,
        out_dir=tmp_path / "art",
        audit_logger=logger,
    )

    lines = audit_path.read_text().strip().split("\n")
    assert len(lines) == 1
    log = json.loads(lines[0])
    assert log["event_type"] == "generation"
    assert log["base_model_key"] == "qwen36"
    assert log["compiler"] == "skidl"
    assert log["seed"] == 42
    assert log["sample_idx"] == 0
    assert log["dsl_parse_ok"] is True
    assert log["compile_ok"] is True
    assert out["compile_ok_rate"] == 1.0
    assert out["dsl_parse_ok_rate"] == 1.0
    assert out["n_attempts"] == 1


def test_run_cell_aggregates_rates_across_seeds(tmp_path, monkeypatch):
    monkeypatch.setattr(hybrid_pipeline, "load_model_and_tokenizer", _stub_load)
    monkeypatch.setattr(hybrid_pipeline, "generate_sample", _stub_generate)

    from scripts.kicad_sch.compilers.result import CompileResult

    class _AltRunner:
        def __init__(self): self.n = 0
        def run(self, dsl, out_dir, **_):
            self.n += 1
            return CompileResult(
                dsl_parse_ok=True,
                compile_ok=(self.n % 2 == 0),
                output_path=None,
                stderr="", wall_time_ms=1,
            )

    monkeypatch.setitem(hybrid_pipeline.RUNNERS, "skidl", _AltRunner())
    logger = AuditLogger(tmp_path / "a.ndjson")
    out = hybrid_pipeline.run_cell(
        base_model_key="qwen36", compiler="skidl",
        prompt="p", seeds=[1, 2, 3, 4], n_samples=1,
        out_dir=tmp_path / "art", audit_logger=logger,
    )
    assert out["n_attempts"] == 4
    assert out["dsl_parse_ok_rate"] == 1.0
    assert out["compile_ok_rate"] == 0.5


def test_run_all_iterates_full_grid(tmp_path, monkeypatch):
    monkeypatch.setattr(hybrid_pipeline, "load_model_and_tokenizer", _stub_load)
    monkeypatch.setattr(hybrid_pipeline, "generate_sample", _stub_generate)
    monkeypatch.setattr(hybrid_pipeline, "unload_model", lambda: None)

    from scripts.kicad_sch.compilers.result import CompileResult

    class _OK:
        def run(self, dsl, out_dir, **_):
            return CompileResult(
                dsl_parse_ok=True, compile_ok=True,
                output_path=None, stderr="", wall_time_ms=1,
            )

    for c in hybrid_pipeline.COMPILERS:
        monkeypatch.setitem(hybrid_pipeline.RUNNERS, c, _OK())

    logger = AuditLogger(tmp_path / "a.ndjson")
    summary = hybrid_pipeline.run_all(
        prompts=["voltage divider"],
        base_models=list(hybrid_pipeline.BASE_MODELS),
        compilers=list(hybrid_pipeline.COMPILERS),
        seeds=[42],
        n_samples=1,
        out_dir=tmp_path / "art",
        audit_logger=logger,
    )
    # 5 models * 4 compilers * 1 prompt = 20 cells
    assert len(summary["cells"]) == 20
    assert summary["n_attempts_total"] == 20
    assert summary["compile_ok_rate_overall"] == 1.0


def test_run_all_writes_summary_json(tmp_path, monkeypatch):
    monkeypatch.setattr(hybrid_pipeline, "load_model_and_tokenizer", _stub_load)
    monkeypatch.setattr(hybrid_pipeline, "generate_sample", _stub_generate)
    monkeypatch.setattr(hybrid_pipeline, "unload_model", lambda: None)

    from scripts.kicad_sch.compilers.result import CompileResult

    class _OK:
        def run(self, dsl, out_dir, **_):
            return CompileResult(dsl_parse_ok=True, compile_ok=True)

    for c in hybrid_pipeline.COMPILERS:
        monkeypatch.setitem(hybrid_pipeline.RUNNERS, c, _OK())

    summary_path = tmp_path / "summary.json"
    logger = AuditLogger(tmp_path / "a.ndjson")
    hybrid_pipeline.run_all(
        prompts=["led blinker"],
        base_models=["qwen36"],
        compilers=["skidl"],
        seeds=[42],
        n_samples=1,
        out_dir=tmp_path / "art",
        audit_logger=logger,
        summary_path=summary_path,
    )
    payload = json.loads(summary_path.read_text())
    assert payload["cells"][0]["base_model_key"] == "qwen36"
    assert payload["cells"][0]["compiler"] == "skidl"


def test_cli_smoke_mode_runs_one_cell(tmp_path, monkeypatch):
    monkeypatch.setattr(hybrid_pipeline, "load_model_and_tokenizer", _stub_load)
    monkeypatch.setattr(hybrid_pipeline, "generate_sample", _stub_generate)
    monkeypatch.setattr(hybrid_pipeline, "unload_model", lambda: None)

    from scripts.kicad_sch.compilers.result import CompileResult

    class _OK:
        def run(self, dsl, out_dir, **_):
            return CompileResult(dsl_parse_ok=True, compile_ok=True)

    for c in hybrid_pipeline.COMPILERS:
        monkeypatch.setitem(hybrid_pipeline.RUNNERS, c, _OK())

    rc = hybrid_pipeline.main([
        "--mode", "smoke",
        "--out-dir", str(tmp_path / "art"),
        "--audit-path", str(tmp_path / "audit.ndjson"),
        "--summary-path", str(tmp_path / "summary.json"),
    ])
    assert rc == 0
    assert (tmp_path / "summary.json").exists()
    payload = json.loads((tmp_path / "summary.json").read_text())
    # smoke = 1 model x 1 compiler x 1 prompt x 1 seed x 1 sample = 1 cell
    assert len(payload["cells"]) == 1
