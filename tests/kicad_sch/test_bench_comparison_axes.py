"""Tests for bench_comparison.py --metric-axes extension."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

# Repo-relative path (tests/kicad_sch/ -> repo root -> scripts/). The
# previous Path.home()/"ailiance/scripts" pointed at a production deploy
# location absent in CI; the script lives in the repo itself.
BENCH = Path(__file__).resolve().parents[2] / "scripts" / "bench_comparison.py"


def _write_ppl(path: Path, model: str, domain: str, ppl: float, n=30):
    rows = [{"model_key": model, "domain": domain,
             "perplexity": ppl, "n_samples": n}]
    path.write_text(json.dumps(rows))


def _write_axes_validator(path: Path, model: str, domain: str):
    """Mimic run_eval_n3 --out-aggregate output."""
    rows = [{
        "model_key": model, "domain": domain,
        "pass_rate": 0.7, "n_samples": 5,
        "axis_parse_ok": 1.0, "axis_erc_clean": 0.8,
        "axis_sch_render": 0.6, "axis_drc_clean": 0.0,
        "axis_sem_equiv": 0.4,
    }]
    path.write_text(json.dumps(rows))


def test_no_axes_flag_is_backward_compat(tmp_path):
    """Without --metric-axes, output matches PR #24 behavior (no axis cols)."""
    base = tmp_path / "perplexity_base_test.json"
    tuned = tmp_path / "perplexity_v1-only_test.json"
    _write_ppl(base, "m1", "d1", 10.0)
    _write_ppl(tuned, "m1", "d1", 8.0)
    out = tmp_path / "out"

    res = subprocess.run(
        [sys.executable, str(BENCH),
         "--base", str(base), "--tuned", str(tuned),
         "--out-prefix", str(out)],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr
    md = (Path(str(out) + ".md")).read_text()
    assert "parse_ok" not in md
    assert "sem_equiv" not in md
    assert "lift_log" in md  # legacy column preserved


def test_axes_flag_adds_columns(tmp_path):
    base = tmp_path / "perplexity_base_test.json"
    tuned = tmp_path / "perplexity_v1-only_test.json"
    val = tmp_path / "axes_validator.json"
    _write_ppl(base, "m1", "d1", 10.0)
    _write_ppl(tuned, "m1", "d1", 8.0)
    _write_axes_validator(val, "m1", "d1")
    out = tmp_path / "out"

    res = subprocess.run(
        [sys.executable, str(BENCH),
         "--base", str(base), "--tuned", str(tuned),
         "--validator-tuned", str(val),
         "--metric-axes",
         "parse_ok,erc_clean,sch_render,drc_clean,sem_equiv",
         "--out-prefix", str(out)],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr
    md = (Path(str(out) + ".md")).read_text()
    for col in ["parse_ok", "erc_clean", "sch_render",
                "drc_clean", "sem_equiv"]:
        assert col in md, f"missing column {col} in MD output"
    # Composite (= existing validator_lift surrogate) preserved
    assert "lift_log" in md


def test_axes_flag_json_carries_axis_fields(tmp_path):
    base = tmp_path / "perplexity_base_test.json"
    tuned = tmp_path / "perplexity_v1-only_test.json"
    val = tmp_path / "axes_validator.json"
    _write_ppl(base, "m1", "d1", 10.0)
    _write_ppl(tuned, "m1", "d1", 8.0)
    _write_axes_validator(val, "m1", "d1")
    out = tmp_path / "out"

    subprocess.run(
        [sys.executable, str(BENCH),
         "--base", str(base), "--tuned", str(tuned),
         "--validator-tuned", str(val),
         "--metric-axes",
         "parse_ok,erc_clean,sch_render,drc_clean,sem_equiv",
         "--out-prefix", str(out)],
        check=True, capture_output=True, text=True,
    )
    data = json.loads((Path(str(out) + ".json")).read_text())
    assert len(data) == 1
    row = data[0]
    for axis in ["parse_ok", "erc_clean", "sch_render",
                 "drc_clean", "sem_equiv"]:
        assert f"axis_{axis}" in row
