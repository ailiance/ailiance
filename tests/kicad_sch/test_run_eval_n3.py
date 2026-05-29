"""Tests for run_eval_n3 CLI orchestrator."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

# Repo-relative path (tests/kicad_sch/ -> repo root -> scripts/). The
# previous Path.home()/"ailiance/scripts" pointed at a production deploy
# location absent in CI; the script lives in the repo itself.
RUNNER = Path(__file__).resolve().parents[2] / "scripts" / "run_eval_n3.py"


@pytest.mark.skipif(not RUNNER.exists(), reason="runner not yet created")
def test_runner_emits_seed_records(tmp_path, monkeypatch):
    sch_dir = tmp_path / "sch"
    sch_dir.mkdir()
    (sch_dir / "a.kicad_sch").write_text("(kicad_sch (version 20240101))")
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    (ref_dir / "a.kicad_sch").write_text("(kicad_sch (version 20240101))")
    out = tmp_path / "results.json"

    res = subprocess.run(
        [sys.executable, str(RUNNER),
         "--sch-dir", str(sch_dir),
         "--ref-dir", str(ref_dir),
         "--model-key", "test-model",
         "--domain", "kicad-sch",
         "--out", str(out),
         "--audit-dir", str(tmp_path / "audit"),
         "--mock-cli"],  # see implementation: short-circuits real kicad-cli
        capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr
    assert out.exists()
    data = json.loads(out.read_text())
    assert isinstance(data, list)
    # 1 file x 5 seeds = 5 records (one per seed).
    assert len(data) == 5
    for r in data:
        assert r["model_key"] == "test-model"
        assert r["domain"] == "kicad-sch"
        assert r["seed"] in [42, 137, 1024, 8675309, 31415]
        for axis in ["parse_ok", "erc_clean", "sch_render",
                     "drc_clean", "sem_equiv", "composite"]:
            assert axis in r


def test_runner_aggregates_pass_rate_for_bench_comparison(tmp_path):
    """Output must include a `pass_rate` field consumable by bench_comparison.

    Spec: bench_comparison reads validator JSON entries shaped as:
        {"model_key": ..., "domain": ..., "pass_rate": <0..1>, "n_samples": N}
    run_eval_n3 must also emit an aggregate sidecar (--out-aggregate).
    """
    sch_dir = tmp_path / "sch"
    sch_dir.mkdir()
    (sch_dir / "a.kicad_sch").write_text("(kicad_sch (version 20240101))")
    out = tmp_path / "results.json"
    agg = tmp_path / "agg.json"

    res = subprocess.run(
        [sys.executable, str(RUNNER),
         "--sch-dir", str(sch_dir),
         "--ref-dir", str(sch_dir),  # self-ref for sanity
         "--model-key", "m",
         "--domain", "d",
         "--out", str(out),
         "--out-aggregate", str(agg),
         "--audit-dir", str(tmp_path / "audit"),
         "--mock-cli"],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr
    data = json.loads(agg.read_text())
    assert len(data) == 1
    cell = data[0]
    assert cell["model_key"] == "m"
    assert cell["domain"] == "d"
    assert "pass_rate" in cell
    assert "n_samples" in cell
    assert cell["n_samples"] == 5  # 1 file x 5 seeds
