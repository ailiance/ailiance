"""Red tests for scripts.kicad_sch.synth_d2 (TDD C5)."""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.kicad_sch.synth_d2 import TEMPLATES, randomize_values, synth_one


def test_templates_cover_minimum_set() -> None:
    names = {t["name"] for t in TEMPLATES}
    expected = {
        "voltage_divider",
        "rc_lowpass",
        "rlc_series",
        "ne555_astable",
        "opamp_noninv",
        "common_emitter",
    }
    assert expected.issubset(names)


def test_randomize_values_deterministic_with_seed() -> None:
    t = next(t for t in TEMPLATES if t["name"] == "voltage_divider")
    a = randomize_values(t, seed=42)
    b = randomize_values(t, seed=42)
    assert a == b


def test_synth_one_writes_file_when_compile_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_compile(tpl, vals, out):
        out.write_text("(kicad_sch (version 20240101) (generator skidl))")
        return 0

    monkeypatch.setattr(
        "scripts.kicad_sch.synth_d2._compile_skidl", fake_compile
    )
    monkeypatch.setattr(
        "scripts.kicad_sch.synth_d2._kicad_erc", lambda p: 0
    )
    out = synth_one(
        template="voltage_divider",
        compiler="skidl",
        seed=42,
        out_dir=tmp_path,
    )
    assert out is not None
    assert out.exists()


def test_synth_one_returns_none_when_erc_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_compile(tpl, vals, out):
        out.write_text("x")
        return 0

    monkeypatch.setattr(
        "scripts.kicad_sch.synth_d2._compile_skidl", fake_compile
    )
    monkeypatch.setattr(
        "scripts.kicad_sch.synth_d2._kicad_erc", lambda p: 3
    )
    out = synth_one(
        template="voltage_divider",
        compiler="skidl",
        seed=42,
        out_dir=tmp_path,
    )
    assert out is None
