"""Tests for scripts.kicad_sch.mix_d3 (TDD C7)."""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.kicad_sch.mix_d3 import mix, stratify


def test_stratify_balances_compilers(tmp_path: Path) -> None:
    files = (
        [tmp_path / f"voltage_divider-skidl-{i}.kicad_sch"
         for i in range(10)]
        + [tmp_path / f"rc_lowpass-atopile-{i}.kicad_sch"
           for i in range(10)]
        + [tmp_path / f"led-circuit-synth-{i}.kicad_sch"
           for i in range(10)]
    )
    for f in files:
        f.write_text("x")
    picked = stratify(
        files, n=6, key_re=r"-(skidl|atopile|circuit-synth)-"
    )
    keys = []
    for p in picked:
        for k in ("skidl", "atopile", "circuit-synth"):
            if f"-{k}-" in p.name:
                keys.append(k)
                break
    assert len(set(keys)) == 3
    assert len(picked) == 6


def test_mix_symlinks_half_half(tmp_path: Path) -> None:
    d1 = tmp_path / "d1"
    d1.mkdir()
    d2 = tmp_path / "d2"
    d2.mkdir()
    d3 = tmp_path / "d3"
    d3.mkdir()
    manifest = tmp_path / "d3_manifest.csv"
    for i in range(20):
        (d1 / f"hash{i:02d}.kicad_sch").write_text("a")
        (d2 / f"voltage_divider-skidl-{i}.kicad_sch").write_text("b")
    n = mix(d1=d1, d2=d2, d3=d3, n_total=10, seed=42, manifest_path=manifest)
    assert n == 10
    links = list(d3.iterdir())
    assert len(links) == 10
    assert all(link.is_symlink() for link in links)
