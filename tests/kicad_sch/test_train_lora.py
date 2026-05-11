"""Tests for scripts.kicad_sch.train_lora (TDD C9)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from scripts.kicad_sch.train_lora import load_config, run_train


def test_load_config_reads_lora_params(tmp_path: Path) -> None:
    cfg = {
        "model": "m",
        "data": "d",
        "lora_parameters": {"rank": 16, "scale": 2.0},
        "iters": 100,
        "seed": 42,
        "adapter_path": str(tmp_path / "a"),
    }
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    out = load_config(p)
    assert out["lora_parameters"]["rank"] == 16
    assert out["seed"] == 42


def test_run_train_invokes_mlx_lm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "model": "m",
                "data": str(tmp_path),
                "adapter_path": str(tmp_path / "ad"),
                "iters": 1,
                "seed": 42,
                "lora_parameters": {
                    "rank": 16,
                    "scale": 2.0,
                    "dropout": 0.05,
                },
            }
        )
    )
    called: dict[str, list[str]] = {}

    def fake_run(cmd, **kw):
        called["cmd"] = cmd
        r = MagicMock()
        r.returncode = 0
        r.stdout = ""
        r.stderr = ""
        return r

    monkeypatch.setattr(
        "scripts.kicad_sch.train_lora.subprocess.run", fake_run
    )
    rc = run_train(cfg, actually_run=True)
    assert rc == 0
    joined = " ".join(called["cmd"])
    assert "mlx_lm" in joined


def test_run_train_default_is_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "model": "m",
                "data": str(tmp_path),
                "adapter_path": str(tmp_path / "ad"),
                "iters": 1,
                "seed": 42,
                "lora_parameters": {"rank": 16, "scale": 2.0},
            }
        )
    )

    def fail_run(*a, **kw):  # pragma: no cover
        raise AssertionError("subprocess.run must not be called in dry-run")

    monkeypatch.setattr(
        "scripts.kicad_sch.train_lora.subprocess.run", fail_run
    )
    rc = run_train(cfg)
    assert rc == 0
