"""Track C LoRA training orchestrator wrapping ``mlx_lm.lora``.

Defaults to dry-run: actual training is gated behind ``--actually-run``
to prevent accidental GPU kick-off while F1 / other heavy jobs share
the Studio. Audit events are emitted regardless so every invocation
is traceable.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from scripts.kicad_sch.audit_log import AuditLogger


def load_config(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _audit_path(adapter: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = Path.home() / "eu-kiki/output/audit/kicad-sch-2026-05-11"
    return base / f"train-{adapter.name}-{stamp}.ndjson"


def run_train(config_path: Path, actually_run: bool = False) -> int:
    """Run (or dry-print) the mlx_lm.lora command for ``config_path``.

    Returns the underlying ``subprocess`` exit code, or ``0`` in
    dry-run mode.
    """
    cfg = load_config(config_path)
    adapter = Path(cfg["adapter_path"])
    adapter.mkdir(parents=True, exist_ok=True)
    log = AuditLogger(_audit_path(adapter))
    log.log(
        "train_start",
        config=str(config_path),
        adapter=str(adapter),
        model=cfg["model"],
        iters=cfg.get("iters"),
        seed=cfg.get("seed"),
    )
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm.lora",
        "--config",
        str(config_path),
    ]
    if not actually_run:
        log.log("train_dry_run", cmd=" ".join(cmd))
        print(" ".join(cmd))
        return 0
    proc = subprocess.run(cmd, capture_output=False)
    log.log("train_done", rc=proc.returncode)
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument(
        "--actually-run",
        action="store_true",
        help="Disarm dry-run safety and launch real mlx_lm.lora training.",
    )
    a = p.parse_args(argv)
    return run_train(a.config, actually_run=a.actually_run)


if __name__ == "__main__":
    sys.exit(main())
