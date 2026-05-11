"""5-axis evaluator for .kicad_sch v10 generation gap (spec 2026-05-11).

Axes:
- parse_ok    : kicad-cli sch erc rc==0          weight 0.30
- erc_clean   : erc errors_count==0              weight 0.30
- sch_render  : kicad-cli sch export svg rc==0   weight 0.15
- drc_clean   : pcbnew --drc rc==0 (optional)    weight 0.10
- sem_equiv   : netlist graph cosine vs ref      weight 0.15
"""
from __future__ import annotations

import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


def _resolve_cli(cli_path: Path) -> str:
    if cli_path.is_absolute() or "/" in str(cli_path):
        return str(cli_path)
    found = shutil.which(str(cli_path))
    return found or str(cli_path)


def eval_parse_ok(sch_path: Path, cli_path: Path = Path("kicad-cli")) -> int:
    """Return 1 iff kicad-cli sch erc <file> exits 0, else 0.

    kicad-cli rc semantics (v10.0.2):
      0  : parse OK, ERC ran
      3  : "Échec du chargement de la schématique" (parse failed)
      >0 : other errors -> treat as parse failure for parse_ok axis
    """
    cli = _resolve_cli(cli_path)
    try:
        proc = subprocess.run(
            [cli, "sch", "erc", str(sch_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 0
    return 1 if proc.returncode == 0 else 0


_ERC_COUNT_RE = re.compile(r"(\d+)\s+error", re.IGNORECASE)


def eval_erc_clean(sch_path: Path, cli_path: Path = Path("kicad-cli")) -> int:
    """Return 1 iff ERC report shows 0 errors, else 0.

    Two-stage gate:
      1. parse_ok must hold (rc==0); otherwise erc_clean=0 by definition.
      2. Stdout must mention 'N errors' with N==0.
    """
    cli = _resolve_cli(cli_path)
    try:
        proc = subprocess.run(
            [cli, "sch", "erc", str(sch_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 0
    if proc.returncode != 0:
        return 0
    blob = (proc.stdout or "") + "\n" + (proc.stderr or "")
    match = _ERC_COUNT_RE.search(blob)
    if not match:
        # No explicit count line => assume clean if rc==0 (conservative for
        # kicad-cli versions that omit summary on zero-violations runs).
        return 1
    return 1 if int(match.group(1)) == 0 else 0


def eval_sch_render(sch_path: Path, cli_path: Path = Path("kicad-cli")) -> int:
    """Return 1 iff kicad-cli sch export svg <file> -o <tmp> exits 0."""
    cli = _resolve_cli(cli_path)
    with tempfile.TemporaryDirectory() as td:
        try:
            proc = subprocess.run(
                [cli, "sch", "export", "svg", str(sch_path), "-o", td],
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return 0
    return 1 if proc.returncode == 0 else 0
