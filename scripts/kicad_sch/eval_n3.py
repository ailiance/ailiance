"""5-axis evaluator for .kicad_sch v10 generation gap (spec 2026-05-11).

Axes:
- parse_ok    : kicad-cli sch erc rc==0          weight 0.30
- erc_clean   : erc errors_count==0              weight 0.30
- sch_render  : kicad-cli sch export svg rc==0   weight 0.15
- drc_clean   : pcbnew --drc rc==0 (optional)    weight 0.10
- sem_equiv   : netlist graph cosine vs ref      weight 0.15
"""
from __future__ import annotations

import shutil
import subprocess
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
