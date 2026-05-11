"""SKiDL -> kicad_sch runner.

Strategy: write DSL to <out_dir>/circuit.py, run it via subprocess so the
runner is isolated from skidl global state, then scan <out_dir>/*.kicad_sch
for the artefact.
"""
from __future__ import annotations

import ast
import subprocess
import sys
import time
from pathlib import Path

from .result import CompileResult


def run(dsl: str, out_dir: Path, timeout_s: int = 60) -> CompileResult:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    script = out_dir / "circuit.py"
    script.write_text(dsl)

    # Cheap syntax gate via ast before paying for a subprocess.
    try:
        ast.parse(dsl)
    except SyntaxError as e:
        return CompileResult(
            dsl_parse_ok=False, compile_ok=False, output_path=None,
            stderr=f"SyntaxError: {e}", wall_time_ms=0,
        )

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(out_dir),
            capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as e:
        return CompileResult(
            dsl_parse_ok=True, compile_ok=False, output_path=None,
            stderr=f"timeout {timeout_s}s: {e}",
            wall_time_ms=int((time.monotonic() - t0) * 1000),
        )
    wall = int((time.monotonic() - t0) * 1000)

    schs = sorted(out_dir.glob("*.kicad_sch"))
    if proc.returncode == 0 and schs:
        return CompileResult(
            dsl_parse_ok=True, compile_ok=True, output_path=schs[0],
            stderr=proc.stderr, wall_time_ms=wall,
        )
    return CompileResult(
        dsl_parse_ok=True, compile_ok=False, output_path=None,
        stderr=proc.stderr or proc.stdout, wall_time_ms=wall,
    )
