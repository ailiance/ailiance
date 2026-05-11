"""circuit_synth -> kicad_sch runner."""
from __future__ import annotations

import ast
import subprocess
import sys
import time
from pathlib import Path

from .result import CompileResult

_DRIVER = """\
import sys, importlib.util
spec = importlib.util.spec_from_file_location('user_circuit', 'circuit.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
c = mod.build()
c.to_kicad_sch('out.kicad_sch')
"""


def run(dsl: str, out_dir: Path, timeout_s: int = 90) -> CompileResult:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "circuit.py").write_text(dsl)
    (out_dir / "_driver.py").write_text(_DRIVER)

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
            [sys.executable, "_driver.py"],
            cwd=str(out_dir), capture_output=True, text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as e:
        return CompileResult(
            dsl_parse_ok=True, compile_ok=False, output_path=None,
            stderr=f"timeout {timeout_s}s: {e}",
            wall_time_ms=int((time.monotonic() - t0) * 1000),
        )
    wall = int((time.monotonic() - t0) * 1000)

    out = out_dir / "out.kicad_sch"
    if proc.returncode == 0 and out.exists():
        return CompileResult(
            dsl_parse_ok=True, compile_ok=True, output_path=out,
            stderr=proc.stderr, wall_time_ms=wall,
        )
    return CompileResult(
        dsl_parse_ok=True, compile_ok=False, output_path=None,
        stderr=proc.stderr or proc.stdout, wall_time_ms=wall,
    )
