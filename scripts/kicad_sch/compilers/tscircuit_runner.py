"""tscircuit (.tsx) -> kicad_sch runner."""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

from .result import CompileResult


def run(dsl: str, out_dir: Path, timeout_s: int = 180) -> CompileResult:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tsx_file = out_dir / "circuit.tsx"
    tsx_file.write_text(dsl)

    if shutil.which("npx") is None:
        return CompileResult(
            dsl_parse_ok=False, compile_ok=False, output_path=None,
            stderr="npx not installed", wall_time_ms=0,
        )

    # Pre-flight: real TSX contains `import ` or `export `.
    if not any(kw in dsl for kw in ("import ", "export ")):
        return CompileResult(
            dsl_parse_ok=False, compile_ok=False, output_path=None,
            stderr="pre-flight: no import/export keyword",
            wall_time_ms=0,
        )

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            ["npx", "--no-install", "tsci", "build",
             "--input", str(tsx_file),
             "--output-format", "kicad_sch",
             "--output-dir", str(out_dir)],
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

    stderr = proc.stderr or ""
    parse_fail_markers = ("ts1005", "ts1109", "unexpected token",
                          "syntaxerror", "parse error")
    if any(m in stderr.lower() for m in parse_fail_markers):
        return CompileResult(
            dsl_parse_ok=False, compile_ok=False, output_path=None,
            stderr=stderr, wall_time_ms=wall,
        )

    schs = sorted(out_dir.rglob("*.kicad_sch"))
    if proc.returncode == 0 and schs:
        return CompileResult(
            dsl_parse_ok=True, compile_ok=True, output_path=schs[0],
            stderr=stderr, wall_time_ms=wall,
        )
    return CompileResult(
        dsl_parse_ok=True, compile_ok=False, output_path=None,
        stderr=stderr, wall_time_ms=wall,
    )
