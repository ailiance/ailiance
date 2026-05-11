"""Shared result type for every Track-D compiler runner."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class CompileResult:
    """Outcome of a single LLM-DSL -> compiler invocation.

    Three independent booleans match the spec failure-mode taxonomy:
      - dsl_parse_ok : compiler accepted the DSL grammatically
      - compile_ok   : compiler emitted a .kicad_sch artefact
      - kicad_load_ok: filled in downstream by Eval N3, not by the runner

    Runners populate dsl_parse_ok and compile_ok only.
    """

    dsl_parse_ok: bool = False
    compile_ok: bool = False
    output_path: Path | None = None
    stderr: str = ""
    wall_time_ms: int = 0

    def as_dict(self) -> dict:
        return {
            "dsl_parse_ok": self.dsl_parse_ok,
            "compile_ok": self.compile_ok,
            "output_path": str(self.output_path) if self.output_path else None,
            "stderr": self.stderr,
            "wall_time_ms": self.wall_time_ms,
        }
