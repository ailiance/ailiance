"""Track-D orchestrator: 5 base models x 4 compilers = 20 hybrid pipelines.

Exposes `run_cell()` for one (model, compiler, prompt) cell, and `run_all()`
for the full grid. Inference-only (no LoRA). Logs every attempt to NDJSON
via AuditLogger.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.kicad_sch.audit_log import AuditLogger
from scripts.eval_framework import (
    MODELS, load_model_and_tokenizer, generate_sample, unload_model,
)
from scripts.kicad_sch.compilers import (
    skidl_runner, atopile_runner, tscircuit_runner, circuit_synth_runner,
)
from scripts.kicad_sch.compilers.system_prompts import SYSTEM_PROMPTS

# Module-level so tests can monkeypatch a single compiler.
RUNNERS: dict[str, Any] = {
    "skidl": skidl_runner,
    "atopile": atopile_runner,
    "tscircuit": tscircuit_runner,
    "circuit-synth": circuit_synth_runner,
}

BASE_MODELS = ("apertus", "devstral", "eurollm", "qwen36", "medium35")
COMPILERS = ("skidl", "atopile", "tscircuit", "circuit-synth")
DEFAULT_SEEDS = (42, 137, 1024, 8675309, 31415)


def _build_prompt(compiler: str, user_prompt: str) -> str:
    return f"{SYSTEM_PROMPTS[compiler]}\n\nCircuit: {user_prompt}\n"


def run_cell(
    *,
    base_model_key: str,
    compiler: str,
    prompt: str,
    seeds: list[int],
    n_samples: int,
    out_dir: Path,
    audit_logger: AuditLogger,
    model_tok: tuple | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.2,
) -> dict:
    """Run one (model, compiler, prompt) cell across seeds x samples.

    If `model_tok` is provided the caller has already loaded the model
    (used by `run_all` to amortise loads). Otherwise the cell loads on
    demand.
    """
    if compiler not in RUNNERS:
        raise ValueError(f"unknown compiler {compiler!r}")
    if base_model_key not in MODELS:
        raise ValueError(f"unknown base model {base_model_key!r}")

    loaded_here = False
    if model_tok is None:
        model, tok = load_model_and_tokenizer(MODELS[base_model_key]["path"])
        loaded_here = True
    else:
        model, tok = model_tok

    full_prompt = _build_prompt(compiler, prompt)
    runner = RUNNERS[compiler]

    n_parse_ok = 0
    n_compile_ok = 0
    n_attempts = 0
    out_dir = Path(out_dir)

    try:
        for seed in seeds:
            for sample_idx in range(n_samples):
                dsl = generate_sample(
                    model, tok, full_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=seed,
                )
                cell_dir = (
                    out_dir / base_model_key / compiler
                    / f"seed-{seed}" / f"s{sample_idx}"
                )
                result = runner.run(dsl, cell_dir)
                audit_logger.log(
                    "generation",
                    base_model_key=base_model_key,
                    compiler=compiler,
                    prompt=prompt,
                    seed=seed,
                    sample_idx=sample_idx,
                    dsl_parse_ok=result.dsl_parse_ok,
                    compile_ok=result.compile_ok,
                    output_path=(
                        str(result.output_path)
                        if result.output_path else None
                    ),
                    wall_time_ms=result.wall_time_ms,
                    stderr_tail=result.stderr[-500:] if result.stderr else "",
                )
                n_attempts += 1
                if result.dsl_parse_ok:
                    n_parse_ok += 1
                if result.compile_ok:
                    n_compile_ok += 1
    finally:
        if loaded_here:
            unload_model()

    return {
        "base_model_key": base_model_key,
        "compiler": compiler,
        "prompt": prompt,
        "n_attempts": n_attempts,
        "dsl_parse_ok_rate": (
            n_parse_ok / n_attempts if n_attempts else 0.0
        ),
        "compile_ok_rate": (
            n_compile_ok / n_attempts if n_attempts else 0.0
        ),
    }
