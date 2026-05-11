#!/usr/bin/env python3
"""F1 3rd-axis bench: iact-bench validator pass-rate per (model, domain).

Generates N samples per (model, domain, version) cell, validates each
completion through the chain orchestrator IactBenchValidator (or its
StubValidator fallback when iact-bench is not vendored — see issue #23),
and emits JSON arrays in the shape consumed by ``bench_comparison.py``
``--validator-base`` / ``--validator-tuned``.

Path decision (2026-05-11):
  * iact-bench submodule NOT vendored on Studio.
  * IactBenchValidator raises ValidatorUnavailable on import.
  * -> fall back to StubValidator and mark each cell ``stub_mode: true``.
  * Real Docker validators will be wired when issue #23 is closed and the
    submodule lands at ``vendored/iact-bench``.

Covered domain → validator map mirrors
``electron-bench/docs/validators-mapping-2026-05-11.md`` (commit dfe15e7).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# 13 functional-validator-covered domains -> ordered validator list.
# First validator wins; alternates fall through.
DOMAINS_VALIDATORS: dict[str, list[str]] = {
    "cpp": ["compile-cpp"],
    "html-css": ["parse-html-css"],
    "typescript": ["compile-typescript"],
    "shell": ["compile-shell"],
    "yaml-json": ["compile-yaml-json"],
    "sql": ["parse-sql"],
    "rust-embedded": ["compile-rust-embedded"],
    "platformio": ["compile-platformio"],
    "embedded": ["idf-build"],
    "kicad-dsl": ["atopile-build", "skidl", "tscircuit", "circuit-synth"],
    "kicad-pcb": ["kicad-drc", "kicad-erc"],
    "spice-sim": ["ngspice-converge", "xyce", "lcapy"],
    "freecad": ["freecad-script"],
}

MODELS = ["apertus", "devstral", "eurollm", "qwen36", "medium35"]


@dataclass
class CellResult:
    model_key: str
    domain: str
    version: str  # base | tuned
    pass_rate: float
    n_samples: int
    passed: int
    stub_mode: bool
    validator_name: str
    errors: list[dict]
    duration_s: float


def _build_validator():
    """Return (validator, stub_mode_bool, validator_name)."""
    from orchestrator.validators import (
        IactBenchValidator,
        StubValidator,
        ValidatorUnavailable,
    )

    # Try IactBench eagerly: instantiate + force import in async ctx.
    v = IactBenchValidator()
    try:
        v._load_runner()  # type: ignore[attr-defined]
        return v, False, "iact-bench"
    except ValidatorUnavailable as exc:
        print(f"  IactBenchValidator unavailable ({exc}); falling back to StubValidator")
        return StubValidator(), True, "stub"


def load_eval_prompts(domain: str, n: int = 10) -> list[str]:
    """Pull N prompts from data/hf-traced/<domain>/valid.jsonl (user msg)."""
    valid_path = REPO_ROOT / "data" / "hf-traced" / domain / "valid.jsonl"
    prompts: list[str] = []
    if not valid_path.exists():
        # Fallback to eval_framework DOMAIN_PROMPTS
        try:
            from eval_framework import DOMAIN_PROMPTS, DEFAULT_PROMPTS  # type: ignore
            prompts = list(DOMAIN_PROMPTS.get(domain, DEFAULT_PROMPTS))
        except Exception:
            prompts = []
        return prompts[:n]

    with valid_path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            msgs = rec.get("messages") or []
            user_msg = next(
                (m.get("content", "") for m in msgs if m.get("role") == "user"),
                None,
            )
            if user_msg:
                prompts.append(user_msg)
            if len(prompts) >= n:
                break
    return prompts[:n]


def find_adapter(model_key: str, domain: str, version_tag: str = "v1") -> Path | None:
    base = REPO_ROOT / "output" / ("adapters" if version_tag == "v1" else "adapters-v2")
    cand = base / model_key / domain
    if (cand / "adapters.safetensors").exists():
        return cand
    return None


def build_cell_matrix(version: str) -> tuple[list[tuple[str, str, Path | None]], list[dict]]:
    """Return (runnable_cells, skipped_cells)."""
    runnable: list[tuple[str, str, Path | None]] = []
    skipped: list[dict] = []
    for model_key in MODELS:
        for domain in DOMAINS_VALIDATORS:
            adapter = find_adapter(model_key, domain, "v1") if version == "tuned" else None
            if version == "tuned" and adapter is None:
                skipped.append(
                    {"model_key": model_key, "domain": domain, "version": version,
                     "reason": "no_v1_adapter"}
                )
                continue
            runnable.append((model_key, domain, adapter))
    return runnable, skipped


async def _validate(validator, completion: str, domain: str, tool: str):
    return await validator.run(completion, domain=domain, tool=tool)


def run_cell(
    model_key: str,
    domain: str,
    version: str,
    adapter_path: Path | None,
    n_samples: int,
    *,
    smoke: bool = False,
) -> CellResult:
    """Generate n_samples for one cell, validate, return CellResult."""
    from eval_framework import (  # type: ignore
        MODELS as EVAL_MODELS,
        load_model_and_tokenizer,
        generate_sample,
        unload_model,
    )

    model_info = EVAL_MODELS[model_key]
    prompts = load_eval_prompts(domain, n=n_samples)
    if not prompts:
        return CellResult(model_key, domain, version, 0.0, 0, 0, True,
                          "no-prompts", [{"error": "no prompts available"}], 0.0)

    validator, stub_mode, vname = _build_validator()

    t0 = time.time()
    passed = 0
    errors: list[dict] = []
    if smoke:
        # Skip heavy model load — emit synthetic completions.
        completions = [f"// smoke completion {i}" for i in range(len(prompts))]
    else:
        model, tokenizer = load_model_and_tokenizer(
            model_info["path"], str(adapter_path) if adapter_path else None,
        )
        completions = []
        for p in prompts:
            try:
                resp, _, _ = generate_sample(model, tokenizer, p, max_tokens=512)
                completions.append(resp)
            except Exception as exc:
                completions.append("")
                errors.append({"phase": "generate", "error": str(exc)[:200]})
        del model, tokenizer
        unload_model()

    loop = asyncio.new_event_loop()
    try:
        for i, completion in enumerate(completions):
            cell_passed = False
            for tool in DOMAINS_VALIDATORS[domain]:
                try:
                    result = loop.run_until_complete(
                        _validate(validator, completion, domain, tool)
                    )
                except Exception as exc:
                    errors.append({"prompt_idx": i, "validator": tool,
                                   "exc": str(exc)[:200]})
                    continue
                if result.exit_code == 0:
                    cell_passed = True
                    break
                errors.append({"prompt_idx": i, "validator": tool,
                               "stderr": (result.stderr or "")[:200]})
            if cell_passed:
                passed += 1
    finally:
        loop.close()

    elapsed = round(time.time() - t0, 2)
    pass_rate = passed / max(1, len(completions))
    return CellResult(
        model_key=model_key, domain=domain, version=version,
        pass_rate=round(pass_rate, 4), n_samples=len(completions),
        passed=passed, stub_mode=stub_mode, validator_name=vname,
        errors=errors[:5], duration_s=elapsed,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--versions", default="base,tuned",
                   help="comma list of base|tuned")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "output" / "iact_validator_runs"))
    p.add_argument("--date", default="2026-05-11")
    p.add_argument("--smoke", action="store_true",
                   help="skip model load, emit synthetic completions (wire-up test)")
    p.add_argument("--only-cell", default=None,
                   help="MODEL:DOMAIN:VERSION — run a single cell and print result")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.only_cell:
        m, d, v = args.only_cell.split(":")
        adapter = find_adapter(m, d, "v1") if v == "tuned" else None
        if v == "tuned" and adapter is None:
            print(json.dumps({"skipped": True, "reason": "no_v1_adapter"}))
            return 0
        res = run_cell(m, d, v, adapter, args.n_samples, smoke=args.smoke)
        print(json.dumps(asdict(res), indent=2))
        return 0

    versions = [v.strip() for v in args.versions.split(",") if v.strip()]
    for version in versions:
        cells, skipped = build_cell_matrix(version)
        print(f"[{version}] {len(cells)} runnable cells, {len(skipped)} skipped")
        results: list[dict] = []
        for (m, d, adapter) in cells:
            print(f"  -> {m}/{d}/{version} (adapter={adapter})")
            try:
                res = run_cell(m, d, version, adapter, args.n_samples,
                               smoke=args.smoke)
                results.append(asdict(res))
            except Exception as exc:
                tb = traceback.format_exc()
                print(f"     FAILED: {exc}\n{tb}")
                results.append({
                    "model_key": m, "domain": d, "version": version,
                    "pass_rate": 0.0, "n_samples": 0, "passed": 0,
                    "stub_mode": True, "validator_name": "error",
                    "errors": [{"exc": str(exc)[:200]}], "duration_s": 0.0,
                })

        out_main = out_dir / f"{version}_{args.date}.json"
        out_main.write_text(json.dumps(results, indent=2))
        out_skip = out_dir / f"{version}_{args.date}_skipped.json"
        out_skip.write_text(json.dumps(skipped, indent=2))
        print(f"[{version}] wrote {out_main} ({len(results)} rows) + {out_skip} ({len(skipped)} skipped)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
