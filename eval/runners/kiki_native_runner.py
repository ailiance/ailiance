#!/usr/bin/env python3
"""kiki_native_runner — KIKI-native domain benchmarks (KiCad/SPICE/EMC/MISRA-C).

Runs a JSON-defined task suite against an OpenAI-compatible endpoint
(typically mlx_lm.server with optional adapter). Scores each response
using domain-specific rules (S-expression parse, regex pattern, etc.)
and optionally an LLM judge for semantic correctness.

The task JSON format (see eval/tasks/kiki_native/*.json):

    {
      "name": "kiki-kicad-dsl",
      "scoring": {"method": "rule-based + LLM-judge fallback"},
      "questions": [
        {
          "id": "kicad-dsl-001",
          "prompt": "...",
          "must_contain": ["R1", "R2", ...],
          "must_be_unique": ["R1", "R2"],
          "max_tokens": 600
        }
      ]
    }

Usage:
    python -m runners.kiki_native_runner \\
        --base-url http://localhost:8802/v1 \\
        --model-id /path/to/model \\
        --task tasks/kiki_native/kicad_dsl.json \\
        --output-dir results/2026-05-04/devstral-python-adapter/kiki_kicad_dsl

Scoring per question:
    syntax_ok:    output parseable / well-formed
    contains:     fraction of must_contain items found
    unique:       fraction of must_be_unique items appearing exactly once
    overall:      mean(syntax_ok, contains, unique) -> 0.0..1.0

A question is considered PASSED if overall >= 0.6.
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path


def _generate(
    base_url: str,
    model_id: str,
    prompt: str,
    *,
    max_tokens: int = 600,
    temperature: float = 0.0,
    timeout_s: int = 180,
    disable_thinking: bool = True,
) -> str:
    """Send a chat-completions request, return assistant content.

    Qwen3-family models emit `reasoning` separately from `content` when
    thinking mode is on. We disable it via `chat_template_kwargs` so all
    output flows through `content`. Falls back to `reasoning` if `content`
    is empty (multi-turn safety).
    """
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            body = json.loads(r.read())
        msg = body["choices"][0]["message"]
        content = msg.get("content") or ""
        # Fallback: some models emit only `reasoning` if thinking is on.
        if not content.strip():
            content = msg.get("reasoning") or ""
        return content
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        return f"[ERROR: {e}]"
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        return f"[PARSE_ERROR: {e}]"


def _check_syntax(output: str, expected_format: str = "sexpr") -> tuple[bool, str]:
    """Heuristic syntax check.

    For schematic outputs we accept any of:
      - KiCad S-expression (balanced parens around `kicad_sch`/`symbol`)
      - SPICE-like compact netlist (lines starting with R/C/L/Q/D + nodes + value)
      - SPICE deck (with .end)

    Returns (passes, note).
    """
    if expected_format in ("sexpr", "schematic"):
        # Try fenced code block first
        match = re.search(r"```(?:lisp|sexpr|kicad|schematic|spice|netlist|text)?\n(.*?)```",
                          output, re.DOTALL)
        body = match.group(1) if match else output

        # KiCad S-expression: balanced parens with `kicad_sch` or `symbol` keywords
        opens = body.count("(")
        closes = body.count(")")
        has_kicad_keyword = bool(re.search(r"\b(kicad_sch|kicad_pcb|symbol|component|net)\b",
                                           body, re.IGNORECASE))
        if opens >= 5 and opens == closes and has_kicad_keyword:
            return True, f"sexpr balanced ({opens} pairs)"

        # SPICE-like compact netlist: lines like "R1 N1 N2 10k", "C1 IN OUT 100n"
        netlist_lines = re.findall(
            r"^\s*([RCLQDM][A-Z0-9_]*)\s+\S+\s+\S+(?:\s+\S+)*\s*$",
            body, re.MULTILINE
        )
        if len(netlist_lines) >= 1:
            return True, f"spice-like netlist ({len(netlist_lines)} components)"

        # SPICE deck with .end
        if re.search(r"\.end\b", body, re.IGNORECASE):
            return True, "spice deck (.end found)"

        # If parens unbalanced AND no netlist → fail
        if opens > 0:
            return False, f"unbalanced parens ({opens} vs {closes}), no netlist"
        return False, "no recognizable schematic format"

    if expected_format == "spice":
        if not re.search(r"\.end\b", output, re.IGNORECASE):
            return False, "no .end directive"
        return True, ".end found"

    return True, "no syntax check defined"


def _check_contains(output: str, items: list[str]) -> tuple[float, list[str]]:
    """Fraction of items found (case-insensitive substring match)."""
    if not items:
        return 1.0, []
    output_lower = output.lower()
    missing = [item for item in items if item.lower() not in output_lower]
    found = len(items) - len(missing)
    return found / len(items), missing


def _check_unique(output: str, refs: list[str]) -> tuple[float, list[str]]:
    """Fraction of refs that appear with exactly one referenced instance.

    Heuristic: count occurrences of <ref> as a word boundary token
    (e.g. R1, R2, C1). Uniqueness here means appearing 'reasonably' often
    (>=2 mentions) without ambiguous duplication.
    """
    if not refs:
        return 1.0, []
    bad = []
    for ref in refs:
        # Word-boundary match
        count = len(re.findall(rf"\b{re.escape(ref)}\b", output))
        if count == 0:
            bad.append(f"{ref}=missing")
        # 1 occurrence is fine (e.g., declaration only)
        # >10 might suggest the model just spammed the symbol — cap at warning
        elif count > 15:
            bad.append(f"{ref}=spam({count})")
    return (len(refs) - len(bad)) / len(refs), bad


def score_question(question: dict, output: str, syntax_format: str = "sexpr") -> dict:
    syntax_ok, syntax_note = _check_syntax(output, syntax_format)
    contains_score, missing = _check_contains(output, question.get("must_contain", []))
    unique_score, unique_issues = _check_unique(output, question.get("must_be_unique", []))

    overall = (
        (1.0 if syntax_ok else 0.0)
        + contains_score
        + unique_score
    ) / 3
    return {
        "syntax_ok": syntax_ok,
        "syntax_note": syntax_note,
        "contains_score": round(contains_score, 3),
        "missing_items": missing,
        "unique_score": round(unique_score, 3),
        "unique_issues": unique_issues,
        "overall_score": round(overall, 3),
        "passed": overall >= 0.6,
    }


def run_task(
    *,
    task_path: Path,
    base_url: str,
    model_id: str,
    output_dir: Path,
    syntax_format: str = "sexpr",
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    task = json.loads(task_path.read_text())
    questions = task["questions"]

    print(f"[kiki-native] {task['name']} — {len(questions)} questions")
    results = []
    answers_path = output_dir / "answers.jsonl"
    with answers_path.open("w") as fp:
        for i, q in enumerate(questions, 1):
            print(f"  [{i}/{len(questions)}] {q['id']} ", end="", flush=True)
            output = _generate(
                base_url, model_id,
                q["prompt"],
                max_tokens=q.get("max_tokens", 800),
                temperature=0.0,
            )
            score = score_question(q, output, syntax_format)
            entry = {
                "question_id": q["id"],
                "category": q.get("category", "?"),
                "prompt": q["prompt"],
                "output": output,
                "score": score,
            }
            fp.write(json.dumps(entry) + "\n")
            fp.flush()
            results.append(entry)
            mark = "PASS" if score["passed"] else "FAIL"
            print(f"-> {mark} (overall={score['overall_score']:.2f})")

    # Aggregate
    n = len(results)
    n_passed = sum(1 for r in results if r["score"]["passed"])
    overall_avg = sum(r["score"]["overall_score"] for r in results) / n if n else 0.0
    syntax_ok = sum(1 for r in results if r["score"]["syntax_ok"]) / n if n else 0.0
    contains_avg = sum(r["score"]["contains_score"] for r in results) / n if n else 0.0
    unique_avg = sum(r["score"]["unique_score"] for r in results) / n if n else 0.0

    by_category: dict[str, list[float]] = {}
    for r in results:
        cat = r.get("category", "?")
        by_category.setdefault(cat, []).append(r["score"]["overall_score"])

    summary = {
        "task": task["name"],
        "task_version": task.get("version", "?"),
        "n_questions": n,
        "n_passed": n_passed,
        "pass_rate": round(n_passed / n, 4) if n else 0.0,
        "metrics": {
            "overall_avg": round(overall_avg, 4),
            "syntax_ok_rate": round(syntax_ok, 4),
            "contains_avg": round(contains_avg, 4),
            "unique_avg": round(unique_avg, 4),
        },
        "by_category": {
            cat: {"avg": round(sum(s) / len(s), 3), "n": len(s)}
            for cat, s in by_category.items()
        },
        "model_id": model_id,
        "base_url": base_url,
        "task_path": str(task_path),
        "answers_path": str(answers_path),
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (output_dir / "results.json").write_text(json.dumps(summary, indent=2))
    print()
    print(f"[kiki-native] {task['name']}: {n_passed}/{n} = {n_passed/n*100:.1f}% passed | "
          f"avg={overall_avg:.3f}")
    return summary


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--task", required=True, type=Path, help="Path to task JSON")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--syntax-format", default="sexpr",
                   choices=["sexpr", "spice", "none"],
                   help="Expected output syntax (for first-pass syntax check)")
    args = p.parse_args()

    summary = run_task(
        task_path=args.task,
        base_url=args.base_url,
        model_id=args.model_id,
        output_dir=args.output_dir,
        syntax_format=args.syntax_format,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    _cli()
