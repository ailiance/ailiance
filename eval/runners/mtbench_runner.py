#!/usr/bin/env python3
"""mtbench_runner — MT-Bench evaluation via fastchat against an MLX endpoint.

MT-Bench (Zheng et al. 2023) — 80 multi-turn chat prompts across 8 categories
(writing, roleplay, reasoning, math, coding, extraction, stem, humanities).
Each prompt has 2 turns. A strong judge LLM (default: Mistral-Medium-128B-4bit
local on Studio :8500) scores each response 1-10.

The judge model is logged in env.json for reproducibility — never use a
closed-source-only judge if results need to be reproducible.

Usage:
    python -m runners.mtbench_runner \\
        --base-url http://localhost:8810/v1 \\
        --model-id devstral-python \\
        --model-path /Users/clems/KIKI-Mac_tunner/models/Devstral-Small-2-24B-MLX-4bit \\
        --judge-base-url http://localhost:8500/v1 \\
        --judge-model mlx-community/Mistral-Medium-3.5-128B-MLX-4bit \\
        --output-dir results/2026-05-04/devstral-python/mtbench \\
        --questions data/mtbench/question.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def _ensure_fschat() -> None:
    try:
        import fastchat  # noqa: F401
    except ImportError:
        sys.exit(
            "fastchat not installed. Install with: "
            "uv pip install 'fschat[model_worker,llm_judge]'"
        )


# Default MT-Bench question source (HF or fastchat repo)
DEFAULT_QUESTIONS_URL = (
    "https://raw.githubusercontent.com/lm-sys/FastChat/main/"
    "fastchat/llm_judge/data/mt_bench/question.jsonl"
)


def fetch_questions(dest: Path) -> Path:
    """Download MT-Bench questions if not already present."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[mtbench] Downloading questions → {dest}")
    urllib.request.urlretrieve(DEFAULT_QUESTIONS_URL, dest)
    return dest


def generate_answers(
    *,
    base_url: str,
    model_id: str,
    questions_path: Path,
    answer_path: Path,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    n_choices: int = 1,
    max_questions: int | None = None,
) -> Path:
    """Run inference loop: N questions × 2 turns each, write JSONL.

    Use max_questions to limit (smoke test). Default = all 80.
    """
    answer_path.parent.mkdir(parents=True, exist_ok=True)

    # Load questions
    questions = []
    with questions_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    if max_questions is not None:
        questions = questions[:max_questions]

    print(f"[mtbench] {len(questions)} questions × 2 turns = {len(questions)*2} generations")

    answers = []
    import urllib.error
    import urllib.request

    for i, q in enumerate(questions):
        choices = []
        for _ in range(n_choices):
            messages = []
            turns_out = []
            for turn_text in q["turns"]:
                messages.append({"role": "user", "content": turn_text})
                payload = {
                    "model": model_id,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False,
                }
                req = urllib.request.Request(
                    f"{base_url.rstrip('/')}/chat/completions",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                )
                try:
                    with urllib.request.urlopen(req, timeout=120) as r:
                        body = json.loads(r.read())
                except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
                    body = {"error": str(e)}

                content = ""
                if "choices" in body:
                    content = body["choices"][0]["message"]["content"]
                elif "error" in body:
                    content = f"[ERROR: {body['error']}]"
                turns_out.append(content)
                messages.append({"role": "assistant", "content": content})
            choices.append({"index": 0, "turns": turns_out})

        answers.append({
            "question_id": q["question_id"],
            "answer_id": f"{model_id.replace('/', '_')}_{q['question_id']}",
            "model_id": model_id,
            "choices": choices,
            "tstamp": time.time(),
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(questions):
            print(f"[mtbench] {i+1}/{len(questions)} done")

    with answer_path.open("w") as f:
        for a in answers:
            f.write(json.dumps(a) + "\n")
    print(f"[mtbench] Answers → {answer_path}")
    return answer_path


def run_judge(
    *,
    answer_path: Path,
    questions_path: Path,
    judge_base_url: str,
    judge_model: str,
    judgment_path: Path,
    mode: str = "single",
) -> Path:
    """Score answers with judge LLM via single-grading mode (1-10 per turn)."""
    judgment_path.parent.mkdir(parents=True, exist_ok=True)

    questions = {}
    with questions_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                q = json.loads(line)
                questions[q["question_id"]] = q

    answers = []
    with answer_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                answers.append(json.loads(line))

    judgments = []
    import urllib.error
    import urllib.request

    JUDGE_PROMPT_SINGLE = (
        "Please act as an impartial judge and evaluate the quality of the response "
        "provided by an AI assistant to the user question displayed below. "
        "Your evaluation should consider factors such as the helpfulness, relevance, "
        "accuracy, depth, creativity, and level of detail of the response. "
        "Begin your evaluation by providing a short explanation. Be as objective as "
        "possible. After providing your explanation, please rate the response on a "
        "scale of 1 to 10 by strictly following this format: \"[[rating]]\", for "
        "example: \"Rating: [[5]]\".\n\n"
        "[Question]\n{question}\n\n"
        "[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]"
    )

    def _judge_one(question_text: str, answer_text: str) -> dict:
        prompt = JUDGE_PROMPT_SINGLE.format(question=question_text, answer=answer_text)
        payload = {
            "model": judge_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
            "temperature": 0.0,
        }
        req = urllib.request.Request(
            f"{judge_base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                body = json.loads(r.read())
            content = body["choices"][0]["message"]["content"]
        except Exception as e:
            return {"score": None, "raw": f"[ERROR: {e}]"}

        # Extract [[rating]]
        score = None
        import re
        m = re.search(r"\[\[(\d+(?:\.\d+)?)\]\]", content)
        if m:
            try:
                score = float(m.group(1))
            except ValueError:
                pass
        return {"score": score, "raw": content}

    for ans in answers:
        qid = ans["question_id"]
        if qid not in questions:
            continue
        q = questions[qid]
        per_turn = []
        for turn_idx, turn in enumerate(q["turns"]):
            answer_turn = ans["choices"][0]["turns"][turn_idx] if turn_idx < len(ans["choices"][0]["turns"]) else ""
            verdict = _judge_one(turn, answer_turn)
            per_turn.append({
                "turn": turn_idx + 1,
                "question": turn,
                "answer": answer_turn,
                "score": verdict["score"],
                "judge_raw": verdict["raw"],
            })
        judgments.append({
            "question_id": qid,
            "category": q.get("category", "?"),
            "model_id": ans["model_id"],
            "judge_model": judge_model,
            "turns": per_turn,
        })
        scores = [t["score"] for t in per_turn if t["score"] is not None]
        avg = sum(scores) / len(scores) if scores else None
        print(f"[mtbench-judge] qid={qid} cat={q.get('category', '?')} avg={avg}")

    with judgment_path.open("w") as f:
        for j in judgments:
            f.write(json.dumps(j) + "\n")
    print(f"[mtbench] Judgments → {judgment_path}")
    return judgment_path


def aggregate(judgment_path: Path) -> dict:
    """Compute per-category and overall averages."""
    by_category: dict[str, list[float]] = {}
    by_turn: dict[int, list[float]] = {1: [], 2: []}
    overall: list[float] = []

    with judgment_path.open() as f:
        for line in f:
            j = json.loads(line)
            cat = j.get("category", "?")
            for turn in j["turns"]:
                if turn["score"] is None:
                    continue
                overall.append(turn["score"])
                by_category.setdefault(cat, []).append(turn["score"])
                by_turn.setdefault(turn["turn"], []).append(turn["score"])

    def _avg(xs: list[float]) -> float | None:
        return round(sum(xs) / len(xs), 3) if xs else None

    return {
        "overall_score": _avg(overall),
        "n_judged_turns": len(overall),
        "by_category": {cat: {"avg": _avg(scores), "n": len(scores)}
                        for cat, scores in by_category.items()},
        "by_turn": {str(k): {"avg": _avg(v), "n": len(v)}
                    for k, v in by_turn.items()},
    }


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", required=True, help="OpenAI-compat endpoint of model under test")
    p.add_argument("--model-id", required=True, help="Model id to send in API calls")
    p.add_argument("--judge-base-url", required=True, help="OpenAI-compat endpoint of judge LLM")
    p.add_argument("--judge-model", required=True, help="Judge model id")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--questions", type=Path, default=None,
                   help="Path to MT-Bench question.jsonl (auto-downloaded if missing)")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.7,
                   help="MT-Bench standard is 0.7 for chat models")
    p.add_argument("--max-questions", type=int, default=None,
                   help="Limit number of questions (smoke). Default: all 80.")
    p.add_argument("--skip-generate", action="store_true",
                   help="Skip answer generation (re-judge existing answers)")
    p.add_argument("--skip-judge", action="store_true")
    args = p.parse_args()

    _ensure_fschat()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    questions_path = args.questions or (out / "questions.jsonl")
    fetch_questions(questions_path)

    answer_path = out / "answers.jsonl"
    judgment_path = out / "judgments.jsonl"

    if not args.skip_generate:
        generate_answers(
            base_url=args.base_url,
            model_id=args.model_id,
            questions_path=questions_path,
            answer_path=answer_path,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            max_questions=args.max_questions,
        )

    if not args.skip_judge:
        run_judge(
            answer_path=answer_path,
            questions_path=questions_path,
            judge_base_url=args.judge_base_url,
            judge_model=args.judge_model,
            judgment_path=judgment_path,
        )

    summary = aggregate(judgment_path)
    summary["model_id"] = args.model_id
    summary["judge_model"] = args.judge_model
    summary["base_url"] = args.base_url
    summary["judge_base_url"] = args.judge_base_url
    summary["temperature"] = args.temperature
    summary["max_tokens"] = args.max_tokens
    summary["evaluated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (out / "results.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    _cli()
