"""Unit tests for the v0.3.0 ChainOrchestrator.

All tests stay free of network / Docker by injecting a fake LLM call
function and a stub validator. The goal is to lock the Deliberation
loop semantics — number of steps, retry budget, audit-file shape —
not to exercise any worker.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from src.orchestrator.chain_orchestrator import ChainOrchestrator
from src.orchestrator.chain_policy import ChainPolicy
from src.orchestrator.validators import (
    StubValidator,
    Validator,
    ValidatorResult,
    ValidatorUnavailable,
)

POLICIES_PATH = Path("configs/chain_policies.yaml")
REFLECTOR_PATH = Path("configs/reflector_prompts.yaml")


def _make_llm(
    outputs: list[str],
) -> tuple[Callable[..., Awaitable[str]], list[list[dict[str, Any]]]]:
    """Return an async LLM caller that returns scripted outputs.

    Also returns the list of message arrays each call received, so
    tests can assert that reflector prompts include stderr.
    """
    seen: list[list[dict[str, Any]]] = []
    idx = {"i": 0}

    async def call(messages: list[dict[str, Any]], model: str) -> str:
        seen.append(messages)
        i = idx["i"]
        idx["i"] += 1
        if i >= len(outputs):
            return outputs[-1]
        return outputs[i]

    return call, seen


class _ScriptedValidator:
    """Validator that returns scripted exit codes one per call."""

    def __init__(self, exits: list[int], stderr: str = "boom") -> None:
        self.exits = exits
        self.stderr = stderr
        self.calls = 0

    async def run(
        self, output: str, *, domain: str, tool: str
    ) -> ValidatorResult:
        i = self.calls
        self.calls += 1
        code = self.exits[i] if i < len(self.exits) else self.exits[-1]
        return ValidatorResult(
            exit_code=code,
            stdout="",
            stderr=self.stderr if code != 0 else "",
            duration_s=0.001,
            image_digest="sha256:test",
        )


@pytest.mark.asyncio
async def test_direct_policy_passes_through(tmp_path: Path) -> None:
    llm, seen = _make_llm(["hello world"])
    orch = ChainOrchestrator(
        policies_path=POLICIES_PATH,
        reflector_path=REFLECTOR_PATH,
        validator=StubValidator(),
        llm_call=llm,
        audit_dir=tmp_path,
    )
    result = await orch.execute(
        "ping",
        domain="docker-devops",  # mapped to direct in chain_policies.yaml
        model="ailiance",
    )
    assert result.status == "direct"
    assert result.policy == ChainPolicy.DIRECT
    assert result.final_output == "hello world"
    assert len(result.steps) == 1
    assert result.steps[0].kind == "llm"
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_deliberate_first_attempt_passes(tmp_path: Path) -> None:
    llm, _ = _make_llm(["good output"])
    validator = _ScriptedValidator([0])
    orch = ChainOrchestrator(
        policies_path=POLICIES_PATH,
        reflector_path=REFLECTOR_PATH,
        validator=validator,
        llm_call=llm,
        audit_dir=tmp_path,
    )
    result = await orch.execute(
        "make a kicad pcb",
        domain="kicad-pcb",  # deliberate, max_retries=2
        model="ailiance",
    )
    assert result.status == "ok"
    assert result.policy == ChainPolicy.DELIBERATE
    assert result.final_output == "good output"
    # 2 steps: llm + validator.
    assert len(result.steps) == 2
    kinds = [s.kind for s in result.steps]
    assert kinds == ["llm", "validator"]
    assert result.steps[1].success
    assert result.steps[0].attempt == 1


@pytest.mark.asyncio
async def test_deliberate_retry_then_succeed(tmp_path: Path) -> None:
    llm, seen = _make_llm(["bad", "good"])
    validator = _ScriptedValidator([1, 0], stderr="DRC_ERR_HOLE_TOO_SMALL")
    orch = ChainOrchestrator(
        policies_path=POLICIES_PATH,
        reflector_path=REFLECTOR_PATH,
        validator=validator,
        llm_call=llm,
        audit_dir=tmp_path,
    )
    result = await orch.execute(
        "make a kicad pcb",
        domain="kicad-pcb",
        model="ailiance",
    )
    assert result.status == "ok"
    assert result.final_output == "good"
    # 4 steps: llm, validator-fail, reflector, validator-pass.
    kinds = [s.kind for s in result.steps]
    assert kinds == ["llm", "validator", "reflector", "validator"]
    # Reflector prompt must include the stderr.
    second_call_messages = seen[1]
    last_user = second_call_messages[-1]
    assert last_user["role"] == "user"
    assert "DRC_ERR_HOLE_TOO_SMALL" in last_user["content"]
    # Attempt counter advanced.
    assert result.steps[2].attempt == 2


@pytest.mark.asyncio
async def test_deliberate_exhausts_retries(tmp_path: Path) -> None:
    llm, _ = _make_llm(["a", "b", "c"])
    validator = _ScriptedValidator([1, 1, 1])
    orch = ChainOrchestrator(
        policies_path=POLICIES_PATH,
        reflector_path=REFLECTOR_PATH,
        validator=validator,
        llm_call=llm,
        audit_dir=tmp_path,
    )
    # Use a domain with max_retries=2 (kicad-pcb).
    result = await orch.execute(
        "x",
        domain="kicad-pcb",
        model="ailiance",
    )
    assert result.status == "exhausted"
    # 3 attempts × 2 steps = 6 steps total (1 llm + 1 validator each).
    #   attempt 1: llm + validator-fail (2)
    #   attempt 2: reflector + validator-fail (2)
    #   attempt 3: reflector + validator-fail (2)
    # = 6 steps. max_retries=2 means "retry twice after the first
    # attempt", so attempt budget = 3 and each contributes 2 steps.
    assert len(result.steps) == 6
    # Last step must be a failed validator.
    assert result.steps[-1].kind == "validator"
    assert not result.steps[-1].success


@pytest.mark.asyncio
async def test_audit_ndjson_written(tmp_path: Path) -> None:
    llm, _ = _make_llm(["good"])
    validator = _ScriptedValidator([0])
    orch = ChainOrchestrator(
        policies_path=POLICIES_PATH,
        reflector_path=REFLECTOR_PATH,
        validator=validator,
        llm_call=llm,
        audit_dir=tmp_path,
    )
    result = await orch.execute(
        "x",
        domain="kicad-pcb",
        model="ailiance",
    )
    audit_file = tmp_path / "chains" / result.chain_id / "cells.ndjson"
    assert audit_file.exists()
    lines = audit_file.read_text().strip().splitlines()
    assert len(lines) == len(result.steps)
    for line in lines:
        json.loads(line)  # valid JSON per line


@pytest.mark.asyncio
async def test_reflector_template_default_fallback(tmp_path: Path) -> None:
    llm, seen = _make_llm(["bad", "good"])
    validator = _ScriptedValidator(
        [1, 0], stderr="UNKNOWN_DOMAIN_STDERR"
    )
    orch = ChainOrchestrator(
        policies_path=POLICIES_PATH,
        reflector_path=REFLECTOR_PATH,
        validator=validator,
        llm_call=llm,
        audit_dir=None,
    )
    # Force DELIBERATE on a domain with no template entry.
    result = await orch.execute(
        "x",
        domain="some-unknown-domain",
        model="ailiance",
        override_policy=ChainPolicy.DELIBERATE,
    )
    assert result.status == "ok"
    # Reflector prompt must come from _default and include stderr.
    second_call = seen[1]
    assert "UNKNOWN_DOMAIN_STDERR" in second_call[-1]["content"]


class _AlwaysUnavailable:
    """Validator that raises ValidatorUnavailable on every call."""

    async def run(
        self, output: str, *, domain: str, tool: str
    ) -> ValidatorResult:
        raise ValidatorUnavailable("submodule missing")


@pytest.mark.asyncio
async def test_validator_unavailable_falls_back_to_direct(
    tmp_path: Path,
) -> None:
    llm, _ = _make_llm(["the answer"])
    orch = ChainOrchestrator(
        policies_path=POLICIES_PATH,
        reflector_path=REFLECTOR_PATH,
        validator=_AlwaysUnavailable(),
        llm_call=llm,
        audit_dir=tmp_path,
    )
    result = await orch.execute(
        "design a board",
        domain="kicad-pcb",
        model="ailiance",
    )
    # We made 1 LLM call before the validator raised, so we must
    # surface that output rather than 500-ing.
    assert result.final_output == "the answer"
    assert result.status == "direct"


@pytest.mark.asyncio
async def test_reflector_lookup_by_tool_when_domain_missing(
    tmp_path: Path,
) -> None:
    """Lookup falls back to the tool key when the domain key is absent.

    Critic finding (MAJOR): ``configs/reflector_prompts.yaml`` keys
    ``ngspice-converge`` / ``compile-shell`` are tool names while the
    orchestrator looks up by domain (``spice-sim`` / ``shell``). The
    fallback keeps both naming conventions working.
    """
    llm, seen = _make_llm(["bad", "good"])
    validator = _ScriptedValidator([1, 0], stderr="LIBRARY_NOT_FOUND")
    orch = ChainOrchestrator(
        policies_path=POLICIES_PATH,
        reflector_path=REFLECTOR_PATH,
        validator=validator,
        llm_call=llm,
        audit_dir=tmp_path,
    )
    # spice-sim has policy=deliberate tool=ngspice-converge; the
    # reflector YAML keys this template under the tool name.
    result = await orch.execute(
        "simulate this netlist",
        domain="spice-sim",
        model="ailiance",
    )
    assert result.status == "ok"
    second_call = seen[1]
    reflector_prompt = second_call[-1]["content"]
    # Must come from the ngspice-converge template, not _default —
    # signature phrase from configs/reflector_prompts.yaml.
    assert "converge" in reflector_prompt.lower()
    assert "LIBRARY_NOT_FOUND" in reflector_prompt


@pytest.mark.asyncio
async def test_reflector_template_with_unknown_placeholder_does_not_crash(
    tmp_path: Path,
) -> None:
    """Stray placeholder in a reflector template must not crash the chain.

    Bonus finding #4: ``format()`` raises KeyError on unknown
    placeholders. Switching to ``format_map(defaultdict(str, ...))``
    means a stray ``{nope}`` resolves to empty string instead.
    """
    llm, seen = _make_llm(["bad", "good"])
    validator = _ScriptedValidator([1, 0], stderr="OOPS")
    orch = ChainOrchestrator(
        policies_path=POLICIES_PATH,
        reflector_path=REFLECTOR_PATH,
        validator=validator,
        llm_call=llm,
        audit_dir=tmp_path,
    )
    # Inject a template with a stray placeholder under the domain key.
    orch._reflector["zzz-unknown-domain"] = (
        "stderr={stderr} prev={previous_output} stray={nope}"
    )
    result = await orch.execute(
        "x",
        domain="zzz-unknown-domain",
        model="ailiance",
        override_policy=ChainPolicy.DELIBERATE,
    )
    assert result.status == "ok"
    second_call = seen[1]
    assert "stray=" in second_call[-1]["content"]


class _CrashingValidator:
    """Validator that raises a generic RuntimeError on every call."""

    async def run(
        self, output: str, *, domain: str, tool: str
    ) -> ValidatorResult:
        raise RuntimeError("docker daemon dead")


@pytest.mark.asyncio
async def test_deliberate_validator_crash_records_error_step(
    tmp_path: Path,
) -> None:
    """Generic validator exception is recorded, not propagated.

    Critic (MAJOR): only ValidatorUnavailable was caught — anything
    else (ConnectionError, TimeoutError, json.JSONDecodeError) became
    a raw 500 with no audit trace.
    """
    llm, _ = _make_llm(["draft"])
    orch = ChainOrchestrator(
        policies_path=POLICIES_PATH,
        reflector_path=REFLECTOR_PATH,
        validator=_CrashingValidator(),
        llm_call=llm,
        audit_dir=tmp_path,
    )
    result = await orch.execute(
        "x",
        domain="kicad-pcb",
        model="ailiance",
    )
    assert result.status == "error"
    assert result.steps[-1].kind == "validator"
    assert result.steps[-1].success is False
    assert result.steps[-1].payload["error"] == "RuntimeError"
    assert "docker daemon dead" in result.steps[-1].payload["message"]
    # NDJSON must be written even on error.
    audit_file = tmp_path / "chains" / result.chain_id / "cells.ndjson"
    assert audit_file.exists()
    lines = audit_file.read_text().strip().splitlines()
    assert len(lines) == len(result.steps)
    for line in lines:
        json.loads(line)


def test_validator_protocol_runtime_check() -> None:
    """StubValidator and our scripted helper satisfy the Protocol."""
    assert isinstance(StubValidator(), Validator)
    assert isinstance(_ScriptedValidator([0]), Validator)


@pytest.mark.asyncio
async def test_audit_manifest_records_submodule_sha(
    tmp_path: Path, monkeypatch
) -> None:
    """Manifest sidecar records validator kind + submodule pin."""
    from src.orchestrator import chain_orchestrator as co

    # Reset the cache and stub out the subprocess call.
    co._submodule_sha_cache = (False, None)
    monkeypatch.setattr(
        co, "_read_submodule_sha", lambda: "deadbeef" * 5
    )

    llm, _ = _make_llm(["good"])
    validator = _ScriptedValidator([0])
    orch = ChainOrchestrator(
        policies_path=POLICIES_PATH,
        reflector_path=REFLECTOR_PATH,
        validator=validator,
        llm_call=llm,
        audit_dir=tmp_path,
    )
    result = await orch.execute(
        "x",
        domain="kicad-pcb",
        model="ailiance",
    )
    manifest_path = (
        tmp_path / "chains" / result.chain_id / "manifest.json"
    )
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["chain_id"] == result.chain_id
    assert manifest["policy"] == "deliberate"
    assert manifest["domain"] == "kicad-pcb"
    assert manifest["tool"] == "kicad-drc"
    assert manifest["validator_kind"] == "_ScriptedValidator"
    assert manifest["submodule_sha"] == "deadbeef" * 5
    assert manifest["status"] == "ok"
    assert isinstance(manifest["started_at"], float)
# ---------------- MIXTURE v0.3.1 tests ------------------------------------


def _make_mixture_llm(
    per_model_outputs: dict[str, str],
    *,
    fail_models: tuple[str, ...] = (),
) -> tuple[Callable[..., Awaitable[str]], list[tuple[str, str]]]:
    """LLM stub that returns ``per_model_outputs[model]`` for each call.

    Records (model, last_user_msg) per call for assertion. Raises
    RuntimeError if model is listed in ``fail_models`` to exercise the
    worker-failure path.
    """
    log_: list[tuple[str, str]] = []

    async def call(messages: list[dict[str, Any]], model: str) -> str:
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            "",
        )
        log_.append((model, last_user))
        if model in fail_models:
            raise RuntimeError(f"simulated failure for {model}")
        if model not in per_model_outputs:
            raise KeyError(f"no scripted output for {model}")
        return per_model_outputs[model]

    return call, log_


@pytest.mark.asyncio
async def test_mixture_calls_all_workers_in_parallel(tmp_path: Path) -> None:
    """MIXTURE fans out to every worker listed in the policy entry."""
    llm, log_ = _make_mixture_llm({
        "ailiance-mistral": "A. Bonjour, je suis Mistral.",
        "ailiance-gemma4": "B. Bonjour, je suis Gemma.",
        "ailiance-eurollm": "C. Bonjour, je suis EuroLLM.",
    })
    orch = ChainOrchestrator(
        policies_path=POLICIES_PATH,
        reflector_path=REFLECTOR_PATH,
        validator=StubValidator(),
        llm_call=llm,
        audit_dir=tmp_path,
    )
    # Inject a mixture policy entry for a synthetic domain.
    orch._policies["test-mixture"] = {
        "policy": "mixture",
        "workers": ["ailiance-mistral", "ailiance-gemma4", "ailiance-eurollm"],
        "judge": None,  # no judge → first successful wins
    }
    result = await orch.execute("Bonjour", domain="test-mixture", model="ailiance")

    assert result.policy == ChainPolicy.MIXTURE
    assert result.status == "ok"
    assert result.metadata["judge_used"] is False
    # All 3 workers got the same prompt.
    user_prompts = [u for (_, u) in log_]
    assert user_prompts == ["Bonjour"] * 3
    models_called = sorted(m for (m, _) in log_)
    assert models_called == sorted(
        ["ailiance-mistral", "ailiance-gemma4", "ailiance-eurollm"]
    )
    # First successful = the first listed worker.
    assert result.final_output.startswith("A.")


@pytest.mark.asyncio
async def test_mixture_judge_picks_index(tmp_path: Path) -> None:
    """Judge selects a non-default index → that worker's output wins."""
    llm, log_ = _make_mixture_llm({
        "ailiance-mistral": "M1 output",
        "ailiance-gemma4": "M2 output",
        # Judge call comes last; respond with {"choice": 1}.
        "judge-model": '{"choice": 1}',
    })
    orch = ChainOrchestrator(
        policies_path=POLICIES_PATH,
        reflector_path=REFLECTOR_PATH,
        validator=StubValidator(),
        llm_call=llm,
        audit_dir=tmp_path,
    )
    orch._policies["test-mix-judge"] = {
        "policy": "mixture",
        "workers": ["ailiance-mistral", "ailiance-gemma4"],
        "judge": "judge-model",
    }
    result = await orch.execute("X", domain="test-mix-judge", model="ailiance")

    assert result.policy == ChainPolicy.MIXTURE
    assert result.status == "ok"
    assert result.metadata["judge_used"] is True
    assert result.metadata["chosen_index"] == 1
    assert result.metadata["chosen_worker"] == "ailiance-gemma4"
    assert result.final_output == "M2 output"
    # Judge step recorded.
    judge_steps = [s for s in result.steps if s.kind == "reflector"]
    assert len(judge_steps) == 1
    assert judge_steps[0].payload["role"] == "judge"


@pytest.mark.asyncio
async def test_mixture_judge_invalid_falls_back_to_index_0(tmp_path: Path) -> None:
    """Garbage judge output → safe fallback to index 0 (first worker)."""
    llm, _ = _make_mixture_llm({
        "ailiance-mistral": "first",
        "ailiance-gemma4": "second",
        "judge-model": "I cannot decide, sorry.",
    })
    orch = ChainOrchestrator(
        policies_path=POLICIES_PATH,
        reflector_path=REFLECTOR_PATH,
        validator=StubValidator(),
        llm_call=llm,
        audit_dir=tmp_path,
    )
    orch._policies["test-mix-bad-judge"] = {
        "policy": "mixture",
        "workers": ["ailiance-mistral", "ailiance-gemma4"],
        "judge": "judge-model",
    }
    result = await orch.execute("X", domain="test-mix-bad-judge", model="ailiance")

    assert result.final_output == "first"
    assert result.metadata["chosen_index"] == 0
    assert result.metadata["judge_ok"] is True  # judge call succeeded, parse failed


@pytest.mark.asyncio
async def test_mixture_one_worker_fails_others_continue(tmp_path: Path) -> None:
    """A failing worker doesn't sink the mixture — judge sees only successes."""
    llm, _ = _make_mixture_llm({
        "ailiance-mistral": "alpha",
        "ailiance-gemma4": "beta",  # this will be killed by fail_models
        "judge-model": '{"choice": 0}',
    }, fail_models=("ailiance-gemma4",))
    orch = ChainOrchestrator(
        policies_path=POLICIES_PATH,
        reflector_path=REFLECTOR_PATH,
        validator=StubValidator(),
        llm_call=llm,
        audit_dir=tmp_path,
    )
    orch._policies["test-mix-failover"] = {
        "policy": "mixture",
        "workers": ["ailiance-mistral", "ailiance-gemma4"],
        "judge": "judge-model",
    }
    result = await orch.execute("X", domain="test-mix-failover", model="ailiance")

    # Only one successful worker → no judge needed (short-circuit).
    assert result.status == "ok"
    assert result.final_output == "alpha"
    assert result.metadata["judge_used"] is False
    assert result.metadata["workers"] == ["ailiance-mistral"]


@pytest.mark.asyncio
async def test_mixture_no_workers_falls_back_to_direct(tmp_path: Path) -> None:
    """Misconfigured mixture (no workers) degrades to DIRECT gracefully."""
    llm, _ = _make_mixture_llm({"ailiance": "direct path output"})
    orch = ChainOrchestrator(
        policies_path=POLICIES_PATH,
        reflector_path=REFLECTOR_PATH,
        validator=StubValidator(),
        llm_call=llm,
        audit_dir=tmp_path,
    )
    orch._policies["test-mix-empty"] = {
        "policy": "mixture",
        "workers": [],
        "judge": "j",
    }
    result = await orch.execute("X", domain="test-mix-empty", model="ailiance")

    assert result.policy == ChainPolicy.DIRECT
    assert result.final_output == "direct path output"


@pytest.mark.asyncio
async def test_mixture_audit_records_all_workers_and_judge(tmp_path: Path) -> None:
    """Audit NDJSON contains one step per worker + one judge step."""
    llm, _ = _make_mixture_llm({
        "ailiance-mistral": "a",
        "ailiance-gemma4": "b",
        "ailiance-qwen": "c",
        "j": '{"choice": 2}',
    })
    orch = ChainOrchestrator(
        policies_path=POLICIES_PATH,
        reflector_path=REFLECTOR_PATH,
        validator=StubValidator(),
        llm_call=llm,
        audit_dir=tmp_path,
    )
    orch._policies["test-mix-audit"] = {
        "policy": "mixture",
        "workers": ["ailiance-mistral", "ailiance-gemma4", "ailiance-qwen"],
        "judge": "j",
    }
    result = await orch.execute("X", domain="test-mix-audit", model="ailiance")

    assert result.metadata["chosen_worker"] == "ailiance-qwen"
    llm_steps = [s for s in result.steps if s.kind == "llm"]
    judge_steps = [s for s in result.steps if s.kind == "reflector"]
    assert len(llm_steps) == 3
    assert len(judge_steps) == 1


# ---------------- SEQUENTIAL v0.4 tests -----------------------------------


def _make_seq_llm(
    per_model_outputs: dict[str, list[str]],
) -> tuple[Callable[..., Awaitable[str]], list[tuple[str, str]]]:
    """LLM stub that returns sequential outputs per model.

    For each model, returns ``per_model_outputs[model][i]`` on the i-th
    call to that model. Records (model, last_user_msg) per call.
    """
    log_: list[tuple[str, str]] = []
    idx: dict[str, int] = {}

    async def call(messages: list[dict[str, Any]], model: str) -> str:
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            "",
        )
        log_.append((model, last_user))
        if model not in per_model_outputs:
            raise KeyError(f"no scripted output for {model}")
        i = idx.get(model, 0)
        outputs = per_model_outputs[model]
        idx[model] = i + 1
        if i >= len(outputs):
            return outputs[-1]
        return outputs[i]

    return call, log_


@pytest.mark.asyncio
async def test_sequential_planner_solver_aggregator(tmp_path: Path) -> None:
    """SEQUENTIAL runs planner → N solver hops → aggregator."""
    llm, log_ = _make_seq_llm({
        "planner-x": ['["step one", "step two", "step three"]'],
        "solver-x": ["r1", "r2", "r3"],
        "agg-x": ["FINAL ANSWER"],
    })
    orch = ChainOrchestrator(
        policies_path=POLICIES_PATH,
        reflector_path=REFLECTOR_PATH,
        validator=StubValidator(),
        llm_call=llm,
        audit_dir=tmp_path,
    )
    orch._policies["test-seq"] = {
        "policy": "sequential",
        "planner": "planner-x",
        "solver": "solver-x",
        "aggregator": "agg-x",
        "max_steps": 5,
    }
    result = await orch.execute("solve X", domain="test-seq", model="ailiance")

    assert result.policy == ChainPolicy.SEQUENTIAL
    assert result.status == "ok"
    assert result.final_output == "FINAL ANSWER"
    # 1 planner + 3 solver + 1 aggregator = 5 steps
    assert len(result.steps) == 5
    models_called = [m for m, _ in log_]
    assert models_called == [
        "planner-x", "solver-x", "solver-x", "solver-x", "agg-x",
    ]
    assert result.metadata["n_steps_planned"] == 3
    assert result.metadata["n_steps_done"] == 3
    assert result.metadata["sub_tasks"] == ["step one", "step two", "step three"]


@pytest.mark.asyncio
async def test_sequential_respects_max_steps(tmp_path: Path) -> None:
    """Planner returning > max_steps tasks is truncated."""
    llm, log_ = _make_seq_llm({
        "p": ['["a","b","c","d","e","f","g"]'],
        "s": ["x"] * 10,
        "a": ["done"],
    })
    orch = ChainOrchestrator(
        policies_path=POLICIES_PATH,
        reflector_path=REFLECTOR_PATH,
        validator=StubValidator(),
        llm_call=llm,
        audit_dir=tmp_path,
    )
    orch._policies["test-seq-cap"] = {
        "policy": "sequential",
        "planner": "p",
        "solver": "s",
        "aggregator": "a",
        "max_steps": 3,
    }
    result = await orch.execute("X", domain="test-seq-cap", model="ailiance")

    assert result.metadata["n_steps_planned"] == 3
    # 1 planner + 3 solver + 1 aggregator
    assert len(result.steps) == 5


@pytest.mark.asyncio
async def test_sequential_planner_unparseable_falls_back_to_lines(tmp_path: Path) -> None:
    """Planner output without JSON list is split on newlines."""
    llm, _ = _make_seq_llm({
        "p": ["1. first thing\n2. second thing\n3. third"],
        "s": ["r"] * 10,
        "a": ["agg"],
    })
    orch = ChainOrchestrator(
        policies_path=POLICIES_PATH,
        reflector_path=REFLECTOR_PATH,
        validator=StubValidator(),
        llm_call=llm,
        audit_dir=tmp_path,
    )
    orch._policies["t-noparse"] = {
        "policy": "sequential",
        "planner": "p",
        "solver": "s",
        "aggregator": "a",
        "max_steps": 5,
    }
    result = await orch.execute("Q", domain="t-noparse", model="x")
    assert result.metadata["n_steps_planned"] == 3
    assert result.final_output == "agg"


@pytest.mark.asyncio
async def test_sequential_missing_planner_falls_back_to_direct(tmp_path: Path) -> None:
    """SEQUENTIAL with no planner key degrades to DIRECT."""
    llm, _ = _make_seq_llm({"ailiance": ["direct out"]})
    orch = ChainOrchestrator(
        policies_path=POLICIES_PATH,
        reflector_path=REFLECTOR_PATH,
        validator=StubValidator(),
        llm_call=llm,
        audit_dir=tmp_path,
    )
    orch._policies["t-broken"] = {"policy": "sequential", "solver": "s"}
    result = await orch.execute("Q", domain="t-broken", model="ailiance")
    assert result.policy == ChainPolicy.DIRECT
    assert result.final_output == "direct out"


@pytest.mark.asyncio
async def test_sequential_aggregator_defaults_to_solver(tmp_path: Path) -> None:
    """When aggregator key absent, solver is reused for synthesis."""
    llm, log_ = _make_seq_llm({
        "p": ['["one","two"]'],
        "s": ["s1", "s2", "synthesis"],
    })
    orch = ChainOrchestrator(
        policies_path=POLICIES_PATH,
        reflector_path=REFLECTOR_PATH,
        validator=StubValidator(),
        llm_call=llm,
        audit_dir=tmp_path,
    )
    orch._policies["t-noagg"] = {
        "policy": "sequential",
        "planner": "p",
        "solver": "s",
        "max_steps": 5,
    }
    result = await orch.execute("Q", domain="t-noagg", model="ailiance")
    assert result.metadata["aggregator"] == "s"
    assert result.final_output == "synthesis"
    # Planner + 2 solver + 1 aggregator (also solver)
    assert len([m for m, _ in log_ if m == "s"]) == 3
