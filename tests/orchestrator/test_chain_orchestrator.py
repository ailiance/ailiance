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
