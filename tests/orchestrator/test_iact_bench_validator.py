"""Unit tests for the iact-bench validator adapter.

The real iact-bench runner spawns Docker; we never touch that here.
Instead we exercise the registry-resolution path (which needs the
submodule on disk) and mock ``run_validator`` to verify field
mapping. A separate e2e suite covers real Docker invocation.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.orchestrator import validators as v_mod
from src.orchestrator.validators import (
    IactBenchValidator,
    StubValidator,
    ValidatorResult,
    ValidatorUnavailable,
    make_validator,
)


SUBMODULE_PRESENT = (
    Path(__file__).resolve().parents[2] / "vendored" / "iact-bench" / "tools"
).is_dir()


pytestmark = pytest.mark.skipif(
    not SUBMODULE_PRESENT,
    reason="vendored/iact-bench submodule not initialised",
)


def test_iact_bench_imports_when_submodule_present() -> None:
    val = IactBenchValidator()
    assert val.known_tools(), "registry should not be empty"
    # A handful of tools we map in chain_policies.yaml must resolve.
    for name in ("kicad-drc", "compile-cpp", "parse-sql"):
        assert name in val.known_tools()


def test_iact_bench_unknown_tool_raises_unavailable() -> None:
    val = IactBenchValidator()
    with pytest.raises(ValidatorUnavailable):
        asyncio.run(val.run("x", domain="d", tool="nonexistent-validator"))


def test_make_validator_auto_falls_back_to_stub(monkeypatch) -> None:
    def boom(*_a, **_kw):
        raise ValidatorUnavailable("forced for test")

    monkeypatch.setattr(v_mod, "IactBenchValidator", boom)
    val = make_validator("auto")
    assert isinstance(val, StubValidator)


def test_make_validator_explicit_kinds() -> None:
    assert isinstance(make_validator("stub"), StubValidator)
    with pytest.raises(ValueError):
        make_validator("does-not-exist")


def test_infrastructure_error_raises_unavailable_not_validation_failure() -> None:
    """Docker missing / image-pull-fail / timeout must NOT look like
    'validator caught a bug'. The runner sets ``error`` on the result
    in that case; we surface ValidatorUnavailable so the orchestrator
    degrades to DIRECT instead of burning LLM calls on retries."""
    val = IactBenchValidator()

    docker_missing = SimpleNamespace(
        score=0.0,
        exit_code=127,
        stdout_head="",
        stderr_head="docker binary not in PATH",
        duration_ms=0,
        image_digest="sha256:abc",
        pass_rule="exit_zero",
        error="docker_missing",
    )

    with patch.object(val._bridge, "run_validator", return_value=docker_missing):
        with pytest.raises(ValidatorUnavailable, match="infrastructure"):
            asyncio.run(val.run("x", domain="cpp", tool="compile-cpp"))


def test_validation_result_field_mapping() -> None:
    val = IactBenchValidator()

    fake = SimpleNamespace(
        score=1.0,
        exit_code=0,
        stdout_head="hello",
        stderr_head="warn",
        duration_ms=1234,
        image_digest="sha256:abc",
        pass_rule="exit_zero",
        error=None,
    )

    with patch.object(val._bridge, "run_validator", return_value=fake):
        result: ValidatorResult = asyncio.run(
            val.run("anything", domain="sql", tool="parse-sql")
        )

    assert result.exit_code == 0
    assert result.stdout == "hello"
    assert result.stderr == "warn"
    assert result.duration_s == pytest.approx(1.234)
    assert result.image_digest == "sha256:abc"
