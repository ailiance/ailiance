"""Real Docker smoke test for the iact-bench validator adapter.

Skipped by default — set ``AILIANCE_DOCKER_E2E=1`` to opt in.

Coverage prioritised on the first production client (electron-rare,
hardware/PCB consulting): ``compile-cpp`` is on their critical path,
``parse-sql`` kept as a lightweight canary.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from src.orchestrator.validators import IactBenchValidator


pytestmark = pytest.mark.skipif(
    not os.environ.get("AILIANCE_DOCKER_E2E"),
    reason="real docker required (set AILIANCE_DOCKER_E2E=1)",
)


def test_parse_sql_smoke() -> None:
    """Lightweight canary — fastest container in the registry."""
    val = IactBenchValidator()
    result = asyncio.run(
        val.run("SELECT 1;", domain="sql", tool="parse-sql")
    )
    assert result.exit_code == 0, (
        f"parse-sql failed: stderr={result.stderr!r}"
    )


def test_compile_cpp_smoke() -> None:
    """Client-critical: electron-rare embedded/firmware consulting
    relies on compile-cpp + compile-arm-gcc as the primary signal
    that a model's C++ output would build at all."""
    val = IactBenchValidator()
    src = (
        "int main() { return 0; }\n"
    )
    result = asyncio.run(
        val.run(src, domain="cpp", tool="compile-cpp")
    )
    assert result.exit_code == 0, (
        f"compile-cpp failed: stderr={result.stderr!r}"
    )
