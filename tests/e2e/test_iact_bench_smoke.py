"""Real Docker smoke test for the iact-bench validator adapter.

Skipped by default — set ``AILIANCE_DOCKER_E2E=1`` to opt in. The
test pulls / runs the parse-sql validator container against a known-
good SQL string and asserts exit_code == 0.
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
    val = IactBenchValidator()
    result = asyncio.run(
        val.run(
            "SELECT 1;",
            domain="sql",
            tool="parse-sql",
        )
    )
    assert result.exit_code == 0, (
        f"parse-sql failed: stderr={result.stderr!r}"
    )
