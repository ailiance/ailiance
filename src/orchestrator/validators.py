"""Validator protocol + stub implementation for the chain orchestrator.

The orchestrator never imports the iact-bench validators directly;
it talks to anything implementing the ``Validator`` Protocol. This
keeps tests fast (no Docker) and lets ailiance boot even when the
iact-bench submodule is absent.

Production wiring (the iact-bench submodule + sandbox runner) is
operational concern; ``IactBenchValidator`` is a thin import shim
that raises ``ValidatorUnavailable`` at call time when the submodule
is not present, so the gateway can degrade gracefully to DIRECT.
"""

from __future__ import annotations

import importlib
import logging
import os
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

log = logging.getLogger(__name__)


@dataclass
class ValidatorResult:
    """Result of one validator run.

    Mirrors the iact-bench ``cells.ndjson`` schema so chain audit
    records are byte-comparable to bench audit records.
    """

    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    image_digest: str | None = None


class ValidatorUnavailable(RuntimeError):
    """Raised when a validator backend cannot be reached.

    The orchestrator catches this and degrades the chain to DIRECT,
    so a missing iact-bench submodule does not crash the gateway.
    """


@runtime_checkable
class Validator(Protocol):
    """Async validator protocol.

    Implementations may dispatch to a Docker sandbox, a local
    interpreter, or a stub. They MUST NOT raise on validation
    failure — exit_code != 0 is the success-path signal that the
    chain should retry. They MAY raise ``ValidatorUnavailable`` if
    the backend is structurally unreachable (missing submodule,
    Docker daemon down).
    """

    async def run(
        self, output: str, *, domain: str, tool: str
    ) -> ValidatorResult: ...


class StubValidator:
    """Test double — always reports success.

    Useful as the default validator when running the gateway without
    iact-bench installed, and as a baseline for unit tests.
    """

    async def run(
        self, output: str, *, domain: str, tool: str
    ) -> ValidatorResult:
        return ValidatorResult(
            exit_code=0,
            stdout="",
            stderr="",
            duration_s=0.0,
            image_digest=None,
        )


class IactBenchValidator:
    """Bridge to the iact-bench validator runner (operational).

    The submodule is expected at ``vendored/iact-bench`` (configurable
    via ``AILIANCE_IACT_BENCH_PATH``). This class does NOT import the
    runner at construction time so the gateway boots cleanly even if
    iact-bench is not vendored. The first ``run()`` call performs the
    import and raises :class:`ValidatorUnavailable` on failure.

    Wiring the actual sandbox dispatch is operational: the iact-bench
    submodule must be added separately, and its image digests must be
    pinned per call. See ``docs/router-v0.3-deliberate.md`` for the
    expected layout.
    """

    def __init__(self, submodule_path: str | None = None) -> None:
        self.submodule_path = submodule_path or os.environ.get(
            "AILIANCE_IACT_BENCH_PATH",
            "vendored/iact-bench/src/iact_bench/validators/runner.py",
        )
        self._runner = None

    def _load_runner(self):
        # Lazy import: keep the gateway boot path free of iact-bench.
        try:
            mod = importlib.import_module(
                "iact_bench.validators.runner"
            )
        except Exception as exc:  # ImportError, ModuleNotFoundError, etc.
            raise ValidatorUnavailable(
                f"iact-bench validators not importable "
                f"(path={self.submodule_path}): {exc}"
            ) from exc
        return mod

    async def run(
        self, output: str, *, domain: str, tool: str
    ) -> ValidatorResult:
        if self._runner is None:
            self._runner = self._load_runner()
        # Real wiring goes here; the runner is expected to expose an
        # async ``run_validator(output, domain, tool)`` callable that
        # returns a dict matching ValidatorResult fields.
        result = await self._runner.run_validator(  # type: ignore[attr-defined]
            output, domain=domain, tool=tool
        )
        return ValidatorResult(**result)
