"""Validator protocol + implementations for the chain orchestrator.

The orchestrator never imports the iact-bench validators directly;
it talks to anything implementing the ``Validator`` Protocol. This
keeps tests fast (no Docker) and lets ailiance boot even when the
iact-bench submodule is absent.

``IactBenchValidator`` is a real adapter onto the vendored
``iact-bench`` runner (``vendored/iact-bench``). The runner is
synchronous (Docker subprocess) so we dispatch via
``asyncio.to_thread`` to keep the gateway event loop responsive.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

log = logging.getLogger(__name__)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SUBMODULE = _REPO_ROOT / "vendored" / "iact-bench"
_DEFAULT_REGISTRY = _DEFAULT_SUBMODULE / "configs" / "domain_validators.yaml"


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
    so a missing iact-bench submodule (or an unknown tool name) does
    not crash the gateway.
    """


@runtime_checkable
class Validator(Protocol):
    """Async validator protocol."""

    async def run(
        self, output: str, *, domain: str, tool: str
    ) -> ValidatorResult: ...


class StubValidator:
    """Test double — always reports success."""

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


# ---------------------------------------------------------------------------
# iact-bench bridge
# ---------------------------------------------------------------------------


def _load_iact_bench(submodule_path: Path) -> Any:
    """Import the iact-bench runner + base lazily.

    Adds ``<submodule>/tools`` to ``sys.path`` (idempotent) and returns
    a small namespace object exposing the symbols we need. Raises
    :class:`ValidatorUnavailable` on any failure so the gateway can
    fall back gracefully.
    """
    tools_dir = submodule_path / "tools"
    if not tools_dir.is_dir():
        raise ValidatorUnavailable(
            f"iact-bench submodule not initialised at {submodule_path} "
            f"(expected directory {tools_dir})"
        )
    tools_str = str(tools_dir)
    if tools_str not in sys.path:
        sys.path.insert(0, tools_str)
    try:
        from iact_bench.validators import base as _base  # noqa: WPS433
        from iact_bench.validators import runner as _runner  # noqa: WPS433
    except Exception as exc:  # noqa: BLE001
        raise ValidatorUnavailable(
            f"iact-bench validators not importable from {tools_dir}: {exc}"
        ) from exc

    class _Bridge:
        run_validator = staticmethod(_runner.run_validator)
        load_registry = staticmethod(_base.load_registry)
        ValidatorRegistry = _base.ValidatorRegistry
        ValidatorConfig = _base.ValidatorConfig
        ValidationResult = _base.ValidationResult

    return _Bridge


class IactBenchValidator:
    """Real adapter onto the iact-bench Docker sandbox runner.

    Loads the validator registry once at construction time. Each
    ``run()`` call resolves the ``tool`` name against the registry
    and dispatches the synchronous Docker invocation via
    ``asyncio.to_thread``.
    """

    def __init__(
        self,
        registry_path: Path | None = None,
        submodule_path: Path | None = None,
    ) -> None:
        self.submodule_path = Path(
            submodule_path
            or os.environ.get("AILIANCE_IACT_BENCH_PATH")
            or _DEFAULT_SUBMODULE
        )
        self.registry_path = Path(
            registry_path
            or (self.submodule_path / "configs" / "domain_validators.yaml")
        )
        self._bridge = _load_iact_bench(self.submodule_path)
        try:
            self._registry = self._bridge.load_registry(self.registry_path)
        except Exception as exc:  # noqa: BLE001
            raise ValidatorUnavailable(
                f"failed to load iact-bench registry "
                f"({self.registry_path}): {exc}"
            ) from exc

    def known_tools(self) -> list[str]:
        return list(self._registry.by_name.keys())

    async def run(
        self, output: str, *, domain: str, tool: str
    ) -> ValidatorResult:
        cfg = self._registry.get(tool)
        if cfg is None:
            raise ValidatorUnavailable(
                f"validator '{tool}' not in iact-bench registry "
                f"(domain={domain})"
            )
        result = await asyncio.to_thread(
            self._bridge.run_validator, cfg, output
        )
        return ValidatorResult(
            exit_code=result.exit_code,
            stdout=result.stdout_head,
            stderr=result.stderr_head,
            duration_s=result.duration_ms / 1000.0,
            image_digest=result.image_digest,
        )


def make_validator(kind: str = "auto") -> Validator:
    """Factory selecting a validator implementation.

    - ``"stub"``: always-pass test double.
    - ``"iact_bench"``: real Docker-sandboxed runner. Raises
      :class:`ValidatorUnavailable` if the submodule is missing.
    - ``"auto"`` (default): prefer iact-bench, fall back to stub on
      any construction failure (logged as a warning).
    """
    if kind == "stub":
        log.info("validator: StubValidator (kind=stub)")
        return StubValidator()
    if kind == "iact_bench":
        log.info("validator: IactBenchValidator (kind=iact_bench)")
        return IactBenchValidator()
    if kind == "auto":
        try:
            v = IactBenchValidator()
            log.info(
                "validator: IactBenchValidator (kind=auto, %d tools)",
                len(v.known_tools()),
            )
            return v
        except ValidatorUnavailable as exc:
            log.warning(
                "validator: iact-bench unavailable, falling back to stub: %s",
                exc,
            )
            return StubValidator()
    raise ValueError(f"unknown validator kind: {kind!r}")
