"""ChainOrchestrator — Deliberation pattern entrypoint for v0.3.0.

Implements the loop described in
``docs/plans/2026-05-11-router-agentic-v0.3.md`` lines 116-131:

    1. LLM call    -> raw output
    2. Validator   -> exit_code
    3. Branch:
         - 0          -> return ok
         - non-0 + retries_left -> reflector prompt + loop
         - non-0 + exhausted    -> return last output, status=exhausted

The orchestrator is decoupled from any HTTP backend: callers inject an
async ``llm_call(messages, model)`` closure that returns the assistant
content string. This makes unit tests free of FastAPI / httpx and lets
the gateway plug in a real worker proxy at runtime.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from src.orchestrator.chain_policy import (
    ChainPolicy,
    ChainResult,
    ChainStep,
)
from src.orchestrator.validators import (
    Validator,
    ValidatorResult,
    ValidatorUnavailable,
)

log = logging.getLogger(__name__)

# Truncate validator stderr inside reflector prompts so we don't blow
# the worker's context window on a multi-MB linker spew.
_STDERR_HEAD_BYTES = 4 * 1024

# Truncate stderr/stdout payloads inside the audit NDJSON for the same
# reason — auditor can read full logs from the validator backend if
# needed.
_AUDIT_PAYLOAD_HEAD = 8 * 1024


# Type alias for the injected LLM caller. We intentionally use ``Any``
# for messages so the orchestrator does not pull pydantic models from
# ``src.worker.schemas`` — keeps the import graph thin.
LLMCall = Callable[[list[dict[str, Any]], str], Awaitable[str]]


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n... [truncated {len(value) - limit} bytes]"


class ChainOrchestrator:
    """Domain-policy-aware orchestrator for v0.3.0 Deliberation."""

    def __init__(
        self,
        policies_path: Path,
        reflector_path: Path,
        validator: Validator,
        llm_call: LLMCall,
        audit_dir: Path | None = None,
    ) -> None:
        self.policies_path = Path(policies_path)
        self.reflector_path = Path(reflector_path)
        self.validator = validator
        self.llm_call = llm_call
        self.audit_dir = Path(audit_dir) if audit_dir else None

        self._policies = self._load_policies()
        self._reflector = self._load_reflector()

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    def _load_policies(self) -> dict[str, dict]:
        raw = yaml.safe_load(self.policies_path.read_text()) or {}
        policies = raw.get("policies", {})
        if not isinstance(policies, dict):
            raise ValueError(
                f"chain_policies.yaml: 'policies' must be a mapping, "
                f"got {type(policies).__name__}"
            )
        return policies

    def _load_reflector(self) -> dict[str, str]:
        raw = yaml.safe_load(self.reflector_path.read_text()) or {}
        prompts = raw.get("prompts", {})
        if not isinstance(prompts, dict):
            raise ValueError(
                f"reflector_prompts.yaml: 'prompts' must be a mapping, "
                f"got {type(prompts).__name__}"
            )
        return {k: str(v) for k, v in prompts.items()}

    # ------------------------------------------------------------------
    # Policy lookup
    # ------------------------------------------------------------------

    def policy_for_domain(self, domain: str) -> tuple[ChainPolicy, dict]:
        """Return ``(ChainPolicy, entry)`` for a domain.

        Falls back to the ``_default`` entry (or DIRECT) if the domain
        is unknown.
        """
        entry = self._policies.get(domain) or self._policies.get(
            "_default"
        ) or {"policy": "direct"}
        try:
            policy = ChainPolicy(entry.get("policy", "direct"))
        except ValueError:
            log.warning(
                "unknown chain policy %r for domain %s; using DIRECT",
                entry.get("policy"),
                domain,
            )
            policy = ChainPolicy.DIRECT
        return policy, entry

    # ------------------------------------------------------------------
    # Reflector prompt
    # ------------------------------------------------------------------

    def _reflector_prompt(
        self,
        domain: str,
        *,
        stderr: str,
        previous_output: str,
        tool: str = "",
    ) -> str:
        # Lookup order: domain key first (preferred convention), then
        # the tool name (some YAML entries are keyed by tool, e.g.
        # ``ngspice-converge`` / ``compile-shell``), then ``_default``.
        # Both naming conventions resolve cleanly so editors can pick
        # either one without silently falling through.
        template = (
            self._reflector.get(domain)
            or (self._reflector.get(tool) if tool else None)
            or self._reflector.get("_default")
        )
        if template is None:
            # Last-ditch hard-coded fallback so we never crash on a
            # missing reflector config.
            template = (
                "Your previous attempt failed validation:\n\n"
                "{stderr}\n\nPrevious output:\n```\n{previous_output}\n"
                "```\n\nFix and retry."
            )
        # Use defaultdict via format_map so stray placeholders authored
        # in YAML (e.g. ``{nope}``) resolve to "" instead of crashing
        # the whole chain on KeyError.
        from collections import defaultdict

        values: defaultdict[str, str] = defaultdict(str)
        values["stderr"] = _truncate(stderr, _STDERR_HEAD_BYTES)
        values["previous_output"] = previous_output
        return template.format_map(values)

    # ------------------------------------------------------------------
    # Public dispatch
    # ------------------------------------------------------------------

    async def execute(
        self,
        prompt: str,
        *,
        domain: str,
        model: str,
        override_policy: ChainPolicy | None = None,
    ) -> ChainResult:
        """Dispatch to the right pattern based on domain policy.

        ``override_policy`` (typically from ``extra_body.chain_policy``)
        wins when set. Unsupported policies (MIXTURE, SEQUENTIAL) are
        logged and silently degraded to DIRECT in v0.3.0.
        """
        policy, entry = self.policy_for_domain(domain)
        if override_policy is not None:
            policy = override_policy

        if policy == ChainPolicy.DELIBERATE:
            tool = entry.get("tool", "")
            max_retries = int(entry.get("max_retries", 2))
            return await self.deliberate(
                prompt,
                domain=domain,
                model=model,
                max_retries=max_retries,
                tool=tool,
            )

        if policy == ChainPolicy.VALIDATE:
            tool = entry.get("tool", "")
            return await self._validate_only(
                prompt, domain=domain, model=model, tool=tool
            )

        if policy in (ChainPolicy.MIXTURE, ChainPolicy.SEQUENTIAL):
            log.info(
                "chain policy %s not implemented in v0.3.0 — "
                "falling back to DIRECT for domain=%s",
                policy.value,
                domain,
            )
            policy = ChainPolicy.DIRECT

        # DIRECT path.
        return await self._direct(prompt, domain=domain, model=model)

    # ------------------------------------------------------------------
    # Patterns
    # ------------------------------------------------------------------

    async def _direct(
        self, prompt: str, *, domain: str, model: str
    ) -> ChainResult:
        chain_id = uuid.uuid4().hex
        steps: list[ChainStep] = []
        t0 = time.perf_counter()
        started = time.time()
        output = await self.llm_call(
            [{"role": "user", "content": prompt}], model
        )
        steps.append(
            ChainStep(
                step_idx=0,
                attempt=1,
                kind="llm",
                started_at=started,
                duration_s=time.perf_counter() - t0,
                payload={
                    "model": model,
                    "output_head": _truncate(output, _AUDIT_PAYLOAD_HEAD),
                },
                success=True,
            )
        )
        result = ChainResult(
            chain_id=chain_id,
            final_output=output,
            status="direct",
            steps=steps,
            policy=ChainPolicy.DIRECT,
            domain=domain,
        )
        self._write_audit(result)
        return result

    async def _validate_only(
        self, prompt: str, *, domain: str, model: str, tool: str
    ) -> ChainResult:
        chain_id = uuid.uuid4().hex
        steps: list[ChainStep] = []
        t0 = time.perf_counter()
        started = time.time()
        output = await self.llm_call(
            [{"role": "user", "content": prompt}], model
        )
        steps.append(
            ChainStep(
                step_idx=0,
                attempt=1,
                kind="llm",
                started_at=started,
                duration_s=time.perf_counter() - t0,
                payload={
                    "model": model,
                    "output_head": _truncate(output, _AUDIT_PAYLOAD_HEAD),
                },
                success=True,
            )
        )
        try:
            v_started = time.time()
            v_t0 = time.perf_counter()
            vr = await self.validator.run(
                output, domain=domain, tool=tool
            )
            steps.append(
                self._validator_step(
                    step_idx=1,
                    attempt=1,
                    started_at=v_started,
                    duration_s=time.perf_counter() - v_t0,
                    tool=tool,
                    result=vr,
                )
            )
            status: str = "ok" if vr.exit_code == 0 else "exhausted"
        except ValidatorUnavailable as exc:
            log.warning(
                "validator unavailable (tool=%s); degrading to direct: %s",
                tool,
                exc,
            )
            status = "direct"

        result = ChainResult(
            chain_id=chain_id,
            final_output=output,
            status=status,  # type: ignore[arg-type]
            steps=steps,
            policy=ChainPolicy.VALIDATE,
            domain=domain,
        )
        self._write_audit(result)
        return result

    async def deliberate(
        self,
        prompt: str,
        *,
        domain: str,
        model: str,
        max_retries: int,
        tool: str,
    ) -> ChainResult:
        """Run the Deliberation loop (LLM -> validator -> reflector)."""
        chain_id = uuid.uuid4().hex
        steps: list[ChainStep] = []
        attempt = 1
        current_messages: list[dict[str, Any]] = [
            {"role": "user", "content": prompt}
        ]
        last_output = ""
        last_stderr = ""
        step_idx = 0

        # Pre-flight: if the validator can't load at all, degrade to
        # DIRECT so a broken iact-bench submodule doesn't dead-end the
        # caller.
        validator_ok = True

        while True:
            # 1. LLM call.
            llm_started = time.time()
            llm_t0 = time.perf_counter()
            try:
                last_output = await self.llm_call(current_messages, model)
            except Exception as exc:
                steps.append(
                    ChainStep(
                        step_idx=step_idx,
                        attempt=attempt,
                        kind="llm",
                        started_at=llm_started,
                        duration_s=time.perf_counter() - llm_t0,
                        payload={"error": str(exc)},
                        success=False,
                    )
                )
                step_idx += 1
                result = ChainResult(
                    chain_id=chain_id,
                    final_output="",
                    status="exhausted",
                    steps=steps,
                    policy=ChainPolicy.DELIBERATE,
                    domain=domain,
                )
                self._write_audit(result)
                return result

            llm_kind = "llm" if attempt == 1 else "reflector"
            steps.append(
                ChainStep(
                    step_idx=step_idx,
                    attempt=attempt,
                    kind=llm_kind,  # type: ignore[arg-type]
                    started_at=llm_started,
                    duration_s=time.perf_counter() - llm_t0,
                    payload={
                        "model": model,
                        "output_head": _truncate(
                            last_output, _AUDIT_PAYLOAD_HEAD
                        ),
                    },
                    success=True,
                )
            )
            step_idx += 1

            # 2. Validator.
            v_started = time.time()
            v_t0 = time.perf_counter()
            try:
                vr = await self.validator.run(
                    last_output, domain=domain, tool=tool
                )
            except ValidatorUnavailable as exc:
                log.warning(
                    "validator unavailable mid-chain (tool=%s); "
                    "returning last output as direct: %s",
                    tool,
                    exc,
                )
                validator_ok = False
                break
            except Exception as exc:
                # Critic (MAJOR): only ValidatorUnavailable was caught
                # before — ConnectionError / TimeoutError / JSON decode
                # errors propagated as raw 500 with no audit step. Now
                # we record a failed validator step and return
                # status="error" so the caller sees a structured
                # response and the NDJSON trace is complete.
                log.warning(
                    "validator raised %s, aborting chain %s",
                    type(exc).__name__,
                    chain_id,
                )
                steps.append(
                    ChainStep(
                        step_idx=step_idx,
                        attempt=attempt,
                        kind="validator",
                        started_at=v_started,
                        duration_s=time.perf_counter() - v_t0,
                        payload={
                            "tool": tool,
                            "error": type(exc).__name__,
                            "message": str(exc)[:512],
                        },
                        success=False,
                    )
                )
                step_idx += 1
                result = ChainResult(
                    chain_id=chain_id,
                    final_output=last_output,
                    status="error",
                    steps=steps,
                    policy=ChainPolicy.DELIBERATE,
                    domain=domain,
                )
                self._write_audit(result)
                return result

            steps.append(
                self._validator_step(
                    step_idx=step_idx,
                    attempt=attempt,
                    started_at=v_started,
                    duration_s=time.perf_counter() - v_t0,
                    tool=tool,
                    result=vr,
                )
            )
            step_idx += 1
            last_stderr = vr.stderr

            # 3. Branch.
            if vr.exit_code == 0:
                result = ChainResult(
                    chain_id=chain_id,
                    final_output=last_output,
                    status="ok",
                    steps=steps,
                    policy=ChainPolicy.DELIBERATE,
                    domain=domain,
                )
                self._write_audit(result)
                return result

            if attempt > max_retries:
                result = ChainResult(
                    chain_id=chain_id,
                    final_output=last_output,
                    status="exhausted",
                    steps=steps,
                    policy=ChainPolicy.DELIBERATE,
                    domain=domain,
                )
                self._write_audit(result)
                return result

            # Build reflector prompt for next attempt.
            reflector = self._reflector_prompt(
                domain,
                stderr=last_stderr,
                previous_output=last_output,
                tool=tool,
            )
            current_messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": last_output},
                {"role": "user", "content": reflector},
            ]
            attempt += 1

        # Reached only if validator went unavailable mid-chain.
        result = ChainResult(
            chain_id=chain_id,
            final_output=last_output,
            status="direct" if not validator_ok else "exhausted",
            steps=steps,
            policy=ChainPolicy.DELIBERATE,
            domain=domain,
        )
        self._write_audit(result)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validator_step(
        self,
        *,
        step_idx: int,
        attempt: int,
        started_at: float,
        duration_s: float,
        tool: str,
        result: ValidatorResult,
    ) -> ChainStep:
        return ChainStep(
            step_idx=step_idx,
            attempt=attempt,
            kind="validator",
            started_at=started_at,
            duration_s=duration_s,
            payload={
                "tool": tool,
                "exit_code": result.exit_code,
                "stdout_head": _truncate(
                    result.stdout, _AUDIT_PAYLOAD_HEAD
                ),
                "stderr_head": _truncate(
                    result.stderr, _AUDIT_PAYLOAD_HEAD
                ),
                "image_digest": result.image_digest,
                "validator_duration_s": result.duration_s,
            },
            success=result.exit_code == 0,
        )

    def _write_audit(self, result: ChainResult) -> None:
        if not self.audit_dir:
            return
        chain_dir = self.audit_dir / "chains" / result.chain_id
        chain_dir.mkdir(parents=True, exist_ok=True)
        path = chain_dir / "cells.ndjson"
        with path.open("w", encoding="utf-8") as fh:
            for step in result.steps:
                fh.write(json.dumps(asdict(step)) + "\n")
