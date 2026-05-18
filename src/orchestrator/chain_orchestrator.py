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

import asyncio
import json
import logging
import subprocess
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml


# Repo-relative path to the iact-bench submodule. Cached at module
# import time so the SHA lookup runs at most once per process.
_SUBMODULE_PATH = Path(__file__).resolve().parents[2] / "vendored" / "iact-bench"
_submodule_sha_cache: tuple[bool, str | None] = (False, None)


def _read_submodule_sha() -> str | None:
    """Return the iact-bench submodule HEAD SHA, or None on any error.

    Cached at module level so chain runs do not fork a subprocess per
    request. ``git -C <path> rev-parse HEAD`` is the cheapest reliable
    way to read submodule pin (works whether the submodule is checked
    out as a worktree or as a packed gitdir).
    """
    global _submodule_sha_cache
    cached, value = _submodule_sha_cache
    if cached:
        return value
    try:
        result = subprocess.run(
            ["git", "-C", str(_SUBMODULE_PATH), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        sha: str | None = result.stdout.strip() or None
        if result.returncode != 0:
            sha = None
    except Exception:  # noqa: BLE001
        sha = None
    _submodule_sha_cache = (True, sha)
    return sha

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
        max_retries: int | None = None,
    ) -> ChainResult:
        """Dispatch to the right pattern based on domain policy.

        ``override_policy`` (typically from ``extra_body.chain_policy``)
        wins when set. Unsupported policies (MIXTURE, SEQUENTIAL) are
        logged and silently degraded to DIRECT in v0.3.0.

        ``max_retries`` (typically from ``extra_body.max_retries``) is
        honoured only on the DELIBERATE branch and overrides the
        policy-YAML default when set.
        """
        policy, entry = self.policy_for_domain(domain)
        if override_policy is not None:
            policy = override_policy

        if policy == ChainPolicy.DELIBERATE:
            tool = entry.get("tool", "")
            effective_retries = (
                max_retries
                if max_retries is not None
                else int(entry.get("max_retries", 2))
            )
            return await self.deliberate(
                prompt,
                domain=domain,
                model=model,
                max_retries=effective_retries,
                tool=tool,
            )

        if policy == ChainPolicy.VALIDATE:
            tool = entry.get("tool", "")
            return await self._validate_only(
                prompt, domain=domain, model=model, tool=tool
            )

        if policy == ChainPolicy.MIXTURE:
            return await self._mixture(
                prompt, domain=domain, model=model, entry=entry
            )

        if policy == ChainPolicy.SEQUENTIAL:
            return await self._sequential(
                prompt, domain=domain, model=model, entry=entry
            )

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
        await self._write_audit(result)
        return result

    async def _mixture(
        self,
        prompt: str,
        *,
        domain: str,
        model: str,
        entry: dict,
    ) -> ChainResult:
        """MIXTURE policy (v0.3.1).

        Runs every worker in ``entry["workers"]`` in parallel against
        the same prompt, then asks ``entry["judge"]`` to pick the best
        candidate by index. If no judge is configured or the judge fails
        to return a valid index, falls back to the first worker's output.

        Each worker call and the judge call are recorded as ``ChainStep``
        rows so the audit trail captures the full deliberation.
        """
        import asyncio as _asyncio
        import re as _re

        chain_id = uuid.uuid4().hex
        steps: list[ChainStep] = []

        workers = entry.get("workers") or []
        if not isinstance(workers, list) or not workers:
            log.warning(
                "MIXTURE policy for domain=%s has no workers — "
                "falling back to DIRECT (model=%s)",
                domain, model,
            )
            return await self._direct(prompt, domain=domain, model=model)

        judge = entry.get("judge")

        # 1. Fan out — parallel llm_call against each worker.
        async def _call_worker(w: str) -> tuple[str, str, float, bool]:
            t0 = time.perf_counter()
            try:
                out = await self.llm_call(
                    [{"role": "user", "content": prompt}], w
                )
                return (w, out, time.perf_counter() - t0, True)
            except Exception as exc:  # noqa: BLE001
                log.warning("MIXTURE worker %s failed: %s", w, exc)
                return (w, str(exc), time.perf_counter() - t0, False)

        started = time.time()
        results = await _asyncio.gather(
            *(_call_worker(w) for w in workers),
            return_exceptions=False,
        )
        for idx, (w, out, dur, ok) in enumerate(results):
            steps.append(
                ChainStep(
                    step_idx=idx,
                    attempt=1,
                    kind="llm",
                    started_at=started,
                    duration_s=dur,
                    payload={
                        "model": w,
                        "output_head": _truncate(out, _AUDIT_PAYLOAD_HEAD),
                    },
                    success=ok,
                )
            )

        successful = [(w, out) for (w, out, _, ok) in results if ok]
        if not successful:
            log.error(
                "MIXTURE: all %d workers failed for domain=%s — "
                "returning first error",
                len(workers), domain,
            )
            return ChainResult(
                chain_id=chain_id,
                final_output=results[0][1],
                status="error",
                steps=steps,
                policy=ChainPolicy.MIXTURE,
                domain=domain,
            )

        # 2. Single worker → no judging needed.
        if len(successful) == 1 or not judge:
            chosen_idx = 0
            chosen_output = successful[0][1]
            return ChainResult(
                chain_id=chain_id,
                final_output=chosen_output,
                status="ok",
                steps=steps,
                policy=ChainPolicy.MIXTURE,
                domain=domain,
                metadata={
                    "workers": [w for w, _ in successful],
                    "chosen_index": chosen_idx,
                    "judge_used": False,
                },
            )

        # 3. Judge: build prompt and call.
        candidates_md = "\n\n".join(
            f"[{i}] (from `{w}`):\n{out}"
            for i, (w, out) in enumerate(successful)
        )
        judge_prompt = (
            "You are an expert judge. Given the user prompt below and "
            f"{len(successful)} candidate answers, return ONLY a JSON "
            'object of the form {"choice": <int>} where <int> is the '
            "0-based index of the best answer based on accuracy, "
            "helpfulness, and language quality. Do not explain.\n\n"
            f"User prompt:\n{prompt}\n\n"
            f"Candidates:\n{candidates_md}\n\n"
            'Return only valid JSON like: {"choice": 0}'
        )
        t_judge = time.perf_counter()
        try:
            judge_out = await self.llm_call(
                [{"role": "user", "content": judge_prompt}], judge
            )
            judge_ok = True
        except Exception as exc:  # noqa: BLE001
            log.warning("MIXTURE judge %s failed: %s", judge, exc)
            judge_out = str(exc)
            judge_ok = False

        steps.append(
            ChainStep(
                step_idx=len(steps),
                attempt=1,
                kind="reflector",  # reuse the reflector kind for the judge step
                started_at=time.time(),
                duration_s=time.perf_counter() - t_judge,
                payload={
                    "model": judge,
                    "role": "judge",
                    "output_head": _truncate(judge_out, _AUDIT_PAYLOAD_HEAD),
                },
                success=judge_ok,
            )
        )

        # Parse {"choice": <int>} from the judge output (robust to wrapping prose).
        chosen_idx = 0
        if judge_ok:
            match = _re.search(r'\{[^{}]*"choice"\s*:\s*(\d+)[^{}]*\}', judge_out)
            if match:
                try:
                    candidate = int(match.group(1))
                    if 0 <= candidate < len(successful):
                        chosen_idx = candidate
                except (TypeError, ValueError):
                    pass

        chosen_output = successful[chosen_idx][1]
        chosen_worker = successful[chosen_idx][0]

        result = ChainResult(
            chain_id=chain_id,
            final_output=chosen_output,
            status="ok",
            steps=steps,
            policy=ChainPolicy.MIXTURE,
            domain=domain,
            metadata={
                "workers": [w for w, _ in successful],
                "chosen_index": chosen_idx,
                "chosen_worker": chosen_worker,
                "judge": judge,
                "judge_used": True,
                "judge_ok": judge_ok,
            },
        )
        await self._write_audit(result)
        return result

    async def _sequential(
        self,
        prompt: str,
        *,
        domain: str,
        model: str,
        entry: dict,
    ) -> ChainResult:
        """SEQUENTIAL policy (v0.4).

        Planner decomposes the user request into <= max_steps sub-tasks,
        Solver executes each sub-task in order (with accumulated context),
        Aggregator (or Solver if None) synthesizes the final answer.

        Each LLM hop is recorded as a ``ChainStep`` so the audit trail
        captures planner output, every solver step, and the aggregator.
        """
        import json as _json
        import re as _re

        chain_id = uuid.uuid4().hex
        steps: list[ChainStep] = []

        planner = entry.get("planner")
        solver = entry.get("solver")
        aggregator = entry.get("aggregator") or solver
        max_steps = int(entry.get("max_steps", 5))

        if not planner or not solver:
            log.warning(
                "SEQUENTIAL policy for domain=%s missing planner/solver "
                "— falling back to DIRECT (model=%s)",
                domain, model,
            )
            return await self._direct(prompt, domain=domain, model=model)

        # 1. Planner — decompose into JSON array of sub-tasks.
        plan_user_prompt = (
            "Decompose the user request below into an ordered list of "
            f"sub-tasks. Return STRICT JSON: a list of at most {max_steps} "
            "short strings, no prose, no markdown fences.\n\n"
            f"User request:\n{prompt}\n\n"
            'Example output: ["step one", "step two"]'
        )
        p_started = time.time()
        p_t0 = time.perf_counter()
        try:
            planner_out = await self.llm_call(
                [{"role": "user", "content": plan_user_prompt}], planner
            )
            planner_ok = True
        except Exception as exc:  # noqa: BLE001
            log.warning("SEQUENTIAL planner %s failed: %s", planner, exc)
            planner_out = str(exc)
            planner_ok = False
        steps.append(
            ChainStep(
                step_idx=0,
                attempt=1,
                kind="llm",
                started_at=p_started,
                duration_s=time.perf_counter() - p_t0,
                payload={
                    "model": planner,
                    "role": "planner",
                    "output_head": _truncate(planner_out, _AUDIT_PAYLOAD_HEAD),
                },
                success=planner_ok,
            )
        )

        if not planner_ok:
            result = ChainResult(
                chain_id=chain_id,
                final_output=planner_out,
                status="error",
                steps=steps,
                policy=ChainPolicy.SEQUENTIAL,
                domain=domain,
            )
            await self._write_audit(result)
            return result

        # Parse JSON array — robust to wrapping prose / code fences.
        sub_tasks: list[str] = []
        match = _re.search(r"\[.*\]", planner_out, _re.DOTALL)
        if match:
            try:
                parsed = _json.loads(match.group(0))
                if isinstance(parsed, list):
                    sub_tasks = [str(s).strip() for s in parsed if str(s).strip()]
            except Exception:  # noqa: BLE001
                sub_tasks = []
        if not sub_tasks:
            # Fallback: split on newlines, strip enumerators.
            sub_tasks = [
                _re.sub(r"^[\s\-\*0-9\.\)]+", "", line).strip()
                for line in planner_out.splitlines()
                if line.strip()
            ]
        sub_tasks = [s for s in sub_tasks if s][:max_steps]

        if not sub_tasks:
            log.warning(
                "SEQUENTIAL planner returned no parseable steps for "
                "domain=%s — falling back to DIRECT",
                domain,
            )
            return await self._direct(prompt, domain=domain, model=model)

        # 2. Solver — execute each sub-task sequentially with accumulated context.
        accumulated = ""
        step_idx = 1
        for i, task in enumerate(sub_tasks, start=1):
            solver_prompt = (
                f"Original user request:\n{prompt}\n\n"
                f"Previous step results:\n{accumulated or '(none)'}\n\n"
                f"Now execute step {i} of {len(sub_tasks)}:\n{task}\n\n"
                "Be concise and produce only the output for this step."
            )
            s_started = time.time()
            s_t0 = time.perf_counter()
            try:
                step_out = await self.llm_call(
                    [{"role": "user", "content": solver_prompt}], solver
                )
                step_ok = True
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "SEQUENTIAL solver %s failed on step %d: %s",
                    solver, i, exc,
                )
                step_out = str(exc)
                step_ok = False
            steps.append(
                ChainStep(
                    step_idx=step_idx,
                    attempt=1,
                    kind="llm",
                    started_at=s_started,
                    duration_s=time.perf_counter() - s_t0,
                    payload={
                        "model": solver,
                        "role": f"solver-step-{i}",
                        "task": task[:512],
                        "output_head": _truncate(step_out, _AUDIT_PAYLOAD_HEAD),
                    },
                    success=step_ok,
                )
            )
            step_idx += 1
            if not step_ok:
                # Hard-fail one step — return what we have so far.
                result = ChainResult(
                    chain_id=chain_id,
                    final_output=step_out,
                    status="error",
                    steps=steps,
                    policy=ChainPolicy.SEQUENTIAL,
                    domain=domain,
                    metadata={
                        "planner": planner,
                        "solver": solver,
                        "aggregator": aggregator,
                        "n_steps_planned": len(sub_tasks),
                        "n_steps_done": i - 1,
                        "failed_step": i,
                    },
                )
                await self._write_audit(result)
                return result
            accumulated += f"\nStep {i} ({task}):\n{step_out}\n"

        # 3. Aggregator — synthesize final answer.
        agg_prompt = (
            f"Original user request:\n{prompt}\n\n"
            f"All sub-task results:\n{accumulated}\n\n"
            "Synthesize a single concise final answer for the user. "
            "Do not repeat the step labels."
        )
        a_started = time.time()
        a_t0 = time.perf_counter()
        try:
            final_out = await self.llm_call(
                [{"role": "user", "content": agg_prompt}], aggregator
            )
            agg_ok = True
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "SEQUENTIAL aggregator %s failed: %s", aggregator, exc
            )
            final_out = str(exc)
            agg_ok = False
        steps.append(
            ChainStep(
                step_idx=step_idx,
                attempt=1,
                kind="reflector",  # reuse reflector kind for the synthesis hop
                started_at=a_started,
                duration_s=time.perf_counter() - a_t0,
                payload={
                    "model": aggregator,
                    "role": "aggregator",
                    "output_head": _truncate(final_out, _AUDIT_PAYLOAD_HEAD),
                },
                success=agg_ok,
            )
        )

        result = ChainResult(
            chain_id=chain_id,
            final_output=final_out,
            status="ok" if agg_ok else "error",
            steps=steps,
            policy=ChainPolicy.SEQUENTIAL,
            domain=domain,
            metadata={
                "planner": planner,
                "solver": solver,
                "aggregator": aggregator,
                "n_steps_planned": len(sub_tasks),
                "n_steps_done": len(sub_tasks),
                "sub_tasks": sub_tasks,
            },
        )
        await self._write_audit(result)
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
        await self._write_audit(result)
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
                await self._write_audit(result)
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
                await self._write_audit(result)
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
                await self._write_audit(result)
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
                await self._write_audit(result)
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
        await self._write_audit(result)
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

    async def _write_audit(self, result: ChainResult) -> None:
        # Audit writes (mkdir + two file writes) are blocking I/O. Offload
        # them to a worker thread so the orchestrator coroutine — and every
        # other coroutine sharing the event loop — keeps progressing.
        if not self.audit_dir:
            return
        await asyncio.to_thread(self._write_audit_sync, result)

    def _write_audit_sync(self, result: ChainResult) -> None:
        if not self.audit_dir:
            return
        chain_dir = self.audit_dir / "chains" / result.chain_id
        chain_dir.mkdir(parents=True, exist_ok=True)
        path = chain_dir / "cells.ndjson"
        with path.open("w", encoding="utf-8") as fh:
            for step in result.steps:
                fh.write(json.dumps(asdict(step)) + "\n")
        self._write_manifest(chain_dir, result)

    def _write_manifest(self, chain_dir: Path, result: ChainResult) -> None:
        # First validator step (if any) carries the tool name; first
        # step carries the chain start_at.
        tool: str | None = None
        for step in result.steps:
            if step.kind == "validator":
                tool = step.payload.get("tool") or None
                break
        started_at: float | None = (
            result.steps[0].started_at if result.steps else None
        )
        manifest = {
            "chain_id": result.chain_id,
            "policy": result.policy.value,
            "domain": result.domain,
            "tool": tool,
            "started_at": started_at,
            "validator_kind": type(self.validator).__name__,
            "submodule_sha": _read_submodule_sha(),
            "status": result.status,
        }
        (chain_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
