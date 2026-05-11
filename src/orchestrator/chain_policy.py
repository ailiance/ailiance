"""Chain policy enum + dataclasses shared across the orchestrator.

Mirrors the schema described in
``docs/plans/2026-05-11-router-agentic-v0.3.md``:

- ``ChainPolicy`` selects which orchestration pattern runs.
- ``RouteDecision`` is the router's structured output: which worker,
  which fallback workers, which validator tools, which chain policy.
- ``ChainStep`` records a single hop (LLM call, validator, reflector).
- ``ChainResult`` is the orchestrator's final report.

These types stay free of any I/O so they can be imported from tests
without pulling FastAPI / httpx.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal


class ChainPolicy(StrEnum):
    """Supported orchestration patterns.

    v0.3.0 implements DIRECT, VALIDATE, DELIBERATE. MIXTURE and
    SEQUENTIAL are scaffolded so config loaders accept them today and
    silently degrade to DIRECT until v0.3.1 / v0.4.
    """

    DIRECT = "direct"
    VALIDATE = "validate"
    DELIBERATE = "deliberate"
    MIXTURE = "mixture"
    SEQUENTIAL = "sequential"


@dataclass(frozen=True)
class RouteDecision:
    """Router's structured output for a single prompt."""

    primary_worker: str
    fallback_workers: list[str]
    recommended_tools: list[str]
    chain_policy: ChainPolicy
    domain: str
    confidence: float


@dataclass
class ChainStep:
    """One observable hop in a chain run.

    ``payload`` carries truncated stdout/stderr heads and any
    metadata the auditor needs without blowing up the NDJSON record.
    """

    step_idx: int
    attempt: int
    kind: Literal["llm", "validator", "reflector"]
    started_at: float
    duration_s: float
    payload: dict
    success: bool


@dataclass
class ChainResult:
    """Final orchestrator output for one request."""

    chain_id: str
    final_output: str
    status: Literal["ok", "exhausted", "direct"]
    steps: list[ChainStep]
    policy: ChainPolicy
    domain: str
    # Default empty for forward compat; tests construct without it.
    metadata: dict = field(default_factory=dict)
