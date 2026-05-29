"""Langfuse cost-tracking for the ailiance gateway.

The gateway accepts unauthenticated public traffic on :9300 — but we still
want a single pane of glass for cost, latency and token usage per model.
Langfuse already runs on the cluster (langfuse.saillant.cc); this module
ships a no-op if the LANGFUSE_PUBLIC_KEY env var is absent, so dev / CI
runs never need credentials.

Tracing is *fire-and-forget*: each completion spawns an asyncio task that
posts to Langfuse in the background. Client latency is not impacted.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import structlog

log = structlog.get_logger()

_client = None
_initialized = False


def _get_client():
    """Lazy-init the Langfuse client. Returns None when keys are missing."""
    global _client, _initialized
    if _initialized:
        return _client
    _initialized = True
    pub = os.environ.get("LANGFUSE_PUBLIC_KEY")
    sec = os.environ.get("LANGFUSE_SECRET_KEY")
    host = os.environ.get("LANGFUSE_HOST", "https://langfuse.saillant.cc")
    if not pub or not sec:
        log.info("langfuse.disabled", reason="missing LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY")
        return None
    try:
        from langfuse import Langfuse  # type: ignore[import-not-found]
    except ImportError:
        log.warning("langfuse.disabled", reason="langfuse SDK not installed")
        return None
    try:
        _client = Langfuse(public_key=pub, secret_key=sec, host=host)
        log.info("langfuse.enabled", host=host)
    except Exception as exc:  # noqa: BLE001
        log.warning("langfuse.init_failed", error=str(exc))
        _client = None
    return _client


def _extract_messages(request_body: dict) -> list[dict]:
    """Trim messages to {role, content} for cost-tracking payload."""
    out: list[dict] = []
    for m in request_body.get("messages", []) or []:
        if hasattr(m, "model_dump"):
            d = m.model_dump(exclude_none=True)
        elif isinstance(m, dict):
            d = m
        else:
            continue
        out.append({"role": d.get("role"), "content": d.get("content", "")})
    return out


async def _send_trace(
    *,
    model_alias: str,
    domain: str,
    kind: str,
    request_body: dict,
    response_body: dict,
    latency_ms: float,
    upstream_model: str | None = None,
    chain_id: str | None = None,
    error: str | None = None,
    served_model: str | None = None,
) -> None:
    """Background task: emit a Langfuse trace + generation observation."""
    client = _get_client()
    if client is None:
        return
    try:
        trace_name = f"chat.{kind}"  # chat.direct / chat.chain / chat.stream
        messages = _extract_messages(request_body)
        # OpenAI-style usage block if the worker returned one.
        usage_raw = response_body.get("usage") or {}
        choices = response_body.get("choices") or []
        output_content = ""
        if choices:
            msg = choices[0].get("message") or {}
            output_content = msg.get("content", "")
        trace = client.trace(
            name=trace_name,
            input=messages,
            output=output_content,
            metadata={
                "alias": model_alias,
                "upstream_model": upstream_model or response_body.get("model"),
                "served_model": served_model,
                "domain": domain,
                "kind": kind,
                "latency_ms": round(latency_ms, 1),
                "chain_id": chain_id,
                "error": error,
            },
            tags=[f"alias:{model_alias}", f"domain:{domain}", f"kind:{kind}"],
        )
        # The generation observation gets the cost-tracking payload Langfuse
        # uses to look up per-model pricing.
        trace.generation(
            name=f"{model_alias}.completion",
            model=upstream_model or model_alias,
            input=messages,
            output=output_content,
            usage={
                "input": usage_raw.get("prompt_tokens", 0),
                "output": usage_raw.get("completion_tokens", 0),
                "total": usage_raw.get("total_tokens", 0),
                "unit": "TOKENS",
            },
            metadata={
                "alias": model_alias,
                "kind": kind,
                "latency_ms": round(latency_ms, 1),
            },
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("langfuse.trace_failed", error=str(exc), alias=model_alias)


def track_chat(
    *,
    model_alias: str,
    domain: str,
    kind: str,
    request_body: dict,
    response_body: dict,
    started_at: float,
    upstream_model: str | None = None,
    chain_id: str | None = None,
    error: str | None = None,
    served_model: str | None = None,
) -> None:
    """Public entry point: spawn the background trace and return immediately."""
    latency_ms = (time.perf_counter() - started_at) * 1000.0
    try:
        asyncio.get_event_loop().create_task(
            _send_trace(
                model_alias=model_alias,
                domain=domain,
                kind=kind,
                request_body=request_body,
                response_body=response_body,
                latency_ms=latency_ms,
                upstream_model=upstream_model,
                chain_id=chain_id,
                error=error,
                served_model=served_model,
            )
        )
    except RuntimeError:
        # No running loop — sync context. Skip silently (would block).
        pass
