"""LiteLLM streaming adapter.

Thin wrapper around ``POST /v1/chat/completions?stream=true`` against
the LiteLLM proxy already configured by the ailiance gateway. Yields
incremental ``content`` strings (the assistant deltas) for the session
FSM to forward as ``response.audio_transcript.delta`` events.

Endpoint, key, and default model come from env vars so a single config
change can swap (e.g. from npc-fast to a hints-deep alias for longer
deliberation).
"""
from __future__ import annotations

import json
import logging
import os
from typing import AsyncIterator

import httpx


log = logging.getLogger(__name__)

LITELLM_URL = os.getenv("LITELLM_URL", "http://100.116.92.12:4000")
LITELLM_KEY = os.getenv("LITELLM_MASTER_KEY", "sk-zacus-local-dev-do-not-share")
LITELLM_DEFAULT_MODEL = os.getenv("REALTIME_LITELLM_MODEL", "npc-fast")


async def stream_completion(
    user_text: str,
    *,
    model: str = LITELLM_DEFAULT_MODEL,
    instructions: str | None = None,
    history: list[dict] | None = None,
    timeout_s: float = 60.0,
) -> AsyncIterator[str]:
    """POST a single-turn completion in streaming mode, yield content deltas.

    Args:
        user_text: the just-transcribed user turn (Kyutai final).
        model: LiteLLM alias to call. Defaults to ``npc-fast`` because
            sub-second TTFT matters more than depth for a Realtime loop.
        instructions: optional system prompt (Realtime
            ``session.update.instructions``). When None, no system
            message is sent and the LiteLLM-side default applies.
        history: optional prior messages [{role, content}, ...].
        timeout_s: hard ceiling for the whole HTTP exchange.

    Yields:
        Each non-empty ``delta.content`` string from the streaming
        response. The caller concatenates for the transcript echo.

    Raises:
        ``RuntimeError`` on non-200, malformed SSE, or network error.
        Callers should map to the OpenAI Realtime ``error`` event.
    """
    messages: list[dict] = []
    if instructions:
        messages.append({"role": "system", "content": instructions})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_text})

    body = {"model": model, "messages": messages, "stream": True}
    headers = {
        "Authorization": f"Bearer {LITELLM_KEY}",
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        async with client.stream(
            "POST",
            f"{LITELLM_URL}/v1/chat/completions",
            headers=headers,
            json=body,
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                raise RuntimeError(
                    f"litellm {resp.status_code}: {body[:200]!r}"
                )
            async for raw in resp.aiter_lines():
                # OpenAI-compatible SSE: lines start with ``data: ``.
                # Empty lines separate events, ``data: [DONE]`` ends.
                if not raw or not raw.startswith("data:"):
                    continue
                payload = raw[5:].strip()
                if payload == "[DONE]":
                    return
                try:
                    evt = json.loads(payload)
                except json.JSONDecodeError:
                    log.warning("litellm bad SSE chunk: %r", payload[:120])
                    continue
                choices = evt.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    yield content
