"""OpenAI function-calling shim for text-completion workers.

Translates between OpenAI tools spec (input) and OpenAI tool_calls
response (output) by:
1. Injecting the tools schema into the system prompt with formatting
   instructions for the model.
2. Parsing the model's text output for an embedded JSON tool call.
3. Emitting OpenAI-compatible SSE chunks (one for tool_calls or content,
   one final with usage).
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

# Matches a top-level JSON object containing both "name" and "arguments"
# keys. Tolerates one level of nested braces inside "arguments".
_TOOL_CALL_RE = re.compile(
    r'\{[^{}]*"name"\s*:\s*"[^"]+"[^{}]*"arguments"\s*:\s*'
    r'(\{(?:[^{}]|\{[^{}]*\})*\})[^{}]*\}',
    re.DOTALL,
)

_INJECT_TEMPLATE = (
    "You have access to tools. To call a tool, respond with EXACTLY this "
    "JSON on a single line, nothing else:\n\n"
    '{{"name": "<tool_name>", "arguments": {{<args matching the tool '
    'schema>}}}}\n\n'
    "If no tool call is needed, respond in plain text. Available tools:\n\n"
    "{tools_json}"
)


def _tool_to_dict(tool: Any) -> dict:
    """Coerce a ToolDef pydantic model OR a plain dict to a serializable dict."""
    if hasattr(tool, "model_dump"):
        return tool.model_dump(exclude_none=True)
    if isinstance(tool, dict):
        return tool
    raise TypeError(f"Unsupported tool type: {type(tool)!r}")


def _msg_to_dict(msg: Any) -> dict:
    if hasattr(msg, "model_dump"):
        return msg.model_dump(exclude_none=True)
    if isinstance(msg, dict):
        return dict(msg)
    raise TypeError(f"Unsupported message type: {type(msg)!r}")


def inject_tools_into_messages(
    messages: list[Any], tools: list[Any]
) -> list[dict]:
    """Append/insert a system message describing available tools.

    Returns plain-dict messages (role/content) so they can flow directly
    into the worker's chat-template path.
    """
    tools_serialized = [_tool_to_dict(t) for t in tools]
    tools_json = json.dumps(tools_serialized, ensure_ascii=False, indent=2)
    block = _INJECT_TEMPLATE.format(tools_json=tools_json)

    out: list[dict] = []
    injected = False
    for m in messages:
        d = _msg_to_dict(m)
        if not injected and d.get("role") == "system":
            existing = d.get("content") or ""
            d = {**d, "content": f"{existing}\n\n{block}" if existing else block}
            injected = True
        out.append(d)

    if not injected:
        out.insert(0, {"role": "system", "content": block})
    return out


def generate_tool_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:8]}"


def parse_tool_call_from_text(
    text: str,
) -> tuple[str, list[dict] | None]:
    """Extract a tool call JSON object from model output.

    Returns (content, tool_calls). If no tool call detected, returns
    (text, None). Otherwise returns (text_before_json_stripped, [call_dict]).
    The arguments field is serialized as a JSON STRING per OpenAI spec.
    """
    if not text:
        return text or "", None

    match = _TOOL_CALL_RE.search(text)
    if not match:
        return text, None

    candidate = match.group(0)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return text, None

    name = parsed.get("name")
    args = parsed.get("arguments")
    if not isinstance(name, str) or not isinstance(args, dict):
        return text, None

    content_before = text[: match.start()].strip()
    tool_call = {
        "id": generate_tool_call_id(),
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(args, ensure_ascii=False),
        },
    }
    return content_before, [tool_call]


def _chunk_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:12]}"


def _now() -> int:
    return int(time.time())


def _format_chunk(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def sse_chunk_for_tool_call(
    model: str, tool_calls: list[dict], chunk_id: str | None = None
) -> str:
    """First SSE chunk: assistant role + tool_calls delta."""
    cid = chunk_id or _chunk_id()
    deltas = []
    for i, tc in enumerate(tool_calls):
        deltas.append({
            "index": i,
            "id": tc["id"],
            "type": tc.get("type", "function"),
            "function": {
                "name": tc["function"]["name"],
                "arguments": tc["function"]["arguments"],
            },
        })
    payload = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": _now(),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "tool_calls": deltas},
            "finish_reason": None,
        }],
    }
    return _format_chunk(payload)


def sse_chunk_for_content(
    model: str, content: str, chunk_id: str | None = None
) -> str:
    cid = chunk_id or _chunk_id()
    payload = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": _now(),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": content},
            "finish_reason": None,
        }],
    }
    return _format_chunk(payload)


def sse_chunk_finish(
    model: str,
    finish_reason: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    chunk_id: str | None = None,
) -> str:
    """Final SSE chunk: empty delta + finish_reason + usage."""
    cid = chunk_id or _chunk_id()
    payload = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": _now(),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return _format_chunk(payload)


# Backwards-friendly alias
def sse_chunk_for_usage(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    finish_reason: str = "stop",
    chunk_id: str | None = None,
) -> str:
    return sse_chunk_finish(
        model, finish_reason, prompt_tokens, completion_tokens, chunk_id
    )


def sse_done() -> str:
    return "data: [DONE]\n\n"
