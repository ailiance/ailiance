"""Unit tests for function-calling shim."""

from __future__ import annotations

import json

from src.worker.function_calling import (
    inject_tools_into_messages,
    parse_tool_call_from_text,
    sse_chunk_finish,
    sse_chunk_for_content,
    sse_chunk_for_tool_call,
    sse_done,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]


def test_parse_tool_call_extracts_json():
    text = (
        'thinking about this... '
        '{"name":"read_file","arguments":{"path":"/x"}} '
        'done.'
    )
    content, calls = parse_tool_call_from_text(text)
    assert calls is not None
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "read_file"
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "/x"}
    assert content == "thinking about this..."
    assert calls[0]["id"].startswith("call_")
    assert calls[0]["type"] == "function"


def test_parse_tool_call_no_call():
    content, calls = parse_tool_call_from_text("plain answer with no tool")
    assert calls is None
    assert content == "plain answer with no tool"


def test_parse_tool_call_empty():
    content, calls = parse_tool_call_from_text("")
    assert calls is None
    assert content == ""


def test_inject_tools_appends_system():
    msgs = [{"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"}]
    out = inject_tools_into_messages(msgs, TOOLS)
    assert len(out) == 2
    assert out[0]["role"] == "system"
    assert "You are helpful." in out[0]["content"]
    assert "Available tools" in out[0]["content"]
    assert "read_file" in out[0]["content"]
    assert out[1] == {"role": "user", "content": "Hi"}


def test_inject_tools_prepends_when_no_system():
    msgs = [{"role": "user", "content": "Hi"}]
    out = inject_tools_into_messages(msgs, TOOLS)
    assert len(out) == 2
    assert out[0]["role"] == "system"
    assert "read_file" in out[0]["content"]
    assert out[1]["role"] == "user"


def test_sse_chunk_for_tool_call_format():
    calls = [{
        "id": "call_abc12345",
        "type": "function",
        "function": {"name": "read_file", "arguments": '{"path":"/x"}'},
    }]
    chunk = sse_chunk_for_tool_call("ailiance", calls)
    assert chunk.startswith("data: ")
    assert chunk.endswith("\n\n")
    payload = json.loads(chunk[len("data: "):].strip())
    assert payload["object"] == "chat.completion.chunk"
    assert payload["model"] == "ailiance"
    delta = payload["choices"][0]["delta"]
    assert delta["role"] == "assistant"
    assert delta["tool_calls"][0]["function"]["name"] == "read_file"
    assert payload["choices"][0]["finish_reason"] is None


def test_sse_chunk_for_content_format():
    chunk = sse_chunk_for_content("ailiance", "hello")
    payload = json.loads(chunk[len("data: "):].strip())
    assert payload["choices"][0]["delta"]["content"] == "hello"
    assert payload["choices"][0]["finish_reason"] is None


def test_sse_chunk_finish_includes_usage():
    chunk = sse_chunk_finish("ailiance", "tool_calls", 10, 5)
    payload = json.loads(chunk[len("data: "):].strip())
    assert payload["choices"][0]["finish_reason"] == "tool_calls"
    assert payload["usage"]["total_tokens"] == 15


def test_sse_done():
    assert sse_done() == "data: [DONE]\n\n"
