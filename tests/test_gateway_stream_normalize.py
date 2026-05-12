"""Tests for the SSE stream normalizer.

Covers :func:`_normalize_sse_stream` and :func:`_rewrite_sse_event`:
- delta.reasoning → delta.content promotion
- intra-event [THINK] stripping
- passthrough for [DONE], comments, malformed JSON, partial events
- byte/string input variants
"""

from __future__ import annotations

import asyncio
import json

import pytest

from src.gateway.server import _normalize_sse_stream, _rewrite_sse_event


def _make_data_event(payload: dict) -> str:
    return "data: " + json.dumps(payload)


def _delta(**kw) -> str:
    return _make_data_event({"choices": [{"index": 0, "delta": kw}]})


async def _drain(async_iter) -> list[bytes]:
    return [b async for b in async_iter]


async def _stream_from_strings(parts: list[str]):
    for p in parts:
        yield p


def _run(coro):
    return asyncio.run(coro)


class TestEventRewrite:
    def test_reasoning_promoted_to_content(self):
        evt = _delta(reasoning="Hello")
        out = _rewrite_sse_event(evt).decode()
        obj = json.loads(out[len("data: "):])
        assert obj["choices"][0]["delta"]["content"] == "Hello"

    def test_existing_content_kept(self):
        evt = _delta(content="answer", reasoning="thoughts")
        out = _rewrite_sse_event(evt).decode()
        obj = json.loads(out[len("data: "):])
        assert obj["choices"][0]["delta"]["content"] == "answer"

    def test_think_block_stripped(self):
        evt = _delta(content="[THINK]hidden[/THINK]visible")
        out = _rewrite_sse_event(evt).decode()
        obj = json.loads(out[len("data: "):])
        assert obj["choices"][0]["delta"]["content"] == "visible"

    def test_done_marker_passthrough(self):
        evt = "data: [DONE]"
        assert _rewrite_sse_event(evt) == b"data: [DONE]"

    def test_comment_line_passthrough(self):
        evt = ": keep-alive"
        assert _rewrite_sse_event(evt) == b": keep-alive"

    def test_malformed_json_passthrough(self):
        evt = "data: {not json"
        assert _rewrite_sse_event(evt) == b"data: {not json"

    def test_terminal_message_frame_normalized(self):
        """Some workers send a final frame with `message` instead of `delta`."""
        evt = "data: " + json.dumps({
            "choices": [{"index": 0, "message": {"content": "", "reasoning": "done"}}]
        })
        out = _rewrite_sse_event(evt).decode()
        obj = json.loads(out[len("data: "):])
        assert obj["choices"][0]["message"]["content"] == "done"


class TestStream:
    def test_promotes_across_multiple_events(self):
        events = [
            _delta(reasoning="Hello "),
            _delta(reasoning="world"),
            "data: [DONE]",
        ]
        wire = "\n\n".join(events) + "\n\n"
        out = b"".join(_run(_drain(_normalize_sse_stream(_stream_from_strings([wire])))))
        text = out.decode()
        # Both reasoning chunks promoted to content.
        assert '"content": "Hello "' in text or '"content":"Hello "' in text
        assert '"content": "world"' in text or '"content":"world"' in text
        assert "[DONE]" in text

    def test_event_split_across_chunks(self):
        """Worker may flush bytes that don't end on a \\n\\n boundary."""
        full = _delta(reasoning="ok") + "\n\n"
        # Cut at an arbitrary mid-event byte.
        cut = len(full) // 2
        parts = [full[:cut], full[cut:]]
        out = b"".join(_run(_drain(_normalize_sse_stream(_stream_from_strings(parts)))))
        text = out.decode()
        assert "content" in text and "ok" in text

    def test_byte_input_decoded(self):
        async def byte_stream():
            yield (_delta(reasoning="bx") + "\n\n").encode("utf-8")
        out = b"".join(_run(_drain(_normalize_sse_stream(byte_stream()))))
        assert b"bx" in out

    def test_trailing_fragment_without_terminator_flushed(self):
        """A worker that closes mid-event must still flush bytes to the client."""
        parts = ["data: partial-no-newline"]
        out = b"".join(_run(_drain(_normalize_sse_stream(_stream_from_strings(parts)))))
        assert out == b"data: partial-no-newline"

    def test_multiple_events_in_single_chunk(self):
        wire = (
            _delta(reasoning="a") + "\n\n"
            + _delta(reasoning="b") + "\n\n"
            + "data: [DONE]\n\n"
        )
        out = b"".join(_run(_drain(_normalize_sse_stream(_stream_from_strings([wire])))))
        text = out.decode()
        assert text.count("data:") == 3
        assert "[DONE]" in text
