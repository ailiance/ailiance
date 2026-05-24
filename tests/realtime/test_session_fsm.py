"""Session FSM tests with mocked adapters.

The full Realtime session is harder to test end-to-end without a live
Kyutai + LiteLLM + voice-bridge stack (covered by the opt-in
integration test next to this file). This module verifies the FSM
transitions and adapter dispatch via injected mocks.
"""
from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from src.realtime import session as session_mod
from src.realtime.session import RealtimeSession, SessionState


class FakeWS:
    """Minimal FastAPI WebSocket double — queues incoming json, captures sent."""

    def __init__(self, incoming: list[dict]) -> None:
        # `incoming` is consumed FIFO; when exhausted we raise to end the loop.
        self._incoming = list(incoming)
        self.sent: list[dict] = []

    async def receive_json(self) -> dict:
        if not self._incoming:
            from fastapi.websockets import WebSocketDisconnect

            raise WebSocketDisconnect()
        return self._incoming.pop(0)

    async def send_json(self, payload: dict) -> None:
        self.sent.append(payload)


class FakeKyutai:
    """Stand-in for KyutaiSession — replays a canned transcript."""

    def __init__(self, transcript: str = "bonjour le test") -> None:
        self._transcript = transcript
        self.fed_bytes = 0
        self._marker = 0

    async def connect(self) -> None:
        return None

    async def feed_pcm(self, pcm: bytes) -> None:
        self.fed_bytes += len(pcm)

    async def commit(self) -> int:
        self._marker += 1
        return self._marker

    async def words_until_marker(self, marker_id: int) -> AsyncIterator[dict]:
        for w in self._transcript.split():
            yield {"text": w, "start_time": 0.0}

    async def close(self) -> None:
        return None


async def _fake_stream_completion(*args: Any, **kwargs: Any) -> AsyncIterator[str]:
    for chunk in ["Bon", "jour", " ", "à", " toi."]:
        yield chunk


async def _fake_synthesise_chunks(text: str, **kwargs: Any) -> AsyncIterator[str]:
    yield base64.b64encode(b"\x00\x01" * 50).decode("ascii")
    yield base64.b64encode(b"\x02\x03" * 50).decode("ascii")


@pytest.fixture(autouse=True)
def _patch_adapters(monkeypatch):
    monkeypatch.setattr(session_mod.llm_adapter, "stream_completion", _fake_stream_completion)
    monkeypatch.setattr(session_mod.tts_adapter, "synthesise_chunks", _fake_synthesise_chunks)
    yield


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_end_to_end_round_trip_with_mocks() -> None:
    ws = FakeWS(
        [
            {"type": "session.update", "session": {"instructions": "tu es un test"}},
            {"type": "input_audio_buffer.append",
             "audio": base64.b64encode(b"\x00" * 320).decode("ascii")},
            {"type": "input_audio_buffer.commit"},
            {"type": "response.create", "response": {}},
        ]
    )
    sess = RealtimeSession(ws)  # type: ignore[arg-type]
    sess._stt = FakeKyutai("bonjour le test")
    asyncio.run(sess.run())

    types = [e["type"] for e in ws.sent]
    assert "conversation.item.input_audio_transcription.completed" in types
    assert any(t == "response.audio_transcript.delta" for t in types)
    assert "response.audio_transcript.done" in types
    assert any(t == "response.audio.delta" for t in types)
    assert "response.audio.done" in types
    # Final state should be back to LISTENING after the response cycle.
    assert sess._state == SessionState.CLOSED  # closed by WS disconnect


def test_transcript_propagated_to_event() -> None:
    ws = FakeWS(
        [
            {"type": "input_audio_buffer.commit"},
        ]
    )
    sess = RealtimeSession(ws)  # type: ignore[arg-type]
    sess._stt = FakeKyutai("ceci est un essai")
    asyncio.run(sess.run())
    completed = [e for e in ws.sent
                 if e["type"] == "conversation.item.input_audio_transcription.completed"]
    assert len(completed) == 1
    assert completed[0]["transcript"] == "ceci est un essai"


def test_response_create_without_transcript_errors() -> None:
    ws = FakeWS([{"type": "response.create", "response": {}}])
    sess = RealtimeSession(ws)  # type: ignore[arg-type]
    sess._stt = FakeKyutai()
    asyncio.run(sess.run())
    errors = [e for e in ws.sent if e["type"] == "error"]
    assert len(errors) == 1
    assert errors[0]["error"]["code"] == "no_input"


def test_unknown_client_event_is_silently_skipped() -> None:
    ws = FakeWS(
        [
            {"type": "session.create"},  # not in v1 mapping
            {"type": "input_audio_buffer.commit"},
        ]
    )
    sess = RealtimeSession(ws)  # type: ignore[arg-type]
    sess._stt = FakeKyutai("ok")
    asyncio.run(sess.run())
    # Unknown event produced no error to client (silent skip).
    assert not any(e["type"] == "error" for e in ws.sent), ws.sent
