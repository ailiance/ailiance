"""Unit tests for the Realtime API event protocol layer."""
from __future__ import annotations

import pytest

from src.realtime import protocol


def test_parse_known_client_events() -> None:
    cases = [
        ({"type": "session.update", "session": {"voice": "ff_siwis"}}, protocol.SessionUpdate),
        ({"type": "input_audio_buffer.append", "audio": "aGVsbG8="}, protocol.InputAudioBufferAppend),
        ({"type": "input_audio_buffer.commit"}, protocol.InputAudioBufferCommit),
        ({"type": "response.create", "response": {}}, protocol.ResponseCreate),
    ]
    for raw, cls in cases:
        evt = protocol.parse_client_event(raw)
        assert isinstance(evt, cls), f"{raw} → expected {cls}, got {type(evt)}"


def test_parse_unknown_event_returns_none() -> None:
    assert protocol.parse_client_event({"type": "conversation.delete"}) is None
    assert protocol.parse_client_event({"type": "session.create"}) is None


def test_parse_missing_type_returns_none() -> None:
    assert protocol.parse_client_event({}) is None
    assert protocol.parse_client_event({"audio": "x"}) is None
    assert protocol.parse_client_event({"type": 42}) is None  # non-string


def test_audio_append_validates_audio_field() -> None:
    # OpenAI semantics: ``audio`` is a string (b64). Missing → 422-equivalent.
    with pytest.raises(Exception):
        protocol.parse_client_event({"type": "input_audio_buffer.append"})


def test_server_event_round_trip() -> None:
    evt = protocol.TranscriptionCompleted(
        event_id="event_abc", item_id="item_xyz", transcript="bonjour"
    )
    dumped = evt.model_dump()
    assert dumped["type"] == (
        "conversation.item.input_audio_transcription.completed"
    )
    assert dumped["transcript"] == "bonjour"
    assert dumped["content_index"] == 0  # default field
