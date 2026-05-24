"""OpenAI Realtime API event schemas — minimal v1 subset.

Pydantic models for the 9 events we currently support. Field names
match OpenAI's docs verbatim so JSON round-trips with their official
clients (openai-python, openai-realtime-console, etc.) work without
adapters.

Out of scope for v1 (silently dropped with a log line if received):
function calling, multi-modal inputs, persistent sessions, response
queues, in-flight cancellation. Add explicit error events later when
we know which subset our consumers actually use.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ── Client → Server ────────────────────────────────────────────────


class SessionUpdate(BaseModel):
    """``session.update`` — configure voice, instructions, modalities."""

    type: Literal["session.update"] = "session.update"
    session: dict[str, Any] = Field(default_factory=dict)


class InputAudioBufferAppend(BaseModel):
    """``input_audio_buffer.append`` — forward PCM to Kyutai.

    OpenAI ships PCM16 24 kHz little-endian, b64-encoded. We expect
    the same; the STT adapter handles any internal resampling.
    """

    type: Literal["input_audio_buffer.append"] = "input_audio_buffer.append"
    audio: str  # b64-encoded PCM16


class InputAudioBufferCommit(BaseModel):
    """``input_audio_buffer.commit`` — flush + ask Kyutai to finalise."""

    type: Literal["input_audio_buffer.commit"] = "input_audio_buffer.commit"


class ResponseCreate(BaseModel):
    """``response.create`` — trigger the LLM + TTS chain."""

    type: Literal["response.create"] = "response.create"
    response: dict[str, Any] = Field(default_factory=dict)


# ── Server → Client ────────────────────────────────────────────────


class TranscriptionCompleted(BaseModel):
    """``conversation.item.input_audio_transcription.completed``."""

    type: Literal[
        "conversation.item.input_audio_transcription.completed"
    ] = "conversation.item.input_audio_transcription.completed"
    event_id: str
    item_id: str
    content_index: int = 0
    transcript: str


class ResponseAudioDelta(BaseModel):
    """``response.audio.delta`` — one TTS chunk (b64-encoded PCM16)."""

    type: Literal["response.audio.delta"] = "response.audio.delta"
    event_id: str
    response_id: str
    item_id: str
    output_index: int = 0
    content_index: int = 0
    delta: str  # b64-encoded PCM16


class ResponseAudioDone(BaseModel):
    """``response.audio.done`` — end of TTS stream."""

    type: Literal["response.audio.done"] = "response.audio.done"
    event_id: str
    response_id: str
    item_id: str
    output_index: int = 0
    content_index: int = 0


class ResponseAudioTranscriptDelta(BaseModel):
    """``response.audio_transcript.delta`` — LLM token echoed for caption."""

    type: Literal[
        "response.audio_transcript.delta"
    ] = "response.audio_transcript.delta"
    event_id: str
    response_id: str
    item_id: str
    output_index: int = 0
    content_index: int = 0
    delta: str


class ResponseAudioTranscriptDone(BaseModel):
    """``response.audio_transcript.done``."""

    type: Literal[
        "response.audio_transcript.done"
    ] = "response.audio_transcript.done"
    event_id: str
    response_id: str
    item_id: str
    output_index: int = 0
    content_index: int = 0
    transcript: str


class ErrorEvent(BaseModel):
    """``error`` — upstream failure surfaced to the client."""

    type: Literal["error"] = "error"
    event_id: str
    error: dict[str, Any]


# ── Helpers ────────────────────────────────────────────────────────


def parse_client_event(raw: dict[str, Any]) -> Optional[BaseModel]:
    """Decode a raw client JSON message into the matching pydantic event.

    Returns None for unknown event types — the caller should log
    + skip (forward-compat with newer OpenAI events we don't yet
    handle). Returns None for missing/non-string ``type`` field too.
    """
    etype = raw.get("type")
    if not isinstance(etype, str):
        return None
    mapping = {
        "session.update": SessionUpdate,
        "input_audio_buffer.append": InputAudioBufferAppend,
        "input_audio_buffer.commit": InputAudioBufferCommit,
        "response.create": ResponseCreate,
    }
    cls = mapping.get(etype)
    if cls is None:
        return None
    return cls.model_validate(raw)
