"""Per-connection Realtime session FSM.

Glues the three adapters around an OpenAI Realtime event protocol.

States:
    LISTENING : audio is streaming in, no response in flight
    THINKING  : commit received → Kyutai final → LLM call
    SPEAKING  : TTS chunks streaming to the client
    CLOSED    : socket dead

State transitions are intentionally lossy — if the client fires a new
``response.create`` while we're already SPEAKING, we drop the new one
with an ``error`` event (response queues are v2).
"""
from __future__ import annotations

import asyncio
import base64
import logging
import uuid
from enum import Enum, auto
from typing import Any

from fastapi import WebSocket
from fastapi.websockets import WebSocketDisconnect

from . import llm_adapter, tts_adapter
from .protocol import (
    ErrorEvent,
    InputAudioBufferAppend,
    InputAudioBufferCommit,
    ResponseAudioDelta,
    ResponseAudioDone,
    ResponseAudioTranscriptDelta,
    ResponseAudioTranscriptDone,
    ResponseCreate,
    SessionUpdate,
    TranscriptionCompleted,
    parse_client_event,
)
from .stt_adapter import KyutaiSession, KyutaiSttError


log = logging.getLogger(__name__)


class SessionState(Enum):
    LISTENING = auto()
    THINKING = auto()
    SPEAKING = auto()
    CLOSED = auto()


class RealtimeSession:
    """One Realtime WS connection. Owns one Kyutai upstream session."""

    def __init__(self, ws: WebSocket) -> None:
        self._ws = ws
        self._state = SessionState.LISTENING
        self._stt = KyutaiSession()
        self._instructions: str | None = None
        self._voice: str | None = None  # currently ignored — Kokoro voice
        self._last_transcript: str = ""
        self._history: list[dict] = []

    # ── helpers ─────────────────────────────────────────────────

    def _eid(self) -> str:
        return f"event_{uuid.uuid4().hex[:12]}"

    def _rid(self) -> str:
        return f"resp_{uuid.uuid4().hex[:12]}"

    def _iid(self) -> str:
        return f"item_{uuid.uuid4().hex[:12]}"

    async def _send(self, event: Any) -> None:
        # pydantic v2 — model_dump exists on every event class.
        await self._ws.send_json(event.model_dump())

    async def _send_error(self, message: str, code: str = "upstream_error") -> None:
        await self._send(
            ErrorEvent(
                event_id=self._eid(),
                error={"type": "server_error", "code": code, "message": message},
            )
        )

    # ── client event handlers ───────────────────────────────────

    async def _on_session_update(self, evt: SessionUpdate) -> None:
        sess = evt.session
        if isinstance(sess.get("instructions"), str):
            self._instructions = sess["instructions"]
        if isinstance(sess.get("voice"), str):
            self._voice = sess["voice"]

    async def _on_audio_append(self, evt: InputAudioBufferAppend) -> None:
        try:
            pcm = base64.b64decode(evt.audio)
        except Exception as exc:
            await self._send_error(f"bad b64 audio: {exc}", "bad_request")
            return
        try:
            await self._stt.feed_pcm(pcm)
        except KyutaiSttError as exc:
            await self._send_error(f"kyutai feed: {exc}")

    async def _on_audio_commit(self, evt: InputAudioBufferCommit) -> None:
        try:
            marker = await self._stt.commit()
            words: list[str] = []
            async for w in self._stt.words_until_marker(marker):
                words.append(w["text"])
        except KyutaiSttError as exc:
            await self._send_error(f"kyutai commit: {exc}")
            return
        transcript = " ".join(words).strip()
        self._last_transcript = transcript
        item_id = self._iid()
        await self._send(
            TranscriptionCompleted(
                event_id=self._eid(),
                item_id=item_id,
                transcript=transcript,
            )
        )

    async def _on_response_create(self, evt: ResponseCreate) -> None:
        if self._state == SessionState.SPEAKING:
            await self._send_error(
                "response already in flight; queues not supported in v1",
                "response_in_progress",
            )
            return
        if not self._last_transcript:
            await self._send_error(
                "no transcript to respond to (commit first)", "no_input"
            )
            return
        self._state = SessionState.THINKING
        rid = self._rid()
        iid = self._iid()

        # 1. Stream LLM tokens; we accumulate for TTS and echo each
        #    chunk as an audio_transcript.delta so clients showing
        #    captions get live feedback during synthesis.
        reply_chunks: list[str] = []
        try:
            async for delta in llm_adapter.stream_completion(
                self._last_transcript,
                instructions=self._instructions,
                history=self._history,
            ):
                reply_chunks.append(delta)
                await self._send(
                    ResponseAudioTranscriptDelta(
                        event_id=self._eid(),
                        response_id=rid,
                        item_id=iid,
                        delta=delta,
                    )
                )
        except Exception as exc:
            await self._send_error(f"llm: {exc}", "llm_error")
            self._state = SessionState.LISTENING
            return

        full_reply = "".join(reply_chunks).strip()
        await self._send(
            ResponseAudioTranscriptDone(
                event_id=self._eid(),
                response_id=rid,
                item_id=iid,
                transcript=full_reply,
            )
        )
        self._history.append({"role": "user", "content": self._last_transcript})
        self._history.append({"role": "assistant", "content": full_reply})

        # 2. Synthesise + stream audio chunks.
        self._state = SessionState.SPEAKING
        try:
            async for b64chunk in tts_adapter.synthesise_chunks(full_reply):
                await self._send(
                    ResponseAudioDelta(
                        event_id=self._eid(),
                        response_id=rid,
                        item_id=iid,
                        delta=b64chunk,
                    )
                )
        except Exception as exc:
            await self._send_error(f"tts: {exc}", "tts_error")
            self._state = SessionState.LISTENING
            return

        await self._send(
            ResponseAudioDone(event_id=self._eid(), response_id=rid, item_id=iid)
        )
        self._state = SessionState.LISTENING

    # ── lifecycle ──────────────────────────────────────────────

    async def run(self) -> None:
        await self._stt.connect()
        try:
            while True:
                try:
                    raw = await self._ws.receive_json()
                except WebSocketDisconnect:
                    return
                evt = parse_client_event(raw)
                if evt is None:
                    log.info("realtime: unknown event %r", raw.get("type"))
                    continue
                if isinstance(evt, SessionUpdate):
                    await self._on_session_update(evt)
                elif isinstance(evt, InputAudioBufferAppend):
                    await self._on_audio_append(evt)
                elif isinstance(evt, InputAudioBufferCommit):
                    await self._on_audio_commit(evt)
                elif isinstance(evt, ResponseCreate):
                    await self._on_response_create(evt)
        finally:
            self._state = SessionState.CLOSED
            await self._stt.close()
