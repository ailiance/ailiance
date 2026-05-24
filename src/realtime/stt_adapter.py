"""Kyutai STT adapter — incremental PCM in, Word events out.

Wraps the moshi-server WebSocket so a Realtime session can stream
incoming b64 PCM (OpenAI Realtime ``input_audio_buffer.append``)
straight to Kyutai and asynchronously consume ``Word`` events as
they arrive, exposing them as transcript partials/finals.

Differs from the baby-brain / Zacus one-shot helpers
(``kyutai_transcribe_pcm16``) because the Realtime API is
genuinely bidirectional — audio arrives over the *session*
WebSocket, not as a single buffer at the end. So this class owns
its upstream WS for the lifetime of the session and offers a
``feed_pcm`` / ``commit`` / ``words`` async-iterator surface.

Upstream protocol cheatsheet (see Zacus repo's MOSHI_STT_DEPLOY.md
for the runbook) :

  outgoing : {type:"Audio", pcm:[f32]} or {type:"Marker", id:int}
  incoming : Word / EndWord / Step / Marker
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import AsyncIterator, Optional

import msgpack
import numpy as np
import websockets
from scipy.signal import resample_poly


log = logging.getLogger(__name__)

KYUTAI_STT_URL = os.getenv(
    "KYUTAI_STT_URL", "ws://100.116.92.12:8304/api/asr-streaming"
)
KYUTAI_STT_KEY = os.getenv("KYUTAI_STT_KEY", "ailiance-realtime")
# OpenAI Realtime ships PCM16 24 kHz already, which is exactly Mimi's
# input rate, so the resample below is conditional.
KYUTAI_SR = 24_000
FRAME_SAMPLES = 1920  # 80 ms @ 24 kHz


class KyutaiSttError(RuntimeError):
    """Upstream Kyutai WS failed mid-session."""


class KyutaiSession:
    """Per-Realtime-connection Kyutai STT session.

    Lifecycle: ``connect()`` → many ``feed_pcm(...)`` → ``commit()``
    → iterate ``words()`` → ``close()``.

    Words are emitted to an internal queue so the session FSM can
    consume them without backpressuring the audio sender. The
    upstream WS is held open for the whole Realtime session even
    across multiple commit cycles (Kyutai handles re-marking).
    """

    def __init__(
        self,
        *,
        url: str = KYUTAI_STT_URL,
        api_key: str = KYUTAI_STT_KEY,
        input_sample_rate: int = 24_000,
    ) -> None:
        self._url = url
        self._api_key = api_key
        self._input_sr = input_sample_rate
        self._ws: Optional["websockets.WebSocketClientProtocol"] = None
        self._recv_task: Optional[asyncio.Task[None]] = None
        # Bounded queue so a wedged consumer back-pressures rather
        # than letting the WS reader OOM the process.
        self._words: asyncio.Queue[Optional[dict]] = asyncio.Queue(maxsize=512)
        self._marker_counter = 0

    async def connect(self) -> None:
        if self._ws is not None:
            return
        headers = {"kyutai-api-key": self._api_key}
        self._ws = await websockets.connect(
            self._url, additional_headers=headers, max_size=None
        )
        self._recv_task = asyncio.create_task(self._recv_loop())

    async def _recv_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                try:
                    evt = msgpack.unpackb(raw, raw=False)
                except (msgpack.exceptions.UnpackException, ValueError):
                    log.exception("kyutai bad msgpack")
                    continue
                etype = evt.get("type")
                if etype == "Word":
                    await self._words.put(
                        {
                            "kind": "word",
                            "text": (evt.get("text") or "").strip(),
                            "start_time": float(evt.get("start_time", 0.0)),
                        }
                    )
                elif etype == "Marker":
                    await self._words.put({"kind": "marker", "id": evt.get("id")})
                # Step / EndWord ignored
        except websockets.WebSocketException as exc:
            log.warning("kyutai ws closed: %s", exc)
            await self._words.put(None)  # sentinel

    def _resample_to_24k(self, pcm16: bytes) -> np.ndarray:
        f = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        if self._input_sr == KYUTAI_SR:
            return f
        # Generic polyphase — handles 16 kHz inputs and any other.
        from math import gcd

        g = gcd(self._input_sr, KYUTAI_SR)
        return resample_poly(
            f, up=KYUTAI_SR // g, down=self._input_sr // g
        ).astype(np.float32)

    async def feed_pcm(self, pcm16: bytes) -> None:
        """Send a PCM16 chunk upstream. Caller batches at its own cadence."""
        if self._ws is None:
            raise KyutaiSttError("session not connected")
        samples = self._resample_to_24k(pcm16)
        # Slice into Kyutai-friendly frames so the LM stays smooth even
        # if the caller pushes irregular OpenAI chunks.
        for i in range(0, len(samples), FRAME_SAMPLES):
            frame = samples[i : i + FRAME_SAMPLES]
            try:
                await self._ws.send(
                    msgpack.packb(
                        {"type": "Audio", "pcm": [float(x) for x in frame]},
                        use_single_float=True,
                    )
                )
            except websockets.WebSocketException as exc:
                raise KyutaiSttError(f"feed_pcm: {exc}") from exc

    async def commit(self) -> int:
        """Send a Marker; return its id so callers can match echo."""
        if self._ws is None:
            raise KyutaiSttError("session not connected")
        self._marker_counter += 1
        marker_id = self._marker_counter
        try:
            await self._ws.send(
                msgpack.packb(
                    {"type": "Marker", "id": marker_id}, use_single_float=True
                )
            )
            # Post-roll silence so the LM emits trailing words + echoes
            # the marker promptly. Same pattern as the one-shot helpers.
            silence = np.zeros(int(KYUTAI_SR * 3.0), dtype=np.float32)
            for i in range(0, len(silence), FRAME_SAMPLES):
                frame = silence[i : i + FRAME_SAMPLES]
                await self._ws.send(
                    msgpack.packb(
                        {"type": "Audio", "pcm": [float(x) for x in frame]},
                        use_single_float=True,
                    )
                )
        except websockets.WebSocketException as exc:
            raise KyutaiSttError(f"commit: {exc}") from exc
        return marker_id

    async def words_until_marker(
        self, marker_id: int
    ) -> AsyncIterator[dict]:
        """Yield Word events until we see the matching marker echo.

        Each yielded dict has shape ``{"text": str, "start_time": float}``.
        Raises ``KyutaiSttError`` if the upstream closes mid-utterance.
        """
        while True:
            evt = await self._words.get()
            if evt is None:
                raise KyutaiSttError("upstream closed mid-utterance")
            if evt["kind"] == "word":
                yield evt
            elif evt["kind"] == "marker" and evt.get("id") == marker_id:
                return

    async def close(self) -> None:
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        if self._recv_task is not None:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except (asyncio.CancelledError, Exception):
                pass
            self._recv_task = None
