"""TTS adapter — calls voice-bridge /tts, slices the WAV into PCM chunks.

OpenAI Realtime expects audio at 24 kHz PCM16 mono, b64-encoded, in
``response.audio.delta`` events. voice-bridge /tts returns exactly
that format inside a WAV container (F5 native sample rate is 24 kHz,
Kokoro emits 24 kHz too) so we strip the RIFF header and chunk.

Falls through to the Kokoro :8002 endpoint directly if voice-bridge
is configured off — useful when ailiance Realtime is the *only* TTS
consumer in a deployment and we want one less hop.
"""
from __future__ import annotations

import base64
import io
import logging
import os
import struct
from typing import AsyncIterator

import httpx
import numpy as np
import soundfile as sf


log = logging.getLogger(__name__)

VOICE_BRIDGE_URL = os.getenv(
    "REALTIME_VOICE_BRIDGE_URL", "http://100.116.92.12:8200"
)
KOKORO_URL = os.getenv("REALTIME_KOKORO_URL", "http://100.116.92.12:8002")
# Output to the client: 80 ms chunks at 24 kHz mono PCM16 = 3840 bytes.
# Small enough for low jitter on slow networks, big enough that we
# don't drown the WS in messages.
CHUNK_BYTES = int(os.getenv("REALTIME_TTS_CHUNK_BYTES", "3840"))


async def _voice_bridge_synthesise(text: str, timeout_s: float) -> bytes:
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        resp = await client.post(
            f"{VOICE_BRIDGE_URL}/tts",
            json={"text": text},
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"voice-bridge /tts {resp.status_code}: {resp.text[:200]}"
            )
        return resp.content


async def _kokoro_synthesise(text: str, timeout_s: float) -> bytes:
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        resp = await client.post(
            f"{KOKORO_URL}/synthesize",
            json={"text": text},
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"kokoro /synthesize {resp.status_code}: {resp.text[:200]}"
            )
        return resp.content


def _wav_to_pcm16(wav: bytes) -> bytes:
    """Strip the RIFF/WAVE container and return raw mono PCM16 samples.

    Accepts any standards-compliant WAV; uses soundfile so LIST/INFO
    chunks don't trip us. Resamples to 24 kHz if needed (F5 and Kokoro
    both emit 24 kHz already so this is a no-op in practice).
    """
    data, sr = sf.read(io.BytesIO(wav), dtype="int16", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1).astype(np.int16)
    if sr != 24_000:
        # Lazy import scipy only when needed.
        from math import gcd

        from scipy.signal import resample_poly

        f = data.astype(np.float32) / 32768.0
        g = gcd(sr, 24_000)
        f = resample_poly(f, up=24_000 // g, down=sr // g)
        data = (np.clip(f, -1.0, 1.0) * 32767).astype(np.int16)
    return data.tobytes()


async def synthesise_chunks(
    text: str,
    *,
    backend: str = "voice-bridge",
    timeout_s: float = 30.0,
) -> AsyncIterator[str]:
    """Synthesise text → yield b64-encoded PCM16 24 kHz chunks.

    Args:
        text: assistant reply to speak. The caller is expected to send
            the full reply at once; streaming-synthesis (token-level)
            is a separate refactor and not in v1.
        backend: ``voice-bridge`` (default — uses the full F5 → Kokoro
            → Piper chain) or ``kokoro`` (direct, fastest fallback).
        timeout_s: hard ceiling for the upstream call.

    Yields:
        Base64 strings ready to drop into ``response.audio.delta.delta``.
    """
    if not text.strip():
        return
    if backend == "kokoro":
        wav = await _kokoro_synthesise(text, timeout_s)
    else:
        wav = await _voice_bridge_synthesise(text, timeout_s)
    pcm = _wav_to_pcm16(wav)
    for i in range(0, len(pcm), CHUNK_BYTES):
        yield base64.b64encode(pcm[i : i + CHUNK_BYTES]).decode("ascii")
