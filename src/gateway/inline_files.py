"""Inline file extraction + image staging for /v1/chat/completions.

Two pre-forward transforms applied to the request body so callers can
attach files and images directly inside ``messages[].content`` without
a separate round-trip:

1. **input_file blocks** — when content carries
   ``{"type":"input_file","file":{...}}``, the file is decoded
   (base64 data URL or HTTP fetch) and the block is rewritten as a
   plain text block with the extracted markdown inlined. Any text
   LLM downstream then has the document context in-prompt.

2. **image_url with data: URLs** — MLX vision workers (Pixtral on
   Mac Studio :9325) reject ``data:image/*;base64,…`` URLs silently
   (the worker returns hallucinated answers as if blind). We decode
   the data URL, stash the bytes in an in-memory store with a short
   TTL, and rewrite the URL to a public HTTP endpoint
   (:func:`stage_image_url`) served by this same gateway. The worker
   then downloads the image like a normal HTTP URL.

The image store is intentionally process-local and bounded
(``MAX_STAGE_BYTES`` / ``MAX_STAGE_ENTRIES``) — this is a request
auxiliary, not a CDN.
"""

from __future__ import annotations

import base64
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

import httpx

from src.gateway.file_extract import ExtractError, extract as extract_file

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# In-memory staged-image store
# ---------------------------------------------------------------------------

@dataclass
class _StagedImage:
    data: bytes
    mime: str
    expires_at: float  # monotonic seconds


# 5-minute TTL — long enough for a worker to fetch, short enough that a
# leaked URL is uninteresting.
STAGE_TTL_S = 300

# 64 MiB cap on a single staged image; bigger than that is suspicious.
MAX_STAGE_BYTES = 64 * 1024 * 1024

# Total entries we'll keep — older entries are evicted FIFO. Sizing for
# bursts of Playground uploads; bumps cheap if needed.
MAX_STAGE_ENTRIES = 256

_DATA_URL_RE = re.compile(
    r"^data:(?P<mime>[\w./+-]+);base64,(?P<b64>[A-Za-z0-9+/=\s]+)$",
    re.DOTALL,
)


class _ImageStore:
    """Thread-safe in-memory store with TTL eviction.

    Reads from this store happen on the GET /v1/_staged endpoint; writes
    happen inside the chat-completions request handler. Both paths run
    inside the asyncio loop's threadpool / event loop on the same
    process, but a lock keeps invariants simple.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: dict[str, _StagedImage] = {}

    def put(self, data: bytes, mime: str) -> str:
        if len(data) > MAX_STAGE_BYTES:
            raise ValueError(f"staged image exceeds {MAX_STAGE_BYTES} bytes")
        key = uuid.uuid4().hex
        expires = time.monotonic() + STAGE_TTL_S
        with self._lock:
            self._items[key] = _StagedImage(data=data, mime=mime, expires_at=expires)
            self._evict_locked()
        return key

    def get(self, key: str) -> _StagedImage | None:
        with self._lock:
            entry = self._items.get(key)
            if entry is None:
                return None
            if entry.expires_at < time.monotonic():
                self._items.pop(key, None)
                return None
            return entry

    def _evict_locked(self) -> None:
        # Drop expired first, then trim FIFO if still over capacity.
        now = time.monotonic()
        expired = [k for k, v in self._items.items() if v.expires_at < now]
        for k in expired:
            self._items.pop(k, None)
        while len(self._items) > MAX_STAGE_ENTRIES:
            # dict preserves insertion order — pop oldest.
            oldest = next(iter(self._items))
            self._items.pop(oldest, None)

    def __len__(self) -> int:  # pragma: no cover — debug only
        with self._lock:
            return len(self._items)


# Module-level singleton (mirrors how the rest of the gateway holds
# state). Tests can reset via ``image_store._items.clear()``.
image_store = _ImageStore()


# ---------------------------------------------------------------------------
# Content rewrites
# ---------------------------------------------------------------------------

# Filename hints we recognise inside an input_file block when the caller
# doesn't supply ``mime`` / ``filename`` directly.
_FILENAME_KEYS = ("filename", "name", "file_name")
_MIME_KEYS = ("mime", "mime_type", "content_type", "type")


def _decode_data_url(url: str) -> tuple[bytes, str] | None:
    """Return ``(bytes, mime)`` for a ``data:`` URL, or ``None`` if not one."""
    m = _DATA_URL_RE.match(url.strip())
    if not m:
        return None
    try:
        data = base64.b64decode(m.group("b64").encode("ascii"), validate=False)
    except Exception:  # noqa: BLE001 — surface as invalid_request upstream
        return None
    return data, m.group("mime")


async def _fetch_url_bytes(url: str, client: httpx.AsyncClient) -> tuple[bytes, str]:
    """Download ``url`` and return ``(bytes, mime)``.

    Raises :class:`ExtractError` on HTTP error or non-2xx response so the
    caller can map cleanly to a 400.
    """
    try:
        resp = await client.get(url, timeout=15.0, follow_redirects=True)
    except httpx.HTTPError as exc:
        raise ExtractError("fetch_failed", f"failed to fetch {url!r}: {exc}") from exc
    if resp.status_code != 200:
        raise ExtractError(
            "fetch_failed",
            f"GET {url!r} returned HTTP {resp.status_code}",
        )
    return resp.content, resp.headers.get("content-type", "")


def _coerce_to_text_block(filename: str, markdown: str) -> dict[str, Any]:
    body = f"Attached file: {filename}\n\n```markdown\n{markdown}\n```"
    return {"type": "text", "text": body}


async def rewrite_input_files(
    messages: list[Any],
    http_client: httpx.AsyncClient,
) -> tuple[int, list[str]]:
    """Replace ``input_file`` blocks with extracted-text blocks in-place.

    Iterates the ``messages`` list, finds each ``content`` list, and
    rewrites any ``{"type":"input_file","file":{...}}`` entry to a
    ``{"type":"text","text":"…"}`` block carrying the file's markdown.

    Returns ``(extracted_count, warnings)``. Errors bubble up as
    :class:`ExtractError` — the caller maps them to a 400 so the
    client knows their attachment failed.
    """
    extracted = 0
    warnings: list[str] = []
    for msg in messages:
        content = _get_attr_or_key(msg, "content")
        if not isinstance(content, list):
            continue
        for i, part in enumerate(list(content)):
            if not isinstance(part, dict):
                continue
            if part.get("type") != "input_file":
                continue
            file_obj = part.get("file") if isinstance(part.get("file"), dict) else {}
            filename, mime = _file_hints(file_obj)
            data = await _resolve_file_bytes(file_obj, http_client)
            try:
                result = extract_file(data, filename=filename, mime=mime)
            except ExtractError:
                raise
            content[i] = _coerce_to_text_block(filename or "file", result.markdown)
            extracted += 1
            if result.metadata.get("truncated"):
                warnings.append(
                    f"{filename or 'file'}: truncated to {result.metadata.get('kept_chars')} chars"
                )
    return extracted, warnings


async def _resolve_file_bytes(
    file_obj: dict[str, Any],
    http_client: httpx.AsyncClient,
) -> bytes:
    """Return bytes for an input_file payload, regardless of shape.

    Supported shapes:

    * ``{"url": "data:application/pdf;base64,…"}``
    * ``{"url": "https://example.com/doc.pdf"}``
    * ``{"data": "<base64>", "mime": "application/pdf"}``
    """
    url = file_obj.get("url")
    if isinstance(url, str):
        if url.startswith("data:"):
            decoded = _decode_data_url(url)
            if decoded is None:
                raise ExtractError("bad_data_url", "data: URL is malformed")
            return decoded[0]
        if url.startswith(("http://", "https://")):
            data, _ = await _fetch_url_bytes(url, http_client)
            return data
    b64 = file_obj.get("data")
    if isinstance(b64, str):
        try:
            return base64.b64decode(b64.encode("ascii"), validate=False)
        except Exception as exc:  # noqa: BLE001
            raise ExtractError("bad_base64", f"file.data not valid base64: {exc}") from exc
    raise ExtractError(
        "missing_source",
        "input_file requires file.url (http/https or data:) or file.data (base64).",
    )


def _file_hints(file_obj: dict[str, Any]) -> tuple[str | None, str | None]:
    filename = None
    for key in _FILENAME_KEYS:
        v = file_obj.get(key)
        if isinstance(v, str):
            filename = v
            break
    mime = None
    for key in _MIME_KEYS:
        v = file_obj.get(key)
        if isinstance(v, str):
            mime = v
            break
    return filename, mime


def stage_image_url(url: str, public_base: str) -> str:
    """If ``url`` is a ``data:`` image URL, stage and return a public URL.

    Otherwise (regular http/https) returns ``url`` unchanged so callers
    can use this as an unconditional rewrite step.
    """
    if not isinstance(url, str) or not url.startswith("data:image/"):
        return url
    decoded = _decode_data_url(url)
    if decoded is None:
        return url  # malformed — let the worker complain
    data, mime = decoded
    try:
        key = image_store.put(data, mime)
    except ValueError:
        return url
    base = public_base.rstrip("/")
    return f"{base}/v1/_staged/{key}"


def rewrite_image_urls(messages: list[Any], public_base: str) -> int:
    """Rewrite every ``image_url.url`` data URL to a staged HTTP URL.

    Mutates ``messages`` in place. Returns the number of URLs staged.
    """
    staged = 0
    for msg in messages:
        content = _get_attr_or_key(msg, "content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") not in ("image_url", "image", "input_image"):
                continue
            url_obj = part.get("image_url")
            if isinstance(url_obj, dict):
                url = url_obj.get("url")
                if isinstance(url, str) and url.startswith("data:image/"):
                    new_url = stage_image_url(url, public_base)
                    if new_url != url:
                        url_obj["url"] = new_url
                        staged += 1
            elif isinstance(url_obj, str) and url_obj.startswith("data:image/"):
                new_url = stage_image_url(url_obj, public_base)
                if new_url != url_obj:
                    part["image_url"] = new_url
                    staged += 1
    return staged


def _get_attr_or_key(obj: Any, key: str) -> Any:
    """Read ``obj.key`` or ``obj[key]`` — bridges pydantic models and dicts."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)
