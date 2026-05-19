"""Per-tenant KV-cache isolation via leading system prefix.

mlx_lm.server uses a single process-global LRUPromptCache indexed by
token prefix. Two requests sharing a long token prefix will hit the
same cache entries. That is fine for efficiency, but it also means a
timing side-channel could leak that *somebody else* recently submitted
a prompt with that prefix.

This module mitigates that by prepending a stable, per-tenant marker
at the start of every chat-completions request. Identical tenants
keep cache locality (efficiency win for repeat callers). Different
tenants diverge at token 1 in the trie, so their cache entries are
disjoint.

Identity is derived (in order of preference) from:

1. ``CF-Connecting-IP`` — Cloudflare tunnel forwards the real client
   IP here. This is the canonical source on the public deployment
   behind ``gateway.ailiance.fr``.
2. ``X-Real-IP`` — typical reverse-proxy header (nginx/Traefik).
3. ``X-Forwarded-For`` — comma-separated chain, we take the first.
4. ``request.client.host`` — direct TCP peer IP (LAN clients).
5. ``"anon"`` — no identifiable info available.

The IP is hashed with HMAC-SHA256 using ``AILIANCE_TENANT_SALT`` (an
environment variable; falls back to a per-process random salt). Only
the first 8 hex characters of the digest are kept — enough entropy
to split typical traffic mixes, short enough to keep prompt overhead
under 30 tokens.

Disable with ``AILIANCE_TENANT_ISOLATION=0`` (default ``1``).
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from typing import Optional

from src.worker.schemas import ChatMessage

# Resolved once at import. Override via env to make tenants stable
# across gateway restarts (useful for analytics) — otherwise a fresh
# salt is generated each boot.
_SALT = os.environ.get("AILIANCE_TENANT_SALT", "").encode() or secrets.token_bytes(16)

_ISOLATION_ENABLED = os.environ.get("AILIANCE_TENANT_ISOLATION", "1") != "0"

# Format chosen to look like a structural session marker, not part of
# the user content. Mistral / Llama / Qwen / Gemma chat templates all
# treat the system role as meta-context; a single short opaque token
# string has essentially zero impact on output behaviour.
_PREFIX_TEMPLATE = "[ailiance-session:{tenant}]"

# Chat templates that reject a leading ``system`` message: they require
# the first message to be ``user`` and raise on anything else. Mixtral-
# 8x22B-Instruct-v0.1 raises "Conversation roles must alternate user/
# assistant/..." on a leading system message. For these aliases the
# tenant marker is folded into the first user message instead; the per-
# tenant token-prefix divergence that drives cache isolation holds
# either way.
_NO_LEADING_SYSTEM_ALIASES = frozenset({
    "ailiance-mixtral",
    "ailiance-mixtral-8x22b",
})


def _extract_client_ip(headers: dict, peer_host: Optional[str]) -> str:
    """Pick the most trustworthy client identifier from request data.

    Headers are looked up case-insensitively. Returns ``"anon"`` if
    nothing usable is found.
    """
    lower = {k.lower(): v for k, v in headers.items() if isinstance(v, str)}
    for key in ("cf-connecting-ip", "x-real-ip"):
        val = lower.get(key)
        if val and val.strip():
            return val.strip()
    fwd = lower.get("x-forwarded-for")
    if fwd and fwd.strip():
        return fwd.split(",", 1)[0].strip()
    if peer_host:
        return peer_host
    return "anon"


def derive_tenant_id(headers: dict, peer_host: Optional[str]) -> str:
    """HMAC-hash the client identity into a short stable token.

    Same inputs always produce the same tenant id within a salt epoch.
    8 hex chars => 32-bit space, plenty to separate typical traffic.
    """
    ip = _extract_client_ip(headers, peer_host)
    digest = hmac.new(_SALT, ip.encode(), hashlib.sha256).hexdigest()
    return digest[:8]


def isolation_enabled() -> bool:
    """Allow callers to short-circuit the injection cheaply."""
    return _ISOLATION_ENABLED


def _fold_marker_into_first_user(
    messages: list[ChatMessage], marker: str
) -> list[ChatMessage]:
    """Prepend ``marker`` to the first user message's text content.

    Used for chat templates that reject a leading system message. If no
    user message carries plain-string content, fall back to a leading
    system message — keeps a marker present rather than dropping
    isolation. The original list is not mutated.
    """
    out = list(messages)
    for i, msg in enumerate(out):
        if msg.role == "user" and isinstance(msg.content, str):
            out[i] = msg.model_copy(
                update={"content": f"{marker}\n{msg.content}"}
            )
            return out
    return [ChatMessage(role="system", content=marker), *out]


def inject_tenant_prefix(
    messages: list[ChatMessage],
    tenant_id: str,
    alias: str | None = None,
) -> list[ChatMessage]:
    """Return ``messages`` with the per-tenant session marker prepended.

    Default: a leading ``system`` message. For aliases whose chat
    template rejects a leading system role (``_NO_LEADING_SYSTEM_ALIASES``)
    the marker is folded into the first user message instead — the
    token-prefix divergence that drives cache isolation holds either way.

    The original list is *not* mutated — we return a new list so the
    caller can keep an unmodified copy for audit logging.
    """
    marker = _PREFIX_TEMPLATE.format(tenant=tenant_id)
    if alias in _NO_LEADING_SYSTEM_ALIASES:
        return _fold_marker_into_first_user(messages, marker)
    return [ChatMessage(role="system", content=marker), *messages]
