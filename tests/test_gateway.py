# tests/test_gateway.py
import pytest
import httpx
from fastapi.testclient import TestClient


def test_gateway_health():
    from src.gateway.server import make_gateway_app

    app = make_gateway_app(skip_router_load=True)
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_gateway_models_list():
    from src.gateway.server import make_gateway_app

    app = make_gateway_app(skip_router_load=True)
    client = TestClient(app)
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    models = resp.json()["data"]
    ids = set(m["id"] for m in models)
    # Core production workers + the bare "ailiance" auto-router alias.
    # Asserted as subset so adding a new alias does not break the test.
    # ``ailiance-embed`` is intentionally absent: bge-m3 is an embedding
    # model and /v1/models advertises chat-capable aliases only.
    expected_core = {
        "ailiance",
        "ailiance-mistral-medium",
        "ailiance-mistral",
        "ailiance-gemma",
        "ailiance-qwen",
        # Tower Ollama wiring (2026-05-11) — mascarade fine-tunes
        "ailiance-kicad",
    }
    assert expected_core.issubset(ids)
    # And the embed surface stays off the chat listing.
    assert "ailiance-embed" not in ids


def test_gateway_force_map_has_all_workers():
    """MODEL_FORCE_MAP is the single source of truth for force-routing.

    Asserted as subset so adding a new worker alias does not require
    a test edit; only locks the core 5 plus production aliases that
    callers depend on.
    """
    from src.gateway.server import MODEL_FORCE_MAP

    expected_core = {
        "ailiance-mistral-medium",
        "ailiance-mistral",
        "ailiance-gemma",
        "ailiance-qwen",
        "ailiance-kicad",
        # ailiance-embed is in MODEL_FORCE_MAP (route still resolves) but
        # rejected at /v1/chat/completions via _BLOCKED_CHAT_ALIASES.
        "ailiance-embed",
    }
    assert expected_core.issubset(set(MODEL_FORCE_MAP))
    # Mascarade hardware aliases route to Studio MLX :9340 (omlx consolidation 2026-05-29).
    assert MODEL_FORCE_MAP["ailiance-kicad"] == 9340
    # ailiance-embed (bge-m3) remains on Tower Ollama :8004 (embed-only, not a chat model).
    assert MODEL_FORCE_MAP["ailiance-embed"] == 8004
    # ailiance-apertus is preserved as legacy alias → routes to mistral-medium.
    assert MODEL_FORCE_MAP.get("ailiance-apertus") == 9301
    # Qwen is reached via the autossh tunnel on the gateway host (port 8002).
    assert MODEL_FORCE_MAP["ailiance-qwen"] == 8002
    # Gemma sits on tower:9304.
    assert MODEL_FORCE_MAP["ailiance-gemma"] == 9304


def test_models_endpoint_filters_unhealthy_ports(monkeypatch):
    """Fix A — /v1/models must hide aliases whose worker port is unhealthy."""
    import src.gateway.server as gw

    # Expose only port 9301; all aliases pointing elsewhere should vanish.
    monkeypatch.setattr(gw, "_healthy_ports", {9301})

    app = gw.make_gateway_app(skip_router_load=True)
    client = TestClient(app)
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    ids = [m["id"] for m in resp.json()["data"]]

    # Every non-ailiance alias must map to a healthy port.
    for mid in ids:
        if mid == "ailiance":
            continue
        assert gw.MODEL_FORCE_MAP.get(mid) in {9301}, (
            f"alias {mid!r} (port {gw.MODEL_FORCE_MAP.get(mid)}) "
            "was advertised but its port is not in _healthy_ports"
        )

    # ailiance-kicad → 9340, which is NOT in {9301}: must be absent.
    assert "ailiance-kicad" not in ids, (
        "ailiance-kicad (port 9340) should be hidden when only port 9301 is healthy"
    )


def test_models_endpoint_always_includes_ailiance(monkeypatch):
    """Fix A — bare 'ailiance' auto-router alias must survive even with no healthy ports."""
    import src.gateway.server as gw

    monkeypatch.setattr(gw, "_healthy_ports", set())

    app = gw.make_gateway_app(skip_router_load=True)
    client = TestClient(app)
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    ids = [m["id"] for m in resp.json()["data"]]
    assert "ailiance" in ids, "'ailiance' auto-router alias must always be present"


def test_unreachable_worker_returns_503(monkeypatch):
    """Fix B — ConnectError from upstream must produce a clean 503, not 500."""
    # http_client is a local AsyncClient inside make_gateway_app, so we
    # intercept at the httpx.AsyncClient.post level before the app is built.
    async def _fail_post(self, *args, **kwargs):
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr(httpx.AsyncClient, "post", _fail_post)

    from src.gateway.server import make_gateway_app

    app = make_gateway_app(skip_router_load=True)
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "ailiance-gemma",  # port 9304 — explicit force-route
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert resp.status_code == 503, f"Expected 503, got {resp.status_code}"
    detail = resp.json().get("detail", {})
    assert detail.get("type") == "upstream_unreachable", (
        f"Expected type='upstream_unreachable', got detail={detail!r}"
    )


# ---------------------------------------------------------------------------
# Issue #10 — empty completion 502 guard
# ---------------------------------------------------------------------------

def _mock_worker_post(response_body: dict, status_code: int = 200):
    """Return a monkeypatch-compatible async post that replies with *response_body*."""
    import json as _json

    async def _fake_post(self, url, *args, **kwargs):
        content = _json.dumps(response_body).encode()
        return httpx.Response(status_code, content=content,
                              headers={"content-type": "application/json"})

    return _fake_post


def _chat_payload(model: str = "ailiance-gemma") -> dict:
    return {"model": model, "messages": [{"role": "user", "content": "hi"}]}


def _empty_completion(*, tool_calls=None, reasoning=None, usage=None) -> dict:
    """Build a structurally-valid 200 body with empty content."""
    msg: dict = {"role": "assistant", "content": ""}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    if reasoning is not None:
        msg["reasoning"] = reasoning
    body: dict = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
    }
    if usage is not None:
        body["usage"] = usage
    else:
        body["usage"] = {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10}
    return body


def test_empty_completion_returns_502(monkeypatch):
    """Issue #10: empty content + completion_tokens==0 + no tools → 502."""
    monkeypatch.setattr(httpx.AsyncClient, "post",
                        _mock_worker_post(_empty_completion()))

    from src.gateway.server import make_gateway_app

    app = make_gateway_app(skip_router_load=True)
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/v1/chat/completions", json=_chat_payload())
    assert resp.status_code == 502, f"Expected 502, got {resp.status_code}: {resp.text}"
    detail = resp.json().get("detail", {})
    assert detail.get("type") == "empty_completion", (
        f"Expected type='empty_completion', got detail={detail!r}"
    )


def test_empty_completion_with_tool_calls_passes(monkeypatch):
    """tool_calls present → content may legitimately be empty; must NOT 502."""
    tool_calls = [{"id": "call_1", "type": "function",
                   "function": {"name": "my_fn", "arguments": "{}"}}]
    monkeypatch.setattr(
        httpx.AsyncClient, "post",
        _mock_worker_post(_empty_completion(tool_calls=tool_calls)),
    )

    from src.gateway.server import make_gateway_app

    app = make_gateway_app(skip_router_load=True)
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/v1/chat/completions", json=_chat_payload())
    assert resp.status_code == 200, (
        f"tool_calls response must not be rejected; got {resp.status_code}"
    )


def test_empty_content_with_reasoning_passes(monkeypatch):
    """content empty but reasoning non-blank → normalize backfills content → 200."""
    body = _empty_completion(reasoning="my chain of thought")
    # After _normalize_response_body, content will be promoted from reasoning,
    # so the guard must NOT fire.
    monkeypatch.setattr(httpx.AsyncClient, "post", _mock_worker_post(body))

    from src.gateway.server import make_gateway_app

    app = make_gateway_app(skip_router_load=True)
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/v1/chat/completions", json=_chat_payload())
    assert resp.status_code == 200, (
        f"reasoning-backed response must not be rejected; got {resp.status_code}"
    )


def test_non_empty_content_passes(monkeypatch):
    """Normal non-empty response must not trigger the guard."""
    body = {
        "id": "chatcmpl-ok",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": "Hello!"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }
    monkeypatch.setattr(httpx.AsyncClient, "post", _mock_worker_post(body))

    from src.gateway.server import make_gateway_app

    app = make_gateway_app(skip_router_load=True)
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/v1/chat/completions", json=_chat_payload())
    assert resp.status_code == 200, (
        f"Normal response must not be rejected; got {resp.status_code}"
    )


def test_empty_content_no_usage_relayed(monkeypatch):
    """Conservative: usage absent → completion_tokens is None (not 0) → relay 200.

    Documents the deliberate decision: when the worker gives no token info
    we cannot be sure it's a true empty completion, so we relay rather than 502.
    """
    body = _empty_completion()
    del body["usage"]  # remove usage entirely
    monkeypatch.setattr(httpx.AsyncClient, "post", _mock_worker_post(body))

    from src.gateway.server import make_gateway_app

    app = make_gateway_app(skip_router_load=True)
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/v1/chat/completions", json=_chat_payload())
    assert resp.status_code == 200, (
        f"No-usage response must be relayed conservatively; got {resp.status_code}"
    )


# ---------------------------------------------------------------------------
# Issue #117 — mid-stream worker drop: log + telemetry, no crash/500
# ---------------------------------------------------------------------------

def _make_mid_stream_drop_send(first_chunk: str):
    """
    Return a mock for httpx.AsyncClient.send that:
    - Returns a streaming HTTP 200 with content-type text/event-stream.
    - aiter_text() yields *first_chunk* then raises RemoteProtocolError.

    The mock patches the `send` method (not `post`) because the streaming path
    calls `http_client.send(req_stream, stream=True)`.

    The fake response uses a custom stream object that satisfies httpx's
    async iterator protocol without requiring a real network connection.
    """
    import json as _json

    class _DropStream(httpx.AsyncByteStream):
        """Async byte-stream that yields one chunk then simulates a mid-stream drop.

        Must subclass httpx.AsyncByteStream so httpx's aiter_raw() guard passes.
        """

        def __init__(self, chunk: bytes):
            self._chunk = chunk

        async def __aiter__(self):
            yield self._chunk
            raise httpx.RemoteProtocolError(
                "peer closed connection without sending complete message body",
                request=None,
            )

        async def aclose(self):
            pass

    async def _fake_send(self, request, *, stream=False, **kwargs):
        raw_stream = _DropStream(first_chunk.encode("utf-8"))
        resp = httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=raw_stream,
            request=request,
        )
        return resp

    return _fake_send


def test_mid_stream_drop_no_crash(monkeypatch):
    """Issue #117: worker drops SSE connection after first chunk.

    Assertions:
    (a) The client receives the initial SSE chunk (stream starts, no 500).
    (b) The response status is 200 (already committed before drop — cannot 5xx).
    (c) No unhandled exception propagates to the test.

    The log/telemetry side-effect (track_chat with mid_stream_drop error) is
    verified in test_mid_stream_drop_telemetry below.
    """
    import json as _json

    # A single valid SSE chunk with content so saw_content=True (prevents the
    # 'empty_completion' branch in finally — only the drop warning fires).
    chunk = 'data: ' + _json.dumps({
        "choices": [{"index": 0, "delta": {"content": "hello"}}]
    }) + "\n\n"

    monkeypatch.setattr(
        httpx.AsyncClient, "send", _make_mid_stream_drop_send(chunk)
    )

    from src.gateway.server import make_gateway_app

    app = make_gateway_app(skip_router_load=True)
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "ailiance-gemma", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        headers={"Accept": "text/event-stream"},
    )
    # HTTP 200 already committed — mid-stream drop cannot become 5xx.
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text!r}"
    # The first chunk must have reached the client.
    assert b"hello" in resp.content, (
        f"Expected first chunk to be relayed, got: {resp.content!r}"
    )


def test_mid_stream_drop_telemetry(monkeypatch, caplog):
    """Issue #117: worker mid-stream drop must emit a warning log containing
    'mid-stream worker drop' (and not crash).

    Uses caplog to capture the WARNING from relay()'s except clause.
    """
    import json as _json
    import logging

    chunk = 'data: ' + _json.dumps({
        "choices": [{"index": 0, "delta": {"content": "first"}}]
    }) + "\n\n"

    monkeypatch.setattr(
        httpx.AsyncClient, "send", _make_mid_stream_drop_send(chunk)
    )

    from src.gateway.server import make_gateway_app

    app = make_gateway_app(skip_router_load=True)
    client = TestClient(app, raise_server_exceptions=False)

    with caplog.at_level(logging.WARNING, logger="gateway"):
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "ailiance-gemma", "messages": [{"role": "user", "content": "hi"}], "stream": True},
            headers={"Accept": "text/event-stream"},
        )

    assert resp.status_code == 200
    # The except clause in relay() must have logged a warning.
    drop_warnings = [r for r in caplog.records if "mid-stream" in r.message.lower() and r.levelno == logging.WARNING]
    assert drop_warnings, (
        f"Expected a WARNING log containing 'mid-stream', got records: {[(r.levelname, r.message) for r in caplog.records]}"
    )


def test_worker_headers_includes_served_model():
    from src.router.domain_map import QWEN36_PORT
    from src.gateway.server import _worker_headers
    h = _worker_headers(worker_port=QWEN36_PORT, domain="emc", effective_alias="ailiance")
    assert h["X-Ailiance-Served-Model"] == "qwen36-emc-dsp-power"


def test_worker_headers_omits_served_model_when_uninformative():
    from src.gateway.server import _worker_headers
    h = _worker_headers(worker_port=9304, domain="", effective_alias="ailiance")
    assert "X-Ailiance-Served-Model" not in h


def test_track_chat_accepts_served_model(monkeypatch):
    import src.gateway.observability as obs
    import asyncio
    captured = {}

    async def _fake_send_trace(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(obs, "_send_trace", _fake_send_trace)

    async def _run():
        obs.track_chat(model_alias="ailiance", domain="emc", kind="direct",
                       request_body={}, response_body={}, started_at=0.0,
                       served_model="qwen36-emc-dsp-power")
        await asyncio.sleep(0)

    asyncio.run(_run())
    assert captured.get("served_model") == "qwen36-emc-dsp-power"
