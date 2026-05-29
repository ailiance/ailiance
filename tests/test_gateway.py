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
