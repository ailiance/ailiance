# tests/test_gateway.py
import pytest
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
    # Tower Ollama mascarade aliases all share port 8004 (tunnel target).
    assert MODEL_FORCE_MAP["ailiance-kicad"] == 8004
    assert MODEL_FORCE_MAP["ailiance-embed"] == 8004
    # ailiance-apertus is preserved as legacy alias → routes to mistral-medium.
    assert MODEL_FORCE_MAP.get("ailiance-apertus") == 9301
    # Qwen is reached via the autossh tunnel on the gateway host (port 8002).
    assert MODEL_FORCE_MAP["ailiance-qwen"] == 8002
    # Gemma sits on tower:9304.
    assert MODEL_FORCE_MAP["ailiance-gemma"] == 9304
