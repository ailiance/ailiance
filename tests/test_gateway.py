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
    ids = [m["id"] for m in models]
    # 5 production workers + the bare "ailiance" auto-router alias.
    assert "ailiance" in ids
    assert "ailiance-apertus" in ids
    assert "ailiance-devstral" in ids
    assert "ailiance-eurollm" in ids
    assert "ailiance-gemma" in ids
    assert "ailiance-qwen" in ids


def test_gateway_force_map_has_all_workers():
    """MODEL_FORCE_MAP is the single source of truth for force-routing."""
    from src.gateway.server import MODEL_FORCE_MAP

    assert set(MODEL_FORCE_MAP) == {
        "ailiance-apertus",
        "ailiance-devstral",
        "ailiance-eurollm",
        "ailiance-gemma",
        "ailiance-qwen",
    }
    # Qwen is reached via the autossh tunnel on the gateway host (port 8002).
    assert MODEL_FORCE_MAP["ailiance-qwen"] == 8002
    # Gemma sits on tower:9304.
    assert MODEL_FORCE_MAP["ailiance-gemma"] == 9304
