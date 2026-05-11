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
    expected_core = {
        "ailiance",
        "ailiance-apertus",
        "ailiance-mistral",
        "ailiance-eurollm",
        "ailiance-gemma",
        "ailiance-qwen",
    }
    assert expected_core.issubset(ids)


def test_gateway_force_map_has_all_workers():
    """MODEL_FORCE_MAP is the single source of truth for force-routing.

    Asserted as subset so adding a new worker alias does not require
    a test edit; only locks the core 5 plus production aliases that
    callers depend on.
    """
    from src.gateway.server import MODEL_FORCE_MAP

    expected_core = {
        "ailiance-apertus",
        "ailiance-mistral",
        "ailiance-eurollm",
        "ailiance-gemma",
        "ailiance-qwen",
    }
    assert expected_core.issubset(set(MODEL_FORCE_MAP))
    # Qwen is reached via the autossh tunnel on the gateway host (port 8002).
    assert MODEL_FORCE_MAP["ailiance-qwen"] == 8002
    # Gemma sits on tower:9304.
    assert MODEL_FORCE_MAP["ailiance-gemma"] == 9304
