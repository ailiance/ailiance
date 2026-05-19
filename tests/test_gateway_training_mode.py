import os

os.environ.setdefault("AILIANCE_ADMIN_TOKEN", "test-token")

from src.gateway.server import make_gateway_app
from fastapi.testclient import TestClient


def test_app_exposes_training_orchestrator():
    app = make_gateway_app(skip_router_load=True)
    assert hasattr(app.state, "training")
    client = TestClient(app)
    # admin router mounted: 401 (no token) rather than 404 (not found)
    assert client.get("/admin/training/status").status_code == 401
