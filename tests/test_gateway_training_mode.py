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


from unittest.mock import patch

from src.gateway.training.state import CampaignState
from src.gateway.training.studio_ops import MINIMAL_ROUTABLE_PORTS
from src.gateway.server import MODEL_FORCE_MAP


def _pick_unloaded_alias():
    """An explicitly force-mapped alias whose port is NOT a minimal port."""
    for alias, port in MODEL_FORCE_MAP.items():
        if port not in MINIMAL_ROUTABLE_PORTS:
            return alias, port
    raise AssertionError("no force-mapped non-minimal alias found")


def test_explicit_unloaded_alias_returns_503():
    app = make_gateway_app(skip_router_load=True)
    alias, port = _pick_unloaded_alias()
    app.state.training.state = CampaignState(
        status="TRAINING", domains=["kicad-dsl"], unloaded_ports=[port],
        phase=2, iter=10, iter_total=800)
    client = TestClient(app)
    resp = client.post("/v1/chat/completions",
                        json={"model": alias,
                              "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "training_in_progress"


def test_no_503_when_campaign_idle():
    app = make_gateway_app(skip_router_load=True)
    alias, port = _pick_unloaded_alias()
    # campaign IDLE -> state.is_active is False -> no interception
    client = TestClient(app)

    class _Resp:
        status_code = 200
        content = b"{}"
        def json(self):
            return {"id": "x", "choices": [{"message": {"content": "ok"}}]}

    async def fake_post(self, *a, **k):
        return _Resp()

    with patch("httpx.AsyncClient.post", new=fake_post):
        resp = client.post("/v1/chat/completions",
                           json={"model": alias,
                                 "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code != 503
