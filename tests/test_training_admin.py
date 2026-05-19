import os

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.gateway.training.admin import make_training_router


class FakeOrch:
    def __init__(self):
        self.started = None

    async def start(self, domains=None):
        self.started = domains or ["kicad-dsl"]

    async def abort(self):
        self.started = None

    def status(self):
        return {"status": "IDLE"}


def _client(orch, token="secret"):
    os.environ["AILIANCE_ADMIN_TOKEN"] = token
    app = FastAPI()
    app.state.training = orch
    app.include_router(make_training_router())
    return TestClient(app)


def test_status_requires_token():
    c = _client(FakeOrch())
    assert c.get("/admin/training/status").status_code == 401
    ok = c.get("/admin/training/status", headers={"X-Admin-Token": "secret"})
    assert ok.status_code == 200
    assert ok.json()["status"] == "IDLE"


def test_start_triggers_orchestrator():
    orch = FakeOrch()
    c = _client(orch)
    r = c.post("/admin/training/start", json={},
               headers={"X-Admin-Token": "secret"})
    assert r.status_code == 202
    assert orch.started is not None


def test_admin_disabled_without_env(monkeypatch):
    monkeypatch.delenv("AILIANCE_ADMIN_TOKEN", raising=False)
    app = FastAPI()
    app.state.training = FakeOrch()
    app.include_router(make_training_router())
    c = TestClient(app)
    assert c.get("/admin/training/status",
                 headers={"X-Admin-Token": "x"}).status_code == 503
