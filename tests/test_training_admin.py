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


def test_start_with_explicit_domains():
    orch = FakeOrch()
    c = _client(orch)
    r = c.post("/admin/training/start", json={"domains": ["foo", "bar"]},
               headers={"X-Admin-Token": "secret"})
    assert r.status_code == 202
    assert orch.started == ["foo", "bar"]


def test_log_endpoint_tails_studio_log():
    log_text = "\n".join(f"Iter {i}: train loss {1.0 - i*0.01:.3f}" for i in range(1, 51))

    class _Ops:
        async def read_domain_log(self, domain):
            return log_text

    class _Orch:
        _ops = _Ops()
        async def start(self, domains=None): pass
        async def abort(self): pass
        def status(self): return {}

    c = _client(_Orch())
    r = c.get("/admin/training/log/kicad-dsl?tail=5",
              headers={"X-Admin-Token": "secret"})
    assert r.status_code == 200
    lines = r.text.splitlines()
    assert len(lines) == 5
    assert lines[-1].startswith("Iter 50")
    assert r.headers["content-type"].startswith("text/plain")


def test_log_endpoint_token_required():
    class _Orch:
        _ops = None
        async def start(self, domains=None): pass
        async def abort(self): pass
        def status(self): return {}
    c = _client(_Orch())
    assert c.get("/admin/training/log/kicad-dsl").status_code == 401
