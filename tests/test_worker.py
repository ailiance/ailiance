import pytest
from fastapi.testclient import TestClient


def test_worker_health():
    from src.worker.server import make_worker_app
    from src.worker.runtime import WorkerConfig

    cfg = WorkerConfig(
        model_path="/tmp/fake",
        adapters_dir="/tmp/fake",
        domains=["python"],
        port=9202,
    )
    app = make_worker_app(cfg, skip_model_load=True)
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_worker_metrics():
    from src.worker.server import make_worker_app
    from src.worker.runtime import WorkerConfig

    cfg = WorkerConfig(
        model_path="/tmp/fake",
        adapters_dir="/tmp/fake",
        domains=["python"],
    )
    app = make_worker_app(cfg, skip_model_load=True)
    client = TestClient(app)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "ailiance_worker_requests_total" in resp.text
