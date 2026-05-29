import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.gateway.gaia_x.serving import mount_well_known


def test_mount_serves_did_document(tmp_path):
    wk = tmp_path / "well-known"
    wk.mkdir()
    (wk / "did.json").write_text(json.dumps({"id": "did:web:ailiance.fr"}))
    app = FastAPI()
    mounted = mount_well_known(app, wk)
    assert mounted is True
    client = TestClient(app)
    resp = client.get("/.well-known/did.json")
    assert resp.status_code == 200
    assert resp.json()["id"] == "did:web:ailiance.fr"


def test_mount_noop_when_dir_absent(tmp_path):
    app = FastAPI()
    assert mount_well_known(app, tmp_path / "does-not-exist") is False
