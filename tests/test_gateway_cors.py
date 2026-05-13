"""Tests for CORS middleware on the gateway.

The Playground (cockpit-public on www.ailiance.fr) issues
multipart POST against gateway.ailiance.fr. Browsers preflight
those requests; a missing CORSMiddleware returns 405 and the
browser surfaces "Load failed" without sending the real POST.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from src.gateway.server import make_gateway_app

    return TestClient(make_gateway_app(skip_router_load=True))


class TestPreflight:
    def test_options_files_extract_returns_cors_allow(self, client):
        resp = client.options(
            "/v1/files/extract",
            headers={
                "Origin": "https://www.ailiance.fr",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        # Starlette CORSMiddleware returns 200 on preflight when origin
        # is allow-listed; 400 (or 405 from FastAPI) means CORS isn't
        # wired.
        assert resp.status_code == 200, f"unexpected status {resp.status_code}"
        assert resp.headers["access-control-allow-origin"] == "https://www.ailiance.fr"
        assert "POST" in resp.headers["access-control-allow-methods"]

    def test_options_chat_completions_returns_cors_allow(self, client):
        resp = client.options(
            "/v1/chat/completions",
            headers={
                "Origin": "https://www.ailiance.fr",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert resp.status_code == 200
        assert resp.headers["access-control-allow-origin"] == "https://www.ailiance.fr"

    def test_options_preview_origin_allowed(self, client):
        resp = client.options(
            "/v1/files/extract",
            headers={
                "Origin": "https://preview.ailiance.fr",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert resp.status_code == 200
        assert resp.headers["access-control-allow-origin"] == "https://preview.ailiance.fr"

    def test_options_unknown_origin_rejected(self, client):
        # Unlisted origins must not get the allow header, which makes
        # the browser block the request as it should.
        resp = client.options(
            "/v1/files/extract",
            headers={
                "Origin": "https://evil.example.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert resp.headers.get("access-control-allow-origin") != "https://evil.example.com"


class TestActualRequest:
    def test_get_models_returns_cors_for_known_origin(self, client):
        resp = client.get(
            "/v1/models",
            headers={"Origin": "https://www.ailiance.fr"},
        )
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "https://www.ailiance.fr"


class TestEnvOverride:
    def test_custom_origin_via_env(self, monkeypatch):
        from src.gateway import server as gw

        monkeypatch.setenv("AILIANCE_CORS_ORIGINS", "https://x.example,https://y.example")
        origins = gw._cors_origins()
        assert origins == ["https://x.example", "https://y.example"]
