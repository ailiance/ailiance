"""Auto-router tests — chain policy auto-engages when ``req.model``
hits the bare ``ailiance`` alias (router-driven path, no MODEL_FORCE_MAP
entry) AND the YAML default policy for the classified domain is
non-DIRECT. Forced aliases (``ailiance-mistral`` etc.) keep the legacy
1-shot behaviour unless the caller explicitly opts in.

Critical contract for first production client (electron-rare):
``model="ailiance"`` + a KiCad/SPICE/freecad prompt MUST run through
the deliberation chain without any extra_body opt-in.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.orchestrator.chain_orchestrator import ChainOrchestrator
from src.orchestrator.validators import StubValidator


class _FakeRouter:
    """Minimal classifier stub returning a fixed (domain, score) list."""

    def __init__(self, domain: str, score: float = 0.99) -> None:
        self._domain = domain
        self._score = score

    def route(self, _prompt: str):
        return [(self._domain, self._score)]


def _build_orch(audit_dir: Path) -> ChainOrchestrator:
    async def fake_llm(messages, model: str) -> str:
        return "draft-output"

    return ChainOrchestrator(
        policies_path=Path("configs/chain_policies.yaml"),
        reflector_path=Path("configs/reflector_prompts.yaml"),
        validator=StubValidator(),
        llm_call=fake_llm,
        audit_dir=audit_dir,
    )


def test_fc_force_route_falls_back_to_omlx_when_primary_down(tmp_path: Path) -> None:
    """tools[] force-routes to the FC backend (8002). When 8002 is unhealthy,
    the gateway must fall back to omlx Qwen3-Coder-30B (valid tool_calls) and
    rewrite the model field — instead of hanging on the dead backend."""
    from src.gateway import server as gw

    app = gw.make_gateway_app(skip_router_load=True)
    app.state.router = _FakeRouter(domain="math-gsm8k")  # DIRECT, non-FC port
    app.state.orchestrator = _build_orch(tmp_path)
    client = TestClient(app)

    captured: dict = {}

    class _FakeResp:
        status_code = 200
        content = b"{}"

        def json(self):
            return {"id": "x", "choices": [{"message": {"content": "ok"}}]}

    async def fake_post(self, *args, **kwargs):
        captured["url"] = args[0] if args else kwargs.get("url")
        captured["model"] = (kwargs.get("json") or {}).get("model")
        return _FakeResp()

    tools = [{"type": "function", "function": {
        "name": "ls", "description": "list", "parameters": {"type": "object", "properties": {}}}}]

    # Primary FC backend (8002) down; omlx (8500) healthy.
    with patch.object(gw, "_healthy_ports", {gw.OMLX_PORT, gw.HEALTH_FALLBACK_PORT}), \
         patch("httpx.AsyncClient.post", new=fake_post):
        resp = client.post("/v1/chat/completions", json={
            "model": "ailiance",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": tools,
        })

    assert resp.status_code == 200
    assert str(gw.OMLX_PORT) in str(captured["url"])      # forwarded to omlx :8500
    assert captured["model"] == gw.FC_FALLBACK_MODEL        # model rewritten to Qwen3-Coder


def test_fc_force_route_uses_primary_when_healthy(tmp_path: Path) -> None:
    """When 8002 is healthy, tools[] still pins to the primary FC backend and
    does NOT rewrite the model to the omlx fallback."""
    from src.gateway import server as gw

    app = gw.make_gateway_app(skip_router_load=True)
    app.state.router = _FakeRouter(domain="math-gsm8k")
    app.state.orchestrator = _build_orch(tmp_path)
    client = TestClient(app)

    captured: dict = {}

    class _FakeResp:
        status_code = 200
        content = b"{}"

        def json(self):
            return {"id": "x", "choices": [{"message": {"content": "ok"}}]}

    async def fake_post(self, *args, **kwargs):
        captured["url"] = args[0] if args else kwargs.get("url")
        captured["model"] = (kwargs.get("json") or {}).get("model")
        return _FakeResp()

    tools = [{"type": "function", "function": {
        "name": "ls", "description": "list", "parameters": {"type": "object", "properties": {}}}}]

    with patch.object(gw, "_healthy_ports", {gw.FC_FORCE_ROUTE_PORT, gw.OMLX_PORT}), \
         patch("httpx.AsyncClient.post", new=fake_post):
        resp = client.post("/v1/chat/completions", json={
            "model": "ailiance",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": tools,
        })

    assert resp.status_code == 200
    assert str(gw.FC_FORCE_ROUTE_PORT) in str(captured["url"])  # pinned to primary
    assert captured["model"] != gw.FC_FALLBACK_MODEL            # no omlx rewrite


def test_auto_engages_for_ailiance_alias_on_deliberate_domain(
    tmp_path: Path,
) -> None:
    """model='ailiance' + KiCad prompt + no extra_body → DELIBERATE
    auto-engages from the YAML map. ailiance_chain.auto_engaged=True
    so observability can distinguish from explicit opt-in."""
    from src.gateway import server as gw

    app = gw.make_gateway_app(skip_router_load=True)
    app.state.router = _FakeRouter(domain="kicad-pcb")
    app.state.orchestrator = _build_orch(tmp_path)
    client = TestClient(app)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "ailiance",
            "messages": [
                {"role": "user", "content": "draw me a power supply pcb"}
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["ailiance_chain"]["policy"] == "deliberate"
    assert data["ailiance_chain"]["auto_engaged"] is True
    assert data["ailiance_chain"]["status"] == "ok"


def test_auto_does_not_engage_for_forced_alias(tmp_path: Path) -> None:
    """model='ailiance-mistral' bypasses the classifier entirely, so
    the auto-router has no domain to look up. Legacy proxy path runs
    even though kicad-pcb policy in YAML is deliberate."""
    from src.gateway import server as gw

    app = gw.make_gateway_app(skip_router_load=True)
    app.state.router = _FakeRouter(domain="kicad-pcb")
    app.state.orchestrator = _build_orch(tmp_path)
    client = TestClient(app)

    class _FakeResp:
        status_code = 200
        content = b"{}"

        def json(self):
            return {
                "id": "chatcmpl-x",
                "choices": [{"message": {"content": "from-worker"}}],
            }

    async def fake_post(self, *args, **kwargs):
        return _FakeResp()

    with patch("httpx.AsyncClient.post", new=fake_post):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "ailiance-mistral",
                "messages": [
                    {"role": "user", "content": "draw me a power supply pcb"}
                ],
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    # Legacy proxy path: no ailiance_chain envelope.
    assert "ailiance_chain" not in body
    assert body["choices"][0]["message"]["content"] == "from-worker"


def test_auto_does_not_engage_for_direct_domain(tmp_path: Path) -> None:
    """A classified domain whose YAML policy is 'direct' must NOT enter
    the orchestrator. e.g. math-gsm8k → direct."""
    from src.gateway import server as gw

    app = gw.make_gateway_app(skip_router_load=True)
    app.state.router = _FakeRouter(domain="math-gsm8k")
    app.state.orchestrator = _build_orch(tmp_path)
    client = TestClient(app)

    class _FakeResp:
        status_code = 200
        content = b"{}"

        def json(self):
            return {
                "id": "chatcmpl-x",
                "choices": [{"message": {"content": "42"}}],
            }

    async def fake_post(self, *args, **kwargs):
        return _FakeResp()

    with patch("httpx.AsyncClient.post", new=fake_post):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "ailiance",
                "messages": [
                    {"role": "user", "content": "what is 6 times 7?"}
                ],
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert "ailiance_chain" not in body


def test_auto_silently_degrades_to_direct_on_stream(
    tmp_path: Path,
) -> None:
    """stream=true + ailiance + deliberate-domain must NOT 400. The
    user did not opt in, so we silently fall back to legacy SSE proxy.
    Only explicit extra_body.chain_policy on a stream returns 400."""
    from src.gateway import server as gw

    app = gw.make_gateway_app(skip_router_load=True)
    app.state.router = _FakeRouter(domain="kicad-pcb")
    app.state.orchestrator = _build_orch(tmp_path)
    client = TestClient(app)

    # Patch the streaming path so we can assert it was reached.
    captured = {"streamed": False}

    class _FakeStreamResp:
        status_code = 200
        headers = {"content-type": "text/event-stream"}

        async def aiter_raw(self):
            captured["streamed"] = True
            yield b"data: {}\n\n"

        async def aiter_text(self):
            # Gateway switched to aiter_text() in PR #81 (SSE normalizer);
            # the test mock keeps aiter_raw for back-compat callers but the
            # relay now consumes this method.
            captured["streamed"] = True
            yield "data: {}\n\n"

        async def aclose(self):
            return None

    def fake_build_request(self, *args, **kwargs):
        return object()

    async def fake_send(self, request, **kwargs):
        return _FakeStreamResp()

    with patch("httpx.AsyncClient.build_request", new=fake_build_request), \
         patch("httpx.AsyncClient.send", new=fake_send):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "ailiance",
                "messages": [
                    {"role": "user", "content": "stream me a pcb"}
                ],
                "stream": True,
            },
        )
    assert resp.status_code == 200
    assert captured["streamed"] is True


def test_metrics_distinguish_chain_proxy_and_auto(tmp_path: Path) -> None:
    """Critic M1: ailiance_gw_requests_total must label chain vs proxy
    and auto vs explicit, otherwise the auto-router activation hides a
    multi-second latency regression behind a single counter."""
    from src.gateway import server as gw

    app = gw.make_gateway_app(skip_router_load=True)
    app.state.router = _FakeRouter(domain="kicad-pcb")
    app.state.orchestrator = _build_orch(tmp_path)
    client = TestClient(app)

    # Auto-engaged chain.
    client.post(
        "/v1/chat/completions",
        json={
            "model": "ailiance",
            "messages": [{"role": "user", "content": "design a pcb"}],
        },
    )

    # Proxy path on a forced alias.
    class _FakeResp:
        status_code = 200
        content = b"{}"

        def json(self):
            return {"id": "x", "choices": [{"message": {"content": "ok"}}]}

    async def fake_post(self, *args, **kwargs):
        return _FakeResp()

    with patch("httpx.AsyncClient.post", new=fake_post):
        client.post(
            "/v1/chat/completions",
            json={
                "model": "ailiance-mistral",
                "messages": [{"role": "user", "content": "x"}],
            },
        )

    metrics = client.get("/metrics").text
    # Chain path with auto=1 must be present.
    assert (
        'ailiance_gw_requests_total{auto="1",model="ailiance",path="chain",status="200"}'
        in metrics
    ), metrics
    # Proxy path with auto=0 must be present (forced alias call).
    assert (
        'path="proxy"' in metrics and 'auto="0"' in metrics
    ), metrics


def test_explicit_extra_body_still_overrides_for_forced_alias(
    tmp_path: Path,
) -> None:
    """Backward-compat: a forced alias + explicit extra_body must
    still engage the chain — the existing v0.3.0 contract is
    preserved by the auto-router refactor."""
    from src.gateway import server as gw

    app = gw.make_gateway_app(skip_router_load=True)
    app.state.router = _FakeRouter(domain="kicad-pcb")
    app.state.orchestrator = _build_orch(tmp_path)
    client = TestClient(app)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "ailiance-mistral",
            "messages": [{"role": "user", "content": "x"}],
            "extra_body": {
                "chain_policy": "deliberate",
                "include_audit": True,
            },
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ailiance_chain"]["policy"] == "deliberate"
    # Forced alias path: domain="" → orch falls back to "_default" so
    # auto_engaged is False (this was an explicit opt-in).
    assert body["ailiance_chain"]["auto_engaged"] is False
